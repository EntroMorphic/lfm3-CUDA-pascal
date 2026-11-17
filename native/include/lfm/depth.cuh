#pragma once
// Native LFM3-Omni depth decoder (HF `Lfm3OmniDepthDecoder`): from the language model's hidden state and a frame's
// revealed codes, the 2017-way logits of each of the frame's 12 codebooks. #2 step 3, quality bars A1/B1
// (journal/lfm3-tts-quality-bar.md).
//
// Per frame: depth_linear projects the hidden state (2048) to 12 slots of 1024, once (HF recomputes it every pass;
// the input is the same, so is the result). Per pass: each slot adds its codebook's code embedding, or the mask
// embedding while hidden; a 3-layer bidirectional LFM3 trunk (full attention, 32 query / 8 KV heads of 32, FFN
// 2816, no sandwich norm) and its final norm; then each requested codebook's head: RMSNorm with the codebook's own
// weight, and logits against its tied embedding [2017, 1024].
//
// Precision as the codec: FP32 activations over the stored BF16 weights (lfm/fp32_ops.cuh), rotary in FP64.
//
// Batched (concurrent speech, validation/speech-batch-v1): up to `max_frames` frames at once, one per request. The
// row-wise operators (linear, norms, elementwise) run over every frame's 12 slots in one call; those that couple a
// frame's slots (slot inputs, rotary positions 0..11, attention, the heads' per-codebook norm) run per frame on its
// own slots. The FP32 operators compute each row the same way whatever the row count, so a frame's logits do not
// depend on the frames beside it: begin_frame/pass are the batch of one.
#include <lfm/fp32_ops.cuh>
#include <lfm/safetensors.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm::depth {

constexpr int kModelHidden = 2048, kCodebooks = 12, kCodes = 2017, kHidden = 1024, kFfn = 2816, kLayers = 3;
constexpr int kHeads = 32, kKvHeads = 8, kHeadDim = 32;
constexpr double kRopeTheta = 1e6;
constexpr float kNormEps = 1e-5f;

namespace kernels {
using bf16 = __nv_bfloat16;

__global__ void widen(const bf16* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __bfloat162float(x[i]);
}

struct Tables { const bf16* embedding[kCodebooks]; };
// slots[c] = projected[c] + (masked[c] ? mask_embedding : embedding_c[code[c]])
__global__ void slot_inputs(const float* projected, Tables tables, const bf16* mask_embedding, const int32_t* state,
                            float* slots) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCodebooks * kHidden) return;
  const int c = i / kHidden, d = i % kHidden, code = state[c], masked = state[kCodebooks + c];
  const bf16 e = masked ? mask_embedding[d] : tables.embedding[c][size_t(code) * kHidden + d];
  slots[i] = __fadd_rn(projected[i], __bfloat162float(e));
}

__global__ void add(float* a, const float* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __fadd_rn(a[i], b[i]);
}

// One warp per (slot, query head), a lane per head dimension: bidirectional softmax(q.k * 32^-0.5) over the 12
// slots, times V, times sigmoid(gate). Query head h reads KV head h / 4.
__global__ void attention(const float* q, const float* k, const float* v, const float* gate, float* out) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= kCodebooks * kHeads) return;
  const int t = warp / kHeads, h = warp % kHeads, kv = h / (kHeads / kKvHeads);
  const float qv = q[size_t(warp) * kHeadDim + lane], scale = 0.17677669529663687f;  // float(32 ** -0.5), as HF
  float scores[kCodebooks], top = -INFINITY;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) {
    float dot = __fmul_rn(qv, k[(size_t(s) * kKvHeads + kv) * kHeadDim + lane]);
    for (int o = 16; o; o /= 2) dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, o));
    scores[s] = __fmul_rn(__shfl_sync(0xffffffffu, dot, 0), scale);
    top = fmaxf(top, scores[s]);
  }
  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) total = __fadd_rn(total, scores[s] = expf(__fsub_rn(scores[s], top)));
  float o = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s)
    o = __fmaf_rn(__fdiv_rn(scores[s], total), v[(size_t(s) * kKvHeads + kv) * kHeadDim + lane], o);
  const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate[size_t(t) * kHeads + h])));
  out[size_t(t) * kHeads * kHeadDim + h * kHeadDim + lane] = __fmul_rn(o, g);
}

// Batched forms (one launch for every frame), each element computed exactly as the one-frame kernels compute it.
// attention over n frames: frame f's slots are rows [12f, 12f + 12) of q, k, v, gate and out.
__global__ void attention_frames(const float* q, const float* k, const float* v, const float* gate, float* out, int n) {
  const int all = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (all >= n * kCodebooks * kHeads) return;
  const int f = all / (kCodebooks * kHeads), warp = all % (kCodebooks * kHeads), lane = threadIdx.x % 32;
  const size_t s0 = size_t(f) * kCodebooks;
  q += s0 * kHeads * kHeadDim, k += s0 * kKvHeads * kHeadDim, v += s0 * kKvHeads * kHeadDim, gate += s0 * kHeads;
  out += s0 * kHeads * kHeadDim;
  const int t = warp / kHeads, h = warp % kHeads, kv = h / (kHeads / kKvHeads);
  const float qv = q[size_t(warp) * kHeadDim + lane], scale = 0.17677669529663687f;
  float scores[kCodebooks], top = -INFINITY;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) {
    float dot = __fmul_rn(qv, k[(size_t(s) * kKvHeads + kv) * kHeadDim + lane]);
    for (int o = 16; o; o /= 2) dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, o));
    scores[s] = __fmul_rn(__shfl_sync(0xffffffffu, dot, 0), scale);
    top = fmaxf(top, scores[s]);
  }
  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s) total = __fadd_rn(total, scores[s] = expf(__fsub_rn(scores[s], top)));
  float o = 0.0f;
#pragma unroll
  for (int s = 0; s < kCodebooks; ++s)
    o = __fmaf_rn(__fdiv_rn(scores[s], total), v[(size_t(s) * kKvHeads + kv) * kHeadDim + lane], o);
  const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate[size_t(t) * kHeads + h])));
  out[size_t(t) * kHeads * kHeadDim + h * kHeadDim + lane] = __fmul_rn(o, g);
}

// fp32::rotary's angles for positions 0..11 and the 16 pairs of a 32-wide head, computed once on the device by
// fp32::kernels::rotary's own expressions: table[(t * 16 + d) * 2] = sin, [+ 1] = cos.
__global__ void rotary_table(double* table, double theta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCodebooks * (kHeadDim / 2)) return;
  const int t = i / (kHeadDim / 2), d = i % (kHeadDim / 2);
  const double inverse = 1.0 / pow(theta, double(2 * d) / kHeadDim);
  double sn, cs;
  sincos(double(0 + t) * inverse, &sn, &cs);
  table[2 * i] = sn, table[2 * i + 1] = cs;
}
// rotary over n frames' slots ([n * 12 rows, heads, 32]): row r at position r % 12, as fp32::rotary with position0 0
// per frame (its a * c - b * s and b * c + a * s, rounded once each step, built without contraction).
__global__ void rotary_frames(float* x, int rows, int heads, const double* table) {
  constexpr int half = kHeadDim / 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * heads * half) return;
  const int d = i % half, head_row = i / half, t = (head_row / heads) % kCodebooks;
  const double sn = table[2 * (t * half + d)], cs = table[2 * (t * half + d) + 1];
  float* v = x + size_t(head_row) * kHeadDim;
  const double a = v[d], b = v[d + half];
  v[d] = float(__dsub_rn(__dmul_rn(a, cs), __dmul_rn(b, sn)));
  v[d + half] = float(__dadd_rn(__dmul_rn(b, cs), __dmul_rn(a, sn)));
}

// The heads' norm over n frames: row r uses codebook r % 12's weight (fp32::kernels::rms_norm's arithmetic).
__global__ void head_norm_frames(const float* x, const bf16* w, float* y, int rows, int width, float eps) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  const float* xr = x + size_t(row) * width;
  const bf16* wr = w + size_t(row % kCodebooks) * width;
  float sum = 0.0f;
  for (int d = lane; d < width; d += 32) sum = __fmaf_rn(xr[d], xr[d], sum);
  for (int o = 16; o; o /= 2) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, o));
  sum = __shfl_sync(0xffffffffu, sum, 0);
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(width)), eps)));
  for (int d = lane; d < width; d += 32) y[size_t(row) * width + d] = __fmul_rn(__bfloat162float(wr[d]), __fmul_rn(xr[d], inv));
}

// Gather rows `index[i]` of x (width wide) into y's row i; scatter is the reverse.
__global__ void gather_rows(const float* x, const int* index, float* y, int count, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(count) * width) return;
  y[i] = x[size_t(index[i / width]) * width + i % width];
}
__global__ void scatter_rows(const float* x, const int* index, float* y, int count, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(count) * width) return;
  y[size_t(index[i / width]) * width + i % width] = x[i];
}
}  // namespace kernels

class Decoder {
 public:
  using bf16 = __nv_bfloat16;

  // `weights`: the main checkpoint (model.safetensors), shared with the text engine when both run. `max_frames`: the
  // most frames one begin_frames/pass_frames call takes.
  explicit Decoder(std::shared_ptr<Safetensors> weights, int max_frames = 1)
      : weights_(std::move(weights)), max_frames_(max_frames) {
    if (max_frames < 1) throw std::runtime_error("depth: max_frames must be positive");
    const size_t F = size_t(max_frames), S = F * kCodebooks;
    state_ = fp32::DeviceArray<int32_t>(F * 2 * kCodebooks);
    hidden_ = fp32::DeviceArray<float>(F * kModelHidden);
    projected_ = fp32::DeviceArray<float>(S * kHidden);
    x_ = fp32::DeviceArray<float>(S * kHidden);
    normed_ = fp32::DeviceArray<float>(S * kHidden);
    q_ = fp32::DeviceArray<float>(S * kHeads * kHeadDim);
    k_ = fp32::DeviceArray<float>(S * kKvHeads * kHeadDim);
    v_ = fp32::DeviceArray<float>(S * kKvHeads * kHeadDim);
    gate_ = fp32::DeviceArray<float>(S * kHeads);
    attended_ = fp32::DeviceArray<float>(S * kHeads * kHeadDim);
    out_ = fp32::DeviceArray<float>(S * kHidden);
    ffn_a_ = fp32::DeviceArray<float>(S * kFfn);
    ffn_b_ = fp32::DeviceArray<float>(S * kFfn);
    head_in_ = fp32::DeviceArray<float>(S * kHidden);
    gathered_ = fp32::DeviceArray<float>(S * kHidden);
    gathered_logits_ = fp32::DeviceArray<float>(S * kCodes);
    index_ = fp32::DeviceArray<int>(S);
    rotary_ = fp32::DeviceArray<double>(size_t(kCodebooks) * kHeadDim);
    kernels::rotary_table<<<1, 256>>>(rotary_.p, kRopeTheta);
    fp32::check(cudaGetLastError(), "depth rotary table");
    const std::string p = "model.depth_decoder.";
    depth_linear_ = tensor(p + "depth_linear.weight", {kCodebooks * kHidden, kModelHidden});
    depth_linear_bias_ = tensor(p + "depth_linear.bias", {kCodebooks * kHidden});
    mask_embedding_ = tensor(p + "mask_embedding", {kHidden});
    final_norm_ = tensor(p + "model.embedding_norm.weight", {kHidden});
    std::vector<bf16> head_norms(size_t(kCodebooks) * kHidden);
    for (int c = 0; c < kCodebooks; ++c) {
      const std::string e = p + "depth_embeddings." + std::to_string(c) + ".";
      tables_.embedding[c] = tensor(e + "embedding.weight", {kCodes, kHidden});
      fp32::check(cudaMemcpy(head_norms.data() + size_t(c) * kHidden, tensor(e + "embedding_norm.weight", {kHidden}),
                             kHidden * sizeof(bf16), cudaMemcpyDeviceToHost), "depth head norms");
    }
    head_norms_ = fp32::DeviceArray<bf16>(head_norms.size());  // contiguous, one row per codebook (24 KiB)
    fp32::check(cudaMemcpy(head_norms_.p, head_norms.data(), head_norms.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                "depth head norms");
    for (int i = 0; i < kLayers; ++i) {
      const std::string l = p + "model.layers." + std::to_string(i) + ".";
      Layer& y = layers_[i];
      y.operator_norm = tensor(l + "operator_norm.weight", {kHidden});
      y.ffn_norm = tensor(l + "ffn_norm.weight", {kHidden});
      y.q = tensor(l + "self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden});
      y.k = tensor(l + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden});
      y.v = tensor(l + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden});
      y.o = tensor(l + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim});
      y.gate = tensor(l + "self_attn.gate_proj.weight", {kHeads, kHidden});
      y.q_norm = tensor(l + "self_attn.q_norm.weight", {kHeadDim});
      y.k_norm = tensor(l + "self_attn.k_norm.weight", {kHeadDim});
      y.w1 = tensor(l + "feed_forward.w1.weight", {kFfn, kHidden});
      y.w3 = tensor(l + "feed_forward.w3.weight", {kFfn, kHidden});
      y.w2 = tensor(l + "feed_forward.w2.weight", {kHidden, kFfn});
    }
    if (weights_->has(p + "model.layers.3.operator_norm.weight")) throw std::runtime_error("depth: expected 3 layers");
  }

  // Start a frame from the language model's hidden state (BF16 [2048], on the device).
  void begin_frame(const bf16* hidden) { begin_frames(&hidden, 1); }

  // One pass: `codes` and `masked` (host, 12 each). Writes the FP32 logits of the codebooks c with score[c] set to
  // logits + c * 2017 (device). The others are not computed.
  void pass(const int32_t* codes, const bool* masked, const bool* score, float* logits) {
    pass_frames(1, codes, masked, score, logits);
  }

  // Start `n` frames, frame f from hidden[f] (each BF16 [2048], on the device).
  void begin_frames(const bf16* const* hidden, int n) {
    if (n < 1 || n > max_frames_) throw std::runtime_error("depth: frame count out of range");
    for (int f = 0; f < n; ++f)
      kernels::widen<<<fp32::blocks(kModelHidden), 256>>>(hidden[f], hidden_.p + size_t(f) * kModelHidden, kModelHidden);
    linear(hidden_.p, depth_linear_, depth_linear_bias_, projected_.p, n, kCodebooks * kHidden, kModelHidden);
    frames_ = n;
  }

  // One pass over the frames begun: frame f's `codes`, `masked` and `score` at [f * 12, f * 12 + 12) (host). Writes
  // frame f's codebook c logits to logits + (f * 12 + c) * 2017 (device) where score is set; the others are not
  // computed.
  void pass_frames(int n, const int32_t* codes, const bool* masked, const bool* score, float* logits) {
    if (n != frames_) throw std::runtime_error("depth: pass over a different frame count than begun");
    std::vector<int32_t> state(size_t(n) * 2 * kCodebooks);
    for (int f = 0; f < n; ++f)
      for (int c = 0; c < kCodebooks; ++c) {
        const int i = f * kCodebooks + c;
        if (codes[i] < 0 || codes[i] >= kCodes) throw std::runtime_error("depth: code out of range [0, 2017)");
        state[size_t(f) * 2 * kCodebooks + c] = codes[i];
        state[size_t(f) * 2 * kCodebooks + kCodebooks + c] = masked[i];
      }
    fp32::check(cudaMemcpy(state_.p, state.data(), state.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "depth state");
    // the heads' rows, uploaded before the trunk's kernels (validation/speech-speed-v1: a copy after them waited for
    // them, and the device idled until the next launch): per codebook, the frames that score it
    std::vector<int> index;
    std::vector<std::pair<int, int>> spans;  // per codebook scored: first gathered row, count
    for (int c = 0; c < kCodebooks; ++c) {
      const int first = int(index.size());
      for (int f = 0; f < n; ++f)
        if (score[f * kCodebooks + c]) index.push_back(f * kCodebooks + c);
      spans.emplace_back(first, int(index.size()) - first);
    }
    if (!index.empty())
      fp32::check(cudaMemcpy(index_.p, index.data(), index.size() * sizeof(int), cudaMemcpyHostToDevice), "depth heads");
    const int R = n * kCodebooks;
    for (int f = 0; f < n; ++f) {
      const size_t at = size_t(f) * kCodebooks * kHidden;
      kernels::slot_inputs<<<fp32::blocks(kCodebooks * kHidden), 256>>>(projected_.p + at, tables_, mask_embedding_,
                                                                        state_.p + size_t(f) * 2 * kCodebooks, x_.p + at);
    }
    for (const Layer& y : layers_) {
      fp32::rms_norm(x_.p, y.operator_norm, normed_.p, nullptr, R, kHidden, kNormEps);
      float *q = q_.p, *k = k_.p, *v = v_.p, *gate = gate_.p;
      linear(normed_.p, y.q, nullptr, q, R, kHeads * kHeadDim, kHidden);
      linear(normed_.p, y.k, nullptr, k, R, kKvHeads * kHeadDim, kHidden);
      linear(normed_.p, y.v, nullptr, v, R, kKvHeads * kHeadDim, kHidden);
      linear(normed_.p, y.gate, nullptr, gate, R, kHeads, kHidden);
      fp32::rms_norm(q, y.q_norm, q, nullptr, R * kHeads, kHeadDim, kNormEps);
      fp32::rms_norm(k, y.k_norm, k, nullptr, R * kKvHeads, kHeadDim, kNormEps);
      // a frame's slots are positions 0..11 and attend only to each other
      kernels::rotary_frames<<<fp32::blocks(size_t(R) * kHeads * kHeadDim / 2), 256>>>(q, R, kHeads, rotary_.p);
      kernels::rotary_frames<<<fp32::blocks(size_t(R) * kKvHeads * kHeadDim / 2), 256>>>(k, R, kKvHeads, rotary_.p);
      kernels::attention_frames<<<fp32::blocks(size_t(R) * kHeads * 32), 256>>>(q, k, v, gate, attended_.p, n);
      linear(attended_.p, y.o, nullptr, out_.p, R, kHidden, kHeads * kHeadDim);
      kernels::add<<<fp32::blocks(R * kHidden), 256>>>(x_.p, out_.p, R * kHidden);
      fp32::rms_norm(x_.p, y.ffn_norm, normed_.p, nullptr, R, kHidden, kNormEps);
      linear(normed_.p, y.w1, nullptr, ffn_a_.p, R, kFfn, kHidden);
      linear(normed_.p, y.w3, nullptr, ffn_b_.p, R, kFfn, kHidden);
      fp32::swiglu(ffn_a_.p, ffn_b_.p, size_t(R) * kFfn);
      linear(ffn_a_.p, y.w2, nullptr, out_.p, R, kHidden, kFfn);
      kernels::add<<<fp32::blocks(R * kHidden), 256>>>(x_.p, out_.p, R * kHidden);
    }
    fp32::rms_norm(x_.p, final_norm_, normed_.p, nullptr, R, kHidden, kNormEps);
    // the heads' norm: each codebook's own weight (HF Lfm3OmniRMSNorm; in FP32 the same arithmetic as above)
    kernels::head_norm_frames<<<fp32::blocks(size_t(R) * 32), 256>>>(normed_.p, head_norms_.p, head_in_.p, R, kHidden,
                                                                      kNormEps);
    // the heads, one call per codebook over every frame that scores it (its tied embedding read once)
    if (!index.empty()) {
      const int m = int(index.size());
      kernels::gather_rows<<<fp32::blocks(size_t(m) * kHidden), 256>>>(head_in_.p, index_.p, gathered_.p, m, kHidden);
      for (int c = 0; c < kCodebooks; ++c)
        if (spans[size_t(c)].second)
          linear(gathered_.p + size_t(spans[size_t(c)].first) * kHidden, tables_.embedding[c], nullptr,
                 gathered_logits_.p + size_t(spans[size_t(c)].first) * kCodes, spans[size_t(c)].second, kCodes, kHidden);
      kernels::scatter_rows<<<fp32::blocks(size_t(m) * kCodes), 256>>>(gathered_logits_.p, index_.p, logits, m, kCodes);
    }
    fp32::check(cudaGetLastError(), "depth pass");
  }

  size_t buffer_bytes() const {
    return hidden_.bytes() + projected_.bytes() + state_.bytes() + x_.bytes() + normed_.bytes() + q_.bytes() +
           k_.bytes() + v_.bytes() + gate_.bytes() + attended_.bytes() + out_.bytes() + ffn_a_.bytes() +
           ffn_b_.bytes() + head_in_.bytes() + head_norms_.bytes() + gathered_.bytes() + gathered_logits_.bytes() +
           index_.bytes() + rotary_.bytes();
  }

 private:
  // fp32::linear (a 32-row block was tried: the same outputs, but slower on Thor: validation/speech-batch-v1)
  template <class W>
  static void linear(const float* x, const W* w, const bf16* bias, float* y, int M, int N, int K) {
    fp32::linear(x, w, bias, y, M, N, K);
  }
  struct Layer {
    const bf16 *operator_norm, *ffn_norm, *q, *k, *v, *o, *gate, *q_norm, *k_norm, *w1, *w2, *w3;
  };
  const bf16* tensor(const std::string& name, std::vector<int64_t> shape) const {
    const auto& t = weights_->at(name);
    if (t.dtype != "BF16" || t.shape != shape) throw std::runtime_error("depth: unexpected tensor " + name);
    return t.device_data;
  }

  std::shared_ptr<Safetensors> weights_;
  int max_frames_ = 1, frames_ = 0;
  const bf16 *depth_linear_, *depth_linear_bias_, *mask_embedding_, *final_norm_;
  kernels::Tables tables_{};
  Layer layers_[kLayers];
  fp32::DeviceArray<bf16> head_norms_;
  fp32::DeviceArray<int32_t> state_;
  fp32::DeviceArray<float> hidden_, projected_, x_, normed_, q_, k_, v_, gate_, attended_, out_, ffn_a_, ffn_b_,
      head_in_, gathered_, gathered_logits_;
  fp32::DeviceArray<int> index_;
  fp32::DeviceArray<double> rotary_;
};

}  // namespace lfm::depth
