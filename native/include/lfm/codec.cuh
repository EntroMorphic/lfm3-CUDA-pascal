#pragma once
// Native LFM3-Omni codec decoder (audio_tokenizer/, HF `LiquidCodecModel.decode`): 12 FSQ codes per 80 ms frame ->
// 24 kHz audio. #2 step 2, quality bar A2 (journal/lfm3-tts-quality-bar.md).
//
// Precision: every activation is FP32 and every weight is used at its stored BF16 value (exact in FP32). The HF
// reference runs this codec in BF16 (24-33 dB SNR against FP64, validation/codec-v1); the FP32 anchor reaches
// 100-114 dB. Rotary angles and the polar step are computed in FP64. The inverse FFT is a direct DFT against a
// basis built in FP64 (no cuFFT, no cuBLAS).
//
// Streaming: `decode` takes any number of frames at a time. The state carried between calls is the last 6 rows of
// each conv layer's gated input, the last 29 keys and values of each attention layer, and the 960 unfinished
// overlap-add samples. Every kernel computes each row the same way whatever the chunk, so a streamed decode is
// bitwise equal to a whole-sequence one.
#include <lfm/fp32_ops.cuh>
#include <lfm/safetensors.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lfm::codec {

constexpr int kCodebooks = 12, kCodebookSize = 2016, kFsqDim = 4, kLatent = kCodebooks * kFsqDim;  // 48
constexpr int kQuantizer = 256, kHidden = 512, kFfn = 2304, kLayers = 8;
constexpr int kHeads = 8, kKvHeads = 2, kHeadDim = 64, kWindow = 30, kPast = kWindow - 1;  // 29 past + self
constexpr int kConv = 7, kConvPast = kConv - 1;
constexpr int kStack = 6;                     // trunk rows (STFT frames) per codec frame
constexpr int kFft = 1280, kHop = 320, kBins = kFft / 2 + 1, kSpec = 2 * kBins;  // 641 bins, 1282 values
constexpr int kSpecPad = 1288;  // the spectrum's row stride and the DFT's K: 1282 padded with zeros to a multiple of 8
constexpr int kTrim = (kFft - kHop) / 2, kTail = 2 * kTrim;                      // 480, 960
constexpr int kSamplesPerFrame = kStack * kHop;                                  // 1920
constexpr double kRopeTheta = 1e6;
constexpr float kNormEps = 1e-5f;
inline bool is_attention_layer(int i) { return i == 2 || i == 4 || i == 6; }

using fp32::check;
template <class T>
using DeviceArray = fp32::DeviceArray<T>;

// Page-locked host memory, for asynchronous copies.
template <class T>
struct PinnedArray {
  T* p = nullptr;
  PinnedArray() = default;
  explicit PinnedArray(size_t count) { check(cudaMallocHost(&p, count * sizeof(T)), "codec cudaMallocHost"); }
  PinnedArray(PinnedArray&& o) noexcept : p(o.p) { o.p = nullptr; }
  PinnedArray& operator=(PinnedArray&& o) noexcept { std::swap(p, o.p); return *this; }
  PinnedArray(const PinnedArray&) = delete;
  ~PinnedArray() { if (p) cudaFreeHost(p); }
};

namespace kernels {
using bf16 = __nv_bfloat16;
using fp32::kernels::value;

// latents[t][q*4+j] = ((code // basis_j) % level_j - half_j) / half_j, as HF (integer ops, then FP32 division).
__global__ void fsq(const int32_t* codes, float* latents, int frames) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frames * kLatent) return;
  const int j = i % kFsqDim, code = codes[i / kFsqDim];
  const int basis[4] = {1, 8, 56, 336}, level[4] = {8, 7, 6, 6}, half[4] = {4, 3, 3, 3};
  latents[i] = float((code / basis[j]) % level[j] - half[j]) / float(half[j]);
}

// rows[t*6 + r] = frames[t] (nearest-exact upsampling by an integer factor).
__global__ void repeat_rows(const float* frames, float* rows, int frame_count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(frame_count) * kStack * kHidden) rows[i] = frames[(i / kHidden) / kStack * kHidden + i % kHidden];
}

// Gated short conv. P = in_proj output [M, 3*512] = (B, C, x). y[t] = C[t] * sum_k w[k] * (B*x)[t-6+k], with the
// rows before the chunk taken from `history` (6 rows of B*x, zeros at the stream start).
__device__ inline float gated_input(const float* P, int row, int d) {
  return __fmul_rn(P[size_t(row) * 3 * kHidden + d], P[size_t(row) * 3 * kHidden + 2 * kHidden + d]);
}
__global__ void conv7(const float* P, const bf16* w, const float* history, float* y, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * kHidden) return;
  const int t = i / kHidden, d = i % kHidden;
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kConv; ++k) {
    const int src = t - kConvPast + k;
    const float bx = src >= 0 ? gated_input(P, src, d) : history[(kConvPast + src) * kHidden + d];
    acc = __fmaf_rn(value(w[d * kConv + k]), bx, acc);
  }
  y[i] = __fmul_rn(P[size_t(t) * 3 * kHidden + kHidden + d], acc);
}
__global__ void conv7_history(const float* P, const float* old_history, float* new_history, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kConvPast * kHidden) return;
  const int j = i / kHidden, d = i % kHidden, row = M - kConvPast + j;
  new_history[i] = row >= 0 ? gated_input(P, row, d) : old_history[(kConvPast + row) * kHidden + d];
}

// Keys/values of this chunk into cache slots [29, 29+M); slot s holds position position0 - 29 + s.
__global__ void store_kv(const float* k, const float* v, float* kc, float* vc, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * kKvHeads * kHeadDim) return;
  kc[size_t(kPast) * kKvHeads * kHeadDim + i] = k[i];
  vc[size_t(kPast) * kKvHeads * kHeadDim + i] = v[i];
}
// The last 29 slots of this chunk's cache become the first 29 of the next one.
__global__ void carry_kv(const float* kc, const float* vc, float* next_kc, float* next_vc, int M) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kPast * kKvHeads * kHeadDim) return;
  next_kc[i] = kc[size_t(M) * kKvHeads * kHeadDim + i];
  next_vc[i] = vc[size_t(M) * kKvHeads * kHeadDim + i];
}

// One warp per (row, query head): softmax(q.k / 8) over the window (29 past + self, never before position 0),
// times V, times sigmoid(gate). GQA: query head h reads KV head h / 4.
__global__ void window_attention(const float* q, const float* kc, const float* vc, const float* gate, float* out,
                                 int M, long long position0) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= M * kHeads) return;
  const int t = warp / kHeads, h = warp % kHeads, kv = h / (kHeads / kKvHeads);
  const float* qr = q + size_t(warp) * kHeadDim;
  const float q0 = qr[lane], q1 = qr[lane + 32];
  const int history = int(position0 < kPast ? position0 : kPast);
  const int lo = t > kPast - history ? t : kPast - history, hi = kPast + t;
  float scores[kWindow], top = -INFINITY;
  for (int s = lo; s <= hi; ++s) {
    const float* kr = kc + (size_t(s) * kKvHeads + kv) * kHeadDim;
    float dot = __fmaf_rn(q1, kr[lane + 32], __fmul_rn(q0, kr[lane]));
    for (int o = 16; o; o /= 2) dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, o));
    dot = __fmul_rn(__shfl_sync(0xffffffffu, dot, 0), 0.125f);  // 1/sqrt(64), exact
    scores[s - lo] = dot;
    top = fmaxf(top, dot);
  }
  float total = 0.0f;
  for (int s = lo; s <= hi; ++s) total = __fadd_rn(total, scores[s - lo] = expf(__fsub_rn(scores[s - lo], top)));
  float o0 = 0.0f, o1 = 0.0f;
  for (int s = lo; s <= hi; ++s) {
    const float p = __fdiv_rn(scores[s - lo], total);
    const float* vr = vc + (size_t(s) * kKvHeads + kv) * kHeadDim;
    o0 = __fmaf_rn(p, vr[lane], o0);
    o1 = __fmaf_rn(p, vr[lane + 32], o1);
  }
  const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate[size_t(t) * kHeads + h])));
  out[size_t(t) * kHidden + h * kHeadDim + lane] = __fmul_rn(o0, g);
  out[size_t(t) * kHidden + h * kHeadDim + lane + 32] = __fmul_rn(o1, g);
}

// lin output [M, 1282] = (log-magnitude[641], phase[641]) -> spectrum (Re[641], Im[641], 6 zeros) at row stride
// 1288, in FP64.
__global__ void polar(const float* lin, float* spec, int M) {
  constexpr int per_row = kSpecPad / 2;  // 644: 641 bins, then 3 threads writing the 6 padding zeros
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * per_row) return;
  const int t = i / per_row, k = i % per_row;
  if (k >= kBins) {
    spec[size_t(t) * kSpecPad + kSpec + 2 * (k - kBins)] = spec[size_t(t) * kSpecPad + kSpec + 2 * (k - kBins) + 1] = 0.0f;
    return;
  }
  const double magnitude = exp(double(lin[size_t(t) * kSpec + k]));
  double s, c;
  sincos(double(lin[size_t(t) * kSpec + kBins + k]), &s, &c);
  spec[size_t(t) * kSpecPad + k] = float(magnitude * c);
  spec[size_t(t) * kSpecPad + kBins + k] = float(magnitude * s);
}

// Overlap-add of M windowed frames onto the carried tail. sums[i] (audio) and sums[L + i] (window^2 envelope) for
// local samples i in [0, L), L = 320*M + 960; frames add in ascending order after the tail, so the sums do not
// depend on how the stream was chunked.
__global__ void overlap_add(const float* frames, const float* window, const float* tail, float* sums, int M) {
  const int L = kHop * M + kTail, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  float a = i < kTail ? tail[i] : 0.0f, e = i < kTail ? tail[kTail + i] : 0.0f;
  const int first = i < kFft ? 0 : (i - kFft) / kHop + 1, last = i / kHop < M - 1 ? i / kHop : M - 1;
  for (int m = first; m <= last; ++m) {
    const int n = i - kHop * m;
    a = __fadd_rn(a, frames[size_t(m) * kFft + n]);
    e = __fadd_rn(e, __fmul_rn(window[n], window[n]));
  }
  sums[i] = a;
  sums[L + i] = e;
}
// out[j] = audio / envelope for local samples [begin, end); the samples from 320*M on become the next tail.
__global__ void emit(const float* sums, float* out, float* tail, int M, int begin, int end) {
  const int L = kHop * M + kTail, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (begin + i < end) out[i] = __fdiv_rn(sums[begin + i], sums[L + begin + i]);
  if (i < kTail) tail[i] = sums[kHop * M + i], tail[kTail + i] = sums[L + kHop * M + i];
}
}  // namespace kernels

class Decoder {
 public:
  using bf16 = __nv_bfloat16;

  // `codec_dir` holds the codec's model.safetensors (audio_tokenizer/). Calls decode at most `max_chunk_frames`
  // frames at a time.
  Decoder(const std::string& codec_dir, int max_chunk_frames)
      : weights_(std::make_shared<Safetensors>(codec_dir + "/model.safetensors")), max_frames_(max_chunk_frames) {
    if (max_chunk_frames < 1) throw std::runtime_error("codec: max_chunk_frames must be positive");
    const size_t rows = size_t(max_chunk_frames) * kStack;
    codes_ = DeviceArray<int32_t>(size_t(max_chunk_frames) * kCodebooks);
    latents_ = DeviceArray<float>(size_t(max_chunk_frames) * kLatent);
    quantized_ = DeviceArray<float>(size_t(max_chunk_frames) * kQuantizer);
    embedded_ = DeviceArray<float>(size_t(max_chunk_frames) * kHidden);
    hidden_ = DeviceArray<float>(rows * kHidden);
    normed_ = DeviceArray<float>(rows * kHidden);
    mixed_ = DeviceArray<float>(rows * kHidden);
    projected_ = DeviceArray<float>(rows * kHidden);
    wide_ = DeviceArray<float>(rows * (3 * kHidden > kSpec ? 3 * kHidden : kSpec));  // in_proj out, then lin out
    ffn_a_ = DeviceArray<float>(rows * kFfn);
    ffn_b_ = DeviceArray<float>(rows * kFfn);  // also k, v, gate (attention) and the spectrum
    frames_ = DeviceArray<float>(rows * kFft);
    sums_ = DeviceArray<float>(2 * (kHop * rows + kTail));
    samples_ = DeviceArray<float>(kHop * rows + kTail);
    codes_host_ = PinnedArray<int32_t>(size_t(max_chunk_frames) * kCodebooks);
    samples_host_ = PinnedArray<float>(kHop * rows + kTail);
    tail_ = DeviceArray<float>(2 * kTail);
    for (int i = 0; i < kLayers; ++i) {
      if (is_attention_layer(i)) {
        const size_t slots = (kPast + rows) * kKvHeads * kHeadDim;
        for (int b = 0; b < 2; ++b) keys_[i][b] = DeviceArray<float>(slots), values_[i][b] = DeviceArray<float>(slots);
      } else {
        for (int b = 0; b < 2; ++b) history_[i][b] = DeviceArray<float>(kConvPast * kHidden);
      }
    }
    load_weights();
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "codec event");  // last: nothing after it throws
  }
  ~Decoder() {
    if (submitted_) cudaEventSynchronize(done_);  // the pinned buffers outlive the copies queued into them
    if (done_) cudaEventDestroy(done_);
  }
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // Start a new stream.
  void reset() {
    if (submitted_) throw std::runtime_error("codec: reset with a chunk not collected");
    for (int i = 0; i < kLayers; ++i)
      for (int b = 0; b < 2; ++b) {
        if (is_attention_layer(i)) {
          check(cudaMemsetAsync(keys_[i][b].p, 0, keys_[i][b].bytes(), stream_), "codec reset");
          check(cudaMemsetAsync(values_[i][b].p, 0, values_[i][b].bytes(), stream_), "codec reset");
        } else {
          check(cudaMemsetAsync(history_[i][b].p, 0, history_[i][b].bytes(), stream_), "codec reset");
        }
      }
    check(cudaMemsetAsync(tail_.p, 0, tail_.bytes(), stream_), "codec reset");
    frames_done_ = 0;
    current_ = 0;
    started_ = finished_ = false;
  }

  // Decodes `frame_count` frames (host, frame-major [frame][12], codes in [0, 2016)) and appends the samples that
  // are final to `out`: 1920 per frame overall, the last 480 only when `finalize`. submit() then collect().
  void decode(const int32_t* codes, int frame_count, bool finalize, std::vector<float>& out) {
    submit(codes, frame_count, finalize);
    collect(out);
  }

  // The work of decode() queued on the decoder's stream (set_stream; the legacy default stream unless set), its samples
  // copied into pinned memory: collect() appends them. One chunk at a time: a submit must be collected before the next
  // submit or a reset. The kernels and their order are decode()'s, so the samples are the same.
  void submit(const int32_t* codes, int frame_count, bool finalize) {
    if (submitted_) throw std::runtime_error("codec: submit with a chunk not collected");
    if (finished_) throw std::runtime_error("codec: stream already finalized; reset() first");
    if (frame_count < 0 || frame_count > max_frames_) throw std::runtime_error("codec: chunk exceeds max_chunk_frames");
    for (int i = 0; i < frame_count * kCodebooks; ++i)
      if (codes[i] < 0 || codes[i] >= kCodebookSize) throw std::runtime_error("codec: code out of range [0, 2016)");
    if (!frame_count && !(finalize && started_)) {
      finished_ = finalize;
      return;
    }
    const int M = frame_count * kStack;
    if (frame_count) {
      std::copy(codes, codes + size_t(frame_count) * kCodebooks, codes_host_.p);
      check(cudaMemcpyAsync(codes_.p, codes_host_.p, size_t(frame_count) * kCodebooks * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream_), "codec codes");
      trunk(frame_count);
    }
    const int L = kHop * M + kTail;
    kernels::overlap_add<<<blocks(L), 256, 0, stream_>>>(frames_.p, window_, tail_.p, sums_.p, M);
    const int begin = started_ ? 0 : kTrim, end = finalize ? L - kTrim : kHop * M;
    kernels::emit<<<blocks(std::max(end - begin, kTail)), 256, 0, stream_>>>(sums_.p, samples_.p, tail_.p, M, begin, end);
    check(cudaGetLastError(), "codec kernels");
    check(cudaMemcpyAsync(samples_host_.p, samples_.p, size_t(end - begin) * sizeof(float), cudaMemcpyDeviceToHost,
                          stream_), "codec samples");
    check(cudaEventRecord(done_, stream_), "codec event");
    submitted_ = end - begin;
    frames_done_ += frame_count;
    started_ = true;
    finished_ = finalize;
  }
  // Whether a chunk is submitted and not yet collected; whether its samples are ready (collect would not wait).
  bool pending() const { return submitted_ > 0; }
  bool ready() const { return submitted_ == 0 || cudaEventQuery(done_) == cudaSuccess; }
  // Waits for the submitted chunk and appends its samples to `out` (nothing when none is pending).
  void collect(std::vector<float>& out) {
    if (!submitted_) return;
    check(cudaEventSynchronize(done_), "codec samples");
    out.insert(out.end(), samples_host_.p, samples_host_.p + submitted_);
    submitted_ = 0;
  }
  // Drops a submitted chunk: waits for it, ignoring errors (after a CUDA fault), so the decoder can be reset.
  void abandon() noexcept {
    if (submitted_) cudaEventSynchronize(done_);
    submitted_ = 0;
  }
  // The stream the decoder's work goes to (0: the legacy default stream). Not while a chunk is pending.
  void set_stream(cudaStream_t s) {
    if (submitted_) throw std::runtime_error("codec: set_stream with a chunk not collected");
    stream_ = s;
  }

  // Device bytes the decoder holds: the codec checkpoint (encoder included) and its own buffers.
  size_t device_bytes() const {
    size_t n = weights_->device_bytes() + basis_.bytes() + codes_.bytes() + latents_.bytes() + quantized_.bytes() +
               embedded_.bytes() + hidden_.bytes() + normed_.bytes() + mixed_.bytes() + projected_.bytes() +
               wide_.bytes() + ffn_a_.bytes() + ffn_b_.bytes() + frames_.bytes() + sums_.bytes() + samples_.bytes() +
               tail_.bytes() + lin_copy_.bytes() + from_quantizer_copy_.bytes();
    for (int i = 0; i < kLayers; ++i)
      for (int b = 0; b < 2; ++b) n += keys_[i][b].bytes() + values_[i][b].bytes() + history_[i][b].bytes();
    return n;
  }

  // Debug: when set, the first trunk call writes its intermediates as raw FP32 files into this directory.
  std::string dump_dir;

 private:
  struct Layer {
    const bf16 *operator_norm, *post_operator_norm, *ffn_norm, *post_ffn_norm, *w1, *w2, *w3;
    const bf16 *in_proj = nullptr, *conv = nullptr, *out_proj = nullptr;                      // conv layers
    const bf16 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *q_norm = nullptr,
               *k_norm = nullptr;                                                              // attention layers
  };

  static int blocks(size_t threads) { return fp32::blocks(threads); }

  const bf16* tensor(const std::string& name, std::vector<int64_t> shape, const char* dtype = "BF16") const {
    const auto& t = weights_->at(name);
    if (t.dtype != dtype || t.shape != shape) throw std::runtime_error("codec: unexpected tensor " + name);
    return t.device_data;
  }

  // A weight at a 16-byte aligned address: the checkpoint's own when it is, else a copy (the same values). In this
  // checkpoint from_quantizer and decoder.lin start 4 bytes off; fp32::linear's faster kernel needs 16
  // (validation/speech-speed-v1).
  static const bf16* aligned(const bf16* w, size_t count, DeviceArray<bf16>& copy) {
    if (reinterpret_cast<uintptr_t>(w) % 16 == 0) return w;
    copy = DeviceArray<bf16>(count);
    check(cudaMemcpy(copy.p, w, count * sizeof(bf16), cudaMemcpyDeviceToDevice), "codec aligned weight");
    return copy.p;
  }

  void load_weights() {
    project_out_ = tensor("quantizer.project_out.weight", {kQuantizer, kLatent});
    project_out_bias_ = tensor("quantizer.project_out.bias", {kQuantizer});
    from_quantizer_ = aligned(tensor("from_quantizer.weight", {kHidden, kQuantizer}), size_t(kHidden) * kQuantizer,
                              from_quantizer_copy_);
    from_quantizer_bias_ = tensor("from_quantizer.bias", {kHidden});
    final_norm_ = tensor("decoder.lfm.embedding_norm.weight", {kHidden});
    lin_ = aligned(tensor("decoder.lin.weight", {kSpec, kHidden}), size_t(kSpec) * kHidden, lin_copy_);
    lin_bias_ = tensor("decoder.lin.bias", {kSpec});
    window_ = reinterpret_cast<const float*>(tensor("decoder.istft.window", {kFft}, "F32"));
    for (int i = 0; i < kLayers; ++i) {
      const std::string p = "decoder.lfm.layers." + std::to_string(i) + ".";
      Layer& l = layers_[i];
      l.operator_norm = tensor(p + "operator_norm.weight", {kHidden});
      l.post_operator_norm = tensor(p + "post_operator_norm.weight", {kHidden});
      l.ffn_norm = tensor(p + "ffn_norm.weight", {kHidden});
      l.post_ffn_norm = tensor(p + "post_ffn_norm.weight", {kHidden});
      l.w1 = tensor(p + "feed_forward.w1.weight", {kFfn, kHidden});
      l.w3 = tensor(p + "feed_forward.w3.weight", {kFfn, kHidden});
      l.w2 = tensor(p + "feed_forward.w2.weight", {kHidden, kFfn});
      if (is_attention_layer(i)) {
        l.q = tensor(p + "self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden});
        l.k = tensor(p + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden});
        l.v = tensor(p + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden});
        l.o = tensor(p + "self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim});
        l.gate = tensor(p + "self_attn.gate_proj.weight", {kHeads, kHidden});
        l.q_norm = tensor(p + "self_attn.q_norm.weight", {kHeadDim});
        l.k_norm = tensor(p + "self_attn.k_norm.weight", {kHeadDim});
      } else {
        l.in_proj = tensor(p + "conv.in_proj.weight", {3 * kHidden, kHidden});
        l.conv = tensor(p + "conv.conv.weight", {kHidden, 1, kConv});
        l.out_proj = tensor(p + "conv.out_proj.weight", {kHidden, kHidden});
      }
    }
    // Inverse real DFT (torch.fft.irfft, norm="backward") times the synthesis window, as a [1280, 1288] matrix over
    // (Re[0..640], Im[0..640], 6 zero columns). Built in FP64 with exact argument reduction; the imaginary parts of bins 0 and 640
    // are ignored, as irfft does.
    std::vector<float> window(kFft), basis(size_t(kFft) * kSpecPad);
    check(cudaMemcpy(window.data(), window_, kFft * sizeof(float), cudaMemcpyDeviceToHost), "codec window");
    const double pi = 3.14159265358979323846;
    for (int n = 0; n < kFft; ++n)
      for (int k = 0; k < kBins; ++k) {
        const double angle = 2.0 * pi * double((int64_t(k) * n) % kFft) / kFft;
        const double weight = (k == 0 || k == kBins - 1 ? 1.0 : 2.0) / kFft * double(window[n]);
        basis[size_t(n) * kSpecPad + k] = float(weight * std::cos(angle));
        basis[size_t(n) * kSpecPad + kBins + k] = k == 0 || k == kBins - 1 ? 0.0f : float(-weight * std::sin(angle));
      }
    basis_ = DeviceArray<float>(basis.size());
    check(cudaMemcpy(basis_.p, basis.data(), basis.size() * sizeof(float), cudaMemcpyHostToDevice), "codec basis");
  }

  template <class W>
  void linear(const float* x, const W* w, const bf16* bias, float* y, int M, int N, int K) {
    fp32::linear(x, w, bias, y, M, N, K, stream_);
  }
  void norm(const float* x, const bf16* w, float* y, const float* residual, int rows, int width) {
    fp32::rms_norm(x, w, y, residual, rows, width, kNormEps, 0, stream_);
  }

  void dump(const char* name, const float* p, size_t n) {
    if (dump_dir.empty()) return;
    std::vector<float> h(n);
    check(cudaStreamSynchronize(stream_), "codec dump");
    check(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost), "codec dump");
    const std::string path = dump_dir + "/" + name + ".f32";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(h.data(), sizeof(float), n, f) != n) throw std::runtime_error("codec dump " + path);
    std::fclose(f);
  }

  // Codes (already on the device) -> windowed STFT frames in frames_.
  void trunk(int frame_count) {
    const int M = frame_count * kStack;
    const long long position0 = frames_done_ * kStack;
    kernels::fsq<<<blocks(size_t(frame_count) * kLatent), 256, 0, stream_>>>(codes_.p, latents_.p, frame_count);
    dump("fsq", latents_.p, size_t(frame_count) * kLatent);
    linear(latents_.p, project_out_, project_out_bias_, quantized_.p, frame_count, kQuantizer, kLatent);
    dump("latents", quantized_.p, size_t(frame_count) * kQuantizer);  // HF's quantizer.decode output
    linear(quantized_.p, from_quantizer_, from_quantizer_bias_, embedded_.p, frame_count, kHidden, kQuantizer);
    dump("embeddings", embedded_.p, size_t(frame_count) * kHidden);
    kernels::repeat_rows<<<blocks(size_t(M) * kHidden), 256, 0, stream_>>>(embedded_.p, hidden_.p, frame_count);
    const int next = 1 - current_;
    for (int i = 0; i < kLayers; ++i) {
      const Layer& l = layers_[i];
      norm(hidden_.p, l.operator_norm, normed_.p, nullptr, M, kHidden);
      if (is_attention_layer(i)) {
        float *q = mixed_.p, *k = ffn_b_.p, *v = k + size_t(M) * kKvHeads * kHeadDim,
              *gate = v + size_t(M) * kKvHeads * kHeadDim;
        linear(normed_.p, l.q, nullptr, q, M, kHeads * kHeadDim, kHidden);
        linear(normed_.p, l.k, nullptr, k, M, kKvHeads * kHeadDim, kHidden);
        linear(normed_.p, l.v, nullptr, v, M, kKvHeads * kHeadDim, kHidden);
        linear(normed_.p, l.gate, nullptr, gate, M, kHeads, kHidden);
        norm(q, l.q_norm, q, nullptr, M * kHeads, kHeadDim);
        norm(k, l.k_norm, k, nullptr, M * kKvHeads, kHeadDim);
        fp32::rotary(q, M, kHeads, kHeadDim, position0, kRopeTheta, stream_);
        fp32::rotary(k, M, kKvHeads, kHeadDim, position0, kRopeTheta, stream_);
        auto &kc = keys_[i][current_], &vc = values_[i][current_];
        kernels::store_kv<<<blocks(size_t(M) * kKvHeads * kHeadDim), 256, 0, stream_>>>(k, v, kc.p, vc.p, M);
        kernels::window_attention<<<blocks(size_t(M) * kHeads * 32), 256, 0, stream_>>>(q, kc.p, vc.p, gate, projected_.p, M,
                                                                             position0);
        kernels::carry_kv<<<blocks(size_t(kPast) * kKvHeads * kHeadDim), 256, 0, stream_>>>(kc.p, vc.p, keys_[i][next].p,
                                                                                 values_[i][next].p, M);
        linear(projected_.p, l.o, nullptr, mixed_.p, M, kHidden, kHeads * kHeadDim);
      } else {
        linear(normed_.p, l.in_proj, nullptr, wide_.p, M, 3 * kHidden, kHidden);
        kernels::conv7<<<blocks(size_t(M) * kHidden), 256, 0, stream_>>>(wide_.p, l.conv, history_[i][current_].p, projected_.p, M);
        kernels::conv7_history<<<blocks(size_t(kConvPast) * kHidden), 256, 0, stream_>>>(wide_.p, history_[i][current_].p,
                                                                              history_[i][next].p, M);
        linear(projected_.p, l.out_proj, nullptr, mixed_.p, M, kHidden, kHidden);
      }
      norm(mixed_.p, l.post_operator_norm, hidden_.p, hidden_.p, M, kHidden);
      norm(hidden_.p, l.ffn_norm, normed_.p, nullptr, M, kHidden);
      linear(normed_.p, l.w1, nullptr, ffn_a_.p, M, kFfn, kHidden);
      linear(normed_.p, l.w3, nullptr, ffn_b_.p, M, kFfn, kHidden);
      fp32::swiglu(ffn_a_.p, ffn_b_.p, size_t(M) * kFfn, stream_);
      linear(ffn_a_.p, l.w2, nullptr, mixed_.p, M, kHidden, kFfn);
      norm(mixed_.p, l.post_ffn_norm, hidden_.p, hidden_.p, M, kHidden);
      char name[16];
      std::snprintf(name, sizeof name, "layer%d", i);
      dump(name, hidden_.p, size_t(M) * kHidden);
    }
    current_ = next;
    norm(hidden_.p, final_norm_, normed_.p, nullptr, M, kHidden);
    dump("final_norm", normed_.p, size_t(M) * kHidden);
    linear(normed_.p, lin_, lin_bias_, wide_.p, M, kSpec, kHidden);
    dump("lin", wide_.p, size_t(M) * kSpec);
    kernels::polar<<<blocks(size_t(M) * kSpecPad / 2), 256, 0, stream_>>>(wide_.p, ffn_b_.p, M);
    linear(ffn_b_.p, basis_.p, nullptr, frames_.p, M, kFft, kSpecPad);
    dump("frames", frames_.p, size_t(M) * kFft);
    check(cudaGetLastError(), "codec trunk");
    dump_dir.clear();
  }

  std::shared_ptr<Safetensors> weights_;
  int max_frames_;
  const bf16 *project_out_, *project_out_bias_, *from_quantizer_, *from_quantizer_bias_, *final_norm_, *lin_, *lin_bias_;
  const float* window_;
  Layer layers_[kLayers];
  DeviceArray<float> basis_;
  DeviceArray<bf16> lin_copy_, from_quantizer_copy_;  // aligned copies (aligned())
  DeviceArray<int32_t> codes_;
  DeviceArray<float> latents_, quantized_, embedded_, hidden_, normed_, mixed_, projected_, wide_, ffn_a_, ffn_b_,
      frames_, sums_, samples_, tail_;
  DeviceArray<float> keys_[kLayers][2], values_[kLayers][2], history_[kLayers][2];
  long long frames_done_ = 0;
  int current_ = 0;
  bool started_ = false, finished_ = false;
  cudaStream_t stream_ = 0;
  PinnedArray<int32_t> codes_host_;
  PinnedArray<float> samples_host_;
  cudaEvent_t done_ = nullptr;
  size_t submitted_ = 0;  // samples of the chunk submitted and not yet collected
};

}  // namespace lfm::codec
