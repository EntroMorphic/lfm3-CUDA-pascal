#pragma once
// Audio input (#3 audio in): the LFM3-Omni audio encoder and projector, natively (modeling_lfm3_omni.py:
// Lfm3OmniDownsampler, Lfm3OmniEncoder, Lfm3OmniProjector; Lfm3OmniModel.get_audio_features for one clip).
// Include after src/lfm3_main.cu (Buffer, cuda_check, grid, read_json).
//
// One clip's log-mel features [128][T] (lfm/audio_features.hpp) become ceil(T/8) vectors of the language model's
// width, which replace the clip's <|audio|> placeholders in the prompt:
// - downsampler: conv_in (1 -> 256 channels, 3x3, stride 2, pad 1) and SiLU; conv_mid (depthwise 3x3 s2) and lin_mid
//   (1x1) and SiLU; conv_end and lin_end and SiLU; the (256 channel, 16 mel) plane of each time step flattened channel-
//   major and lin_out (4096 -> 768). Over a single clip HF's padding masks change nothing (each stage's output length
//   is exactly its masked length), so a clip is encoded alone, as HF encodes it inside a batch.
// - encoder: an LFM2 model (17 layers, width 768; short convolutions of 3 taps; attention with 8 query and 4 key/value
//   heads of 96, q/k RMSNorm, RoPE theta 1e6, causal within a window of 32 keys: key s serves query t when
//   t - 32 < s <= t), pre-norm residual blocks, SwiGLU 1792, and the final embedding_norm.
// - projector: LayerNorm(768), linear 768 -> 2048, exact GELU, linear 2048 -> 2048.
// Precision: FP32 activations throughout, the weights at their stored BF16 values (lfm/fp32_ops.cuh, as the codec and
// depth decoder; the features are not rounded to BF16 on the way in). Only the output is rounded to BF16, the
// language model's input. validation/audio-in-v1 round 1 ran the encoder at HF's BF16 rounding points instead and
// matched HF's error to the FP32 model only on average (median ratio 1.02, worst clip 1.25).
#include <lfm/audio_features.hpp>
#include <lfm/fp32_ops.cuh>
#include <lfm/safetensors_header.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfm3_audio_in {

namespace kernels {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }
__device__ __forceinline__ float w(const bf16* p, size_t i) { return __bfloat162float(p[i]); }

// mel [128][T] -> [W1][64][256] (time, mel, channel): conv + bias, SiLU
__global__ void conv_in(const float* mel, int T, const bf16* wt, const bf16* bias, float* out, int W1) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(W1) * 64 * 256) return;
  const int c = i % 256, h = (i / 256) % 64, t = int(i / (256 * 64));
  float sum = w(bias, c);
  for (int a = 0; a < 3; ++a) {
    const int y = 2 * h - 1 + a;
    if (y < 0 || y >= 128) continue;
    for (int b = 0; b < 3; ++b) {
      const int x = 2 * t - 1 + b;
      if (x >= 0 && x < T) sum = __fmaf_rn(w(wt, c * 9 + a * 3 + b), mel[size_t(y) * T + x], sum);
    }
  }
  out[i] = silu(sum);
}
// depthwise 3x3 stride 2 pad 1 over [W][H][256] -> [W2][H/2][256]: conv + bias (no activation)
__global__ void depthwise(const float* in, int W, int H, const bf16* wt, const bf16* bias, float* out, int W2) {
  const int H2 = H / 2;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(W2) * H2 * 256) return;
  const int c = i % 256, h = (i / 256) % H2, t = int(i / (256 * H2));
  float sum = w(bias, c);
  for (int a = 0; a < 3; ++a) {
    const int y = 2 * h - 1 + a;
    if (y < 0 || y >= H) continue;
    for (int b = 0; b < 3; ++b) {
      const int x = 2 * t - 1 + b;
      if (x >= 0 && x < W) sum = __fmaf_rn(w(wt, c * 9 + a * 3 + b), in[(size_t(x) * H + y) * 256 + c], sum);
    }
  }
  out[i] = sum;
}
// in place: 1 SiLU, 2 exact GELU
__global__ void activate(float* x, size_t n, int act) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  x[i] = act == 1 ? silu(v) : __fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.0f, erff(__fmul_rn(v, 0.70710678118654752f))));
}
// [W3][16][256] -> [W3][256 * 16], channel-major (flatten(1, 2) of (batch, channel, mel, time))
__global__ void flatten(const float* in, float* out, int W3) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(W3) * 4096) return;
  const int t = int(i / 4096), j = i % 4096, c = j / 16, f = j % 16;
  out[i] = in[(size_t(t) * 16 + f) * 256 + c];
}
// LFM2 short convolution over one sequence from an empty state: in_proj's (B, C, x) per row; C * conv3(B * x), causal
__global__ void short_conv(const float* projected, const bf16* wt, float* out, int rows, int width) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * width) return;
  const int d = i % width, t = int(i / width);
  float sum = 0;
  for (int k = 0; k < 3; ++k) {
    const int s = t + k - 2;
    if (s >= 0)
      sum = __fmaf_rn(w(wt, d * 3 + k),
                      __fmul_rn(projected[size_t(s) * 3 * width + d], projected[size_t(s) * 3 * width + 2 * width + d]), sum);
  }
  out[i] = __fmul_rn(projected[size_t(t) * 3 * width + width + d], sum);
}
__global__ void add(float* x, const float* y, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __fadd_rn(x[i], y[i]);
}
// q [rows][heads][dim], k/v [rows][kv_heads][dim]; one warp per (query, head) over the 32 keys t-31..t (those >= 0)
__global__ void window_attention(const float* q, const float* k, const float* v, float* out, int rows, int heads,
                                 int kv_heads, int dim, float scale) {
  const int warp = int((size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
  if (warp >= rows * heads) return;
  const int t = warp / heads, h = warp % heads, g = h / (heads / kv_heads);
  const int s = t - 31 + lane;  // a window of 32: one key per lane
  const float* qr = q + (size_t(t) * heads + h) * dim;
  float score = -INFINITY;
  if (s >= 0) {
    const float* kr = k + (size_t(s) * kv_heads + g) * dim;
    float dot = 0;
    for (int d = 0; d < dim; ++d) dot = __fmaf_rn(qr[d], kr[d], dot);
    score = __fmul_rn(dot, scale);
  }
  float m = score;
  for (int o = 16; o; o /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  float p = s >= 0 ? expf(score - m) : 0.0f, total = p;
  for (int o = 16; o; o /= 2) total += __shfl_xor_sync(0xffffffff, total, o);
  p = __fdiv_rn(p, total);
  for (int d = lane; d < dim; d += 32) {
    float acc = 0;
    for (int j = 0; j < 32; ++j) {
      const float pj = __shfl_sync(0xffffffff, p, j);
      const int sj = t - 31 + j;
      if (sj >= 0) acc = __fmaf_rn(pj, v[(size_t(sj) * kv_heads + g) * dim + d], acc);
    }
    out[(size_t(t) * heads + h) * dim + d] = acc;
  }
}
// LayerNorm with weight and bias, two-pass FP32 statistics; one warp per row
__global__ void layer_norm(const float* x, const bf16* wt, const bf16* b, float* y, int rows, int width, float eps) {
  const int row = int((size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
  if (row >= rows) return;
  const float* r = x + size_t(row) * width;
  float s = 0;
  for (int d = lane; d < width; d += 32) s = __fadd_rn(s, r[d]);
  for (int o = 16; o; o /= 2) s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, o));
  const float mean = __fdiv_rn(s, float(width));
  s = 0;
  for (int d = lane; d < width; d += 32) { const float c = __fsub_rn(r[d], mean); s = __fmaf_rn(c, c, s); }
  for (int o = 16; o; o /= 2) s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, o));
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, float(width)), eps)));
  for (int d = lane; d < width; d += 32)
    y[size_t(row) * width + d] = __fmaf_rn(__fmul_rn(__fsub_rn(r[d], mean), inv), w(wt, d), w(b, d));
}
__global__ void to_bf16(const float* x, bf16* y, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}
}  // namespace kernels

// Host copies of the encoder's stages for one clip (the parity check against HF; tools/lfm3_audio_in_hf.py "dump").
using Dumps = lfm::audio_features::Dump;

class Encoder {
 public:
  using bf16 = __nv_bfloat16;
  static constexpr int kChannels = 256, kWidth = 768, kLayers = 17, kHeads = 8, kKvHeads = 4, kHeadDim = 96,
                       kMlp = 1792, kWindow = 32, kOut = 2048;

  // The architecture this code implements, checked against the checkpoint's own description (config.json's root) and
  // its tensor table (`lookup`: a tensor's dtype and shape by name, or nothing when absent). Throws on any difference.
  // Used at startup (WorkerCore: whether audio input is supported, from the safetensors header alone) and by the
  // constructor, so a checkpoint the encoder cannot run is known before any request.
  static void check_contract(const lfm::Json& root,
                             const std::function<std::optional<lfm::safetensors_header::Entry>(const std::string&)>& lookup) {
    const auto& ac = root.at("audio_encoder_config");
    const auto& tc = ac.at("text_config");
    auto need = [](bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("audio encoder: ") + what); };
    need(ac.at("conv_channels").integer() == kChannels && ac.at("hidden_size").integer() == kWidth &&
         ac.at("num_mel_bins").integer() == lfm::audio_features::kMels && ac.at("sliding_window").integer() == kWindow,
         "unexpected encoder geometry");
    need(tc.at("model_type").string() == "lfm2" && tc.at("hidden_size").integer() == kWidth &&
         tc.at("num_hidden_layers").integer() == kLayers && tc.at("num_attention_heads").integer() == kHeads &&
         tc.at("num_key_value_heads").integer() == kKvHeads && tc.at("conv_L_cache").integer() == 3 &&
         !tc.at("conv_bias").boolean(), "unexpected encoder LFM2 configuration");
    (void)tc.at("norm_eps").number(), (void)tc.at("rope_parameters").at("rope_theta").number();
    const auto& types = tc.at("layer_types").array();
    need(types.size() == size_t(kLayers), "layer_types length");
    std::vector<bool> attention;
    for (const auto& t : types) {
      need(t.string() == "conv" || t.string() == "full_attention", "layer type");
      attention.push_back(t.string() == "full_attention");
    }
    need(root.at("text_config").at("hidden_size").integer() == kOut, "projector width");
    auto check = [&](const std::string& name, std::initializer_list<int64_t> shape) {
      const auto e = lookup(name);
      if (!e) throw std::runtime_error("audio encoder: missing " + name);
      if (e->dtype != "BF16" || e->shape != std::vector<int64_t>(shape))
        throw std::runtime_error("audio encoder weight contract: " + name);
    };
    const std::string D = "model.audio_encoder.downsample.";
    check(D + "conv_in.weight", {kChannels, 1, 3, 3}); check(D + "conv_in.bias", {kChannels});
    for (const char* s : {"mid", "end"}) {
      check(D + "conv_" + s + ".weight", {kChannels, 1, 3, 3}); check(D + "conv_" + s + ".bias", {kChannels});
      check(D + "lin_" + s + ".weight", {kChannels, kChannels, 1, 1}); check(D + "lin_" + s + ".bias", {kChannels});
    }
    check(D + "lin_out.weight", {kWidth, kChannels * 16, 1}); check(D + "lin_out.bias", {kWidth});
    for (int i = 0; i < kLayers; ++i) {
      const std::string p = layer(i);
      check(p + "operator_norm.weight", {kWidth}); check(p + "ffn_norm.weight", {kWidth});
      check(p + "feed_forward.w1.weight", {kMlp, kWidth}); check(p + "feed_forward.w3.weight", {kMlp, kWidth});
      check(p + "feed_forward.w2.weight", {kWidth, kMlp});
      if (attention[i]) {
        check(p + "self_attn.q_proj.weight", {kWidth, kWidth}); check(p + "self_attn.out_proj.weight", {kWidth, kWidth});
        check(p + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kWidth});
        check(p + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kWidth});
        check(p + "self_attn.q_layernorm.weight", {kHeadDim}); check(p + "self_attn.k_layernorm.weight", {kHeadDim});
      } else {
        check(p + "conv.conv.weight", {kWidth, 1, 3}); check(p + "conv.in_proj.weight", {3 * kWidth, kWidth});
        check(p + "conv.out_proj.weight", {kWidth, kWidth});
      }
    }
    check("model.audio_encoder.lfm.embedding_norm.weight", {kWidth});
    const std::string P = "model.audio_projector.";
    check(P + "layer_norm.weight", {kWidth}); check(P + "layer_norm.bias", {kWidth});
    check(P + "linear_1.weight", {kOut, kWidth}); check(P + "linear_1.bias", {kOut});
    check(P + "linear_2.weight", {kOut, kOut}); check(P + "linear_2.bias", {kOut});
  }

  Encoder(std::shared_ptr<lfm::Safetensors> weights, const std::string& model_dir) : weights_(std::move(weights)) {
    const auto root = read_json(model_dir + "/config.json");
    check_contract(root, [&](const std::string& name) -> std::optional<lfm::safetensors_header::Entry> {
      if (!weights_->has(name)) return std::nullopt;
      const auto& t = weights_->at(name);
      return lfm::safetensors_header::Entry{t.dtype, t.shape};
    });
    const auto& tc = root.at("audio_encoder_config").at("text_config");
    eps_ = float(tc.at("norm_eps").number());
    theta_ = tc.at("rope_parameters").at("rope_theta").number();
    for (const auto& t : tc.at("layer_types").array()) attention_.push_back(t.string() == "full_attention");
  }

  // Encodes one clip's features (mel: [128][T], from lfm::audio_features::log_mel) into `out` (device, [N][2048] BF16,
  // N = ceil(T / 8)); returns N. `dumps`, when set, receives host copies of every stage.
  int encode(const std::vector<float>& mel, bf16* out, Dumps* dumps = nullptr) {
    using namespace kernels;
    namespace f = lfm::fp32;
    constexpr int M = lfm::audio_features::kMels;
    if (mel.empty() || mel.size() % M) throw std::runtime_error("audio encoder: features are not [128][T]");
    const int T = int(mel.size() / M), W1 = (T + 1) / 2, W2 = (W1 + 1) / 2, N = (W2 + 1) / 2;
    if (N != int((size_t(T) + 7) / 8)) throw std::runtime_error("audio encoder: token count");  // unreachable
    auto& in = scratch(0, mel.size());
    cuda_check(cudaMemcpy(in.p, mel.data(), mel.size() * sizeof(float), cudaMemcpyHostToDevice));
    const std::string D = "model.audio_encoder.downsample.";
    // downsampler; f::linear(x, w, bias, y, rows, out, in)
    auto& a1 = scratch(1, size_t(W1) * 64 * kChannels);
    conv_in<<<grid(size_t(W1) * 64 * kChannels), 256>>>(in.p, T, w(D + "conv_in.weight"), w(D + "conv_in.bias"), a1.p, W1);
    auto stage = [&](const float* x, int W, int H, int W2_, const char* s, Buffer<float>& y) {
      const size_t n = size_t(W2_) * (H / 2) * kChannels;
      auto& c = scratch(2, n);
      depthwise<<<grid(n), 256>>>(x, W, H, w(D + "conv_" + s + ".weight"), w(D + "conv_" + s + ".bias"), c.p, W2_);
      f::linear(c.p, w(D + "lin_" + s + ".weight"), w(D + "lin_" + s + ".bias"), y.p, int(n / kChannels), kChannels, kChannels);
      activate<<<grid(n), 256>>>(y.p, n, 1);
    };
    auto& a2 = scratch(3, size_t(W2) * 32 * kChannels);
    stage(a1.p, W1, 64, W2, "mid", a2);
    auto& a3 = scratch(1, size_t(N) * 16 * kChannels);
    stage(a2.p, W2, 32, N, "end", a3);
    auto& flat = scratch(2, size_t(N) * 4096);
    flatten<<<grid(size_t(N) * 4096), 256>>>(a3.p, flat.p, N);
    auto& x = scratch(4, size_t(N) * kWidth);
    f::linear(flat.p, w(D + "lin_out.weight"), w(D + "lin_out.bias"), x.p, N, kWidth, 4096);
    if (dumps) dumps->mel = mel, dumps->down = host(x.p, size_t(N) * kWidth);
    // encoder
    const size_t NW = size_t(N) * kWidth;
    const int kv = kKvHeads * kHeadDim;
    auto& n = scratch(5, NW);
    auto& tmp = scratch(6, NW * 3);  // in_proj (3 x 768), or q/k/v
    auto& op = scratch(7, NW);
    auto& ff_a = scratch(8, size_t(N) * kMlp);
    auto& ff_b = scratch(9, size_t(N) * kMlp);
    auto& att = scratch(10, NW);
    for (int i = 0; i < kLayers; ++i) {
      const std::string p = layer(i);
      f::rms_norm(x.p, w(p + "operator_norm.weight"), n.p, nullptr, N, kWidth, eps_);
      if (!attention_[i]) {
        f::linear(n.p, w(p + "conv.in_proj.weight"), nullptr, tmp.p, N, 3 * kWidth, kWidth);
        short_conv<<<grid(NW), 256>>>(tmp.p, w(p + "conv.conv.weight"), att.p, N, kWidth);
        f::linear(att.p, w(p + "conv.out_proj.weight"), nullptr, op.p, N, kWidth, kWidth);
      } else {
        float *q = tmp.p, *k = tmp.p + NW, *v = tmp.p + NW + size_t(N) * kv;
        f::linear(n.p, w(p + "self_attn.q_proj.weight"), nullptr, q, N, kWidth, kWidth);
        f::linear(n.p, w(p + "self_attn.k_proj.weight"), nullptr, k, N, kv, kWidth);
        f::linear(n.p, w(p + "self_attn.v_proj.weight"), nullptr, v, N, kv, kWidth);
        f::rms_norm(q, w(p + "self_attn.q_layernorm.weight"), q, nullptr, N * kHeads, kHeadDim, eps_);
        f::rms_norm(k, w(p + "self_attn.k_layernorm.weight"), k, nullptr, N * kKvHeads, kHeadDim, eps_);
        f::rotary(q, N, kHeads, kHeadDim, 0, theta_);
        f::rotary(k, N, kKvHeads, kHeadDim, 0, theta_);
        window_attention<<<grid(size_t(N) * kHeads * 32), 256>>>(q, k, v, att.p, N, kHeads, kKvHeads, kHeadDim,
                                                                 float(1.0 / std::sqrt(double(kHeadDim))));
        f::linear(att.p, w(p + "self_attn.out_proj.weight"), nullptr, op.p, N, kWidth, kWidth);
      }
      add<<<grid(NW), 256>>>(x.p, op.p, NW);
      f::rms_norm(x.p, w(p + "ffn_norm.weight"), n.p, nullptr, N, kWidth, eps_);
      f::linear(n.p, w(p + "feed_forward.w1.weight"), nullptr, ff_a.p, N, kMlp, kWidth);
      f::linear(n.p, w(p + "feed_forward.w3.weight"), nullptr, ff_b.p, N, kMlp, kWidth);
      f::swiglu(ff_a.p, ff_b.p, size_t(N) * kMlp);
      f::linear(ff_a.p, w(p + "feed_forward.w2.weight"), nullptr, op.p, N, kWidth, kMlp);
      add<<<grid(NW), 256>>>(x.p, op.p, NW);
    }
    f::rms_norm(x.p, w("model.audio_encoder.lfm.embedding_norm.weight"), n.p, nullptr, N, kWidth, eps_);
    if (dumps) dumps->encoded = host(n.p, NW);
    // projector
    const std::string P = "model.audio_projector.";
    layer_norm<<<grid(size_t(N) * 32), 256>>>(n.p, w(P + "layer_norm.weight"), w(P + "layer_norm.bias"), att.p, N, kWidth,
                                               1e-5f);
    auto& hid = scratch(11, size_t(N) * kOut);
    f::linear(att.p, w(P + "linear_1.weight"), w(P + "linear_1.bias"), hid.p, N, kOut, kWidth);
    activate<<<grid(size_t(N) * kOut), 256>>>(hid.p, size_t(N) * kOut, 2);
    auto& proj = scratch(12, size_t(N) * kOut);
    f::linear(hid.p, w(P + "linear_2.weight"), w(P + "linear_2.bias"), proj.p, N, kOut, kOut);
    to_bf16<<<grid(size_t(N) * kOut), 256>>>(proj.p, out, size_t(N) * kOut);
    cuda_check(cudaGetLastError());
    // the scratch is released after every clip: a long clip's stages take hundreds of MB (the first is W1 x 64 x 256
    // floats), which would otherwise stay allocated for the process's life (validation/audio-in-v1 red-team)
    cuda_check(cudaDeviceSynchronize());
    for (auto& b : scratch_) b.reset();
    std::fill(sizes_.begin(), sizes_.end(), 0);
    if (dumps) {
      std::vector<bf16> b(size_t(N) * kOut);
      cuda_check(cudaMemcpy(b.data(), out, b.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
      dumps->features.resize(b.size());
      for (size_t i = 0; i < b.size(); ++i) dumps->features[i] = __bfloat162float(b[i]);
    }
    return N;
  }

 private:
  std::shared_ptr<lfm::Safetensors> weights_;
  std::vector<bool> attention_;
  float eps_ = 1e-5f;
  double theta_ = 1e6;
  std::vector<std::unique_ptr<Buffer<float>>> scratch_ = std::vector<std::unique_ptr<Buffer<float>>>(13);
  std::vector<size_t> sizes_ = std::vector<size_t>(13, 0);

  static std::string layer(int i) { return "model.audio_encoder.lfm.layers." + std::to_string(i) + "."; }
  const bf16* w(const std::string& name) const { return weights_->at(name).device_data; }
  // device scratch for one encode, released at its end; within an encode a slot is reused only after its contents are
  // consumed (kernels run in stream order, and a reallocation's cudaFree waits for the device)
  Buffer<float>& scratch(int slot, size_t n) {
    if (sizes_[slot] < n) scratch_[slot] = std::make_unique<Buffer<float>>(n), sizes_[slot] = n;
    return *scratch_[slot];
  }
  static std::vector<float> host(const float* p, size_t n) {
    std::vector<float> f(n);
    cuda_check(cudaMemcpy(f.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return f;
  }
};

}  // namespace lfm3_audio_in
