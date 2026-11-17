#pragma once
// Batched decode for the LFM3 text engine (P0 speed). Include AFTER lfm3_main.cu built with LFM3_RUNTIME_ONLY and
// LFM3_NATIVE_GEMM: it reuses the runner's kernels and weights.
//
// B sequences decode in lockstep. Prefill stays the validated Runner path: each prompt is prefilled by one Runner
// and its KV cache, conv state and last-position logits are copied into a row (load). A decode step then runs every
// row through each layer at once, so each weight is read once per step for all rows.
//
// Row independence (the batch gate): a row's tokens do not depend on which rows share its batch, or how many.
// - Projections use lfm::gemm's wide (tensor-core) kernel at every row count; its per-element arithmetic does not
//   depend on M (bench/lfm3_decode_gemm_bench.cu checks it bitwise). The serial runner's decode uses thin (M <= 8),
//   so a row here is not bitwise the serial runner's decode; it is its own engine path, gated on its own.
// - Norms run each row with the one-row block width (the serial decode's), not the row-count-dependent one.
// - Attention is split-K over fixed 256-key splits relative to the row's first key, combined in split order: it
//   depends only on the row's own cache and position.
// - Token selection runs on the device, one block per row, in index order.
// Attention keeps scores, the online softmax and P.V in FP32 (the serial path rounds probabilities to BF16 for P.V).
#include <lfm/gemm.cuh>
#include <lfm/gemm_lossless.cuh>

#include <climits>
#include <cstdint>
#include <type_traits>
#include <memory>
#include <vector>

namespace lfm3_batch {

constexpr int kSplit = 256;   // keys per attention split
constexpr int kGroupMax = 8;  // query heads per KV head supported by the attention kernel

// Y = X W^T (+ FP32 output): the wide kernel whatever the row count, in the tile shape measured fastest for the
// projection and row bucket (native/bench/lfm3_decode_tile_sweep.cu, Thor, distinct weights per instance). Every tile
// shape accumulates each output over the same k16 steps in the same order, so the choice never changes a value
// (the sweep checks it bitwise); a tile's row count is at least the batch's, so weights are read once per step.
namespace tiles {
using lfm::gemm::detail::Tile;
using S16 = Tile<16, 64, 64, 1, 4, 8>;    // narrow outputs and the 2048-wide projections, rows <= 16
using C32 = Tile<32, 64, 64, 1, 4, 6>;    // 2048 x 2048 and the long-K down projection
using W32 = Tile<32, 128, 64, 2, 2, 6>;   // wide outputs (MLP up, LM head), rows <= 32
using N32 = Tile<32, 64, 64, 2, 2, 6>;    // the long-K down projection, rows 17..32
using A64 = Tile<64, 64, 64, 2, 2, 4>;    // MLP up, rows 33..64
using B64 = Tile<64, 64, 64, 4, 2, 4>;    // 2048-wide and the down projection, rows 33..64
using H64 = Tile<64, 128, 64, 2, 2, 3>;   // conv input and LM head, rows 33..64
}  // namespace tiles
// Split-K for long reductions: K is cut into `splits` equal slices (a function of K alone, never of the batch), each
// slice's FP32 partial is the wide kernel over that slice (the slices ride on its batch dimension), and the partials
// are added in slice order. More blocks in flight for the 2048-wide, 7680-deep down projection.
__global__ void sum_splits(const float* parts, int splits, size_t count, bf16* y, float* y32) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float v = parts[i];
  for (int s = 1; s < splits; ++s) v = __fadd_rn(v, parts[size_t(s) * count + i]);
  if (y32) y32[i] = v; else y[i] = __float2bfloat16_rn(v);
}
inline int k_splits(int in) {
  static const int forced = [] { const char* v = std::getenv("LFM3_BATCH_SPLITK"); return v ? std::atoi(v) : -1; }();
  const int n = forced >= 0 ? forced : 1;  // measured: helps at most at batch 1, costs 10-20% at 32-64 rows
  return n > 1 && in % (64 * n) == 0 ? n : 1;
}
inline float* split_workspace(size_t floats) {
  static float* p = nullptr;
  static size_t size = 0;
  if (floats > size) {
    if (p) cuda_check(cudaFree(p));
    cuda_check(cudaMalloc(&p, floats * sizeof(float)));
    size = floats;
  }
  return p;
}
inline void linear_tiled(const lfm::gemm::Problem& p, int rows, int in, int out);
// LFM3_LOSSLESS=1: weights packed by pack_model() are read compressed (bitwise the same outputs).
inline bool lossless_enabled() {
  static const bool on = [] { const char* v = std::getenv("LFM3_LOSSLESS"); return v && std::string(v) == "1"; }();
  return on;
}
// Lossless tiles, measured (lfm3_lossless_check, Thor): the decode needs resident warps, so smaller, shallower tiles
// than the plain table. Above 32 rows the plain kernel is used (not yet measured for lossless).
using LosslessS = lfm::gemm::detail::Tile<16, 64, 64, 1, 4, 3>;  // rows <= 16: 16.5 vs 19.2 ms a step's projections
using LosslessM = lfm::gemm::detail::Tile<32, 64, 64, 2, 4, 3>;  // rows <= 32: 17.1 vs 19.3 ms
template <class T>
inline void launch(const lfm::gemm::Problem& p) {
  if (lossless_enabled() && p.m <= 32) {
    auto& table = lfm::gemm::packed_weights();
    auto it = table.find(p.b);
    if (it != table.end()) {
      if (p.m <= 16) return lfm::gemm::detail::launch_wide_lossless<LosslessS>(p, it->second, 0);
      return lfm::gemm::detail::launch_wide_lossless<LosslessM>(p, it->second, 0);
    }
  }
  lfm::gemm::detail::launch_wide<T, true>(p, 0);
}
// Packs every projection and the tied LM head once (LFM3_LOSSLESS=1). Returns the packed and the original bytes.
inline std::pair<size_t, size_t> pack_model(Runner& base) {
  static std::pair<size_t, size_t> done{0, 0};
  if (!lossless_enabled() || done.second) return done;
  const auto& s = base.s;
  auto add = [&](const std::string& name, int n, int k) {
    const bf16* w = base.w(name);
    if (lfm::gemm::packed_weights().count(w)) return;
    auto packed = lfm::gemm::pack(w, n, k);
    done.second += size_t(n) * k * 2;
    if (packed) lfm::gemm::packed_weights()[w] = packed, done.first += size_t(n) * k * 104 / 64;
    else done.first += size_t(n) * k * 2;
  };
  for (int i = 0; i < s.layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    if (base.config.attention[i]) {
      add(p + "self_attn.q_proj.weight", s.hidden, s.hidden); add(p + "self_attn.o_proj.weight", s.hidden, s.hidden);
      add(p + "self_attn.k_proj.weight", s.kv_width(), s.hidden); add(p + "self_attn.v_proj.weight", s.kv_width(), s.hidden);
      add(p + "self_attn.gate_proj.weight", s.query_heads, s.hidden);
    } else {
      add(p + "conv.in_proj.weight", 3 * s.hidden, s.hidden); add(p + "conv.out_proj.weight", s.hidden, s.hidden);
    }
    add(p + "feed_forward.w1.weight", s.mlp, s.hidden); add(p + "feed_forward.w3.weight", s.mlp, s.hidden);
    add(p + "feed_forward.w2.weight", s.hidden, s.mlp);
  }
  add("embed_tokens.weight", s.vocabulary, s.hidden);
  cuda_check(cudaDeviceSynchronize());
  return done;
}
inline void linear(const bf16* x, const bf16* w, bf16* y, int rows, int in, int out, float* y32 = nullptr) {
  const int splits = k_splits(in);
  if (splits > 1) {
    const int slice = in / splits;
    float* parts = split_workspace(size_t(splits) * rows * out);
    auto p = lfm::gemm::linear(x, w, nullptr, rows, slice, out);
    p.lda = in, p.ldb = in, p.c_f32 = parts;
    p.batch = splits, p.stride_a = slice, p.stride_b = slice, p.stride_c = (long long)rows * out;
    lfm::gemm::check(p);
    linear_tiled(p, rows, slice, out);
    sum_splits<<<grid(size_t(rows) * out), 256>>>(parts, splits, size_t(rows) * out, y, y32);
    return;
  }
  auto p = lfm::gemm::linear(x, w, y, rows, in, out);
  if (y32) p.c = nullptr, p.c_f32 = y32;
  lfm::gemm::check(p);
  linear_tiled(p, rows, in, out);
}
inline void linear_tiled(const lfm::gemm::Problem& p, int rows, int in, int out) {
  using namespace lfm::gemm::detail;
  using namespace tiles;
  // LFM3_BATCH_TILES=base: Wide32/64/128 by row count only (for paired measurement against the tuned table)
  static const bool base_tiles = [] { const char* v = std::getenv("LFM3_BATCH_TILES"); return v && std::string(v) == "base"; }();
  if (base_tiles) {
    if (rows <= 32) launch<Wide32>(p);
    else if (rows <= 64) launch<Wide64>(p);
    else launch<Wide128>(p);
    return;
  }
  const bool head = out >= 65536, up = out >= 6144 && out < 65536, longk = in > 4096;
  if (rows <= 16) {
    if (head || (up && out == 7680)) launch<W32>(p);
    else if (longk || out == 2048) launch<C32>(p);
    else launch<S16>(p);  // conv input (6144), K/V (512), gates
  } else if (rows <= 32) {
    if (head || up) launch<W32>(p);
    else if (longk) launch<N32>(p);
    else if (out == 2048) launch<C32>(p);
    else launch<lfm::gemm::detail::Wide32>(p);
  } else if (rows <= 64) {
    if (head || out == 6144) launch<H64>(p);
    else if (up) launch<A64>(p);
    else launch<B64>(p);
  } else {
    launch<lfm::gemm::detail::Wide128>(p);
  }
}

// K/V of row b into its cache [b][kv_head][capacity][dim] at position cursor[b]: active rows only. An inactive row
// (finished, or never loaded) is not written: its cursor can be at the capacity (a request that ran to its cap), where
// the write would land in the next head's or the next row's cache (validation/speech-batch-v1, red-team).
__global__ void scatter_rows(const bf16* k, const bf16* v, bf16* kc, bf16* vc, int rows, int heads, int dim,
                             int capacity, const int* cursor, const unsigned char* active) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * heads * dim) return;
  const int d = i % dim, h = (i / dim) % heads, b = i / (size_t(heads) * dim);
  if (!active[b] || cursor[b] < 0 || cursor[b] >= capacity) return;
  const size_t dst = ((size_t(b) * heads + h) * capacity + cursor[b]) * dim + d;
  kc[dst] = k[i];
  vc[dst] = v[i];
}

// One block per (split, KV head, row), 256 threads, head_dim 128: for the GROUP query heads of the KV head, the
// scores of the split's keys (FP32 dot, lane-strided then butterfly), their maximum m and sum l of exp(s - m), and
// the FP32 P.V partial over the split's keys in key order.
template <int GROUP>
__global__ void __launch_bounds__(256) attention_partial(const bf16* q, const bf16* kc, const bf16* vc,
                                                         const int* cursor, const unsigned char* active, int window,
                                                         int capacity, int kv_heads, int max_splits, float* part_o,
                                                         float* part_ml) {
  constexpr int D = 128;
  const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  if (!active[b]) return;
  const int pos = cursor[b], begin = window ? max(0, pos - window + 1) : 0, end = pos + 1;
  const int k0 = begin + split * kSplit;
  if (k0 >= end) return;
  const int n = min(kSplit, end - k0);
  const int heads = kv_heads * GROUP, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __shared__ float qs[GROUP][D];
  __shared__ float e[GROUP][kSplit];
  __shared__ float stats[GROUP][2];
  for (int i = threadIdx.x; i < GROUP * D; i += 256)
    qs[i / D][i % D] = __bfloat162float(q[(size_t(b) * heads + kvh * GROUP + i / D) * D + i % D]);
  __syncthreads();
  const bf16* keys = kc + ((size_t(b) * kv_heads + kvh) * capacity + k0) * D;
  const bf16* values = vc + ((size_t(b) * kv_heads + kvh) * capacity + k0) * D;
  const float scale = rsqrtf(float(D));
  for (int j = warp; j < n; j += 8) {
    const uint2 raw = *reinterpret_cast<const uint2*>(keys + size_t(j) * D + lane * 4);
    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 ka = __bfloat1622float2(k2[0]), kb = __bfloat1622float2(k2[1]);
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const float* qq = qs[g] + lane * 4;
      float s = __fmul_rn(qq[0], ka.x);
      s = __fmaf_rn(qq[1], ka.y, s);
      s = __fmaf_rn(qq[2], kb.x, s);
      s = __fmaf_rn(qq[3], kb.y, s);
#pragma unroll
      for (int o = 16; o; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (lane == 0) e[g][j] = __fmul_rn(s, scale);
    }
  }
  __syncthreads();
  if (warp < GROUP) {  // warp g: maximum, exponentials and their sum for query head g
    const int g = warp;
    float m = -INFINITY;
    for (int j = lane; j < n; j += 32) m = fmaxf(m, e[g][j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for (int j = lane; j < n; j += 32) {
      const float x = expf(__fsub_rn(e[g][j], m));
      e[g][j] = x;
      l = __fadd_rn(l, x);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, o));
    if (lane == 0) stats[g][0] = m, stats[g][1] = l;
  }
  __syncthreads();
  // P.V: thread t covers query head t / 64 (and t / 64 + 4, ...) and dims 2 (t % 64), +1, in key order.
  for (int g = threadIdx.x / 64; g < GROUP; g += 4) {
    const int d = (threadIdx.x % 64) * 2;
    float o0 = 0.f, o1 = 0.f;
    for (int j = 0; j < n; ++j) {
      const float2 vv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(values + size_t(j) * D + d));
      const float p = e[g][j];
      o0 = __fmaf_rn(p, vv.x, o0);
      o1 = __fmaf_rn(p, vv.y, o1);
    }
    const size_t slot = (size_t(b) * heads + kvh * GROUP + g) * max_splits + split;
    part_o[slot * D + d] = o0;
    part_o[slot * D + d + 1] = o1;
    if (d == 0) part_ml[slot * 2] = stats[g][0], part_ml[slot * 2 + 1] = stats[g][1];
  }
}

// One block per (query head, row), 128 threads: combine the row's splits in split order; inactive rows give zeros.
__global__ void attention_combine(const float* part_o, const float* part_ml, const int* cursor,
                                  const unsigned char* active, int window, int heads, int max_splits, bf16* out) {
  constexpr int D = 128;
  const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  bf16* o = out + (size_t(b) * heads + h) * D;
  if (!active[b]) { o[d] = __float2bfloat16_rn(0.f); return; }
  const int pos = cursor[b], begin = window ? max(0, pos - window + 1) : 0;
  const int splits = (pos + 1 - begin + kSplit - 1) / kSplit;
  const size_t base = (size_t(b) * heads + h) * max_splits;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, part_ml[(base + s) * 2]);
  float l = 0.f, acc = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float w = expf(__fsub_rn(part_ml[(base + s) * 2], m));
    l = __fmaf_rn(part_ml[(base + s) * 2 + 1], w, l);
    acc = __fmaf_rn(part_o[(base + s) * D + d], w, acc);
  }
  o[d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
}

// Greedy selection, one block of 1024 per row: the largest logit not suppressed, lowest id on ties (torch.argmax).
// Row b's suppression mask is suppress + b * suppress_stride (0: one mask for every row).
__global__ void __launch_bounds__(1024) greedy_rows(const float* logits, int vocab, const unsigned char* suppress,
                                                    const unsigned char* active, int* out, size_t suppress_stride = 0) {
  const int b = blockIdx.x;
  if (!active[b]) { if (threadIdx.x == 0) out[b] = -1; return; }
  const float* x = logits + size_t(b) * vocab;
  if (suppress) suppress += size_t(b) * suppress_stride;
  float best = -INFINITY;
  int id = INT_MAX;
  for (int i = threadIdx.x; i < vocab; i += 1024) {
    if (suppress && suppress[i]) continue;
    const float v = x[i];
    if (v > best || (v == best && i < id)) best = v, id = i;
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, id, o);
    if (ov > best || (ov == best && oi < id)) best = ov, id = oi;
  }
  if (threadIdx.x % 32 == 0) sv[threadIdx.x / 32] = best, si[threadIdx.x / 32] = id;
  __syncthreads();
  if (threadIdx.x < 32) {
    best = sv[threadIdx.x], id = si[threadIdx.x];
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, id, o);
      if (ov > best || (ov == best && oi < id)) best = ov, id = oi;
    }
    if (threadIdx.x == 0) out[b] = id;
  }
}

__device__ __forceinline__ uint64_t mix64(uint64_t x) {  // splitmix64
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Temperature sampling (no top-k/top-p/min-p, no penalty), one block of 1024 per row: p_i = exp((x_i - max) / T) /
// total over the unsuppressed vocabulary, in FP64; one uniform from (seed[b], step[b]); the inverse CDF in index
// order. Thread t owns ids [t C, (t + 1) C); its chunk sum is summed in index order, the chunk prefix by one thread.
__global__ void __launch_bounds__(1024) sample_rows(const float* logits, int vocab, const unsigned char* suppress,
                                                    const unsigned char* active, double temperature,
                                                    const uint64_t* seed, const uint64_t* step, int* out) {
  const int b = blockIdx.x, t = threadIdx.x;
  if (!active[b]) { if (t == 0) out[b] = -1; return; }
  const float* x = logits + size_t(b) * vocab;
  const int chunk = (vocab + 1023) / 1024, lo = min(vocab, t * chunk), hi = min(vocab, lo + chunk);
  float m = -INFINITY;
  for (int i = lo; i < hi; ++i) if (!(suppress && suppress[i])) m = fmaxf(m, x[i]);
  __shared__ float sm[32];
  __shared__ double sums[1024];
  __shared__ int found;
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (t % 32 == 0) sm[t / 32] = m;
  __syncthreads();
  if (t < 32) {
    m = sm[t];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (t == 0) sm[0] = m;
  }
  __syncthreads();
  m = sm[0];
  double s = 0;
  for (int i = lo; i < hi; ++i)
    if (!(suppress && suppress[i])) s += exp((double(x[i]) - double(m)) / temperature);
  sums[t] = s;
  __syncthreads();
  if (t == 0) {
    double acc = 0;
    for (int i = 0; i < 1024; ++i) { const double c = sums[i]; sums[i] = acc; acc += c; }  // exclusive prefix
    const double u = double(mix64(seed[b] ^ mix64(step[b])) >> 11) * (1.0 / 9007199254740992.0);
    const double target = u * acc;
    int c = 0;
    while (c + 1 < 1024 && sums[c + 1] <= target) ++c;
    double cum = sums[c];
    int pick = -1;
    const int clo = min(vocab, c * chunk), chi = min(vocab, clo + chunk);
    for (int i = clo; i < chi && pick < 0; ++i) {
      if (suppress && suppress[i]) continue;
      cum += exp((double(x[i]) - double(m)) / temperature);
      if (target < cum) pick = i;
    }
    // Rounding at the end of the CDF: the last unsuppressed id at or before this chunk.
    for (int i = chi - 1; pick < 0 && i >= 0; --i) if (!(suppress && suppress[i])) pick = i;
    found = pick;
  }
  __syncthreads();
  if (t == 0) out[b] = found;
}

// A3 (journal/lfm3-tts-quality-bar.md): a speech frame's input embedding, as HF's get_audio_out_embeds: the sum of
// its 12 codebook rows (row = code + codebook x 2017 of model.codebook_embedding), accumulated in FP32 in codebook
// order and rounded to BF16 once. Rows whose first code is negative are text rows (left as embedded).
__global__ void frame_embed(const bf16* table, const int* codes, bf16* hidden, int rows, int width, int codebooks,
                            int codebook_size) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * width) return;
  const int r = i / width, d = i % width;
  const int* c = codes + size_t(r) * codebooks;
  if (c[0] < 0) return;
  float sum = 0.f;
  for (int k = 0; k < codebooks; ++k)
    sum = __fadd_rn(sum, __bfloat162float(table[(size_t(c[k]) + size_t(k) * codebook_size) * width + d]));
  hidden[i] = __float2bfloat16_rn(sum);
}

class BatchDecoder {
 public:
  Runner& base;  // weights, shape and the prefill engine; its capacity is every row's
  const lfm::RuntimeShape s;
  const int rows;
  Buffer<bf16> hidden, normalized, projected, operated, updated, q, k, v, gates, attended, gated, a, b, act;
  Buffer<float> logits, part_o, part_ml;  // logits: with defer_head set, call ensure_logits() before touching them
  Buffer<int> tokens, positions, cursor_dev, picked, frame_codes;
  Buffer<unsigned char> active_dev;
  Buffer<uint64_t> seeds_dev, steps_dev;
  std::vector<std::unique_ptr<Buffer<bf16>>> keys, values, states;  // per layer, all rows
  std::vector<int> cursor;
  std::vector<unsigned char> active;
  int max_splits;
  std::vector<std::vector<uint16_t>>* embed_dump = nullptr;  // when set, step() appends row 0's input embedding

  BatchDecoder(Runner& runner, int batch)
      : base(runner), s(runner.s), rows(batch),
        hidden(size_t(batch) * s.hidden), normalized(size_t(batch) * s.hidden), projected(size_t(batch) * 3 * s.hidden),
        operated(size_t(batch) * s.hidden), updated(size_t(batch) * s.hidden), q(size_t(batch) * s.hidden),
        k(size_t(batch) * s.kv_width()), v(size_t(batch) * s.kv_width()), gates(size_t(batch) * s.query_heads),
        attended(size_t(batch) * s.hidden), gated(size_t(batch) * s.hidden), a(size_t(batch) * s.mlp),
        b(size_t(batch) * s.mlp), act(size_t(batch) * s.mlp), logits(size_t(batch) * s.vocabulary),
        part_o(size_t(batch) * s.query_heads * ((s.context + kSplit - 1) / kSplit) * 128),
        part_ml(size_t(batch) * s.query_heads * ((s.context + kSplit - 1) / kSplit) * 2),
        tokens(batch), positions(batch), cursor_dev(batch), picked(batch), frame_codes(size_t(batch) * 12),
        active_dev(batch), seeds_dev(batch),
        steps_dev(batch), cursor(batch, 0), active(batch, 0), max_splits((s.context + kSplit - 1) / kSplit) {
    if (batch < 1 || batch > 128) throw std::invalid_argument("batch must be 1..128");
    if (s.head_dim != 128 || s.query_heads % s.kv_heads || s.query_heads / s.kv_heads > kGroupMax ||
        s.query_heads / s.kv_heads != 4)
      throw std::invalid_argument("batched decode supports head_dim 128 with 4 query heads per KV head");
    for (int i = 0; i < s.layers; ++i) {
      if (base.config.attention[i]) {
        keys.emplace_back(new Buffer<bf16>(size_t(batch) * s.kv_layer_stride()));
        values.emplace_back(new Buffer<bf16>(size_t(batch) * s.kv_layer_stride()));
        states.emplace_back(nullptr);
      } else {
        keys.emplace_back(nullptr), values.emplace_back(nullptr);
        states.emplace_back(new Buffer<bf16>(size_t(batch) * s.hidden * 7));
      }
    }
    cuda_check(cudaMemset(active_dev.p, 0, batch));
    pack_model(base);
  }

  size_t device_bytes() const {
    size_t n = size_t(rows) * (s.hidden * 9 + 3 * s.hidden + 2 * s.kv_width() + s.query_heads + 3 * s.mlp) * 2 +
               size_t(rows) * s.vocabulary * 4 + size_t(rows) * s.query_heads * max_splits * 130 * 4;
    for (int i = 0; i < s.layers; ++i)
      n += base.config.attention[i] ? 2 * size_t(rows) * s.kv_layer_stride() * 2 : size_t(rows) * s.hidden * 7 * 2;
    return n;
  }

  // Row `row` takes the base runner's state after its prefill: KV caches, conv states, last-position logits.
  void load(int row) {
    if (row < 0 || row >= rows || base.cursor < 1) throw std::invalid_argument("load: invalid row or empty runner");
    ensure_logits();  // the other rows' deferred logits, from the states before this row's are replaced
    for (int i = 0; i < s.layers; ++i) {
      if (keys[i]) {
        cuda_check(cudaMemcpy(keys[i]->p + size_t(row) * s.kv_layer_stride(), base.keys[i]->p,
                              s.kv_layer_stride() * sizeof(bf16), cudaMemcpyDeviceToDevice));
        cuda_check(cudaMemcpy(values[i]->p + size_t(row) * s.kv_layer_stride(), base.values[i]->p,
                              s.kv_layer_stride() * sizeof(bf16), cudaMemcpyDeviceToDevice));
      } else {
        cuda_check(cudaMemcpy(states[i]->p + size_t(row) * s.hidden * 7, base.states[i]->p,
                              size_t(s.hidden) * 7 * sizeof(bf16), cudaMemcpyDeviceToDevice));
      }
    }
    static_assert(std::is_same_v<LFM3_LOGIT, float>, "batched decode needs the native-GEMM FP32 logits");
    cuda_check(cudaMemcpy(logits.p + size_t(row) * s.vocabulary, base.logits.p, size_t(s.vocabulary) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    // the prefill's last final-normed hidden state: what a speech frame sampled first reads (HF: hidden[:, -1])
    cuda_check(cudaMemcpy(normalized.p + size_t(row) * s.hidden, base.normalized.p + size_t(base.cursor - 1) * s.hidden,
                          size_t(s.hidden) * sizeof(bf16), cudaMemcpyDeviceToDevice));
    cursor[row] = base.cursor;
    active[row] = 1;
  }
  void finish(int row) { active[row] = 0; }

  // Deferred head (concurrent speech, validation/speech-speed-v1): with `defer_head` set, step() leaves the language
  // model's head (the vocabulary projection) to the first read of the logits: a selection, or a load, which overwrites
  // its own row afterwards. A step whose logits no one reads skips it. When computed, it is the same GEMM over the same
  // rows and the same final-normed states as at the end of the step, so the logits are the same.
  bool defer_head = false;
  void ensure_logits() {
    if (!head_pending_) return;
    head_pending_ = false;
    linear(normalized.p, base.w("embed_tokens.weight"), nullptr, rows, s.hidden, s.vocabulary, logits.p);
    cuda_check(cudaGetLastError());
  }

  // Next tokens from the current logits: greedy (temperature 0) or temperature sampling. -1 for inactive rows.
  std::vector<int> select(double temperature, const std::vector<uint64_t>& seed, const std::vector<uint64_t>& step,
                          const unsigned char* suppress_dev = nullptr) {
    ensure_logits();
    cuda_check(cudaMemcpy(active_dev.p, active.data(), rows, cudaMemcpyHostToDevice));
    if (temperature > 0) {
      cuda_check(cudaMemcpy(seeds_dev.p, seed.data(), rows * sizeof(uint64_t), cudaMemcpyHostToDevice));
      cuda_check(cudaMemcpy(steps_dev.p, step.data(), rows * sizeof(uint64_t), cudaMemcpyHostToDevice));
      sample_rows<<<rows, 1024>>>(logits.p, s.vocabulary, suppress_dev, active_dev.p, temperature, seeds_dev.p,
                                  steps_dev.p, picked.p);
    } else {
      greedy_rows<<<rows, 1024>>>(logits.p, s.vocabulary, suppress_dev, active_dev.p, picked.p);
    }
    cuda_check(cudaGetLastError());
    std::vector<int> out(rows);
    cuda_check(cudaMemcpy(out.data(), picked.p, rows * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
  }

  // Greedy picks with a suppression mask per row (rows x vocabulary bytes on the device): concurrent speech, where each
  // row's mode machine suppresses its own set. -1 for inactive rows.
  std::vector<int> select_greedy_rows(const unsigned char* suppress_rows_dev) {
    ensure_logits();
    cuda_check(cudaMemcpy(active_dev.p, active.data(), rows, cudaMemcpyHostToDevice));
    greedy_rows<<<rows, 1024>>>(logits.p, s.vocabulary, suppress_rows_dev, active_dev.p, picked.p, size_t(s.vocabulary));
    cuda_check(cudaGetLastError());
    std::vector<int> out(rows);
    cuda_check(cudaMemcpy(out.data(), picked.p, rows * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
  }

  // Row r's final-normed hidden state after the last step (or the load): the depth decoder's input for a frame.
  const bf16* hidden_state(int r) const { return normalized.p + size_t(r) * s.hidden; }

  // One decode step: token[r] enters row r at its cursor (inactive rows: ignored); logits for every active row.
  // frames (optional, rows x 12): a row whose first code is >= 0 enters as that speech frame's embedding (A3)
  // instead of its token's.
  void step(const std::vector<int>& token, const std::vector<int>* frames = nullptr) {
    std::vector<int> ids(rows, 0);
    for (int r = 0; r < rows; ++r) {
      if (!active[r]) continue;
      if (token[r] < 0 || token[r] >= s.vocabulary) throw std::invalid_argument("step: invalid token");
      if (cursor[r] >= s.context) throw std::runtime_error("step: row capacity exhausted");
      ids[r] = token[r];
    }
    cuda_check(cudaMemcpy(tokens.p, ids.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(cursor_dev.p, cursor.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(positions.p, cursor.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(active_dev.p, active.data(), rows, cudaMemcpyHostToDevice));
    const int R = rows, H = s.hidden;
    const int t_hidden = norm_threads(1, H), t_q = norm_threads(s.query_heads, s.head_dim),
              t_k = norm_threads(s.kv_heads, s.head_dim);
    auto w = [&](const std::string& n) { return base.w(n); };
    int longest = 0;  // full-attention splits: as many as the longest live row needs
    for (int r = 0; r < rows; ++r) if (active[r]) longest = std::max(longest, cursor[r] + 1);
    const int full_splits = std::max(1, (longest + kSplit - 1) / kSplit);
    embed<<<grid(size_t(R) * H), 256>>>(w("embed_tokens.weight"), tokens.p, hidden.p, R, H);
    if (frames) {
      if (frames->size() != size_t(R) * 12) throw std::invalid_argument("step: frames must be rows x 12");
      cuda_check(cudaMemcpy(frame_codes.p, frames->data(), size_t(R) * 12 * sizeof(int), cudaMemcpyHostToDevice));
      frame_embed<<<grid(size_t(R) * H), 256>>>(base.weights->at("model.codebook_embedding.weight").device_data,
                                                frame_codes.p, hidden.p, R, H, 12, 2017);
    }
    if (embed_dump) {  // diagnostics: row 0's input embedding
      std::vector<uint16_t> bits(H);
      cuda_check(cudaMemcpy(bits.data(), hidden.p, size_t(H) * 2, cudaMemcpyDeviceToHost));
      embed_dump->push_back(std::move(bits));
    }
    for (int layer = 0; layer < s.layers; ++layer) {
      const std::string p = "layers." + std::to_string(layer) + ".";
      const auto spec = base.config.layer_specs[layer];
      norm<<<R, t_hidden>>>(hidden.p, w(p + "operator_norm.weight"), normalized.p, H, s.epsilon);
      if (!base.config.attention[layer]) {
        linear(normalized.p, w(p + "conv.in_proj.weight"), projected.p, R, H, 3 * H);
        lfm::lfm3::convolution7<<<grid(size_t(R) * H), 256>>>(projected.p, w(p + "conv.conv.weight"), states[layer]->p,
                                                             operated.p, H, R);
        linear(operated.p, w(p + "conv.out_proj.weight"), updated.p, R, H, H);
      } else {
        const std::string at = p + "self_attn.";
        linear(normalized.p, w(at + "q_proj.weight"), q.p, R, H, H);
        linear(normalized.p, w(at + "k_proj.weight"), k.p, R, H, s.kv_width());
        linear(normalized.p, w(at + "v_proj.weight"), v.p, R, H, s.kv_width());
        norm<<<R * s.query_heads, t_q>>>(q.p, w(at + "q_norm.weight"), q.p, s.head_dim, s.epsilon);
        norm<<<R * s.kv_heads, t_k>>>(k.p, w(at + "k_norm.weight"), k.p, s.head_dim, s.epsilon);
        rotary_from_frequencies<<<grid(size_t(R) * H / 2), 256>>>(q.p, positions.p, base.frequencies[layer]->p, R,
                                                                  s.query_heads, s.head_dim, spec.rotary_pairs);
        rotary_from_frequencies<<<grid(size_t(R) * s.kv_width() / 2), 256>>>(k.p, positions.p, base.frequencies[layer]->p,
                                                                             R, s.kv_heads, s.head_dim, spec.rotary_pairs);
        scatter_rows<<<grid(size_t(R) * s.kv_width()), 256>>>(k.p, v.p, keys[layer]->p, values[layer]->p, R, s.kv_heads,
                                                             s.head_dim, s.context, cursor_dev.p, active_dev.p);
        const int splits = spec.window ? (spec.window + kSplit - 1) / kSplit : full_splits;
        attention_partial<4><<<dim3(splits, s.kv_heads, R), 256>>>(q.p, keys[layer]->p, values[layer]->p, cursor_dev.p,
                                                                   active_dev.p, spec.window, s.context, s.kv_heads,
                                                                   max_splits, part_o.p, part_ml.p);
        attention_combine<<<dim3(s.query_heads, R), 128>>>(part_o.p, part_ml.p, cursor_dev.p, active_dev.p, spec.window,
                                                           s.query_heads, max_splits, attended.p);
        linear(normalized.p, w(at + "gate_proj.weight"), gates.p, R, H, s.query_heads);
        lfm::lfm3::head_gate<<<grid(size_t(R) * H), 256>>>(attended.p, gates.p, gated.p, size_t(R) * H, s.head_dim);
        linear(gated.p, w(at + "o_proj.weight"), updated.p, R, H, H);
      }
      norm<<<R, t_hidden>>>(updated.p, w(p + "post_operator_norm.weight"), hidden.p, H, s.epsilon, hidden.p);
      norm<<<R, t_hidden>>>(hidden.p, w(p + "ffn_norm.weight"), normalized.p, H, s.epsilon);
      linear(normalized.p, w(p + "feed_forward.w1.weight"), a.p, R, H, s.mlp);
      linear(normalized.p, w(p + "feed_forward.w3.weight"), b.p, R, H, s.mlp);
      swiglu<<<grid(size_t(R) * s.mlp), 256>>>(a.p, b.p, act.p, size_t(R) * s.mlp);
      linear(act.p, w(p + "feed_forward.w2.weight"), updated.p, R, s.mlp, H);
      norm<<<R, t_hidden>>>(updated.p, w(p + "post_ffn_norm.weight"), hidden.p, H, s.epsilon, hidden.p);
    }
    norm<<<R, t_hidden>>>(hidden.p, w("embedding_norm.weight"), normalized.p, H, s.epsilon);
    head_pending_ = true;
    if (!defer_head) ensure_logits();
    cuda_check(cudaGetLastError());
    for (int r = 0; r < rows; ++r) if (active[r]) ++cursor[r];
  }

 private:
  bool head_pending_ = false;
};

}  // namespace lfm3_batch
