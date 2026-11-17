#pragma once
// Image input (#4 images in): the LFM3-Omni vision encoder and projector, natively (modeling_lfm3_vl.py:
// Lfm3VlVisionModel; modeling_lfm3_omni.py: Lfm3OmniMultiModalProjector, Lfm3OmniModel.get_image_features), for one
// image at a time (HF encodes several images frame by frame: attention and convolution never cross images).
//
// One image's pixel_values [N][1536] (lfm/image_preprocess.hpp; N = grid_h * grid_w patches in 2x2-merge-block order)
// become N / 4 vectors of the language model's width, which replace the image's <|image|> placeholders:
// - patchifier (linear 1536 -> 1024, no bias), embedding_norm (RMSNorm).
// - 14 pre-norm residual layers of width 1024, each a mixer then SwiGLU 2560 (RMSNorm before each, eps 1e-5):
//   * conv layers: in_proj (1024 -> 3 x 1024) split B, C, x; a depthwise 3x3 convolution (zero padding) of B * x over
//     the patch grid in raster order; times C; out_proj.
//   * attention layers: 16 query and 8 key/value heads of 64, RMSNorm on each query and key head, 2D axial rotary
//     embedding (dims 0-15 and 32-47 rotate by the patch row, 16-31 and 48-63 by the column; theta 1e6), bidirectional
//     over all of the image's patches, scale 1/8.
//   No final norm.
// - projector: LayerNorm(1024, eps 1e-6), the 4 tokens of each merge block concatenated (4096), linear 4096 -> 4096,
//   exact GELU, linear 4096 -> 2048 (biases on all three).
// Precision (validation/images-speed-v1): the residual stream, norms, softmax and nonlinearities in FP32; the matrix
// multiplies and attention on the tensor cores with every activation split into two BF16 parts (hi + lo, ~2^-16 of
// it) against the stored BF16 weights, FP32 accumulation: FP32's accuracy at the output (Precision::kX2, the default;
// kFp32, the CUDA-core path of images-v1, remains). Only the output is rounded to BF16, the language model's input. The rotary angle is the model's as HF
// defines it (float32 position times float32 inverse frequency); its cosine and sine, and the rotation, in FP64,
// rounded once. Attention: exact softmax per query (running maximum, FP32 sums), keys in a fixed order.
#include <lfm/fp32_ops.cuh>
#include <lfm/gemm.cuh>
#include <lfm/model_config.hpp>
#include <lfm/safetensors.hpp>
#include <lfm/safetensors_header.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm3_vision {

// How the encoder's matrix multiplies and attention compute (validation/images-speed-v1): kFp32, FP32 activations on
// the CUDA cores (images-v1, rule 2); kX1, BF16-rounded activations on the tensor cores, FP32 accumulation; kX2,
// activations split hi + lo into two BF16 parts on the tensor cores. Norms, softmax, nonlinearities and the residual
// stream are FP32 in every mode.
struct Precision {
  enum Mode { kFp32, kX1, kX2 };
  // the default: images-speed-v1 rule 1 (x2 on all 31 images, error 0.023-0.112 of HF BF16's, as FP32's)
  Mode linear = kX2, attention = kX2;
  static Mode parse(const std::string& s) {
    if (s == "fp32") return kFp32;
    if (s == "x1") return kX1;
    if (s == "x2") return kX2;
    throw std::runtime_error("vision precision must be fp32, x1 or x2: " + s);
  }
};

namespace kernels {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float w(const bf16* p, size_t i) { return __bfloat162float(p[i]); }

// Token t (2x2-merge-block order) <-> patch row y, column x of a grid `width` patches wide (HF's patchify order).
__device__ __forceinline__ int token_of(int y, int x, int width) {
  return (((y >> 1) * (width >> 1) + (x >> 1)) * 2 + (y & 1)) * 2 + (x & 1);
}
__device__ __forceinline__ void patch_of(int t, int width, int& y, int& x) {
  const int block = t >> 2, mh = (t >> 1) & 1, mw = t & 1, blocks_w = width >> 1;
  y = (block / blocks_w) * 2 + mh;
  x = (block % blocks_w) * 2 + mw;
}

// gated[t][c] = B[t][c] * x[t][c] from in_proj's [N][3 * D] (B, C, x)
__global__ void gate(const float* proj, float* gated, int n, int d) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(n) * d) return;
  const size_t t = i / d, c = i % d;
  gated[i] = __fmul_rn(proj[t * 3 * d + c], proj[t * 3 * d + 2 * d + c]);
}
// out[t][c] = C[t][c] * sum_{a,b} w[c][a][b] gated[patch (y - 1 + a, x - 1 + b)][c] (zero padding; torch's conv2d is
// a cross-correlation)
__global__ void spatial_conv(const float* gated, const float* proj, const bf16* wt, float* out, int height, int width,
                             int d) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = height * width;
  if (i >= size_t(n) * d) return;
  const int t = int(i / d), c = int(i % d);
  int y, x;
  patch_of(t, width, y, x);
  float sum = 0.0f;
  for (int a = 0; a < 3; ++a) {
    const int yy = y - 1 + a;
    if (yy < 0 || yy >= height) continue;
    for (int b = 0; b < 3; ++b) {
      const int xx = x - 1 + b;
      if (xx < 0 || xx >= width) continue;
      sum = __fmaf_rn(w(wt, size_t(c) * 9 + a * 3 + b), gated[size_t(token_of(yy, xx, width)) * d + c], sum);
    }
  }
  out[i] = __fmul_rn(proj[size_t(t) * 3 * d + d + c], sum);
}
// Axial rotary embedding in place over x [N][heads][64]: pair (k, k + 32), k < 32, rotated by angle[t][k]
// (cos_sin[t][k] = (cos, sin), FP64, from the host). rotate_half: out_k = a c - b s, out_{k+32} = b c + a s.
__global__ void axial_rotary(float* x, const double2* cos_sin, int n, int heads) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(n) * heads * 32) return;
  const int k = int(i % 32);
  const size_t head_row = i / 32, t = head_row / heads;
  const double2 cs = cos_sin[t * 32 + k];
  float* v = x + head_row * 64;
  const double a = v[k], b = v[k + 32];
  v[k] = float(a * cs.x - b * cs.y);
  v[k + 32] = float(b * cs.x + a * cs.y);
}
// Bidirectional attention over all n tokens: q [n][heads][64], k/v [n][kv_heads][64], out [n][heads][64]. A block of
// 64 threads owns 64 queries of one head (one query per thread); keys pass in tiles of 32 through shared memory, in
// ascending order. Per tile: the 32 scores, the running maximum, the old sums rescaled once, then the tile's terms.
constexpr int kAttnQueries = 64, kAttnKeys = 32;
__global__ void __launch_bounds__(kAttnQueries) attention(const float* q, const float* k, const float* v, float* out,
                                                          int n, int heads, int kv_heads, float scale) {
  __shared__ float ks[kAttnKeys][64], vs[kAttnKeys][64];
  const int h = blockIdx.y, g = h / (heads / kv_heads);
  const int query = blockIdx.x * kAttnQueries + threadIdx.x;
  const bool active = query < n;
  float qr[64], acc[64];
  for (int d = 0; d < 64; ++d) {
    qr[d] = active ? q[(size_t(query) * heads + h) * 64 + d] : 0.0f;
    acc[d] = 0.0f;
  }
  float running_max = -INFINITY, total = 0.0f;
  for (int key0 = 0; key0 < n; key0 += kAttnKeys) {
    __syncthreads();
    for (int e = threadIdx.x; e < kAttnKeys * 64; e += kAttnQueries) {
      const int j = e / 64, d = e % 64, key = key0 + j;
      ks[j][d] = key < n ? k[(size_t(key) * kv_heads + g) * 64 + d] : 0.0f;
      vs[j][d] = key < n ? v[(size_t(key) * kv_heads + g) * 64 + d] : 0.0f;
    }
    __syncthreads();
    const int count = min(kAttnKeys, n - key0);
    float s[kAttnKeys];
    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < kAttnKeys; ++j) {
      float dot = 0.0f;
#pragma unroll
      for (int d = 0; d < 64; ++d) dot = __fmaf_rn(qr[d], ks[j][d], dot);
      s[j] = j < count ? __fmul_rn(dot, scale) : -INFINITY;
      tile_max = fmaxf(tile_max, s[j]);
    }
    const float new_max = fmaxf(running_max, tile_max);
    const float rescale = expf(running_max - new_max);  // 0 on the first tile (exp(-inf))
    total = __fmul_rn(total, rescale);
#pragma unroll
    for (int d = 0; d < 64; ++d) acc[d] = __fmul_rn(acc[d], rescale);
#pragma unroll
    for (int j = 0; j < kAttnKeys; ++j) {
      const float p = j < count ? expf(s[j] - new_max) : 0.0f;
      total = __fadd_rn(total, p);
#pragma unroll
      for (int d = 0; d < 64; ++d) acc[d] = __fmaf_rn(p, vs[j][d], acc[d]);
    }
    running_max = new_max;
  }
  if (!active) return;
  for (int d = 0; d < 64; ++d) out[(size_t(query) * heads + h) * 64 + d] = __fdiv_rn(acc[d], total);
}
// ---- the tensor-core paths (validation/images-speed-v1) ----
// A float as BF16 parts: hi = bf16(v); lo = bf16(v - hi) (x2), so hi + lo holds v to about 2^-16 of it.
__device__ __forceinline__ void split_bf16(float v, bf16& hi, bf16& lo) {
  hi = __float2bfloat16_rn(v);
  lo = __float2bfloat16_rn(__fsub_rn(v, __bfloat162float(hi)));
}
// x [rows][in] FP32 -> lfm::gemm's A operand [rows][parts * in]: hi in columns 0..in, lo (x2) in in..2 in, so one
// GEMM against [W | W] (double_rows) sums hi·W + lo·W.
__global__ void split_rows(const float* x, bf16* out, int rows, int in, int parts) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * in) return;
  const size_t r = i / in, c = i % in;
  bf16 hi, lo;
  split_bf16(x[i], hi, lo);
  out[r * parts * in + c] = hi;
  if (parts == 2) out[r * parts * in + in + c] = lo;
}
// [W | W]: each row of W [rows][in] twice, the B operand of the split multiply
__global__ void double_rows(const bf16* w, bf16* out, int rows, int in) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * in) return;
  const size_t r = i / in, c = i % in;
  out[r * 2 * in + c] = out[r * 2 * in + in + c] = w[i];
}
// y[i] = c[i] + bias[i % cols]
__global__ void add_bias(const float* c, const bf16* bias, size_t n, int cols, float* y) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __fadd_rn(c[i], __bfloat162float(bias[i % cols]));
}
// K and V split once per layer into the attention tiles' layouts: ks [kv][npad][PARTS * 64] (part p in columns
// 64 p ..), vt [kv][PARTS][64][npad] (V transposed); keys at or past n are zeros. A block: 64 keys of one kv head,
// loaded coalesced, V transposed through shared memory.
__global__ void split_kv(const float* k, const float* v, bf16* ks, bf16* vt, int n, int npad, int kv_heads, int parts) {
  __shared__ float tile[64][65];
  const int key0 = blockIdx.x * 64, g = blockIdx.y;
  for (int e = threadIdx.x; e < 64 * 64; e += blockDim.x) {
    const int j = e / 64, d = e % 64, key = key0 + j;
    const float kx = key < n ? k[(size_t(key) * kv_heads + g) * 64 + d] : 0.0f;
    tile[j][d] = key < n ? v[(size_t(key) * kv_heads + g) * 64 + d] : 0.0f;
    bf16 hi, lo;
    split_bf16(kx, hi, lo);
    bf16* row = ks + (size_t(g) * npad + key) * (parts * 64);
    row[d] = hi;
    if (parts == 2) row[64 + d] = lo;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 64 * 64; e += blockDim.x) {
    const int d = e / 64, j = e % 64;
    bf16 hi, lo;
    split_bf16(tile[j][d], hi, lo);
    vt[(size_t(g) * parts * 64 + d) * npad + key0 + j] = hi;
    if (parts == 2) vt[(size_t(g) * parts * 64 + 64 + d) * npad + key0 + j] = lo;
  }
}
__device__ __forceinline__ uint32_t pack_bf16(float lo_col, float hi_col) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo_col, hi_col);
  return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Fused bidirectional attention on the tensor cores (mma.sync m16n8k16, BF16 operands, FP32 accumulation), head
// dimension 64: q [n][heads][64], k/v [n][kv_heads][64] FP32, out [n][heads][64] FP32. A block's 4 warps own 16
// queries each of one head; keys pass in tiles of 64 through shared memory (K as [key][dim], V transposed as
// [dim][key], rows padded to 72 so a warp's fragment loads hit distinct banks), in ascending order. Per tile: S = Q·Kᵀ
// into FP32 accumulators, the scale, the running maximum and the rescale of the old sums in FP32 (as the CUDA-core
// kernel), then O += P·V with P taken from S's accumulators (their layout is the next product's A operand).
// PARTS 1: Q, K, V and P rounded to BF16; PARTS 2: each split hi + lo, three products (hi·hi + hi·lo + lo·hi).
// K and V come split (split_kv): ks [kv][npad][PARTS * 64], vt [kv][PARTS][64][npad]. Tiles arrive by cp.async,
// double buffered (the next tile loads while this one computes), into dynamic shared memory; B fragments come by
// ldmatrix.x4 (four 8x8 matrices a load). The products and their order are those of the kernel this replaced
// (images-speed-v1: outputs byte-identical): S per key block nb, k steps 0..3, hi·hi, hi·lo, lo·hi; then P·V per
// k step t, dimension blocks 0..7, the same three.
constexpr int kTcWarps = 8, kTcQueries = 16 * kTcWarps, kTcKeys = 64, kTcPad = 72;
template <int PARTS>
constexpr int tc_shared_bytes() { return 2 * 2 * PARTS * kTcKeys * kTcPad * int(sizeof(bf16)); }
__device__ __forceinline__ unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
template <int PARTS>
__global__ void __launch_bounds__(32 * kTcWarps) attention_tc(const float* q, const bf16* ksg, const bf16* vtg,
                                                              float* out, int n, int npad, int heads, int kv_heads,
                                                              float scale) {
  using Tile = bf16[PARTS][kTcKeys][kTcPad];
  extern __shared__ __align__(16) unsigned char smem[];
  Tile* ks = reinterpret_cast<Tile*>(smem);                                        // [stage] K tile [part][key][dim]
  Tile* vt = reinterpret_cast<Tile*>(smem + 2 * sizeof(Tile));                     // [stage] V tile [part][dim][key]
  const int h = blockIdx.y, g = h / (heads / kv_heads);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gid = lane >> 2, tig = lane & 3;
  const int q0 = blockIdx.x * kTcQueries + warp * 16;
  const int mi = lane >> 3, mr = lane & 7;  // ldmatrix: this lane addresses row mr of matrix mi
  uint32_t qa[PARTS][4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = q0 + gid + (r & 1) * 8, d = s * 16 + tig * 2 + (r >> 1) * 8;
      float a = 0.0f, b = 0.0f;
      if (row < n) {
        const float* src = q + (size_t(row) * heads + h) * 64 + d;
        a = src[0], b = src[1];
      }
      bf16 ah, al, bh, bl;
      split_bf16(a, ah, al);
      split_bf16(b, bh, bl);
      qa[0][s][r] = pack_bf16(__bfloat162float(ah), __bfloat162float(bh));
      if constexpr (PARTS == 2) qa[PARTS - 1][s][r] = pack_bf16(__bfloat162float(al), __bfloat162float(bl));
    }
  const bf16* kg = ksg + size_t(g) * npad * (PARTS * 64);
  const bf16* vg = vtg + size_t(g) * PARTS * 64 * npad;
  auto load = [&](int stage, int key0) {
    for (int e = threadIdx.x; e < kTcKeys * PARTS * 8; e += 32 * kTcWarps) {
      const int j = e / (PARTS * 8), c = e % (PARTS * 8);
      lfm::fp32::kernels::copy16_async(&ks[stage][c / 8][j][(c % 8) * 8], kg + size_t(key0 + j) * (PARTS * 64) + c * 8);
    }
    for (int e = threadIdx.x; e < PARTS * 64 * 8; e += 32 * kTcWarps) {
      const int pd = e / 8, c = e % 8;  // pd = part * 64 + dim
      lfm::fp32::kernels::copy16_async(&vt[stage][pd / 64][pd % 64][c * 8], vg + size_t(pd) * npad + key0 + c * 8);
    }
    lfm::fp32::kernels::commit_async();
  };
  float o[8][4] = {};
  float row_max[2] = {-INFINITY, -INFINITY}, row_sum[2] = {0.0f, 0.0f};
  load(0, 0);
  for (int key0 = 0, stage = 0; key0 < n; key0 += kTcKeys, stage ^= 1) {
    if (key0 + kTcKeys < n) {
      load(stage ^ 1, key0 + kTcKeys);
      lfm::fp32::kernels::wait_async<1>();
    } else {
      lfm::fp32::kernels::wait_async<0>();
    }
    __syncthreads();
    const Tile& K = ks[stage];
    const Tile& Vt = vt[stage];
    float s[8][4] = {};
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int st = 0; st < 4; st += 2) {
        uint32_t b[PARTS][4];
#pragma unroll
        for (int part = 0; part < PARTS; ++part)
          lfm::gemm::detail::ldmatrix_x4(shared_address(&K[part][nb * 8 + mr][(st + (mi >> 1)) * 16 + (mi & 1) * 8]),
                                         b[part][0], b[part][1], b[part][2], b[part][3]);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          mma_bf16(s[nb], qa[0][st + u], b[0][2 * u], b[0][2 * u + 1]);
          if constexpr (PARTS == 2) {
            mma_bf16(s[nb], qa[0][st + u], b[PARTS - 1][2 * u], b[PARTS - 1][2 * u + 1]);
            mma_bf16(s[nb], qa[PARTS - 1][st + u], b[0][2 * u], b[0][2 * u + 1]);
          }
        }
      }
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int key = key0 + nb * 8 + tig * 2 + (c & 1);
        s[nb][c] = key < n ? __fmul_rn(s[nb][c], scale) : -INFINITY;
        tile_max[c >> 1] = fmaxf(tile_max[c >> 1], s[nb][c]);
      }
    float rescale[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
      const float new_max = fmaxf(row_max[r], tile_max[r]);
      rescale[r] = expf(row_max[r] - new_max);  // 0 on the first tile (exp(-inf))
      row_max[r] = new_max;
      row_sum[r] = __fmul_rn(row_sum[r], rescale[r]);
    }
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        o[nb][c] = __fmul_rn(o[nb][c], rescale[c >> 1]);
        const float p = expf(s[nb][c] - row_max[c >> 1]);
        s[nb][c] = p;
        row_sum[c >> 1] = __fadd_rn(row_sum[c >> 1], p);
      }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      uint32_t pa[PARTS][4];
      const float* lo_blk = s[2 * t];
      const float* hi_blk = s[2 * t + 1];
      const float vals[4][2] = {{lo_blk[0], lo_blk[1]}, {lo_blk[2], lo_blk[3]}, {hi_blk[0], hi_blk[1]}, {hi_blk[2], hi_blk[3]}};
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        bf16 ah, al, bh, bl;
        split_bf16(vals[r][0], ah, al);
        split_bf16(vals[r][1], bh, bl);
        pa[0][r] = pack_bf16(__bfloat162float(ah), __bfloat162float(bh));
        if constexpr (PARTS == 2) pa[PARTS - 1][r] = pack_bf16(__bfloat162float(al), __bfloat162float(bl));
      }
#pragma unroll
      for (int nb = 0; nb < 8; nb += 2) {
        uint32_t b[PARTS][4];
#pragma unroll
        for (int part = 0; part < PARTS; ++part)
          lfm::gemm::detail::ldmatrix_x4(shared_address(&Vt[part][(nb + (mi >> 1)) * 8 + mr][t * 16 + (mi & 1) * 8]),
                                         b[part][0], b[part][1], b[part][2], b[part][3]);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          mma_bf16(o[nb + u], pa[0], b[0][2 * u], b[0][2 * u + 1]);
          if constexpr (PARTS == 2) {
            mma_bf16(o[nb + u], pa[0], b[PARTS - 1][2 * u], b[PARTS - 1][2 * u + 1]);
            mma_bf16(o[nb + u], pa[PARTS - 1], b[0][2 * u], b[0][2 * u + 1]);
          }
        }
      }
    }
    __syncthreads();  // this stage is reloaded two tiles on
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    row_sum[r] = __fadd_rn(row_sum[r], __shfl_xor_sync(0xffffffffu, row_sum[r], 1));
    row_sum[r] = __fadd_rn(row_sum[r], __shfl_xor_sync(0xffffffffu, row_sum[r], 2));
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = q0 + gid + r * 8;
    if (row >= n) continue;
    float* dst = out + (size_t(row) * heads + h) * 64;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
      dst[nb * 8 + tig * 2] = __fdiv_rn(o[nb][r * 2], row_sum[r]);
      dst[nb * 8 + tig * 2 + 1] = __fdiv_rn(o[nb][r * 2 + 1], row_sum[r]);
    }
  }
}
__global__ void add(float* x, const float* y, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __fadd_rn(x[i], y[i]);
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
// exact GELU, x/2 (1 + erf(x / sqrt 2)), in FP64, rounded once
__global__ void gelu(float* x, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { const double v = x[i]; x[i] = float(0.5 * v * (1.0 + erf(v * 0.70710678118654752440))); }
}
__global__ void to_bf16(const float* x, bf16* y, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}
}  // namespace kernels

// Host copies of every stage for one image (the parity check against HF; tools/dump_lfm3_vision_reference.py names).
struct Dumps {
  std::vector<float> patchifier, embedding_norm, cos, sin, proj_layer_norm, proj_linear_1, proj_gelu, proj_linear_2,
      features;
  std::vector<std::vector<float>> layers;
};

class Encoder {
 public:
  using bf16 = __nv_bfloat16;
  static constexpr int kWidth = 1024, kLayers = 14, kHeads = 16, kKvHeads = 8, kHeadDim = 64, kMlp = 2560,
                       kPatchDim = 1536, kMerge = 2, kProjHidden = 4096, kOut = 2048;

  // The architecture this code implements, checked against config.json's root and the tensor table; throws on any
  // difference, so a checkpoint the encoder cannot run is known before any request.
  // `lookup`: a tensor's dtype and shape by name (nullopt: absent), from the loaded weights or from the safetensors
  // header alone (WorkerCore at startup: whether image input is supported, before any weights load).
  static void check_contract(const lfm::Json& root,
                             const std::function<std::optional<lfm::safetensors_header::Entry>(const std::string&)>& lookup) {
    const auto& vc = root.at("vision_config");
    auto need = [](bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("vision encoder: ") + what); };
    need(vc.at("hidden_size").integer() == kWidth && vc.at("intermediate_size").integer() == kMlp &&
         vc.at("num_hidden_layers").integer() == kLayers && vc.at("num_attention_heads").integer() == kHeads &&
         vc.at("num_key_value_heads").integer() == kKvHeads && vc.at("patch_size").integer() == 16 &&
         vc.at("temporal_patch_size").integer() == 2 && vc.at("in_channels").integer() == 3 &&
         vc.at("spatial_merge_size").integer() == kMerge && vc.at("conv_kernel_size").integer() == 3 &&
         vc.at("rope_parameters").at("rope_type").string() == "axial", "unexpected vision geometry");
    need(root.at("downsample_factor").integer() == kMerge && root.at("projector_hidden_size").integer() == kProjHidden &&
         root.at("projector_hidden_act").string() == "gelu" && root.at("projector_bias").boolean() &&
         root.at("text_config").at("hidden_size").integer() == kOut, "unexpected projector");
    (void)vc.at("norm_eps").number(), (void)vc.at("rope_parameters").at("rope_theta").number();
    (void)root.at("projector_layer_norm_eps").number();
    const auto& types = vc.at("layer_types").array();
    need(types.size() == size_t(kLayers), "layer_types length");
    auto check = [&](const std::string& name, std::vector<int64_t> shape) {
      const auto t = lookup(name);
      if (!t) throw std::runtime_error("vision encoder: missing " + name);
      if (t->dtype != "BF16" || t->shape != shape) throw std::runtime_error("vision encoder weight contract: " + name);
    };
    const std::string V = "model.vision_tower.";
    check(V + "patchifier.weight", {kWidth, kPatchDim});
    check(V + "embedding_norm.weight", {kWidth});
    for (int i = 0; i < kLayers; ++i) {
      const auto& t = types[i].string();
      need(t == "conv" || t == "full_attention", "layer type");
      const std::string p = layer(i);
      check(p + "operator_norm.weight", {kWidth}); check(p + "ffn_norm.weight", {kWidth});
      check(p + "feed_forward.w1.weight", {kMlp, kWidth}); check(p + "feed_forward.w3.weight", {kMlp, kWidth});
      check(p + "feed_forward.w2.weight", {kWidth, kMlp});
      if (t == "full_attention") {
        check(p + "self_attn.q_proj.weight", {kHeads * kHeadDim, kWidth});
        check(p + "self_attn.k_proj.weight", {kKvHeads * kHeadDim, kWidth});
        check(p + "self_attn.v_proj.weight", {kKvHeads * kHeadDim, kWidth});
        check(p + "self_attn.out_proj.weight", {kWidth, kHeads * kHeadDim});
        check(p + "self_attn.q_layernorm.weight", {kHeadDim}); check(p + "self_attn.k_layernorm.weight", {kHeadDim});
      } else {
        check(p + "conv.in_proj.weight", {3 * kWidth, kWidth}); check(p + "conv.out_proj.weight", {kWidth, kWidth});
        check(p + "conv.spatial_conv.weight", {kWidth, 1, 3, 3});
      }
    }
    const std::string P = "model.multi_modal_projector.";
    check(P + "layer_norm.weight", {kWidth}); check(P + "layer_norm.bias", {kWidth});
    check(P + "linear_1.weight", {kProjHidden, kWidth * kMerge * kMerge}); check(P + "linear_1.bias", {kProjHidden});
    check(P + "linear_2.weight", {kOut, kProjHidden}); check(P + "linear_2.bias", {kOut});
  }

  Encoder(std::shared_ptr<lfm::Safetensors> weights, const lfm::Json& root, Precision precision = {})
      : weights_(std::move(weights)), precision_(precision) {
    check_contract(root, [&](const std::string& name) -> std::optional<lfm::safetensors_header::Entry> {
      if (!weights_->has(name)) return std::nullopt;
      const auto& t = weights_->at(name);
      return lfm::safetensors_header::Entry{t.dtype, t.shape};
    });
    const auto& vc = root.at("vision_config");
    eps_ = float(vc.at("norm_eps").number());
    theta_ = vc.at("rope_parameters").at("rope_theta").number();
    projector_eps_ = float(root.at("projector_layer_norm_eps").number());
    for (const auto& t : vc.at("layer_types").array()) attention_.push_back(t.string() == "full_attention");
    // HF's inverse frequencies, as float32 arithmetic gives them: 1 / (theta ** (arange(0, 32, 2) / 32))
    for (int i = 0; i < 16; ++i) inv_freq_[i] = 1.0f / std::pow(float(theta_), float(2 * i) / 32.0f);
    // the tensor-core attention's dynamic shared memory (above the 48 KB default), on the device this Encoder runs on
    // (set per Encoder, not once per process: a function attribute belongs to the device it is set on)
    if (cudaFuncSetAttribute(kernels::attention_tc<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kernels::tc_shared_bytes<2>()) != cudaSuccess ||
        cudaFuncSetAttribute(kernels::attention_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kernels::tc_shared_bytes<1>()) != cudaSuccess)
      throw std::runtime_error("vision attention: cannot configure shared memory");
  }

  // Encodes one image (pixel_values: [N][1536], N = grid_h * grid_w, in merge-block order) into `out` (device,
  // [N / 4][2048] BF16); returns N / 4. `dumps`, when set, receives host copies of every stage.
  int encode(const float* pixel_values, int grid_h, int grid_w, bf16* out, Dumps* dumps = nullptr) {
    using namespace kernels;
    namespace f = lfm::fp32;
    if (grid_h <= 0 || grid_w <= 0 || grid_h % kMerge || grid_w % kMerge)
      throw std::runtime_error("vision encoder: the patch grid must be positive multiples of 2");
    const int N = grid_h * grid_w, T = N / (kMerge * kMerge);
    const size_t NW = size_t(N) * kWidth;
    auto& in = scratch(0, size_t(N) * kPatchDim);
    f::check(cudaMemcpy(in.p, pixel_values, size_t(N) * kPatchDim * sizeof(float), cudaMemcpyHostToDevice), "pixels");
    const std::string V = "model.vision_tower.";
    auto& x = scratch(1, NW);
    mul(in.p, w(V + "patchifier.weight"), nullptr, x.p, N, kWidth, kPatchDim);
    if (dumps) dumps->patchifier = host(x.p, NW);
    f::rms_norm(x.p, w(V + "embedding_norm.weight"), x.p, nullptr, N, kWidth, eps_);
    if (dumps) dumps->embedding_norm = host(x.p, NW), dumps->layers.clear();
    // rotary table: token t at patch (row, column); angle_k = float(pos) * inv_freq (float32), pos the row for
    // k < 16 and the column for 16 <= k < 32; cosine and sine in FP64
    std::vector<double2> table(size_t(N) * 32);
    for (int t = 0; t < N; ++t) {
      const int block = t / 4, row = (block / (grid_w / 2)) * 2 + (t / 2) % 2, col = (block % (grid_w / 2)) * 2 + t % 2;
      for (int k = 0; k < 32; ++k) {
        const float angle = float(k < 16 ? row : col) * inv_freq_[k % 16];
        table[size_t(t) * 32 + k] = double2{std::cos(double(angle)), std::sin(double(angle))};
      }
    }
    if (dumps) {
      dumps->cos.resize(size_t(N) * 64), dumps->sin.resize(size_t(N) * 64);
      for (size_t t = 0; t < size_t(N); ++t)
        for (int k = 0; k < 64; ++k) {
          dumps->cos[t * 64 + k] = float(table[t * 32 + k % 32].x);
          dumps->sin[t * 64 + k] = float(table[t * 32 + k % 32].y);
        }
    }
    if (rope_.n < table.size()) rope_ = lfm::fp32::DeviceArray<double2>(table.size());
    auto& rope = rope_;
    f::check(cudaMemcpy(rope.p, table.data(), table.size() * sizeof(double2), cudaMemcpyHostToDevice), "rotary");
    auto& n = scratch(2, NW);
    auto& proj = scratch(3, NW * 3);  // in_proj (3 x 1024), or q/k/v
    auto& op = scratch(4, NW);
    auto& ff_a = scratch(5, size_t(N) * kMlp);
    auto& ff_b = scratch(6, size_t(N) * kMlp);
    auto& mix = scratch(7, NW);
    const int kv = kKvHeads * kHeadDim;
    for (int i = 0; i < kLayers; ++i) {
      const std::string p = layer(i);
      f::rms_norm(x.p, w(p + "operator_norm.weight"), n.p, nullptr, N, kWidth, eps_);
      if (!attention_[i]) {
        mul(n.p, w(p + "conv.in_proj.weight"), nullptr, proj.p, N, 3 * kWidth, kWidth);
        gate<<<f::blocks(NW), 256>>>(proj.p, mix.p, N, kWidth);
        spatial_conv<<<f::blocks(NW), 256>>>(mix.p, proj.p, w(p + "conv.spatial_conv.weight"), op.p, grid_h, grid_w,
                                             kWidth);
        mul(op.p, w(p + "conv.out_proj.weight"), nullptr, mix.p, N, kWidth, kWidth);
      } else {
        float *q = proj.p, *k = proj.p + NW, *v = proj.p + NW + size_t(N) * kv;
        mul(n.p, w(p + "self_attn.q_proj.weight"), nullptr, q, N, kWidth, kWidth);
        mul(n.p, w(p + "self_attn.k_proj.weight"), nullptr, k, N, kv, kWidth);
        mul(n.p, w(p + "self_attn.v_proj.weight"), nullptr, v, N, kv, kWidth);
        f::rms_norm(q, w(p + "self_attn.q_layernorm.weight"), q, nullptr, N * kHeads, kHeadDim, eps_);
        f::rms_norm(k, w(p + "self_attn.k_layernorm.weight"), k, nullptr, N * kKvHeads, kHeadDim, eps_);
        axial_rotary<<<f::blocks(size_t(N) * kHeads * 32), 256>>>(q, rope.p, N, kHeads);
        axial_rotary<<<f::blocks(size_t(N) * kKvHeads * 32), 256>>>(k, rope.p, N, kKvHeads);
        attend(q, k, v, op.p, N);
        mul(op.p, w(p + "self_attn.out_proj.weight"), nullptr, mix.p, N, kWidth, kWidth);
      }
      add<<<f::blocks(NW), 256>>>(x.p, mix.p, NW);
      f::rms_norm(x.p, w(p + "ffn_norm.weight"), n.p, nullptr, N, kWidth, eps_);
      mul(n.p, w(p + "feed_forward.w1.weight"), nullptr, ff_a.p, N, kMlp, kWidth);
      mul(n.p, w(p + "feed_forward.w3.weight"), nullptr, ff_b.p, N, kMlp, kWidth);
      f::swiglu(ff_a.p, ff_b.p, size_t(N) * kMlp);
      mul(ff_a.p, w(p + "feed_forward.w2.weight"), nullptr, mix.p, N, kWidth, kMlp);
      add<<<f::blocks(NW), 256>>>(x.p, mix.p, NW);
      if (dumps) dumps->layers.push_back(host(x.p, NW));
    }
    // projector: the merge block's 4 tokens are consecutive rows, so [N][1024] is [T][4096] as it lies
    const std::string P = "model.multi_modal_projector.";
    layer_norm<<<f::blocks(size_t(N) * 32), 256>>>(x.p, w(P + "layer_norm.weight"), w(P + "layer_norm.bias"), n.p, N,
                                                   kWidth, projector_eps_);
    if (dumps) dumps->proj_layer_norm = host(n.p, NW);
    auto& hid = scratch(5, size_t(T) * kProjHidden);
    mul(n.p, w(P + "linear_1.weight"), w(P + "linear_1.bias"), hid.p, T, kProjHidden, kWidth * 4);
    if (dumps) dumps->proj_linear_1 = host(hid.p, size_t(T) * kProjHidden);
    gelu<<<f::blocks(size_t(T) * kProjHidden), 256>>>(hid.p, size_t(T) * kProjHidden);
    if (dumps) dumps->proj_gelu = host(hid.p, size_t(T) * kProjHidden);
    auto& feat = scratch(6, size_t(T) * kOut);
    mul(hid.p, w(P + "linear_2.weight"), w(P + "linear_2.bias"), feat.p, T, kOut, kProjHidden);
    if (dumps) dumps->proj_linear_2 = host(feat.p, size_t(T) * kOut);
    to_bf16<<<f::blocks(size_t(T) * kOut), 256>>>(feat.p, out, size_t(T) * kOut);
    f::check(cudaGetLastError(), "vision encoder launch");
    f::check(cudaDeviceSynchronize(), "vision encoder");
    if (dumps) {
      std::vector<bf16> b(size_t(T) * kOut);
      f::check(cudaMemcpy(b.data(), out, b.size() * sizeof(bf16), cudaMemcpyDeviceToHost), "features");
      dumps->features.resize(b.size());
      for (size_t i = 0; i < b.size(); ++i) dumps->features[i] = __bfloat162float(b[i]);
    }
    return T;
  }

 private:
  std::shared_ptr<lfm::Safetensors> weights_;
  std::vector<bool> attention_;
  float eps_ = 1e-5f, projector_eps_ = 1e-6f;
  double theta_ = 1e6;
  float inv_freq_[16] = {};
  Precision precision_;
  std::vector<lfm::fp32::DeviceArray<float>> scratch_ = std::vector<lfm::fp32::DeviceArray<float>>(9);
  // scratch: kept between images (images-speed-v1 amendment 1), grown to the largest image seen
  std::vector<lfm::fp32::DeviceArray<bf16>> scratch16_ = std::vector<lfm::fp32::DeviceArray<bf16>>(3);
  std::map<const bf16*, lfm::fp32::DeviceArray<bf16>> doubled_;
  lfm::fp32::DeviceArray<double2> rope_;

  // y [rows][out] = x [rows][in] · Wᵀ (+ bias): the CUDA cores (kFp32), or lfm::gemm on the tensor cores with the
  // activations as one BF16 part (kX1) or two (kX2: [hi | lo] against [W | W], one GEMM over K = 2 in), FP32 out.
  void mul(const float* x, const bf16* W, const bf16* bias, float* y, int rows, int out, int in) {
    namespace f = lfm::fp32;
    if (precision_.linear == Precision::kFp32) {
      f::linear(x, W, bias, y, rows, out, in);
      return;
    }
    const int parts = precision_.linear == Precision::kX2 ? 2 : 1;
    const size_t a = size_t(rows) * parts * in, c = size_t(rows) * out;
    if (scratch16_[0].n < a) scratch16_[0] = f::DeviceArray<bf16>(a);
    kernels::split_rows<<<f::blocks(size_t(rows) * in), 256>>>(x, scratch16_[0].p, rows, in, parts);
    float* target = bias ? scratch(8, c).p : y;
    lfm::gemm::Problem p = lfm::gemm::linear(scratch16_[0].p, parts == 2 ? doubled(W, out, in) : W, nullptr, rows,
                                             parts * in, out);
    p.c = nullptr, p.c_f32 = target;
    lfm::gemm::run(p);
    if (bias) kernels::add_bias<<<f::blocks(c), 256>>>(target, bias, c, out, y);
  }
  // [W | W] for the split multiply, built on first use and kept
  const bf16* doubled(const bf16* W, int rows, int in) {
    auto it = doubled_.find(W);
    if (it == doubled_.end()) {
      it = doubled_.emplace(W, lfm::fp32::DeviceArray<bf16>(size_t(rows) * 2 * in)).first;
      kernels::double_rows<<<lfm::fp32::blocks(size_t(rows) * in), 256>>>(W, it->second.p, rows, in);
    }
    return it->second.p;
  }
  // Bidirectional attention over the image's n tokens (scale 1/8): the CUDA-core kernel, or K and V split once into
  // the tensor-core kernel's tile layouts, then the kernel.
  void attend(const float* q, const float* k, const float* v, float* out, int n) {
    using namespace kernels;
    namespace f = lfm::fp32;
    if (precision_.attention == Precision::kFp32) {
      attention<<<dim3((n + kAttnQueries - 1) / kAttnQueries, kHeads), kAttnQueries>>>(q, k, v, out, n, kHeads,
                                                                                        kKvHeads, 0.125f);
      return;
    }
    const int parts = precision_.attention == Precision::kX2 ? 2 : 1, npad = (n + kTcKeys - 1) / kTcKeys * kTcKeys;
    const size_t each = size_t(kKvHeads) * npad * parts * 64;
    for (int i : {1, 2})
      if (scratch16_[i].n < each) scratch16_[i] = f::DeviceArray<bf16>(each);
    split_kv<<<dim3(npad / 64, kKvHeads), 256>>>(k, v, scratch16_[1].p, scratch16_[2].p, n, npad, kKvHeads, parts);
    const dim3 grid((n + kTcQueries - 1) / kTcQueries, kHeads);
    if (parts == 2)
      attention_tc<2><<<grid, 32 * kTcWarps, tc_shared_bytes<2>()>>>(q, scratch16_[1].p, scratch16_[2].p, out, n, npad,
                                                                     kHeads, kKvHeads, 0.125f);
    else
      attention_tc<1><<<grid, 32 * kTcWarps, tc_shared_bytes<1>()>>>(q, scratch16_[1].p, scratch16_[2].p, out, n, npad,
                                                                     kHeads, kKvHeads, 0.125f);
  }
  static std::string layer(int i) { return "model.vision_tower.layers." + std::to_string(i) + "."; }
  const bf16* w(const std::string& name) const { return weights_->at(name).device_data; }
  lfm::fp32::DeviceArray<float>& scratch(int slot, size_t n) {
    if (scratch_[slot].n < n) scratch_[slot] = lfm::fp32::DeviceArray<float>(n);
    return scratch_[slot];
  }
  static std::vector<float> host(const float* p, size_t n) {
    std::vector<float> h(n);
    lfm::fp32::check(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    return h;
  }
};

}  // namespace lfm3_vision
