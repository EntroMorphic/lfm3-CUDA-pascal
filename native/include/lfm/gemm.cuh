#pragma once
// Native BF16 GEMMs for the LFM3 engine (G4: no cuBLAS). FP32 accumulation, BF16 output rounded to
// nearest even, deterministic (fixed reduction order, no atomics; the decode P·V split-K path sums its
// FP32 partials from a caller-provided workspace in a fixed order).
//
//   C[z] (M x N) = A[z] (M x K, K contiguous) · op(B[z / group])
//     op = transpose for B stored N x K (K contiguous): projections X·Wᵀ and attention Q·Kᵀ
//     op = identity  for B stored K x N (N contiguous): attention P·V
//
// Kernels: `thin` and `thin_k128` for M <= 8 with B stored N x K (decode projections and Q·Kᵀ: weight-
// bandwidth bound), `split_kn` for M <= 8 with B stored K x N (decode P·V), and `wide` otherwise
// (prefill). This is the Pascal (sm_61) engine: `wide` is a plain FP32 shared-memory kernel, where the
// Thor engine it was imported from uses tensor cores (mma.sync, cp.async, ldmatrix).
//
// Prompt position 0 (PLAN decision 5): in a prefill of more than 8 rows, the row of position 0 is computed with
// the arithmetic of Blackwell's tensor cores (lfm/tensor_core_model.hpp), as the Thor engine computes it, and every
// other row with the plain kernel. That row is the same for every such prompt that starts with the same token, and
// every later position attends to it (validation/prefill-arithmetic-v1). Only a caller that says its row 0 is
// prompt position 0 (Problem::row0 = 0) gets this; the text runner does, nothing else. A prefill of 8 rows or
// fewer takes the decode kernels for every row, as in the Thor engine. The tensor-core model does not cover
// infinities and NaN: where the plain kernel's result for position 0 is not finite, it is kept.
// -DLFM_GEMM_TENSOR_CORE_MODEL (the validation build of the parity gate) gives every row that arithmetic;
// -DLFM_GEMM_FIRST_ROW=0 gives none (the engine as it was before decision 5, for reproducing earlier records).
// Quality bar (journal §9): perplexity equal or better than the standard stack within the 95% CI.
// No reproduction of cuBLAS's kernel selection or reduction order is attempted or claimed.
#include <cuda_bf16.h>
#include <lfm/tensor_core_model.hpp>
// Which statement of the tensor-core step the model kernel uses: 1 the fast one, 2 the integer reference.
// The build switches are on when defined (-DNAME or -DNAME=1); a value that means nothing is an error, not a default.
#ifdef LFM_GEMM_TENSOR_CORE_MODEL
#if LFM_GEMM_TENSOR_CORE_MODEL != 1 && LFM_GEMM_TENSOR_CORE_MODEL != 2
#error "LFM_GEMM_TENSOR_CORE_MODEL must be 1 (or bare) or 2; leave it undefined for the shipping build"
#endif
#define LFM_GEMM_MODEL_STEP LFM_GEMM_TENSOR_CORE_MODEL
#else
#define LFM_GEMM_MODEL_STEP 1
#endif
#ifndef LFM_GEMM_FIRST_ROW
#define LFM_GEMM_FIRST_ROW 1
#endif
// The preprocessor reads an unknown word as 0, so a comparison alone would take =on for 0: a value must also be one of these.
#define LFM_GEMM_PASTE_(a, b) a##b
#define LFM_GEMM_PASTE(a, b) LFM_GEMM_PASTE_(a, b)
#define LFM_GEMM_DIGIT_0 1
#define LFM_GEMM_DIGIT_1 1
#define LFM_GEMM_DIGIT_2 1
#if !LFM_GEMM_PASTE(LFM_GEMM_DIGIT_, LFM_GEMM_FIRST_ROW) || (LFM_GEMM_FIRST_ROW != 0 && LFM_GEMM_FIRST_ROW != 1)
#error "LFM_GEMM_FIRST_ROW must be 0 or 1"
#endif
#if defined(LFM_GEMM_ROUND_ONCE) && LFM_GEMM_ROUND_ONCE != 1
#error "LFM_GEMM_ROUND_ONCE must be 1 (or bare); leave it undefined otherwise"
#endif
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace lfm::gemm {

using bf16 = __nv_bfloat16;

struct Problem {
  const bf16* a = nullptr;
  const bf16* b = nullptr;
  bf16* c = nullptr;
  // Optional FP32 output: when set, C is stored here as the unrounded FP32 accumulator and `c` is unused
  // (attention scores kept in FP32 through the softmax). Same layout and reduction order as `c`.
  float* c_f32 = nullptr;
  int m = 0, n = 0, k = 0;  // k: reduction length, a multiple of 8
  // B rows (k index) that exist, for K x N B: rows at or beyond it read as zero (attention P·V, where
  // the probability row is zero-padded to k but the value cache is not).
  int b_rows = 0;
  long long lda = 0, ldb = 0, ldc = 0;
  long long stride_a = 0, stride_b = 0, stride_c = 0;  // per batch item; B indexed by z / group
  int batch = 1, group = 1;
  bool b_k_contiguous = true;
  // FP32 scratch for the split-K path (M <= 8 with a K x N B: decode P·V); see workspace_bytes().
  float* workspace = nullptr;
  size_t workspace_size = 0;  // bytes
  // Use the prefill kernel whatever m is: for a block of rows of a larger prefill, whose last block may have
  // 8 rows or fewer. Every output then has the arithmetic it has when the prefill is one call.
  bool wide = false;
  // Which part of the model this product belongs to (kSite*), for experiments that give parts of the model different
  // prefill arithmetic (validation/prefill-arithmetic-v1). It never changes what the default builds compute.
  int site = 0;
  // The prompt position of row 0, for a product whose rows are the positions of a text prompt; -1 for any other
  // product (the default). The prefill kernel gives position 0 the tensor-core arithmetic (see the top of this
  // file), so only the text runner's prefill sets this; a block of a prompt's rows carries its first position.
  int row0 = -1;
  // Set by this file only, on the launch that recomputes position 0 over the plain kernel's result: an output the
  // plain kernel left non-finite is kept (the tensor-core model does not cover infinities and NaN).
  bool keep_nonfinite = false;
};
inline constexpr int kSiteHead = 1, kSiteMlp = 2, kSiteConv = 4, kSiteAttentionProjection = 8, kSiteAttention = 16, kSiteOther = 32;

// Keys per split of the decode P·V path, and the scratch it needs.
inline constexpr int kSplitKeys = 256;
inline size_t workspace_bytes(int rows, int n, int k, int batch) {
  const size_t splits = (size_t(k) + kSplitKeys - 1) / kSplitKeys;
  return rows <= 8 ? splits * size_t(batch) * rows * n * sizeof(float) : 0;
}

namespace detail {

// Stores output element `index` (z * stride_c + row * ldc + col) as BF16, or as FP32 when c_f32 is set.
__device__ __forceinline__ void put(const Problem& p, long long index, float v) {
  if (p.c_f32) p.c_f32[index] = v;
  else p.c[index] = __float2bfloat16_rn(v);
}

// Pascal (sm_61) has no tensor cores, cp.async or ldmatrix: the prefill kernel is plain FP32.
//
// A block of 8 x TY threads computes a (TY x RM) x (8 x RN) tile of C; each thread owns an RM x RN patch in
// registers. K advances kWideK at a time through shared memory, operands converted to FP32 as they are loaded
// (exact). The reduction order of every output is fixed and does not depend on the tiling: k ascending, one fused
// multiply-add per term. Elements outside the operands read as zero: rows >= m, columns >= n, k >= K (or >= b_rows
// for a K x N B).
//
// Shared memory is laid out [k][row] and [k][column], so that the threads of a warp (8 columns x 4 rows of
// patches) read 4 addresses RM apart and 8 addresses RN apart: with RM = 8 and RN = 4, all in different banks.
// (The first Pascal kernel stored [column][k] with 32 k per row: every thread of a warp hit the same bank.)
inline constexpr int kWideK = 32, kWideTX = 8;
// The plain kernel's multiply-add. Experiments only (validation/prefill-arithmetic-v1): -DLFM_GEMM_PLAIN_ROUND=1
// rounds every multiply-add toward zero, =2 away from zero; the default (nearest) is the engine's.
#ifndef LFM_GEMM_PLAIN_ROUND
#define LFM_GEMM_PLAIN_ROUND 0
#endif
#if !LFM_GEMM_PASTE(LFM_GEMM_DIGIT_, LFM_GEMM_PLAIN_ROUND)
#error "LFM_GEMM_PLAIN_ROUND must be 0, 1 or 2"
#endif
__device__ __forceinline__ float fma_away(float a, float b, float c) {
  const float nearest = fmaf(a, b, c);
  return signbit(nearest) ? __fmaf_rd(a, b, c) : __fmaf_ru(a, b, c);  // signbit: a negative result that rounds to -0
}
#if LFM_GEMM_PLAIN_ROUND == 1
#define LFM_GEMM_PLAIN_FMA(a, b, c) __fmaf_rz((a), (b), (c))
#elif LFM_GEMM_PLAIN_ROUND == 2
#define LFM_GEMM_PLAIN_FMA(a, b, c) fma_away((a), (b), (c))
#else
#define LFM_GEMM_PLAIN_FMA(a, b, c) fmaf((a), (b), (c))
#endif
template <int TY, int RM, int RN, int TX = kWideTX>
struct WideTile {
  static constexpr int THREADS = TX * TY, BM = TY * RM, BN = TX * RN;
};
// Loads one K stage: A rows [m0, m0 + BM) and B columns [n0, n0 + BN) for k in [k0, k0 + kWideK).
// store(is_b, kk, index, value): where the caller keeps it.
template <int THREADS, int BM, int BN, bool BK_CONTIGUOUS, class Store>
__device__ __forceinline__ void wide_load(const Problem& p, const bf16* A, const bf16* B, int m0, int n0, int k0, Store store) {
  const bf16 zero = __float2bfloat16_rn(0.f);
#pragma unroll
  for (int i = threadIdx.x; i < BM * kWideK; i += THREADS) {
    const int r = i / kWideK, kk = i % kWideK;  // consecutive threads: consecutive k of one row
    const long long gm = m0 + r, gk = k0 + kk;
    store(false, kk, r, gm < p.m && gk < p.k ? A[gm * p.lda + gk] : zero);
  }
  if constexpr (BK_CONTIGUOUS) {
#pragma unroll
    for (int i = threadIdx.x; i < BN * kWideK; i += THREADS) {
      const int c = i / kWideK, kk = i % kWideK;
      const long long gn = n0 + c, gk = k0 + kk;
      store(true, kk, c, gn < p.n && gk < p.k ? B[gn * p.ldb + gk] : zero);
    }
  } else {
#pragma unroll
    for (int i = threadIdx.x; i < BN * kWideK; i += THREADS) {
      const int kk = i / BN, c = i % BN;        // a K x N operand: consecutive columns of one k
      const long long gn = n0 + c, gk = k0 + kk;
      store(true, kk, c, gn < p.n && gk < p.b_rows ? B[gk * p.ldb + gn] : zero);
    }
  }
}
template <int TY, int RM, int RN, bool BK_CONTIGUOUS>
__global__ void __launch_bounds__(kWideTX * TY) wide_kernel(Problem p) {
  using T = WideTile<TY, RM, RN>;
  __shared__ float sa[kWideK][T::BM];
  __shared__ float sb[kWideK][T::BN];
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * T::BN, m0 = blockIdx.y * T::BM;
  const int ty = threadIdx.x / kWideTX, tx = threadIdx.x % kWideTX;
  float acc[RM][RN];
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < p.k; k0 += kWideK) {
    wide_load<T::THREADS, T::BM, T::BN, BK_CONTIGUOUS>(p, A, B, m0, n0, k0, [&](bool is_b, int kk, int index, bf16 v) {
      if (is_b) sb[kk][index] = __bfloat162float(v); else sa[kk][index] = __bfloat162float(v);
    });
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kWideK; ++kk) {
      float a[RM], b[RN];
#pragma unroll
      for (int i = 0; i < RM; ++i) a[i] = sa[kk][ty * RM + i];
#pragma unroll
      for (int j = 0; j < RN; ++j) b[j] = sb[kk][tx * RN + j];
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) acc[i][j] = LFM_GEMM_PLAIN_FMA(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      const long long row = m0 + ty * RM + i, col = n0 + tx * RN + j;
      if (row < p.m && col < p.n) put(p, c0 + row * p.ldc + col, acc[i][j]);
    }
}

// The plain prefill kernel, phase 2 (validation/speed-v1): every output is the same chain of FMAs in k order as in
// wide_kernel, so the two give identical bits; only the data movement differs. A block of 256 threads (16 x 16)
// computes a (16 RM) x 128 tile of C, each thread RM x 8 outputs: columns tx*4 + {0..3} and 64 + tx*4 + {0..3}; rows
// ty*RM + {0..RM-1}, or for RM = 8 ty*4 + {0..3} and 64 + ty*4 + {0..3}. RM follows the row count (1, 2, 4 or 8), so
// a block of a few rows does not compute 128. K advances 16 at a time; each thread fetches one 16-byte vector (8 BF16) of A and
// one of B per stage into registers while the stage before is computed, then stores it converted to FP32 into the
// other of two shared-memory buffers: one barrier per stage.
// wide_kernel pads K with zeros to a multiple of 32, and an FMA of zeros turns an accumulator of -0 into +0; this
// kernel pads to a multiple of 16, so where the two differ (K mod 32 = 16) it adds that one FMA at the end.
inline constexpr int kW2N = 128, kW2K = 16, kW2Threads = 256;
template <int BM, bool BK_CONTIGUOUS>
__device__ __forceinline__ void wide2_fetch(const Problem& p, const bf16* A, const bf16* B, int m0, int n0, int k0,
                                            int4& va, int4& vb) {
  const int t = threadIdx.x;
  const int4 zero = make_int4(0, 0, 0, 0);
  if (t < 2 * BM) {  // A: row t % BM, k half t / BM
    const int r = t % BM, kk = (t / BM) * 8;
    const long long gm = m0 + r, gk = k0 + kk;
    va = gm < p.m && gk < p.k ? *reinterpret_cast<const int4*>(A + gm * p.lda + gk) : zero;
  }
  if constexpr (BK_CONTIGUOUS) {  // B stored N x K: column t % 128, k half t / 128
    const int c = t % kW2N, kk = (t / kW2N) * 8;
    const long long gn = n0 + c, gk = k0 + kk;
    vb = gn < p.n && gk < p.k ? *reinterpret_cast<const int4*>(B + gn * p.ldb + gk) : zero;
  } else {  // B stored K x N: k t / 16, columns (t % 16) * 8 .. + 8 (N is a multiple of 8)
    const int kk = t / 16, c = (t % 16) * 8;
    const long long gn = n0 + c, gk = k0 + kk;
    vb = gn < p.n && gk < p.b_rows && gk < p.k ? *reinterpret_cast<const int4*>(B + gk * p.ldb + gn) : zero;
  }
}
template <int BM, bool BK_CONTIGUOUS>
__device__ __forceinline__ void wide2_store(float (*sa)[BM], float (*sb)[kW2N], const int4& va, const int4& vb) {
  const int t = threadIdx.x;
  const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&va);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&vb);
  if (t < 2 * BM) {
    const int r = t % BM, kk = (t / BM) * 8;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(a2[e]);
      sa[kk + 2 * e][r] = f.x; sa[kk + 2 * e + 1][r] = f.y;
    }
  }
  if constexpr (BK_CONTIGUOUS) {
    const int c = t % kW2N, kk = (t / kW2N) * 8;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(b2[e]);
      sb[kk + 2 * e][c] = f.x; sb[kk + 2 * e + 1][c] = f.y;
    }
  } else {
    const int kk = t / 16, c = (t % 16) * 8;
    float4 lo, hi;
    float2 f = __bfloat1622float2(b2[0]); lo.x = f.x; lo.y = f.y;
    f = __bfloat1622float2(b2[1]); lo.z = f.x; lo.w = f.y;
    f = __bfloat1622float2(b2[2]); hi.x = f.x; hi.y = f.y;
    f = __bfloat1622float2(b2[3]); hi.z = f.x; hi.w = f.y;
    *reinterpret_cast<float4*>(&sb[kk][c]) = lo;
    *reinterpret_cast<float4*>(&sb[kk][c + 4]) = hi;
  }
}
template <int RM>
__device__ __forceinline__ int wide2_row(int ty, int i) { return RM == 8 ? (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4) : ty * RM + i; }
template <int RM, bool BK_CONTIGUOUS>
__global__ void __launch_bounds__(kW2Threads, 2) wide2_kernel(Problem p) {
  constexpr int BM = 16 * RM;
  __shared__ __align__(16) float sa[2][kW2K][BM];
  __shared__ __align__(16) float sb[2][kW2K][kW2N];
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * kW2N, m0 = blockIdx.y * BM;
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  float acc[RM][8];
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  int4 va, vb;
  wide2_fetch<BM, BK_CONTIGUOUS>(p, A, B, m0, n0, 0, va, vb);
  wide2_store<BM, BK_CONTIGUOUS>(sa[0], sb[0], va, vb);
  __syncthreads();
  int stage = 0;
  for (int k0 = 0; k0 < p.k; k0 += kW2K) {
    const bool more = k0 + kW2K < p.k;
    if (more) wide2_fetch<BM, BK_CONTIGUOUS>(p, A, B, m0, n0, k0 + kW2K, va, vb);
#pragma unroll
    for (int kk = 0; kk < kW2K; ++kk) {
      float a[RM];
      if constexpr (RM == 8) {
        const float4 a0 = *reinterpret_cast<const float4*>(&sa[stage][kk][ty * 4]);
        const float4 a1 = *reinterpret_cast<const float4*>(&sa[stage][kk][64 + ty * 4]);
        a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w; a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
      } else if constexpr (RM == 4) {
        const float4 a0 = *reinterpret_cast<const float4*>(&sa[stage][kk][ty * 4]);
        a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
      } else if constexpr (RM == 2) {
        const float2 a0 = *reinterpret_cast<const float2*>(&sa[stage][kk][ty * 2]);
        a[0] = a0.x; a[1] = a0.y;
      } else {
        a[0] = sa[stage][kk][ty];
      }
      const float4 b0 = *reinterpret_cast<const float4*>(&sb[stage][kk][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&sb[stage][kk][64 + tx * 4]);
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = LFM_GEMM_PLAIN_FMA(a[i], b[j], acc[i][j]);
    }
    if (more) {
      wide2_store<BM, BK_CONTIGUOUS>(sa[stage ^ 1], sb[stage ^ 1], va, vb);
      __syncthreads();
      stage ^= 1;
    }
  }
  const bool pad_more = p.k % kWideK != 0 && p.k % kW2K == 0;  // wide_kernel's zero padding, see above
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const long long row = m0 + wide2_row<RM>(ty, i);
      const long long col = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      const float v = pad_more ? LFM_GEMM_PLAIN_FMA(0.f, 0.f, acc[i][j]) : acc[i][j];
      if (row < p.m && col < p.n) put(p, c0 + row * p.ldc + col, v);
    }
}

// M <= 8 rows: each warp produces COLS outputs for all rows, streaming B rows with 16-byte loads.
// The reduction order is fixed: per-lane strided partial sums, then a butterfly over the warp.
template <int R>
__global__ void __launch_bounds__(256) thin_kernel(Problem p) {
  constexpr int COLS = 4;
  const int z = blockIdx.y;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  const int col0 = warp * COLS;
  if (col0 >= p.n) return;
  float acc[R][COLS];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < COLS; ++j) acc[r][j] = 0.f;
#pragma unroll 4
  for (int kb = lane * 8; kb < p.k; kb += 256) {
    int4 w[COLS];
#pragma unroll
    for (int j = 0; j < COLS; ++j)
      w[j] = col0 + j < p.n ? __ldcs(reinterpret_cast<const int4*>(B + (col0 + j) * p.ldb + kb)) : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int4 xa = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + kb));
      const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xa);
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&w[j]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 xf = __bfloat1622float2(x2[e]), wf = __bfloat1622float2(w2[e]);
          acc[r][j] = fmaf(xf.x, wf.x, acc[r][j]);
          acc[r][j] = fmaf(xf.y, wf.y, acc[r][j]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < COLS; ++j) {
      float v = acc[r][j];
#pragma unroll
      for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      acc[r][j] = v;
    }
  if (lane < R * COLS) {
    // Lane r*COLS+j stores (r, j): select its value without dynamic register indexing.
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int j = 0; j < COLS; ++j)
        if (lane == r * COLS + j) v = acc[r][j];
    const int r = lane / COLS, j = lane % COLS;
    if (col0 + j < p.n) put(p, c0 + r * p.ldc + col0 + j, v);
  }
}

// M <= 8 rows with B stored K x N: keys split across blocks (kSplitKeys each); every thread owns two
// adjacent columns. FP32 partials per split, then a finish that adds the splits in index order.
// G batch items that share one B (G divides group: query heads of one key/value head) are computed by the same
// thread from one read of B; each output's arithmetic is the same whatever G is (phase 2, validation/speed-v1).
template <int R, int G>
__global__ void __launch_bounds__(128) split_kn_partial(Problem p, int splits) {
  const int split = blockIdx.x, z0 = blockIdx.y * G;
  const bf16* B = p.b + (z0 / p.group) * p.stride_b;
  const int col = 2 * (blockIdx.z * blockDim.x + threadIdx.x);
  if (col >= p.n) return;
  const int k0 = split * kSplitKeys, k1 = min(k0 + kSplitKeys, p.b_rows);
  float acc[G][R][2];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[g][r][0] = acc[g][r][1] = 0.f;
  for (int kk = k0; kk < k1; ++kk) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(B + kk * p.ldb + col));
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float a = __bfloat162float(p.a[(z0 + g) * p.stride_a + r * p.lda + kk]);
        acc[g][r][0] = fmaf(a, v.x, acc[g][r][0]);
        acc[g][r][1] = fmaf(a, v.y, acc[g][r][1]);
      }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float* out = p.workspace + ((size_t(z0 + g) * splits + split) * R) * p.n + col;
#pragma unroll
    for (int r = 0; r < R; ++r) { out[r * p.n] = acc[g][r][0]; out[r * p.n + 1] = acc[g][r][1]; }
  }
}
// One row (decode), several splits per block: each thread owns 8 adjacent columns of one split and reads them with one
// 16-byte load per key; 16 threads cover 128 columns. Each output's partial is the same chain over the split's keys
// in ascending order as split_kn_partial's, written to the same place (phase 2, validation/speed-v1).
#ifndef LFM_GEMM_WIDE_AHEAD
#define LFM_GEMM_WIDE_AHEAD 4
#endif
#ifndef LFM_GEMM_WIDE_SPLITS
#define LFM_GEMM_WIDE_SPLITS 4
#endif
inline constexpr int kWideSplitsPerBlock = LFM_GEMM_WIDE_SPLITS, kWideAhead = LFM_GEMM_WIDE_AHEAD;
#ifndef LFM_GEMM_K128_BLOCKS
#define LFM_GEMM_K128_BLOCKS 256
#endif
template <int G>
__global__ void __launch_bounds__(16 * kWideSplitsPerBlock) split_kn_partial_wide(Problem p, int splits) {
  const int split = blockIdx.x * kWideSplitsPerBlock + threadIdx.x / 16, z0 = blockIdx.y * G;
  const int col = blockIdx.z * 128 + (threadIdx.x % 16) * 8;
  if (split >= splits || col >= p.n) return;
  const bf16* B = p.b + (z0 / p.group) * p.stride_b;
  const int k0 = split * kSplitKeys, k1 = min(k0 + kSplitKeys, p.b_rows);
  float acc[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[g][j] = 0.f;
  // kWideAhead keys are loaded before any is used; the sums still run over the keys in ascending order.
  for (int kb = k0; kb < k1; kb += kWideAhead) {
    int4 raw[kWideAhead];
#pragma unroll
    for (int u = 0; u < kWideAhead; ++u)
      raw[u] = kb + u < k1 ? __ldg(reinterpret_cast<const int4*>(B + (long long)(kb + u) * p.ldb + col)) : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < kWideAhead; ++u) {
      if (kb + u >= k1) break;
      const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&raw[u]);
      float v[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(v2[e]); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float a = __bfloat162float(p.a[(z0 + g) * p.stride_a + kb + u]);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g][j] = fmaf(a, v[j], acc[g][j]);
      }
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float* out = p.workspace + (size_t(z0 + g) * splits + split) * p.n + col;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = acc[g][j];
  }
}
template <int R>
__global__ void split_kn_finish(Problem p, int splits) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t per = size_t(R) * p.n;
  if (i >= per * p.batch) return;
  const int z = int(i / per), r = int(i % per) / p.n, col = int(i % p.n);
  const float* part = p.workspace + size_t(z) * splits * per + size_t(r) * p.n + col;
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) sum += part[size_t(s) * per];
  put(p, z * p.stride_c + r * p.ldc + col, sum);
}
inline constexpr int kPairedSplits = 16;
template <int R>
inline void launch_split_kn(const Problem& p, cudaStream_t stream) {
  const int splits = (p.k + kSplitKeys - 1) / kSplitKeys;
  if (!p.workspace || p.workspace_size < workspace_bytes(R, p.n, p.k, p.batch))
    throw std::runtime_error("gemm: split-K workspace too small");
  const int threads = 64, column_blocks = (p.n / 2 + threads - 1) / threads;
  if (R == 1 && p.n % 8 == 0 && splits >= kPairedSplits) {  // decode over long caches: 16-byte loads, heads together
    const dim3 grid((splits + kWideSplitsPerBlock - 1) / kWideSplitsPerBlock, 0, (p.n + 127) / 128);
    if (p.group % 4 == 0) split_kn_partial_wide<4><<<dim3(grid.x, p.batch / 4, grid.z), 16 * kWideSplitsPerBlock, 0, stream>>>(p, splits);
    else if (p.group % 2 == 0) split_kn_partial_wide<2><<<dim3(grid.x, p.batch / 2, grid.z), 16 * kWideSplitsPerBlock, 0, stream>>>(p, splits);
    else split_kn_partial_wide<1><<<dim3(grid.x, p.batch, grid.z), 16 * kWideSplitsPerBlock, 0, stream>>>(p, splits);
  }
  // Grouped heads divide the blocks: worth it only once the keys fill enough splits to keep the card busy.
  else if (R == 1 && p.group % 4 == 0 && splits >= kPairedSplits) split_kn_partial<R, 4><<<dim3(splits, p.batch / 4, column_blocks), threads, 0, stream>>>(p, splits);
  else if (R <= 2 && p.group % 2 == 0 && splits >= kPairedSplits) split_kn_partial<R, 2><<<dim3(splits, p.batch / 2, column_blocks), threads, 0, stream>>>(p, splits);
  else split_kn_partial<R, 1><<<dim3(splits, p.batch, column_blocks), threads, 0, stream>>>(p, splits);
  const size_t outputs = size_t(R) * p.n * p.batch;
  split_kn_finish<R><<<unsigned((outputs + 255) / 256), 256, 0, stream>>>(p, splits);
}

// M <= 8 rows, K = 128 (attention Q·Kᵀ in decode): eight lanes per B row, each lane 16 consecutive
// elements (two 16-byte loads), four rows per warp step; a fixed three-step butterfly per output.
// G (1, 2 or 4) batch items that share one B (query heads of one key/value head) take one read of each key; each output's
// arithmetic is the same whatever G is (phase 2, validation/speed-v1).
template <int R, int G>
__global__ void __launch_bounds__(256) thin_k128_kernel(Problem p) {
  const int z0 = blockIdx.y * G;
  const bf16* B = p.b + (z0 / p.group) * p.stride_b;
  const int lane = threadIdx.x % 32, sub = lane % 8, slot = lane / 8;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  float q[G][R][16];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const bf16* A = p.a + (z0 + g) * p.stride_a;
    const int4 lo = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16));
    const int4 hi = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16 + 8));
    const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
      q[g][r][2 * e] = a.x; q[g][r][2 * e + 1] = a.y; q[g][r][8 + 2 * e] = b.x; q[g][r][8 + 2 * e + 1] = b.y;
    }
  }
  // The next rows' keys are loaded before this rows' are used (a warp loops when the rows outnumber its grid).
  const int4 zero = make_int4(0, 0, 0, 0);
  int4 lo = zero, hi = zero;
  if (warp * 4 + slot < p.n) {
    lo = __ldg(reinterpret_cast<const int4*>(B + (warp * 4 + slot) * p.ldb + sub * 16));
    hi = __ldg(reinterpret_cast<const int4*>(B + (warp * 4 + slot) * p.ldb + sub * 16 + 8));
  }
  for (int row0 = warp * 4; row0 < p.n; row0 += warps * 4) {
    const int row = row0 + slot, next = row + warps * 4;
    int4 next_lo = zero, next_hi = zero;
    if (next < p.n) {
      next_lo = __ldg(reinterpret_cast<const int4*>(B + next * p.ldb + sub * 16));
      next_hi = __ldg(reinterpret_cast<const int4*>(B + next * p.ldb + sub * 16 + 8));
    }
    float acc[G][R];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int r = 0; r < R; ++r) acc[g][r] = 0.f;
    if (row < p.n) {
      const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
      float k[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
        k[2 * e] = a.x; k[2 * e + 1] = a.y; k[8 + 2 * e] = b.x; k[8 + 2 * e + 1] = b.y;
      }
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int e = 0; e < 16; ++e) acc[g][r] = fmaf(q[g][r][e], k[e], acc[g][r]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float v = acc[g][r];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if (sub == r && row < p.n) put(p, (z0 + g) * p.stride_c + r * p.ldc + row, v);
      }
    lo = next_lo; hi = next_hi;
  }
}
template <int R>
inline void launch_thin_k128(const Problem& p, cudaStream_t stream) {
  static_assert(R <= 8, "one lane of the eight stores each row");
  const int blocks = std::min((p.n + 31) / 32, LFM_GEMM_K128_BLOCKS);
  if (R == 1 && p.group % 4 == 0) thin_k128_kernel<R, 4><<<dim3(blocks, p.batch / 4), 256, 0, stream>>>(p);
  else if (R <= 2 && p.group % 2 == 0) thin_k128_kernel<R, 2><<<dim3(blocks, p.batch / 2), 256, 0, stream>>>(p);
  else thin_k128_kernel<R, 1><<<dim3(blocks, p.batch), 256, 0, stream>>>(p);
}

template <int R>
inline void launch_thin(const Problem& p, cudaStream_t stream) {
  const int warps = (p.n + 3) / 4;
  thin_kernel<R><<<dim3((warps + 7) / 8, p.batch), 256, 0, stream>>>(p);
}

// The prefill kernel with the arithmetic of Blackwell's mma.sync.aligned.m16n8k16 (lfm/tensor_core_model.hpp;
// validation/tensor-core-model-v1), so that outputs equal the Thor engine's bit for bit. The engine uses it for
// prompt position 0 (PLAN decision 5); the validation build of the parity gate (decision 4) for every row. Steps of 16 k chain over k ascending from 0, as the Thor kernel's do. Same tiling as the
// plain kernel; each operand's exponent is kept beside its value.
// -DLFM_GEMM_TENSOR_CORE_MODEL=2 uses the integer statement of the model for every step (the reference).
template <int TY, int RM, int RN, bool BK_CONTIGUOUS, int TX = kWideTX>
__global__ void __launch_bounds__(TX * TY) wide_model_kernel(Problem p) {
  using T = WideTile<TY, RM, RN, TX>;
  namespace tcm = lfm::tcm;
  __shared__ float sa[kWideK][T::BM];
  __shared__ float sb[kWideK][T::BN];
  __shared__ short ea[kWideK][T::BM], eb[kWideK][T::BN];
  // Per step of 16 k: the largest exponent of each row's and each column's operands. Their sum bounds every
  // product's exponent of that output from above.
#if LFM_GEMM_MODEL_STEP != 2
  __shared__ short ra[kWideK / 16][T::BM], cb[kWideK / 16][T::BN];
#endif
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * T::BN, m0 = blockIdx.y * T::BM;
  const int ty = threadIdx.x / TX, tx = threadIdx.x % TX;
  float acc[RM][RN];
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < p.k; k0 += kWideK) {
    wide_load<T::THREADS, T::BM, T::BN, BK_CONTIGUOUS>(p, A, B, m0, n0, k0, [&](bool is_b, int kk, int index, bf16 v) {
      const short e = tcm::exponent_of_bf16_bits(*reinterpret_cast<const unsigned short*>(&v));
      if (is_b) { sb[kk][index] = __bfloat162float(v); eb[kk][index] = e; }
      else { sa[kk][index] = __bfloat162float(v); ea[kk][index] = e; }
    });
    __syncthreads();
#if LFM_GEMM_MODEL_STEP != 2
#pragma unroll
    for (int i = threadIdx.x; i < (T::BM + T::BN) * (kWideK / 16); i += T::THREADS) {
      const int index = i / (kWideK / 16), step = i % (kWideK / 16);
      int e = tcm::kNoExponent;
      if (index < T::BM) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) e = max(e, int(ea[step * 16 + kk][index]));
        ra[step][index] = short(e);
      } else {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) e = max(e, int(eb[step * 16 + kk][index - T::BM]));
        cb[step][index - T::BM] = short(e);
      }
    }
    __syncthreads();
#endif
    for (int step = 0; step < kWideK; step += 16) {
#if LFM_GEMM_MODEL_STEP == 2
      bool fast = false;
#else
      // The window of every output of the patch: the largest exponent among its accumulator and 16 products.
      // Where the accumulator's exponent is at least the bound on the products', it is the window, and the pass
      // over the 16 exponent sums is not needed (the usual case once a sum is under way).
      int emax[RM][RN];
      bool bounded = true;
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) {
          emax[i][j] = tcm::exponent_of_float(acc[i][j]);
          bounded = bounded && emax[i][j] >= int(ra[step / 16][ty * RM + i]) + int(cb[step / 16][tx * RN + j]);
        }
      if (!bounded)
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        int xe[RM], ye[RN];
#pragma unroll
        for (int i = 0; i < RM; ++i) xe[i] = ea[step + kk][ty * RM + i];
#pragma unroll
        for (int j = 0; j < RN; ++j) ye[j] = eb[step + kk][tx * RN + j];
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int j = 0; j < RN; ++j) emax[i][j] = max(emax[i][j], xe[i] + ye[j]);
      }
      // All terms zero (padding): any window gives zero. Otherwise the float form needs the window in range.
      bool fast = true;
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) {
          if (emax[i][j] < tcm::kNoExponent / 2) emax[i][j] = 0;
          fast = fast && emax[i][j] >= -90 && emax[i][j] <= 90;
        }
      if (fast) {
        // In units of q = 2^(emax - 25): 16 products, each below 2^27 in magnitude and their sum below 2^31.
        int sum[RM][RN];
        float inverse[RM][RN];
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int j = 0; j < RN; ++j) { sum[i][j] = 0; inverse[i][j] = __uint_as_float(unsigned(25 - emax[i][j] + 127) << 23); }
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          float a[RM], b[RN];
#pragma unroll
          for (int i = 0; i < RM; ++i) a[i] = sa[step + kk][ty * RM + i];
#pragma unroll
          for (int j = 0; j < RN; ++j) b[j] = sb[step + kk][tx * RN + j];
#pragma unroll
          for (int i = 0; i < RM; ++i)
#pragma unroll
            for (int j = 0; j < RN; ++j) sum[i][j] += __float2int_rz(__fmul_rn(__fmul_rn(a[i], b[j]), inverse[i][j]));
        }
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int j = 0; j < RN; ++j) {
            const long long total = (long long)sum[i][j] + __float2int_rz(__fmul_rn(acc[i][j], inverse[i][j]));
            acc[i][j] = __fmul_rn(__ll2float_rz(total), __uint_as_float(unsigned(emax[i][j] - 25 + 127) << 23));
          }
      }
#endif
      if (!fast) {
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int j = 0; j < RN; ++j)
            acc[i][j] = tcm::mma_step_reference(acc[i][j], &sa[step][ty * RM + i], &ea[step][ty * RM + i],
                                                &sb[step][tx * RN + j], &eb[step][tx * RN + j], T::BM, T::BN);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      const long long row = m0 + ty * RM + i, col = n0 + tx * RN + j;
      if (row >= p.m || col >= p.n) continue;
      const long long index = c0 + row * p.ldc + col;
      if (p.keep_nonfinite && !isfinite(p.c_f32 ? p.c_f32[index] : __bfloat162float(p.c[index]))) continue;
      put(p, index, acc[i][j]);
    }
}

#ifdef LFM_GEMM_ROUND_ONCE
// Experiment build only (validation/round-once-v1): the prefill kernel with every output rounded once. No floating
// point is used for the sum. A BF16 value is an integer of at most 8 bits times a power of two; a product of two is
// an integer of at most 16 bits times 2^e, e in [-266, 240]. Each output has a fixed-point accumulator wide enough to
// hold every such product exactly (kOnceLimbs digits of 32 bits, digit i weighing 2^(32 i - 266)), so the sum is the
// exact dot product whatever the order. It is rounded a single time, to nearest even, to the stored format: BF16, or
// FP32 for c_f32. Operands must be finite; a result outside the normal range of the format is not handled.
inline constexpr int kOnceLimbs = 18, kOnceBias = 266;
struct OnceSum {
  long long limb[kOnceLimbs];  // digits of 32 bits with room for deferred carries; the top one carries the sign
};
__device__ __forceinline__ void once_add(OnceSum& s, unsigned a, unsigned b) {  // BF16 bit patterns
  const int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF;
  const long long product = (long long)((ea ? 128 : 0) + (a & 0x7F)) * (long long)((eb ? 128 : 0) + (b & 0x7F));
  const int position = (ea ? ea : 1) + (eb ? eb : 1) - 2 * 134 + kOnceBias;  // >= 0
  const long long term = product << (position & 31);                         // below 2^47
  s.limb[position >> 5] += ((a ^ b) & 0x8000) ? -term : term;
}
__device__ __forceinline__ void once_carry(OnceSum& s) {
  for (int i = 0; i + 1 < kOnceLimbs; ++i) {
    const long long carry = s.limb[i] >> 32;  // arithmetic shift: floor
    s.limb[i] &= 0xFFFFFFFFll;
    s.limb[i + 1] += carry;
  }
}
__device__ __forceinline__ int once_bit(const OnceSum& s, int index) {
  return index < 0 ? 0 : int((s.limb[index >> 5] >> (index & 31)) & 1);
}
// The sum rounded to nearest even at `precision` significant bits.
__device__ __noinline__ float once_round(OnceSum s, int precision) {
  once_carry(s);
  const bool negative = s.limb[kOnceLimbs - 1] < 0;
  if (negative) {  // negate: the magnitude, digit by digit
    long long borrow = 0;
    for (int i = 0; i < kOnceLimbs; ++i) {
      long long v = -s.limb[i] - borrow;
      if (i + 1 < kOnceLimbs) { borrow = v < 0 ? 1 : 0; v &= 0xFFFFFFFFll; }
      s.limb[i] = v;
    }
  }
  int top = -1;
  for (int i = kOnceLimbs - 1; i >= 0 && top < 0; --i)
    if (s.limb[i]) top = i * 32 + 63 - __clzll(s.limb[i]);
  if (top < 0) return 0.f;
  unsigned q = 0;
  for (int bit = top; bit > top - precision; --bit) q = (q << 1) | unsigned(once_bit(s, bit));
  const int below = top - precision;  // the first bit dropped
  if (below >= 0 && once_bit(s, below)) {
    bool sticky = false;
    for (int i = 0; i < (below >> 5); ++i) sticky = sticky || s.limb[i] != 0;
    sticky = sticky || (s.limb[below >> 5] & ((1ll << (below & 31)) - 1)) != 0;
    if (sticky || (q & 1)) ++q;
  }
  const float value = scalbnf(float(q), top - precision + 1 - kOnceBias);  // q has at most 25 bits and is exact here
  return negative ? -value : value;
}
template <int TY, int RM, int RN, bool BK_CONTIGUOUS>
__global__ void __launch_bounds__(kWideTX * TY) wide_once_kernel(Problem p) {
  using T = WideTile<TY, RM, RN>;
  __shared__ unsigned short sa[kWideK][T::BM];
  __shared__ unsigned short sb[kWideK][T::BN];
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * T::BN, m0 = blockIdx.y * T::BM;
  const int ty = threadIdx.x / kWideTX, tx = threadIdx.x % kWideTX;
  OnceSum sum[RM][RN];
  for (int i = 0; i < RM; ++i)
    for (int j = 0; j < RN; ++j)
      for (int l = 0; l < kOnceLimbs; ++l) sum[i][j].limb[l] = 0;
  for (int k0 = 0; k0 < p.k; k0 += kWideK) {
    wide_load<T::THREADS, T::BM, T::BN, BK_CONTIGUOUS>(p, A, B, m0, n0, k0, [&](bool is_b, int kk, int index, bf16 v) {
      const unsigned short bits = *reinterpret_cast<const unsigned short*>(&v);
      if (is_b) sb[kk][index] = bits; else sa[kk][index] = bits;
    });
    __syncthreads();
    for (int i = 0; i < RM; ++i)
      for (int j = 0; j < RN; ++j) {
        for (int kk = 0; kk < kWideK; ++kk) once_add(sum[i][j], sa[kk][ty * RM + i], sb[kk][tx * RN + j]);
        once_carry(sum[i][j]);  // 32 terms below 2^47 each on digits below 2^32: no overflow
      }
    __syncthreads();
  }
  for (int i = 0; i < RM; ++i)
    for (int j = 0; j < RN; ++j) {
      const long long row = m0 + ty * RM + i, col = n0 + tx * RN + j;
      if (!(row < p.m && col < p.n)) continue;
      const long long index = c0 + row * p.ldc + col;
      if (p.c_f32) p.c_f32[index] = once_round(sum[i][j], 24);
      else p.c[index] = __float2bfloat16_rn(once_round(sum[i][j], 8));
    }
}
#endif

// The prefill tiles. Plain kernel: 8 x 4 outputs per thread. Exact kernel: 2 x 2 (measured on the GTX 1080: it is
// bound by its arithmetic, not its loads, and larger patches run slower). The block's height follows the row count
// so that a short prefill computes fewer rows of padding; the smallest block is 4 patches high, so the launch that
// recomputes position 0 alone computes 8 rows for one (its cost is PLAN's first item for phase 2).
// LFM_GEMM_WIDE_RM / _RN override the old plain kernel's patch (used only with -DLFM_GEMM_WIDE2=0) and LFM_GEMM_MODEL_RM / _RN the exact
// kernel's, for measurement.
#ifndef LFM_GEMM_WIDE_RM
#define LFM_GEMM_WIDE_RM 8
#endif
#ifndef LFM_GEMM_WIDE_RN
#define LFM_GEMM_WIDE_RN 4
#endif
#ifndef LFM_GEMM_MODEL_RM
#define LFM_GEMM_MODEL_RM 2
#endif
#ifndef LFM_GEMM_MODEL_RN
#define LFM_GEMM_MODEL_RN 2
#endif
// sm_61: 48 KiB of shared memory per block; `bytes` per operand held in it.
template <int TY, int RM, int RN, int BYTES>
inline constexpr bool kWideFits = kWideK * (WideTile<TY, RM, RN>::BM + WideTile<TY, RM, RN>::BN) * BYTES + 2048 <= 48 * 1024;
template <int RM, int RN, int BYTES, class Launch>
inline void for_wide_tile(int m, Launch launch) {
  static_assert((kWideFits<4, RM, RN, BYTES>), "the smallest prefill tile must fit shared memory");
  const int rows = (m + RM - 1) / RM;  // patches needed
  if constexpr (kWideFits<32, RM, RN, BYTES>) { if (rows > 16) return launch(std::integral_constant<int, 32>{}); }
  if constexpr (kWideFits<16, RM, RN, BYTES>) { if (rows > 8) return launch(std::integral_constant<int, 16>{}); }
  if constexpr (kWideFits<8, RM, RN, BYTES>) { if (rows > 4) return launch(std::integral_constant<int, 8>{}); }
  launch(std::integral_constant<int, 4>{});
}
#ifdef LFM_GEMM_TENSOR_CORE_MODEL
// Which sites take the tensor-core arithmetic in the validation build: all of them, unless LFM_TCM_SITES (a sum of
// kSite* values) narrows it for an experiment; the others then take the plain kernel.
inline int model_sites() {
  static const int mask = [] {
    const char* v = std::getenv("LFM_TCM_SITES");
    if (!v) return -1;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 0 || n > 63) throw std::runtime_error("LFM_TCM_SITES must be 0..63");
    return int(n);
  }();
  return mask;
}
#endif
#ifdef LFM_GEMM_TENSOR_CORE_MODEL
// Experiment (validation/prefill-arithmetic-v1): LFM_TCM_ROWS=N gives the tensor-core arithmetic to prompt positions
// below N only (every site), the plain kernel to the rest; -N the reverse. 0 or unset: no row rule.
inline int model_rows() {
  static const int rows = [] {
    const char* v = std::getenv("LFM_TCM_ROWS");
    if (!v) return 0;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < -1000000 || n > 1000000) throw std::runtime_error("LFM_TCM_ROWS out of range");
    return int(n);
  }();
  return rows;
}
#endif
// The plain prefill product: wide2_kernel, with as many rows per thread as the row count needs. -DLFM_GEMM_WIDE2=0
// selects wide_kernel, the kernel it replaced, which gives identical bits (for measurement).
#ifndef LFM_GEMM_WIDE2
#define LFM_GEMM_WIDE2 1
#endif
#if !LFM_GEMM_PASTE(LFM_GEMM_DIGIT_, LFM_GEMM_WIDE2) || LFM_GEMM_WIDE2 > 1
#error "LFM_GEMM_WIDE2 must be 0 or 1"
#endif
template <bool KC>
inline void launch_wide2(const Problem& p, cudaStream_t stream) {
  const auto go = [&](auto rm) {
    constexpr int RM = decltype(rm)::value;
    const dim3 grid((p.n + kW2N - 1) / kW2N, (p.m + 16 * RM - 1) / (16 * RM), p.batch);
    wide2_kernel<RM, KC><<<grid, kW2Threads, 0, stream>>>(p);
  };
  if (p.m <= 16) go(std::integral_constant<int, 1>{});
  else if (p.m <= 32) go(std::integral_constant<int, 2>{});
  else if (p.m <= 64) go(std::integral_constant<int, 4>{});
  else go(std::integral_constant<int, 8>{});
}
template <bool KC>
inline void launch_wide_plain(const Problem& p, cudaStream_t stream) {
  if (LFM_GEMM_WIDE2) return launch_wide2<KC>(p, stream);
  for_wide_tile<LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN, 4>(p.m, [&](auto ty) {
    constexpr int TY = decltype(ty)::value;
    using T = WideTile<TY, LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN>;
    const dim3 grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch);
    wide_kernel<TY, LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN, KC><<<grid, T::THREADS, 0, stream>>>(p);
  });
}
// Position 0 alone: one row of outputs, one output per thread, LFM_GEMM_FIRST_TX threads per block (each output's
// arithmetic is the model's whatever the block's shape; first_row_check holds it to the host model).
#ifndef LFM_GEMM_FIRST_TX
#define LFM_GEMM_FIRST_TX 32
#endif
static_assert(LFM_GEMM_FIRST_TX >= 1, "LFM_GEMM_FIRST_TX must be a positive number of threads");
template <bool KC>
inline void launch_first_row_model(const Problem& p, cudaStream_t stream) {
  using T = WideTile<1, 1, 1, LFM_GEMM_FIRST_TX>;
  static_assert(kWideK * (T::BM + T::BN) * 6 + 2048 <= 48 * 1024, "the position-0 tile must fit shared memory");
  const dim3 grid((p.n + T::BN - 1) / T::BN, 1, p.batch);
  wide_model_kernel<1, 1, 1, KC, LFM_GEMM_FIRST_TX><<<grid, T::THREADS, 0, stream>>>(p);
}
template <bool KC>
inline void launch_wide_model(const Problem& p, cudaStream_t stream) {
  for_wide_tile<LFM_GEMM_MODEL_RM, LFM_GEMM_MODEL_RN, 6>(p.m, [&](auto ty) {  // an FP32 value and a 16-bit exponent
    constexpr int TY = decltype(ty)::value;
    using T = WideTile<TY, LFM_GEMM_MODEL_RM, LFM_GEMM_MODEL_RN>;
    const dim3 grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch);
    wide_model_kernel<TY, LFM_GEMM_MODEL_RM, LFM_GEMM_MODEL_RN, KC><<<grid, T::THREADS, 0, stream>>>(p);
  });
}
#ifdef LFM_GEMM_ROUND_ONCE
// LFM_ONCE_ROWS=N: prompt positions below N are computed again by the round-once kernel (they overwrite), at every
// site. Unset or 0: the build computes what the plain build does.
inline int once_rows() {
  static const int rows = [] {
    const char* v = std::getenv("LFM_ONCE_ROWS");
    if (!v) return 0;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 0 || n > 1000000) throw std::runtime_error("LFM_ONCE_ROWS out of range");
    return int(n);
  }();
  return rows;
}
template <bool KC>
inline void launch_wide_once(const Problem& p, cudaStream_t stream) {
  for_wide_tile<2, 2, 2>(p.m, [&](auto ty) {
    constexpr int TY = decltype(ty)::value;
    using T = WideTile<TY, 2, 2>;
    const dim3 grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch);
    wide_once_kernel<TY, 2, 2, KC><<<grid, T::THREADS, 0, stream>>>(p);
  });
}
#endif
template <bool KC>
inline void launch_wide_base(const Problem& p, cudaStream_t stream) {
#ifdef LFM_GEMM_TENSOR_CORE_MODEL
  if (const int rule = model_rows()) {
    // The whole product with one arithmetic, then the first rows again with the other (they overwrite).
    const int first = p.row0 < 0 ? 0 : std::max(0, std::min(p.m, (rule > 0 ? rule : -rule) - p.row0));
    if (rule > 0) launch_wide_plain<KC>(p, stream); else launch_wide_model<KC>(p, stream);
    if (first > 0) {
      Problem head = p;
      head.m = first;
      if (rule > 0) launch_wide_model<KC>(head, stream); else launch_wide_plain<KC>(head, stream);
    }
    return;
  }
  if (model_sites() & (p.site ? p.site : kSiteOther)) return launch_wide_model<KC>(p, stream);
  launch_wide_plain<KC>(p, stream);
#else
  launch_wide_plain<KC>(p, stream);
#if LFM_GEMM_FIRST_ROW
  if (p.row0 == 0) {  // position 0 again, with the tensor-core arithmetic (it overwrites)
    Problem first = p;
    first.m = 1;
    first.keep_nonfinite = true;
    launch_first_row_model<KC>(first, stream);
  }
#endif
#endif
}
template <bool KC>
inline void launch_wide(const Problem& p, cudaStream_t stream) {
  launch_wide_base<KC>(p, stream);
#ifdef LFM_GEMM_ROUND_ONCE
  const int first = p.row0 < 0 ? 0 : std::min(p.m, once_rows() - p.row0);
  if (first > 0) {
    Problem head = p;
    head.m = first;
    launch_wide_once<KC>(head, stream);
  }
#endif
}

}  // namespace detail

inline void check(const Problem& p) {
  const auto aligned = [](const void* q) { return reinterpret_cast<std::uintptr_t>(q) % 16 == 0; };
  if (p.m < 1 || p.n < 1 || p.k < 8 || p.k % 8 || p.batch < 1 || p.group < 1 || p.batch % p.group)
    throw std::runtime_error("gemm: invalid geometry");
  if (!p.c == !p.c_f32) throw std::runtime_error("gemm: exactly one of c and c_f32 must be set");
  if (!aligned(p.a) || !aligned(p.b) || p.lda % 8 || p.ldb % 8 || p.stride_a % 8 || p.stride_b % 8)
    throw std::runtime_error("gemm: operands must be 16-byte aligned with 8-element strides");
  if (!p.b_k_contiguous && (p.b_rows < 0 || p.b_rows > p.k || p.n % 8)) throw std::runtime_error("gemm: invalid K x N operand");
}

// The engine's GEMM: thin (M <= 8, B stored N x K), split-K (M <= 8, B stored K x N) or wide.
inline void run(const Problem& p, cudaStream_t stream = 0) {
  check(p);
  if (p.wide) {
    if (p.b_k_contiguous) detail::launch_wide<true>(p, stream);
    else detail::launch_wide<false>(p, stream);
    return;
  }
  if (p.b_k_contiguous && p.m <= 8 && p.k == 128) {
    switch (p.m) {
      case 1: detail::launch_thin_k128<1>(p, stream); return;
      case 2: detail::launch_thin_k128<2>(p, stream); return;
      case 3: detail::launch_thin_k128<3>(p, stream); return;
      case 4: detail::launch_thin_k128<4>(p, stream); return;
      case 5: detail::launch_thin_k128<5>(p, stream); return;
      case 6: detail::launch_thin_k128<6>(p, stream); return;
      case 7: detail::launch_thin_k128<7>(p, stream); return;
      default: detail::launch_thin_k128<8>(p, stream); return;
    }
  }
  if (p.b_k_contiguous && p.m <= 8) {
    switch (p.m) {
      case 1: detail::launch_thin<1>(p, stream); return;
      case 2: detail::launch_thin<2>(p, stream); return;
      case 3: detail::launch_thin<3>(p, stream); return;
      case 4: detail::launch_thin<4>(p, stream); return;
      case 5: detail::launch_thin<5>(p, stream); return;
      case 6: detail::launch_thin<6>(p, stream); return;
      case 7: detail::launch_thin<7>(p, stream); return;
      default: detail::launch_thin<8>(p, stream); return;
    }
  }
  if (!p.b_k_contiguous && p.m <= 8) {
    switch (p.m) {
      case 1: detail::launch_split_kn<1>(p, stream); return;
      case 2: detail::launch_split_kn<2>(p, stream); return;
      case 3: detail::launch_split_kn<3>(p, stream); return;
      case 4: detail::launch_split_kn<4>(p, stream); return;
      case 5: detail::launch_split_kn<5>(p, stream); return;
      case 6: detail::launch_split_kn<6>(p, stream); return;
      case 7: detail::launch_split_kn<7>(p, stream); return;
      default: detail::launch_split_kn<8>(p, stream); return;
    }
  }
  if (p.b_k_contiguous) detail::launch_wide<true>(p, stream);
  else detail::launch_wide<false>(p, stream);
}

// Row-major Y[rows x out] = X[rows x in] · W[out x in]ᵀ (a projection).
inline Problem linear(const bf16* x, const bf16* w, bf16* y, int rows, int in, int out) {
  Problem p;
  p.a = x; p.b = w; p.c = y; p.m = rows; p.n = out; p.k = in;
  p.lda = in; p.ldb = in; p.ldc = out;
  return p;
}

}  // namespace lfm::gemm
