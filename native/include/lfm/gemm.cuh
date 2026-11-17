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
// Quality bar (journal §9): perplexity equal or better than the standard stack within the 95% CI.
// No reproduction of cuBLAS's kernel selection or reduction order is attempted or claimed.
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
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
};

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
// One block computes a 32 x 32 tile of C; each of its 256 threads owns a 2 x 2 patch. K advances 32 at a
// time through shared memory, operands converted to FP32 as they are loaded (exact). The reduction order
// of every output is fixed and simple: k ascending, one fused multiply-add per term.
// Elements outside the operands read as zero: rows >= m, columns >= n, k >= K (or >= b_rows for a K x N B).
inline constexpr int kWideTile = 32;
template <bool BK_CONTIGUOUS>
__global__ void __launch_bounds__(256) wide_kernel(Problem p) {
  constexpr int T = kWideTile;
  __shared__ float sa[T][T];  // [row][k]
  __shared__ float sb[T][T];  // [column][k]
  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * T, m0 = blockIdx.y * T;
  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int k0 = 0; k0 < p.k; k0 += T) {
#pragma unroll
    for (int i = tid; i < T * T; i += 256) {
      const int r = i / T, kk = i % T;
      const long long gm = m0 + r, gn = n0 + r, gk = k0 + kk;
      sa[r][kk] = gm < p.m && gk < p.k ? __bfloat162float(A[gm * p.lda + gk]) : 0.f;
      if constexpr (BK_CONTIGUOUS) sb[r][kk] = gn < p.n && gk < p.k ? __bfloat162float(B[gn * p.ldb + gk]) : 0.f;
      else sb[r][kk] = gn < p.n && gk < p.b_rows ? __bfloat162float(B[gk * p.ldb + gn]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < T; ++kk) {
      const float a0 = sa[2 * ty][kk], a1 = sa[2 * ty + 1][kk], b0 = sb[2 * tx][kk], b1 = sb[2 * tx + 1][kk];
      acc[0][0] = fmaf(a0, b0, acc[0][0]);
      acc[0][1] = fmaf(a0, b1, acc[0][1]);
      acc[1][0] = fmaf(a1, b0, acc[1][0]);
      acc[1][1] = fmaf(a1, b1, acc[1][1]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const long long row = m0 + 2 * ty + i, col = n0 + 2 * tx + j;
      if (row < p.m && col < p.n) put(p, c0 + row * p.ldc + col, acc[i][j]);
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
template <int R>
__global__ void __launch_bounds__(128) split_kn_partial(Problem p, int splits) {
  const int split = blockIdx.x, z = blockIdx.y;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const int col = 2 * (blockIdx.z * blockDim.x + threadIdx.x);
  if (col >= p.n) return;
  const int k0 = split * kSplitKeys, k1 = min(k0 + kSplitKeys, p.b_rows);
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
  for (int kk = k0; kk < k1; ++kk) {
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(B + kk * p.ldb + col));
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float a = __bfloat162float(A[r * p.lda + kk]);
      acc[r][0] = fmaf(a, v.x, acc[r][0]);
      acc[r][1] = fmaf(a, v.y, acc[r][1]);
    }
  }
  float* out = p.workspace + ((size_t(z) * splits + split) * R) * p.n + col;
#pragma unroll
  for (int r = 0; r < R; ++r) { out[r * p.n] = acc[r][0]; out[r * p.n + 1] = acc[r][1]; }
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
template <int R>
inline void launch_split_kn(const Problem& p, cudaStream_t stream) {
  const int splits = (p.k + kSplitKeys - 1) / kSplitKeys;
  if (!p.workspace || p.workspace_size < workspace_bytes(R, p.n, p.k, p.batch))
    throw std::runtime_error("gemm: split-K workspace too small");
  const int threads = 64, column_blocks = (p.n / 2 + threads - 1) / threads;
  split_kn_partial<R><<<dim3(splits, p.batch, column_blocks), threads, 0, stream>>>(p, splits);
  const size_t outputs = size_t(R) * p.n * p.batch;
  split_kn_finish<R><<<unsigned((outputs + 255) / 256), 256, 0, stream>>>(p, splits);
}

// M <= 8 rows, K = 128 (attention Q·Kᵀ in decode): eight lanes per B row, each lane 16 consecutive
// elements (two 16-byte loads), four rows per warp step; a fixed three-step butterfly per output.
template <int R>
__global__ void __launch_bounds__(256) thin_k128_kernel(Problem p) {
  const int z = blockIdx.y;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int lane = threadIdx.x % 32, sub = lane % 8, slot = lane / 8;
  const int warps = gridDim.x * (blockDim.x / 32);
  const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  float q[R][16];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int4 lo = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16));
    const int4 hi = __ldg(reinterpret_cast<const int4*>(A + r * p.lda + sub * 16 + 8));
    const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
      q[r][2 * e] = a.x; q[r][2 * e + 1] = a.y; q[r][8 + 2 * e] = b.x; q[r][8 + 2 * e + 1] = b.y;
    }
  }
  for (int row0 = warp * 4; row0 < p.n; row0 += warps * 4) {
    const int row = row0 + slot;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    if (row < p.n) {
      const int4 lo = __ldg(reinterpret_cast<const int4*>(B + row * p.ldb + sub * 16));
      const int4 hi = __ldg(reinterpret_cast<const int4*>(B + row * p.ldb + sub * 16 + 8));
      const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lo);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hi);
      float k[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = __bfloat1622float2(l2[e]), b = __bfloat1622float2(h2[e]);
        k[2 * e] = a.x; k[2 * e + 1] = a.y; k[8 + 2 * e] = b.x; k[8 + 2 * e + 1] = b.y;
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[r] = fmaf(q[r][e], k[e], acc[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[r];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (sub == r && row < p.n) put(p, c0 + r * p.ldc + row, v);
    }
  }
}
template <int R>
inline void launch_thin_k128(const Problem& p, cudaStream_t stream) {
  static_assert(R <= 8, "one lane of the eight stores each row");
  const int blocks = std::min((p.n + 31) / 32, 1024);
  thin_k128_kernel<R><<<dim3(blocks, p.batch), 256, 0, stream>>>(p);
}

template <int R>
inline void launch_thin(const Problem& p, cudaStream_t stream) {
  const int warps = (p.n + 3) / 4;
  thin_kernel<R><<<dim3((warps + 7) / 8, p.batch), 256, 0, stream>>>(p);
}

template <bool KC>
inline void launch_wide(const Problem& p, cudaStream_t stream) {
  constexpr int T = kWideTile;
  wide_kernel<KC><<<dim3((p.n + T - 1) / T, (p.m + T - 1) / T, p.batch), 256, 0, stream>>>(p);
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
