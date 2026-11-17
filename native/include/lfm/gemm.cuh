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
// (prefill: tensor cores, mma.sync m16n8k16, a multistage cp.async pipeline through XOR-swizzled shared
// memory, ldmatrix).
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

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Tile configuration: block BM x BN, K per stage BK, warps WARPS_M x WARPS_N, pipeline STAGES.
template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_, int STAGES_>
struct Tile {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, STAGES = STAGES_;
  static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
  static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
  static constexpr int KCH = BK / 8;  // 16-byte chunks per K row
  static constexpr int A_CHUNKS = BM * KCH, B_CHUNKS = BN * KCH;
  static constexpr int SMEM = STAGES * (A_CHUNKS + B_CHUNKS) * 16;
  static_assert(WM % 16 == 0 && WN % 16 == 0 && (KCH == 4 || KCH == 8) && BN % 64 == 0, "tile");
};

// K-row tiles: XOR-swizzle the 16-byte chunk so the eight rows an ldmatrix phase reads land on eight
// distinct bank groups (64-byte rows pair up, 128-byte rows stand alone).
template <int KCH>
__device__ __forceinline__ int swz_k(int row, int chunk) {
  return row * KCH + (chunk ^ (KCH == 8 ? (row & 7) : ((row >> 1) & 3)));
}
// K x BN tiles (BN/8 chunks per row): chunk ^= row & 7.
template <int CPR>
__device__ __forceinline__ int swz_n(int row, int chunk) { return row * CPR + (chunk ^ (row & 7)); }

template <class T, bool BK_CONTIGUOUS>
__global__ void __launch_bounds__(T::THREADS) wide_kernel(Problem p) {
  constexpr int BM = T::BM, BN = T::BN, BK = T::BK, MI = T::MI, NI = T::NI, KCH = T::KCH, STAGES = T::STAGES;
  constexpr int NCH = BN / 8;  // chunks per row of a K x BN tile
  extern __shared__ __align__(128) unsigned char smem[];
  int4* sa = reinterpret_cast<int4*>(smem);
  int4* sb = sa + STAGES * T::A_CHUNKS;

  const int z = blockIdx.z;
  const bf16* A = p.a + z * p.stride_a;
  const bf16* B = p.b + (z / p.group) * p.stride_b;
  const long long c0 = z * p.stride_c;
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wm = warp / T::WARPS_N, wn = warp % T::WARPS_N;
  const int ktiles = (p.k + BK - 1) / BK;

  auto load = [&](int slot, int kt) {
    const int k0 = kt * BK;
    int4* da = sa + slot * T::A_CHUNKS;
#pragma unroll
    for (int i = tid; i < T::A_CHUNKS; i += T::THREADS) {
      const int row = i / KCH, ch = i % KCH;
      const int gm = m0 + row, gk = k0 + ch * 8;
      const bool ok = gm < p.m && gk < p.k;
      cp_async16(smem_u32(da + swz_k<KCH>(row, ch)), ok ? A + gm * p.lda + gk : A, ok);
    }
    int4* db = sb + slot * T::B_CHUNKS;
    if constexpr (BK_CONTIGUOUS) {
#pragma unroll
      for (int i = tid; i < T::B_CHUNKS; i += T::THREADS) {
        const int row = i / KCH, ch = i % KCH;
        const int gn = n0 + row, gk = k0 + ch * 8;
        const bool ok = gn < p.n && gk < p.k;
        cp_async16(smem_u32(db + swz_k<KCH>(row, ch)), ok ? B + gn * p.ldb + gk : B, ok);
      }
    } else {
#pragma unroll
      for (int i = tid; i < T::B_CHUNKS; i += T::THREADS) {
        const int row = i / NCH, ch = i % NCH;
        const int gk = k0 + row, gn = n0 + ch * 8;
        const bool ok = gk < p.b_rows && gn < p.n;
        cp_async16(smem_u32(db + swz_n<NCH>(row, ch)), ok ? B + gk * p.ldb + gn : B, ok);
      }
    }
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) load(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < ktiles) load(next % STAGES, next);
    cp_commit();
    const int slot = kt % STAGES;
    const int4* ta = sa + slot * T::A_CHUNKS;
    const int4* tb = sb + slot * T::B_CHUNKS;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      unsigned af[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int row = wm * T::WM + mi * 16 + (lane % 16);
        ldmatrix_x4(smem_u32(ta + swz_k<KCH>(row, kk * 2 + lane / 16)), af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
      }
      unsigned bfr[NI][2];
#pragma unroll
      for (int pair = 0; pair < NI / 2; ++pair) {
        unsigned r0, r1, r2, r3;
        if constexpr (BK_CONTIGUOUS) {
          const int row = wn * T::WN + pair * 16 + (lane % 8) + (lane / 16) * 8;
          ldmatrix_x4(smem_u32(tb + swz_k<KCH>(row, kk * 2 + (lane / 8) % 2)), r0, r1, r2, r3);
        } else {
          const int row = kk * 16 + (lane % 8) + ((lane / 8) % 2) * 8;
          const int chunk = (wn * T::WN + pair * 16) / 8 + lane / 16;
          ldmatrix_x4_trans(smem_u32(tb + swz_n<NCH>(row, chunk)), r0, r1, r2, r3);
        }
        bfr[pair * 2][0] = r0; bfr[pair * 2][1] = r1;
        bfr[pair * 2 + 1][0] = r2; bfr[pair * 2 + 1][1] = r3;
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[mi], bfr[ni][0], bfr[ni][1]);
    }
  }
  cp_wait<0>();

  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = m0 + wm * T::WM + mi * 16 + g + half * 8;
        const int col = n0 + wn * T::WN + ni * 8 + 2 * t;
        if (row >= p.m) continue;
        const long long at = c0 + row * p.ldc + col;
        const float x = acc[mi][ni][half * 2], y = acc[mi][ni][half * 2 + 1];
        if (p.c_f32) {
          if (col < p.n) p.c_f32[at] = x;
          if (col + 1 < p.n) p.c_f32[at + 1] = y;
          continue;
        }
        bf16* out = p.c + at;
        if (col + 1 < p.n && (reinterpret_cast<std::uintptr_t>(out) & 3) == 0) *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(x, y);
        else {
          if (col < p.n) out[0] = __float2bfloat16_rn(x);
          if (col + 1 < p.n) out[1] = __float2bfloat16_rn(y);
        }
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

template <class T, bool KC>
inline void launch_wide(const Problem& p, cudaStream_t stream) {
  static const bool configured =
      cudaFuncSetAttribute(wide_kernel<T, KC>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM) == cudaSuccess;
  if (!configured) throw std::runtime_error("gemm: cannot configure shared memory");
  wide_kernel<T, KC><<<dim3((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch), T::THREADS, T::SMEM, stream>>>(p);
}

// Measured on Thor (validation/gemm-probe-v1, sweep): 128x128x64 with four 64x64 warp tiles and three
// stages is the fastest on every prefill shape (36-42 TFLOPS, ~70% of mma.sync's 57 TFLOPS ceiling).
using Wide128 = Tile<128, 128, 64, 2, 2, 3>;
using Wide64 = Tile<64, 128, 64, 2, 2, 3>;
using Wide32 = Tile<32, 128, 64, 2, 2, 3>;

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
  const bool kc = p.b_k_contiguous;
  if (p.m <= 32) kc ? detail::launch_wide<detail::Wide32, true>(p, stream) : detail::launch_wide<detail::Wide32, false>(p, stream);
  else if (p.m <= 64) kc ? detail::launch_wide<detail::Wide64, true>(p, stream) : detail::launch_wide<detail::Wide64, false>(p, stream);
  else kc ? detail::launch_wide<detail::Wide128, true>(p, stream) : detail::launch_wide<detail::Wide128, false>(p, stream);
}

// Row-major Y[rows x out] = X[rows x in] · W[out x in]ᵀ (a projection).
inline Problem linear(const bf16* x, const bf16* w, bf16* y, int rows, int in, int out) {
  Problem p;
  p.a = x; p.b = w; p.c = y; p.m = rows; p.n = out; p.k = in;
  p.lda = in; p.ldb = in; p.ldc = out;
  return p;
}

}  // namespace lfm::gemm
