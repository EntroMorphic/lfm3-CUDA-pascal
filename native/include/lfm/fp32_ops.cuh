#pragma once
// FP32-activation operators shared by the native audio-output path (codec decoder, depth decoder): weights are used at
// their stored BF16 (or FP32) values, every activation and accumulator is FP32. No cuBLAS.
//
// Row independence: every operator computes each row the same way whatever the number of rows in the call, so
// results do not depend on batching or chunking. For `linear` this fixes the reduction order of each output to one
// that depends only on K: lane l of a warp sums k = 256 j + 8 l + e in ascending (j, e) with FMA, then the 32 lane
// sums are added by a fixed shuffle tree.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace lfm::fp32 {

inline void check(cudaError_t s, const char* what) {
  if (s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}

template <class T>
struct DeviceArray {
  T* p = nullptr;
  size_t n = 0;
  DeviceArray() = default;
  explicit DeviceArray(size_t count) : n(count) {
    check(cudaMalloc(&p, count * sizeof(T)), "fp32 cudaMalloc");
    check(cudaMemset(p, 0, count * sizeof(T)), "fp32 cudaMemset");
  }
  DeviceArray(DeviceArray&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  DeviceArray& operator=(DeviceArray&& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
  DeviceArray(const DeviceArray&) = delete;
  ~DeviceArray() { if (p) cudaFree(p); }
  size_t bytes() const { return n * sizeof(T); }
};

namespace kernels {
using bf16 = __nv_bfloat16;
__device__ inline float value(float x) { return x; }
__device__ inline float value(bf16 x) { return __bfloat162float(x); }

// w[0..8) for k..k+7 (zeros past K), by one 16-byte load when aligned and in bounds.
__device__ inline void load8(const bf16* w, int remaining, float out[8]) {
  if (remaining >= 8 && (reinterpret_cast<uintptr_t>(w) & 15) == 0) {
    const uint4 raw = *reinterpret_cast<const uint4*>(w);
    const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = __bfloat162float(h[e]);
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = e < remaining ? __bfloat162float(w[e]) : 0.0f;
  }
}
__device__ inline void load8(const float* w, int remaining, float out[8]) {
  if (remaining >= 8 && (reinterpret_cast<uintptr_t>(w) & 15) == 0) {
    const float4 a = reinterpret_cast<const float4*>(w)[0], b = reinterpret_cast<const float4*>(w)[1];
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w, out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = e < remaining ? w[e] : 0.0f;
  }
}

constexpr int kRows = 16, kColumnsPerWarp = 4, kWarps = 8, kChunk = 256;
// Y[M,N] = X[M,K] . W[N,K]^T (+ bias). A block: 16 rows x 32 columns (8 warps x 4 columns).
template <class W>
__global__ void __launch_bounds__(256) linear(const float* X, const W* w, const bf16* bias, float* Y, int M, int N, int K) {
  __shared__ __align__(16) float xs[kRows][kChunk];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = blockIdx.y * kRows;
  const int column0 = (blockIdx.x * kWarps + warp) * kColumnsPerWarp, rows = min(kRows, M - row0);
  float acc[kRows][kColumnsPerWarp] = {};
  for (int k0 = 0; k0 < K; k0 += kChunk) {
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * kChunk; i += 256) {
      const int r = i / kChunk, k = i % kChunk;
      xs[r][k] = r < rows && k0 + k < K ? X[size_t(row0 + r) * K + k0 + k] : 0.0f;
    }
    __syncthreads();
    const int k = k0 + 8 * lane;
    float wv[kColumnsPerWarp][8];
#pragma unroll
    for (int c = 0; c < kColumnsPerWarp; ++c) {
      if (column0 + c < N && k < K) {
        load8(w + size_t(column0 + c) * K + k, K - k, wv[c]);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) wv[c][e] = 0.0f;
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      if (r < rows) {
        const float4 a = *reinterpret_cast<const float4*>(&xs[r][8 * lane]);
        const float4 b = *reinterpret_cast<const float4*>(&xs[r][8 * lane + 4]);
        const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int c = 0; c < kColumnsPerWarp; ++c)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[r][c] = __fmaf_rn(xv[e], wv[c][e], acc[r][c]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int c = 0; c < kColumnsPerWarp; ++c) {
      float v = acc[r][c];
      for (int o = 16; o; o /= 2) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));
      const int n = column0 + c;
      if (lane == 0 && r < rows && n < N) Y[size_t(row0 + r) * N + n] = bias ? __fadd_rn(v, value(bias[n])) : v;
    }
}

// `linear` restaged (validation/speech-speed-v1), for BF16 weights when K is a multiple of 256 and X and w are 16-byte
// aligned. Each thread does exactly `linear`'s arithmetic, so every output is the same sum in the same order, bit for
// bit: lane l sums k = 256 j + 8 l + e in ascending (j, e) with FMA from +0 for each of its rows x columns, then the
// warp's shuffle tree, then the bias. Only the data movement differs: each 256-chunk of X's rows and of the warp's
// weight rows is copied to shared memory asynchronously, one chunk ahead (double buffered; X stored so that each lane's
// 8 values are two float4s at consecutive addresses across the warp); rows past M are zeros there, so the summing loop
// has no branch and loads the next row's values while it sums the current one. Rows: 16 (8 for M < 16).
__device__ inline void copy16_async(void* shared, const void* global) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(global));
}
__device__ inline void commit_async() { asm volatile("cp.async.commit_group;\n" ::); }
template <int Pending>
__device__ inline void wait_async() { asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending)); }

template <int Rows>
constexpr int staged_shared_bytes() { return (2 * Rows * 64 + 2 * kWarps * kColumnsPerWarp * 32) * 16; }

template <int Rows>
__global__ void __launch_bounds__(256, 2) linear_staged(const float* X, const bf16* w, const bf16* bias, float* Y,
                                                        int M, int N, int K) {
  constexpr int Cols = kColumnsPerWarp;
  extern __shared__ float4 staged[];
  auto xs = reinterpret_cast<float4 (*)[Rows][2][32]>(staged);                       // [buffer][row][half][lane]
  auto ws = reinterpret_cast<uint4 (*)[kWarps][Cols][32]>(staged + 2 * Rows * 64);  // [buffer][warp][column][lane]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = blockIdx.y * Rows;
  const int column0 = (blockIdx.x * kWarps + warp) * Cols, rows = min(Rows, M - row0), chunks = K / kChunk;
  for (int i = threadIdx.x; i < 2 * Rows * 64; i += 256)
    if ((i / 64) % Rows >= rows) staged[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  auto fill = [&](int buffer, int k0) {
    for (int i = threadIdx.x; i < Rows * 64; i += 256) {
      const int r = i / 64, q = i % 64;  // the chunk's q-th float4: lane q / 2, half q % 2
      if (r < rows) copy16_async(&xs[buffer][r][q % 2][q / 2], X + size_t(row0 + r) * K + k0 + 4 * q);
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      if (column0 + c < N) copy16_async(&ws[buffer][warp][c][lane], w + size_t(column0 + c) * K + k0 + 8 * lane);
      else ws[buffer][warp][c][lane] = make_uint4(0, 0, 0, 0);
    }
    commit_async();
  };
  float acc[Rows][Cols] = {};
  fill(0, 0);
  for (int j = 0; j < chunks; ++j) {
    const int buffer = j & 1;
    if (j + 1 < chunks) {
      fill(buffer ^ 1, (j + 1) * kChunk);
      wait_async<1>();
    } else {
      wait_async<0>();
    }
    __syncthreads();
    float wv[Cols][8];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      const uint4 raw = ws[buffer][warp][c][lane];
      const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
      for (int e = 0; e < 8; ++e) wv[c][e] = __bfloat162float(h[e]);
    }
    float4 a = xs[buffer][0][0][lane], b = xs[buffer][0][1][lane];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      float4 na, nb;
      if (r + 1 < Rows) na = xs[buffer][r + 1][0][lane], nb = xs[buffer][r + 1][1][lane];
      const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int c = 0; c < Cols; ++c) acc[r][c] = __fmaf_rn(xv[e], wv[c][e], acc[r][c]);
      if (r + 1 < Rows) a = na, b = nb;
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < Rows; ++r)
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      float v = acc[r][c];
      for (int o = 16; o; o /= 2) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));
      const int n = column0 + c;
      if (lane == 0 && r < rows && n < N) Y[size_t(row0 + r) * N + n] = bias ? __fadd_rn(v, value(bias[n])) : v;
    }
}

// One warp per row: y = w * (x / sqrt(mean(x^2) + eps)), plus `residual` when given; in place is allowed. Row r uses
// weights w + r * weight_stride (0: one weight vector for all rows).
__global__ void rms_norm(const float* x, const bf16* w, size_t weight_stride, float* y, const float* residual,
                         int rows, int width, float eps) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  const float* xr = x + size_t(row) * width;
  const bf16* wr = w + size_t(row) * weight_stride;
  float sum = 0.0f;
  for (int d = lane; d < width; d += 32) sum = __fmaf_rn(xr[d], xr[d], sum);
  for (int o = 16; o; o /= 2) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, o));
  sum = __shfl_sync(0xffffffffu, sum, 0);
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(width)), eps)));
  for (int d = lane; d < width; d += 32) {
    const float v = __fmul_rn(value(wr[d]), __fmul_rn(xr[d], inv));
    y[size_t(row) * width + d] = residual ? __fadd_rn(residual[size_t(row) * width + d], v) : v;
  }
}

// a = silu(a) * b
__global__ void swiglu(float* a, const float* b, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __fmul_rn(__fdiv_rn(a[i], __fadd_rn(1.0f, expf(-a[i]))), b[i]);
}

// Rotary embedding over the full head (rotate-half pairs d, d + dim/2), row t at position position0 + t; angle and
// rotation in FP64, rounded once. x is [rows, heads, dim].
__global__ void rotary(float* x, int rows, int heads, int dim, long long position0, double theta) {
  const int half = dim / 2, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * heads * half) return;
  const int d = i % half, head_row = i / half, t = head_row / heads;
  const double inverse = 1.0 / pow(theta, double(2 * d) / dim);
  double s, c;
  sincos(double(position0 + t) * inverse, &s, &c);
  float* v = x + size_t(head_row) * dim;
  const double a = v[d], b = v[d + half];
  v[d] = float(a * c - b * s);
  v[d + half] = float(b * c + a * s);
}
}  // namespace kernels

inline int blocks(size_t threads) { return int((threads + 255) / 256); }

template <int Rows>
void linear_staged(const float* x, const __nv_bfloat16* w, const __nv_bfloat16* bias, float* y, int M, int N, int K,
                   cudaStream_t s) {
  constexpr int bytes = kernels::staged_shared_bytes<Rows>();
  static const bool configured = [] {  // more than the default 48 KiB of dynamic shared memory for 16 rows
    check(cudaFuncSetAttribute(kernels::linear_staged<Rows>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
          "fp32 linear shared memory");
    return true;
  }();
  (void)configured;
  const int per_block = kernels::kWarps * kernels::kColumnsPerWarp;
  kernels::linear_staged<Rows><<<dim3((N + per_block - 1) / per_block, (M + Rows - 1) / Rows), 256, bytes, s>>>(
      x, w, bias, y, M, N, K);
}

template <class W>
void linear(const float* x, const W* w, const __nv_bfloat16* bias, float* y, int M, int N, int K, cudaStream_t s = 0) {
  if (M <= 0 || N <= 0) return;
  if constexpr (std::is_same_v<W, __nv_bfloat16>) {
    const bool aligned = (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16 == 0;
    if (K > 0 && K % kernels::kChunk == 0 && aligned) {  // the same outputs, restaged (kernels::linear_staged)
      if (M < 16) linear_staged<8>(x, w, bias, y, M, N, K, s);
      else linear_staged<16>(x, w, bias, y, M, N, K, s);
      return;
    }
  }
  const int per_block = kernels::kWarps * kernels::kColumnsPerWarp;
  kernels::linear<W><<<dim3((N + per_block - 1) / per_block, (M + kernels::kRows - 1) / kernels::kRows), 256, 0, s>>>(
      x, w, bias, y, M, N, K);
}
inline void rms_norm(const float* x, const __nv_bfloat16* w, float* y, const float* residual, int rows, int width,
                     float eps, size_t weight_stride = 0, cudaStream_t s = 0) {
  kernels::rms_norm<<<blocks(size_t(rows) * 32), 256, 0, s>>>(x, w, weight_stride, y, residual, rows, width, eps);
}
inline void swiglu(float* a, const float* b, size_t n, cudaStream_t s = 0) {
  kernels::swiglu<<<blocks(n), 256, 0, s>>>(a, b, n);
}
inline void rotary(float* x, int rows, int heads, int dim, long long position0, double theta, cudaStream_t s = 0) {
  kernels::rotary<<<blocks(size_t(rows) * heads * dim / 2), 256, 0, s>>>(x, rows, heads, dim, position0, theta);
}

}  // namespace lfm::fp32
