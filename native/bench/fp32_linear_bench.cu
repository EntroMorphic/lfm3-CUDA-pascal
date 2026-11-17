// fp32::linear block shapes for the batched depth decoder (validation/speech-batch-v1): rows x columns per warp,
// timed on the depth trunk's shapes at 8 frames (96 slot rows), and checked bit for bit against fp32::linear (each
// output is the same sum in the same order whatever the shape: lanes' 8-wide slices of each 256-wide chunk of K, in
// order, then the warp's reduction).
//   nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I native/include native/bench/fp32_linear_bench.cu -o fp32-linear-bench
#include <lfm/fp32_ops.cuh>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = lfm::fp32::kernels;
using bf16 = __nv_bfloat16;

template <int Rows, int Cols>
__global__ void __launch_bounds__(256) shaped(const float* X, const bf16* w, float* Y, int M, int N, int K) {
  constexpr int kChunk = 256, kWarps = 8;
  __shared__ __align__(16) float xs[Rows][kChunk];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row0 = blockIdx.y * Rows;
  const int column0 = (blockIdx.x * kWarps + warp) * Cols, rows = min(Rows, M - row0);
  float acc[Rows][Cols] = {};
  for (int k0 = 0; k0 < K; k0 += kChunk) {
    __syncthreads();
    for (int i = threadIdx.x; i < Rows * kChunk; i += 256) {
      const int r = i / kChunk, kk = i % kChunk;
      xs[r][kk] = r < rows && k0 + kk < K ? X[size_t(row0 + r) * K + k0 + kk] : 0.0f;
    }
    __syncthreads();
    const int kk = k0 + 8 * lane;
    float wv[Cols][8];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      if (column0 + c < N && kk < K) k::load8(w + size_t(column0 + c) * K + kk, K - kk, wv[c]);
      else
#pragma unroll
        for (int e = 0; e < 8; ++e) wv[c][e] = 0.0f;
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if (r < rows) {
        const float4 a = *reinterpret_cast<const float4*>(&xs[r][8 * lane]);
        const float4 b = *reinterpret_cast<const float4*>(&xs[r][8 * lane + 4]);
        const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int c = 0; c < Cols; ++c)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[r][c] = __fmaf_rn(xv[e], wv[c][e], acc[r][c]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < Rows; ++r)
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      float v = acc[r][c];
      for (int o = 16; o; o /= 2) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));
      const int n = column0 + c;
      if (lane == 0 && r < rows && n < N) Y[size_t(row0 + r) * N + n] = v;
    }
}

template <int Rows, int Cols>
void run_shaped(const float* x, const bf16* w, float* y, int M, int N, int K) {
  shaped<Rows, Cols><<<dim3((N + 8 * Cols - 1) / (8 * Cols), (M + Rows - 1) / Rows), 256>>>(x, w, y, M, N, K);
}

int main() {
  struct Shape { int M, N, K; const char* name; };
  const Shape shapes[] = {{96, 1024, 1024, "q/o"}, {96, 256, 1024, "k/v"}, {96, 2816, 1024, "w1/w3"},
                          {96, 1024, 2816, "w2"}, {8, 12288, 2048, "depth_linear"}, {8, 2017, 1024, "head"}};
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 1);
  using Fn = void (*)(const float*, const bf16*, float*, int, int, int);
  struct Variant { const char* name; Fn fn; };
  const Variant variants[] = {
      {"16x4 (fp32::linear)", [](const float* x, const bf16* w, float* y, int M, int N, int K) { lfm::fp32::linear(x, w, nullptr, y, M, N, K); }},
      {"8x4", run_shaped<8, 4>}, {"16x2", run_shaped<16, 2>}, {"16x8", run_shaped<16, 8>}, {"32x2", run_shaped<32, 2>},
      {"24x4", run_shaped<24, 4>}, {"12x4", run_shaped<12, 4>}, {"40x2", run_shaped<40, 2>}};
  for (const auto& s : shapes) {
    std::vector<float> hx(size_t(s.M) * s.K);
    std::vector<bf16> hw(size_t(s.N) * s.K);
    for (auto& v : hx) v = nd(rng);
    for (auto& v : hw) v = __float2bfloat16(nd(rng) * 0.05f);
    float *x, *y, *ref;
    bf16* w;
    cudaMalloc(&x, hx.size() * 4), cudaMalloc(&w, hw.size() * 2), cudaMalloc(&y, size_t(s.M) * s.N * 4),
        cudaMalloc(&ref, size_t(s.M) * s.N * 4);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    k::linear<bf16><<<dim3((s.N + 31) / 32, (s.M + 15) / 16), 256>>>(x, w, nullptr, ref, s.M, s.N, s.K);  // the original
    std::vector<float> want(size_t(s.M) * s.N), got(want.size());
    cudaMemcpy(want.data(), ref, want.size() * 4, cudaMemcpyDeviceToHost);
    for (const auto& v : variants) {
      cudaMemset(y, 0, got.size() * 4);
      v.fn(x, w, y, s.M, s.N, s.K);
      cudaDeviceSynchronize();
      cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost);
      const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
      for (int i = 0; i < 20; ++i) v.fn(x, w, y, s.M, s.N, s.K);
      cudaDeviceSynchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 200; ++i) v.fn(x, w, y, s.M, s.N, s.K);
      cudaDeviceSynchronize();
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
      std::printf("%-12s %-20s %8.1f us  %s\n", s.name, v.name, us, same ? "identical" : "DIFFERENT");
    }
    cudaFree(x), cudaFree(w), cudaFree(y), cudaFree(ref);
  }
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("cuda error %s\n", cudaGetErrorString(e)); return 1; }
}
