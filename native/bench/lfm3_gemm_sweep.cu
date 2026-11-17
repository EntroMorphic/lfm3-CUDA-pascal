// G4 tile sweep for lfm::gemm's wide (mma.sync) kernel on the dominant prefill shapes. Every tile
// configuration accumulates each output over the same k16 steps in the same order, so all outputs
// must be bitwise identical to the reference configuration; timing is the median of warmed runs.
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "lfm/gemm.cuh"

using namespace lfm::gemm;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }

template <class T>
static double time_cfg(const Problem& p, bf16* out, const std::vector<uint16_t>& ref, size_t count, const char* name) {
  Problem q = p; q.c = out;
  detail::launch_wide<T, true>(q, 0);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  std::vector<uint16_t> got(count);
  ck(cudaMemcpy(got.data(), out, count * 2, cudaMemcpyDeviceToHost));
  const bool same = ref.empty() || std::memcmp(got.data(), ref.data(), count * 2) == 0;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> t;
  for (int i = 0; i < 9; ++i) {
    cudaEventRecord(a); detail::launch_wide<T, true>(q, 0); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  const double flops = 2.0 * p.m * p.n * p.k;
  std::printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"tile\":\"%s\",\"smem\":%d,\"ms\":%.4f,\"tflops\":%.2f,\"bitwise_equal\":%s}\n",
              p.m, p.n, p.k, name, T::SMEM, t[4], flops / t[4] / 1e9, same ? "true" : "false");
  std::fflush(stdout);
  return t[4];
}

int main() {
  const int shapes[][3] = {{4096, 6144, 2048}, {4096, 2048, 7680}, {32768, 7680, 2048}, {512, 6144, 2048}};
  std::mt19937 rng(7);
  std::normal_distribution<float> d(0.f, 1.f);
  for (const auto& s : shapes) {
    const int m = s[0], n = s[1], k = s[2];
    std::vector<uint16_t> ha(size_t(m) * k), hb(size_t(n) * k);
    for (auto& x : ha) { float f = d(rng); uint32_t u; std::memcpy(&u, &f, 4); x = u >> 16; }
    for (auto& x : hb) { float f = d(rng) * 0.02f; uint32_t u; std::memcpy(&u, &f, 4); x = u >> 16; }
    bf16 *a, *b, *c0, *c1;
    ck(cudaMalloc(&a, ha.size() * 2)); ck(cudaMalloc(&b, hb.size() * 2));
    ck(cudaMalloc(&c0, size_t(m) * n * 2)); ck(cudaMalloc(&c1, size_t(m) * n * 2));
    ck(cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
    Problem p = linear(a, b, c0, m, k, n);
    const size_t count = size_t(m) * n;
    std::vector<uint16_t> ref;
    time_cfg<detail::Tile<128, 128, 32, 2, 4, 4>>(p, c0, ref, count, "128x128x32 w2x4 s4");
    ref.resize(count);
    ck(cudaMemcpy(ref.data(), c0, count * 2, cudaMemcpyDeviceToHost));
    time_cfg<detail::Tile<128, 128, 64, 2, 4, 3>>(p, c1, ref, count, "128x128x64 w2x4 s3");
    time_cfg<detail::Tile<128, 128, 32, 2, 2, 4>>(p, c1, ref, count, "128x128x32 w2x2 s4");
    time_cfg<detail::Tile<128, 128, 64, 2, 2, 3>>(p, c1, ref, count, "128x128x64 w2x2 s3");
    time_cfg<detail::Tile<128, 256, 32, 2, 4, 3>>(p, c1, ref, count, "128x256x32 w2x4 s3");
    time_cfg<detail::Tile<128, 256, 32, 2, 4, 4>>(p, c1, ref, count, "128x256x32 w2x4 s4");
    time_cfg<detail::Tile<128, 256, 64, 2, 4, 3>>(p, c1, ref, count, "128x256x64 w2x4 s3");
    time_cfg<detail::Tile<256, 128, 32, 4, 2, 3>>(p, c1, ref, count, "256x128x32 w4x2 s3");
    time_cfg<detail::Tile<256, 128, 64, 4, 2, 3>>(p, c1, ref, count, "256x128x64 w4x2 s3");
    time_cfg<detail::Tile<64, 128, 32, 2, 4, 4>>(p, c1, ref, count, "64x128x32 w2x4 s4");
    time_cfg<detail::Tile<64, 256, 32, 2, 4, 4>>(p, c1, ref, count, "64x256x32 w2x4 s4");
    cudaFree(a); cudaFree(b); cudaFree(c0); cudaFree(c1);
  }
  return 0;
}
