// fp32::linear (which takes kernels::linear_staged for BF16 weights, K a multiple of 256, aligned operands) against
// kernels::linear, the original kernel, bit for bit (validation/speech-speed-v1): the depth decoder's shapes at 1, 8
// and 16 frames (and 3, 5, 13), the codec's (72 trunk rows: 12 frames; its flush shapes; from_quantizer at 1 and 12
// frames), the audio encoder's (downsampler and conformer, short and long clips), ragged M and N, signed zeros, a
// subnormal, a bias, and misaligned operands (which take the original kernel); then the time of each. Exit 1 on any
// difference.
//   nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I native/include native/tests/fp32_linear_staged_check.cu -o check
#include <lfm/fp32_ops.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = lfm::fp32::kernels;
using bf16 = __nv_bfloat16;

template <class F>
double time_us(F f) {
  for (int i = 0; i < 5; ++i) f();
  cudaDeviceSynchronize();
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) f();
    cudaDeviceSynchronize();
    best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50);
  }
  return best;
}

int main() {
  struct Shape { int M, N, K; bool bias; const char* name; };
  const Shape shapes[] = {
      {12, 1024, 1024, false, "depth q/o 1f"},   {12, 256, 1024, false, "depth k/v 1f"},
      {12, 32, 1024, false, "depth gate 1f"},    {12, 2816, 1024, false, "depth w1/w3 1f"},
      {12, 1024, 2816, false, "depth w2 1f"},    {1, 12288, 2048, true, "depth_linear 1f"},
      {1, 2017, 1024, false, "depth head 1"},    {96, 1024, 1024, false, "depth q/o 8f"},
      {96, 256, 1024, false, "depth k/v 8f"},    {96, 32, 1024, false, "depth gate 8f"},
      {96, 2816, 1024, false, "depth w1/w3 8f"}, {96, 1024, 2816, false, "depth w2 8f"},
      {8, 12288, 2048, true, "depth_linear 8f"}, {8, 2017, 1024, false, "depth head 8"},
      {192, 2816, 1024, false, "depth w1 16f"},  {16, 12288, 2048, true, "depth_linear 16f"},
      {72, 256, 512, true, "codec quantizer"},   {72, 512, 512, false, "codec q/o"},
      {72, 1536, 512, false, "codec in_proj"},   {72, 2304, 512, false, "codec w1/w3"},
      {72, 512, 2304, false, "codec w2"},        {72, 1282, 512, true, "codec lin"},
      {5, 77, 512, true, "ragged 5x77"},         {33, 2017, 256, true, "ragged 33x2017"},
      {100, 1000, 768, false, "ragged 100x1000"}, {17, 31, 1024, true, "ragged 17x31"},
      {36, 2816, 1024, false, "depth w1 3f"},    {60, 1024, 2816, false, "depth w2 5f"},
      {156, 256, 1024, false, "depth k/v 13f"},  {3, 2017, 1024, false, "depth head 3"},
      {13, 12288, 2048, true, "depth_linear 13f"}, {1, 512, 256, true, "codec from_q 1f"},
      {12, 512, 256, true, "codec from_q 12f"},  {6, 2304, 512, false, "codec flush 1f"},
      {66, 512, 2304, false, "codec flush 11f"}, {30, 1536, 512, false, "codec flush 5f"},
      {8000, 256, 256, true, "enc down mid"},    {4000, 256, 256, true, "enc down end"},
      {125, 768, 4096, true, "enc lin_out 10s"}, {125, 2304, 768, false, "enc in_proj"},
      {125, 384, 768, false, "enc k/v"},         {125, 1792, 768, false, "enc w1/w3"},
      {125, 768, 1792, false, "enc w2"},         {125, 2048, 768, true, "enc linear_1"},
      {125, 2048, 2048, true, "enc linear_2"},   {7, 768, 4096, true, "enc lin_out 0.5s"},
      {375, 768, 768, false, "enc o 30s"}};
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  int bad = 0;
  for (const auto& s : shapes) {
    std::vector<float> hx(size_t(s.M) * s.K);
    std::vector<bf16> hw(size_t(s.N) * s.K), hb(s.N);
    for (auto& v : hx) v = nd(rng);
    for (auto& v : hw) v = __float2bfloat16(nd(rng) * 0.05f);
    for (auto& v : hb) v = __float2bfloat16(nd(rng));
    hx[0] = -0.0f, hx[1] = 0.0f;
    if (hx.size() > 3) hx[3] = 1e-40f;
    hw[1] = __float2bfloat16(-0.0f);
    float *x, *y, *ref;
    bf16 *w, *b;
    cudaMalloc(&x, hx.size() * 4), cudaMalloc(&w, hw.size() * 2), cudaMalloc(&b, hb.size() * 2);
    cudaMalloc(&y, size_t(s.M) * s.N * 4), cudaMalloc(&ref, size_t(s.M) * s.N * 4);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
    const bf16* bias = s.bias ? b : nullptr;
    auto original = [&] {
      k::linear<bf16><<<dim3((s.N + 31) / 32, (s.M + 15) / 16), 256>>>(x, w, bias, ref, s.M, s.N, s.K);
    };
    auto current = [&] { lfm::fp32::linear(x, w, bias, y, s.M, s.N, s.K); };
    original();
    cudaMemset(y, 0xff, size_t(s.M) * s.N * 4);
    current();
    cudaDeviceSynchronize();
    std::vector<float> want(size_t(s.M) * s.N), got(want.size());
    cudaMemcpy(want.data(), ref, want.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost);
    const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    bad += !same;
    const double a = time_us(original), c = time_us(current);
    std::printf("%-18s M %4d N %5d K %4d  original %8.1f us  fp32::linear %8.1f us  %s\n", s.name, s.M, s.N, s.K, a, c,
                same ? "identical" : "DIFFERENT");
    cudaFree(x), cudaFree(w), cudaFree(b), cudaFree(y), cudaFree(ref);
  }
  {  // misaligned x and w (4 bytes off): fp32::linear must take the original kernel, so the same outputs
    const int M = 20, N = 300, K = 512;
    std::vector<float> hx(size_t(M) * K + 1);
    std::vector<bf16> hw(size_t(N) * K + 2);
    for (auto& v : hx) v = nd(rng);
    for (auto& v : hw) v = __float2bfloat16(nd(rng) * 0.05f);
    float *x, *y, *ref;
    bf16* w;
    cudaMalloc(&x, hx.size() * 4), cudaMalloc(&w, hw.size() * 2), cudaMalloc(&y, size_t(M) * N * 4),
        cudaMalloc(&ref, size_t(M) * N * 4);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    k::linear<bf16><<<dim3((N + 31) / 32, (M + 15) / 16), 256>>>(x + 1, w + 2, nullptr, ref, M, N, K);
    lfm::fp32::linear(x + 1, w + 2, static_cast<const bf16*>(nullptr), y, M, N, K);
    cudaDeviceSynchronize();
    std::vector<float> want(size_t(M) * N), got(want.size());
    cudaMemcpy(want.data(), ref, want.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost);
    const bool same = std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    bad += !same;
    std::printf("%-18s M %4d N %5d K %4d  %s\n", "misaligned", M, N, K, same ? "identical" : "DIFFERENT");
    cudaFree(x), cudaFree(w), cudaFree(y), cudaFree(ref);
  }
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("cuda error %s\n", cudaGetErrorString(e)); return 1; }
  std::printf("%s\n", bad ? "FP32 LINEAR CHECK FAIL" : "FP32 LINEAR CHECK PASS (all identical)");
  return bad != 0;
}
