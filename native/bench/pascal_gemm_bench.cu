// Prefill GEMM throughput on the GTX 1080 (speed only; never acceptance). Times lfm::gemm::run on the engine's
// prefill shapes with random BF16 operands of the model's magnitudes, and prints a checksum of the output so two
// builds of the same arithmetic can be compared.
//   pascal_gemm_bench [REPEATS=3]      build with or without -DLFM_GEMM_TENSOR_CORE_MODEL
#include <lfm/gemm.cuh>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __nv_bfloat16;
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
bf16* random_device(size_t n, unsigned seed, float scale) {
  std::vector<bf16> h(n);
  unsigned s = seed;
  for (auto& v : h) { s = s * 1664525u + 1013904223u; v = __float2bfloat16((int(s >> 8) % 20001 - 10000) * 1e-4f * scale); }
  bf16* d; check(cudaMalloc(&d, n * sizeof(bf16)));
  check(cudaMemcpy(d, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  return d;
}
}  // namespace

int main(int argc, char** argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 3;
  struct Shape { const char* name; int m, n, k; };
  const Shape shapes[] = {
      {"conv.in_proj  257 rows", 257, 6144, 2048}, {"mlp w1        256 rows", 256, 7680, 2048},
      {"mlp w2        256 rows", 256, 2048, 7680}, {"q_proj       4096 rows", 4096, 2048, 2048},
      {"mlp w1       4096 rows", 4096, 7680, 2048}};
#ifdef LFM_GEMM_TENSOR_CORE_MODEL
  std::printf("kernel: exact (tensor-core model %d)\n", int(LFM_GEMM_TENSOR_CORE_MODEL));
#else
  std::printf("kernel: plain\n");
#endif
  {  // bring the card out of its idle state before anything is timed
    bf16* x = random_device(size_t(2048) * 2048, 3, 1.0f); bf16* y; check(cudaMalloc(&y, size_t(2048) * 2048 * 2));
    auto p = lfm::gemm::linear(x, x, y, 2048, 2048, 2048); p.wide = true;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) { lfm::gemm::run(p); check(cudaDeviceSynchronize()); }
    cudaFree(x); cudaFree(y);
  }
  for (const auto& s : shapes) {
    bf16* x = random_device(size_t(s.m) * s.k, 1, 1.0f);
    bf16* w = random_device(size_t(s.n) * s.k, 2, 0.05f);
    bf16* y; check(cudaMalloc(&y, size_t(s.m) * s.n * sizeof(bf16)));
    auto p = lfm::gemm::linear(x, w, y, s.m, s.k, s.n);
    p.wide = true;
    lfm::gemm::run(p); check(cudaDeviceSynchronize());  // warm
    double best = 1e30;
    for (int r = 0; r < repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      lfm::gemm::run(p); check(cudaDeviceSynchronize());
      best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<unsigned short> out(size_t(s.m) * s.n);
    check(cudaMemcpy(out.data(), y, out.size() * 2, cudaMemcpyDeviceToHost));
    unsigned long long sum = 1469598103934665603ull;
    for (unsigned short v : out) sum = (sum ^ v) * 1099511628211ull;
    std::printf("%s  %5d x %5d x %5d  %8.2f ms  %7.1f GFLOP/s  checksum %016llx\n", s.name, s.m, s.n, s.k, best * 1e3,
                2.0 * s.m * s.n * s.k / best * 1e-9, sum);
    cudaFree(x); cudaFree(w); cudaFree(y);
  }
  return 0;
}
