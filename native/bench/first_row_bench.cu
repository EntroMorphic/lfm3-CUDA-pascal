// What position 0's second launch costs (speed only): a one-row prefill product with row0 = 0 against row0 = -1,
// on the engine's projection shapes. Build with -DLFM_GEMM_FIRST_TX=N to compare block widths.
//   first_row_bench [REPEATS=20]
#include <lfm/gemm.cuh>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
lfm::gemm::bf16* random_device(size_t n, unsigned seed) {
  std::vector<lfm::gemm::bf16> h(n);
  unsigned s = seed;
  for (auto& v : h) { s = s * 1664525u + 1013904223u; v = __float2bfloat16((int(s >> 8) % 20001 - 10000) * 1e-5f); }
  lfm::gemm::bf16* d; check(cudaMalloc(&d, n * 2)); check(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
  return d;
}
}  // namespace
int main(int argc, char** argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 20;
  struct Shape { const char* name; int n, k, per_prefill; };
  // per_prefill: calls per prefill in the engine (24 layers)
  const Shape shapes[] = {{"mlp w1/w3", 7680, 2048, 48}, {"mlp w2", 2048, 7680, 24}, {"conv in_proj", 6144, 2048, 13},
                          {"2048 x 2048", 2048, 2048, 35}, {"k/v", 512, 2048, 22}};
  double total = 0;
  for (const Shape& s : shapes) {
    auto* a = random_device(size_t(16) * s.k, 1); auto* b = random_device(size_t(s.n) * s.k, 2);
    lfm::gemm::bf16* c; check(cudaMalloc(&c, size_t(16) * s.n * 2));
    lfm::gemm::Problem p; p.a = a; p.b = b; p.c = c; p.m = 1; p.n = s.n; p.k = s.k; p.lda = s.k; p.ldb = s.k; p.ldc = s.n; p.wide = true;
    double t[2];
    for (int v = 0; v < 2; ++v) {
      p.row0 = v ? 0 : -1;
      for (int i = 0; i < 3; ++i) lfm::gemm::run(p);
      check(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeats; ++i) lfm::gemm::run(p);
      check(cudaDeviceSynchronize());
      t[v] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / repeats;
    }
    std::printf("%-14s n %5d k %5d: plain %.3f ms, with position 0 %.3f ms; position 0 costs %.3f ms x %d per prefill\n",
                s.name, s.n, s.k, t[0] * 1e3, t[1] * 1e3, (t[1] - t[0]) * 1e3, s.per_prefill);
    total += (t[1] - t[0]) * s.per_prefill;
    check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c));
  }
  std::printf("position 0 per prefill (projections): %.1f ms\n", total * 1e3);
}
