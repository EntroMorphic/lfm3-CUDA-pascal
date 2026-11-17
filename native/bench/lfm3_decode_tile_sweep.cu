// Decode tile sweep (P0 speed): lfm::gemm's wide kernel in several tile shapes on LFM3's decode projections, distinct
// weights per instance (DRAM, not L2), M = 1, 8, 32, 64. Every tile shape must give bitwise the outputs of Wide32
// (the G4 tile contract: the same k16 steps in the same order), so a shape can be chosen per projection freely.
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "lfm/gemm.cuh"

using namespace lfm::gemm;
using detail::Tile;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
struct Shape { const char* name; int n, k, count; };

template <class T> void launch(const Problem& p) { detail::launch_wide<T, true>(p, 0); }
using Launch = void (*)(const Problem&);
struct Cfg { const char* name; Launch f; int bm; };
#define CFG(BM, BN, BK, WM, WN, ST) Cfg{#BM "x" #BN "x" #BK " w" #WM "x" #WN " s" #ST, launch<Tile<BM, BN, BK, WM, WN, ST>>, BM}

int main() {
  const Shape shapes[] = {{"w1/w3", 7680, 2048, 48}, {"w2", 2048, 7680, 24}, {"conv in", 6144, 2048, 13},
                          {"2048sq", 2048, 2048, 35}, {"k/v", 512, 2048, 22}, {"head", 128000, 2048, 1}};
  const Cfg cfgs[] = {CFG(32, 128, 64, 2, 2, 3), CFG(32, 128, 64, 2, 2, 4), CFG(32, 128, 64, 2, 2, 6),
                      CFG(32, 64, 64, 2, 2, 4),  CFG(32, 64, 64, 2, 2, 6),  CFG(32, 64, 64, 1, 4, 6),
                      CFG(16, 64, 64, 1, 2, 6),  CFG(16, 64, 64, 1, 4, 8),  CFG(16, 128, 64, 1, 4, 6),
                      CFG(32, 64, 32, 2, 2, 8),  CFG(16, 64, 32, 1, 4, 8),  CFG(64, 128, 64, 2, 2, 3),
                      CFG(64, 64, 64, 2, 2, 4),  CFG(64, 64, 64, 4, 2, 4)};
  const int ms[] = {1, 8, 32, 64};
  size_t total = 0, max_out = 0;
  for (auto& s : shapes) total += size_t(s.n) * s.k * s.count, max_out = std::max(max_out, size_t(64) * s.n);
  bf16 *w, *x, *y; ck(cudaMalloc(&w, total * 2)); ck(cudaMalloc(&x, size_t(64) * 7680 * 2)); ck(cudaMalloc(&y, max_out * 2));
  {
    std::mt19937 g(1); std::normal_distribution<float> d(0, 0.02f);
    std::vector<bf16> h(size_t(1) << 24); for (auto& v : h) v = __float2bfloat16(d(g));
    for (size_t o = 0; o < total; o += h.size()) ck(cudaMemcpy(w + o, h.data(), std::min(h.size(), total - o) * 2, cudaMemcpyHostToDevice));
    std::vector<bf16> hx(size_t(64) * 7680); for (auto& v : hx) v = __float2bfloat16(d(g) * 50);
    ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  bool all_same = true;
  for (int m : ms) {
    std::printf("M=%d\n", m);
    size_t off = 0;
    for (auto& s : shapes) {
      const size_t start = off; off += size_t(s.n) * s.k * s.count;
      std::vector<uint16_t> ref;
      std::printf("  %-8s", s.name);
      for (auto& c : cfgs) {
        auto go = [&] { for (int i = 0; i < s.count; ++i) c.f(linear(x, w + start + size_t(i) * s.n * s.k, y, m, s.k, s.n)); };
        go(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        std::vector<uint16_t> got(size_t(m) * s.n); ck(cudaMemcpy(got.data(), y, got.size() * 2, cudaMemcpyDeviceToHost));
        const bool same = ref.empty() || got == ref;
        if (ref.empty()) ref = got;
        all_same &= same;
        std::vector<float> t;
        for (int i = 0; i < 7; ++i) { cudaEventRecord(a); go(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms); }
        std::sort(t.begin(), t.end());
        std::printf(" %s=%.0f%s", c.name, 2.0 * s.n * s.k * s.count / t[3] / 1e6, same ? "" : "!");
      }
      std::printf("  (GB/s)\n");
    }
  }
  std::printf("all tile shapes bitwise equal to the first: %s\n", all_same ? "yes" : "NO");
  return all_same ? 0 : 1;
}
