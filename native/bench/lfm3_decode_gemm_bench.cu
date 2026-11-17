// Decode-GEMM bench for batched decode (P0 speed): for each projection shape of LFM3 (hidden 2048, MLP 7680, 16/4
// heads, vocab 128000; 13 conv + 11 attention layers) and each row count M, the time of lfm::gemm's thin kernel (M <= 8)
// and of its wide (tensor-core) kernel forced at every M. Reports per-step GEMM time summed over the model, the
// effective weight bandwidth and the GEMM-only token rate M / step. Also checks that wide's rows do not depend on M:
// row 0 of every M is bitwise the row of M = 1.
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "lfm/gemm.cuh"

using namespace lfm::gemm;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }

struct Shape { const char* name; int n, k, count; };

int main() {
  const Shape shapes[] = {{"mlp w1/w3", 7680, 2048, 48}, {"mlp w2", 2048, 7680, 24}, {"conv in", 6144, 2048, 13},
                          {"conv out", 2048, 2048, 13}, {"attn q/o", 2048, 2048, 22}, {"attn k/v", 512, 2048, 22},
                          {"attn gate", 16, 2048, 11}, {"lm head", 128000, 2048, 1}};
  const int ms[] = {1, 2, 4, 8, 16, 32, 48, 64};
  const int max_m = 64;
  // Distinct weights for every layer instance, as a real decode step reads them (no L2 reuse between projections).
  size_t total = 0, max_out = 0;
  for (auto& s : shapes) total += size_t(s.n) * s.k * s.count, max_out = std::max(max_out, size_t(max_m) * s.n);
  bf16 *w, *x, *y; ck(cudaMalloc(&w, total * 2)); ck(cudaMalloc(&x, size_t(max_m) * 7680 * 2)); ck(cudaMalloc(&y, max_out * 2));
  {
    std::mt19937 g(1); std::normal_distribution<float> d(0, 0.02f);
    std::vector<bf16> h(size_t(1) << 24);
    for (auto& v : h) v = __float2bfloat16(d(g));
    for (size_t o = 0; o < total; o += h.size()) ck(cudaMemcpy(w + o, h.data(), std::min(h.size(), total - o) * 2, cudaMemcpyHostToDevice));
    std::vector<bf16> hx(size_t(max_m) * 7680); for (auto& v : hx) v = __float2bfloat16(d(g) * 50);
    ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto time = [&](auto launch) {
    launch(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < 9; ++i) { cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms); }
    std::sort(t.begin(), t.end()); return double(t[t.size() / 2]);
  };
  const double weight_bytes = 2.0 * total;
  std::printf("model projection weights %.3f GB per step, all distinct\n", weight_bytes / 1e9);
  std::printf("%-4s %-6s %10s %10s %12s %12s %10s\n", "M", "kernel", "step ms", "GB/s", "GEMM tok/s", "layers ms", "head ms");
  bool invariant = true;
  std::vector<uint16_t> ref;
  for (int m : ms) {
    for (int kind = 0; kind < 2; ++kind) {  // 0 thin (the default for M <= 8), 1 wide forced
      if (kind == 0 && m > 8) continue;
      auto go = [&](bool head) {
        size_t off = 0;
        for (auto& s : shapes) {
          const bool is_head = s.n == 128000;
          for (int c = 0; c < s.count; ++c, off += size_t(s.n) * s.k) {
            if (is_head != head) continue;
            Problem p = linear(x, w + off, y, m, s.k, s.n);
            kind ? detail::launch_wide<detail::Wide32, true>(p, 0) : run(p, 0);
          }
        }
      };
      const double layers = time([&] { go(false); }), head = time([&] { go(true); });
      const double step = layers + head;
      if (kind == 1) {  // the head's row 0 at this M against M = 1
        std::vector<uint16_t> got(128000); ck(cudaMemcpy(got.data(), y, 128000 * 2, cudaMemcpyDeviceToHost));
        if (m == 1) ref = got; else if (std::memcmp(ref.data(), got.data(), 128000 * 2)) invariant = false;
      }
      std::printf("%-4d %-6s %10.3f %10.1f %12.0f %12.3f %10.3f\n", m, kind ? "wide" : "thin", step,
                  weight_bytes / step / 1e6, m / step * 1000, layers, head);
    }
  }
  // Per shape at M = 1 and 32, distinct weights per instance, one shape's instances back to back.
  for (int m : {1, 32}) for (int kind = 0; kind < 2; ++kind) {
    if (kind == 0 && m > 8) continue;
    std::printf("per shape M=%d %s:", m, kind ? "wide" : "thin");
    size_t off = 0;
    for (auto& s : shapes) {
      const size_t start = off;
      off += size_t(s.n) * s.k * s.count;
      const double ms = time([&] {
        for (int c = 0; c < s.count; ++c) {
          Problem p = linear(x, w + start + size_t(c) * s.n * s.k, y, m, s.k, s.n);
          kind ? detail::launch_wide<detail::Wide32, true>(p, 0) : run(p, 0);
        }
      });
      std::printf(" %s=%.1fus(%.0fGB/s)", s.name, 1000 * ms / s.count, 2.0 * s.n * s.k * s.count / ms / 1e6);
    }
    std::printf("\n");
  }
  std::printf("wide rows independent of M (head row 0 bitwise equal for every M): %s\n", invariant ? "yes" : "NO");
  return invariant ? 0 : 1;
}
