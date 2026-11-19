// Decode attention's two products through lfm::gemm::run as the runner calls them (speed only): 16 query heads over
// 4 key/value heads, head_dim 128, one row, at several cache lengths. Prints microseconds per call and the rate at
// which the call reads the keys or values (the bytes it must read).
//   decode_attention_bench
#include <lfm/gemm.cuh>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
}
int main() {
  using lfm::gemm::bf16;
  const int heads = 16, kv = 4, head_dim = 128, context = 32768;
  bf16 *q, *cache, *probs, *out; float *scores, *work;
  check(cudaMalloc(&q, heads * head_dim * 2)); check(cudaMalloc(&cache, size_t(kv) * context * head_dim * 2));
  check(cudaMalloc(&probs, size_t(heads) * context * 2)); check(cudaMalloc(&out, heads * head_dim * 2));
  check(cudaMalloc(&scores, size_t(heads) * context * 4));
  const size_t ws = lfm::gemm::workspace_bytes(1, head_dim, context, heads); check(cudaMalloc(&work, ws));
  check(cudaMemset(q, 0, heads * head_dim * 2)); check(cudaMemset(cache, 0, size_t(kv) * context * head_dim * 2)); check(cudaMemset(probs, 0, size_t(heads) * context * 2));
  for (int keys : {1000, 8000, 31000}) {
    const int pitch = (keys + 7) / 8 * 8;
    lfm::gemm::Problem s;
    s.a = q; s.b = cache; s.c_f32 = scores; s.m = 1; s.n = keys; s.k = head_dim; s.lda = heads * head_dim; s.ldb = head_dim; s.ldc = pitch;
    s.stride_a = head_dim; s.stride_b = (long long)context * head_dim; s.stride_c = pitch; s.batch = heads; s.group = heads / kv;
    lfm::gemm::Problem v;
    v.a = probs; v.b = cache; v.c = out; v.m = 1; v.n = head_dim; v.k = pitch; v.b_rows = keys; v.b_k_contiguous = false;
    v.lda = pitch; v.ldb = head_dim; v.ldc = heads * head_dim; v.stride_a = pitch; v.stride_b = (long long)context * head_dim;
    v.stride_c = head_dim; v.batch = heads; v.group = heads / kv; v.workspace = work; v.workspace_size = ws;
    for (int which = 0; which < 2; ++which) {
      const auto& p = which ? v : s;
      for (int i = 0; i < 20; ++i) lfm::gemm::run(p);
      check(cudaDeviceSynchronize());
      cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
      const int n = 200;
      check(cudaEventRecord(a)); for (int i = 0; i < n; ++i) lfm::gemm::run(p); check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
      float ms; check(cudaEventElapsedTime(&ms, a, b));
      const double us = ms * 1e3 / n, bytes = double(kv) * keys * head_dim * 2;
      std::printf("keys %5d %-7s %7.1f us per call, %6.1f GB/s of %s\n", keys, which ? "values" : "scores", us, bytes / us * 1e-3, which ? "values" : "keys");
    }
  }
}
