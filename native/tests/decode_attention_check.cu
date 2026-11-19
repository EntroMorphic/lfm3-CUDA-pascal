// Decode attention's two products with query heads paired on one read of the keys and values (G = 2) against one
// head at a time (G = 1), on the card: every output bit for bit, over cache lengths from 1 to 31,000 keys, one and
// two rows, and groups of 2 and 4 query heads per key/value head. Both forms are launched whatever the engine's
// rule for choosing between them.
//   decode_attention_check        exit 0 if every case is identical
// Build: nvcc [the engine's flags] decode_attention_check.cu
#include <lfm/gemm.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using u16 = unsigned short;
using lfm::gemm::Problem;
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
unsigned long long state = 0x9E3779B97F4A7C15ull;
unsigned next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return unsigned(state >> 33); }
u16 random_bf16() { const unsigned r = next(); return u16((r >> 8 & 1) << 15 | (127 - 6 + (r >> 9) % 8) << 7 | (r >> 16 & 0x7f)); }
template <class T> T* upload(const std::vector<T>& h) { T* d; check(cudaMalloc(&d, h.size() * sizeof(T))); check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> std::vector<T> download(const T* d, size_t n) { std::vector<T> h(n); check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

template <int R, int G> void scores(const Problem& p) {
  const int blocks = std::min((p.n + 31) / 32, 1024);
  lfm::gemm::detail::thin_k128_kernel<R, G><<<dim3(blocks, p.batch / G), 256>>>(p);
}
template <int R, int G> void values(const Problem& p) {
  using namespace lfm::gemm; using namespace lfm::gemm::detail;
  const int splits = (p.k + kSplitKeys - 1) / kSplitKeys, threads = 64, column_blocks = (p.n / 2 + threads - 1) / threads;
  split_kn_partial<R, G><<<dim3(splits, p.batch / G, column_blocks), threads>>>(p, splits);
  const size_t outputs = size_t(R) * p.n * p.batch;
  split_kn_finish<R><<<unsigned((outputs + 255) / 256), 256>>>(p, splits);
}

int cases = 0, failures = 0;
template <int R> void one(int keys, int heads, int group) {
  const int head_dim = 128, kv = heads / group, context = keys + 16, pitch = (keys + 7) / 8 * 8;
  std::vector<u16> q(size_t(R) * heads * head_dim), cache(size_t(kv) * context * head_dim), probs(size_t(heads) * R * pitch);
  for (auto& v : q) v = random_bf16();
  for (auto& v : cache) v = random_bf16();
  for (auto& v : probs) v = random_bf16();
  auto *dq = upload(q), *dk = upload(cache), *dp = upload(probs);
  float* ds; check(cudaMalloc(&ds, size_t(heads) * R * pitch * 4));
  lfm::gemm::bf16* dout; check(cudaMalloc(&dout, size_t(R) * heads * head_dim * 2));
  const size_t ws = lfm::gemm::workspace_bytes(R, head_dim, pitch, heads);
  float* dw; check(cudaMalloc(&dw, ws));
  Problem s;  // scores, as lfm3_main.cu head_scores
  using lfm::gemm::bf16;
  s.a = reinterpret_cast<const bf16*>(dq); s.b = reinterpret_cast<const bf16*>(dk); s.c_f32 = ds; s.m = R; s.n = keys; s.k = head_dim; s.lda = heads * head_dim; s.ldb = head_dim; s.ldc = pitch;
  s.stride_a = head_dim; s.stride_b = (long long)context * head_dim; s.stride_c = (long long)R * pitch; s.batch = heads; s.group = group;
  Problem v;  // values, as head_values
  v.a = reinterpret_cast<const bf16*>(dp); v.b = reinterpret_cast<const bf16*>(dk); v.c = dout; v.m = R; v.n = head_dim; v.k = pitch; v.b_rows = keys; v.b_k_contiguous = false;
  v.lda = pitch; v.ldb = head_dim; v.ldc = heads * head_dim; v.stride_a = (long long)R * pitch; v.stride_b = (long long)context * head_dim;
  v.stride_c = head_dim; v.batch = heads; v.group = group; v.workspace = dw; v.workspace_size = ws;
  std::vector<float> s1, s2; std::vector<lfm::gemm::bf16> v1, v2;
  for (int g = 1; g <= 2; ++g) {
    check(cudaMemset(ds, 0x5a, size_t(heads) * R * pitch * 4)); check(cudaMemset(dout, 0x5a, size_t(R) * heads * head_dim * 2));
    if (g == 1) { scores<R, 1>(s); values<R, 1>(v); } else { scores<R, 2>(s); values<R, 2>(v); }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    (g == 1 ? s1 : s2) = download(ds, size_t(heads) * R * pitch);
    (g == 1 ? v1 : v2) = download(dout, size_t(R) * heads * head_dim);
  }
  const bool same_s = std::memcmp(s1.data(), s2.data(), s1.size() * 4) == 0, same_v = std::memcmp(v1.data(), v2.data(), v1.size() * 2) == 0;
  cases += 2; failures += !same_s + !same_v;
  std::printf("%s rows %d keys %5d heads %d group %d: scores %s, values %s\n", same_s && same_v ? "ok  " : "FAIL", R, keys, heads, group,
              same_s ? "identical" : "DIFFER", same_v ? "identical" : "DIFFER");
  check(cudaFree(dq)); check(cudaFree(dk)); check(cudaFree(dp)); check(cudaFree(ds)); check(cudaFree(dout)); check(cudaFree(dw));
}
}  // namespace

int main() {
  for (int keys : {1, 5, 8, 255, 256, 257, 1000, 4097, 31000}) {
    one<1>(keys, 16, 2);
    one<2>(keys, 16, 2);
    one<1>(keys, 16, 4);
  }
  std::printf("decode_attention_check: %d cases, %d failed: %s\n", cases, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
