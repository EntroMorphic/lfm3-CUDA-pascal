// The phase 2 prefill kernel (wide2_kernel) against the one it replaces (wide_kernel), launched directly on the card:
// every output bit for bit, over the engine's shapes and edge cases (rows, columns and K not multiples of the tiles,
// K mod 32 = 8, 16, 24, a K x N B with fewer rows than K, several heads per call, BF16 and FP32 outputs, strided C,
// NaN and infinity among the operands). The output buffer is filled with a pattern before each run, so outputs
// either kernel leaves unwritten must also agree.
//   wide2_check        exit 0 if every case is identical
// Build: nvcc [the engine's flags] wide2_check.cu
#include <lfm/gemm.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using u16 = unsigned short;
using lfm::gemm::Problem;
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
unsigned long long state = 0x2545F4914F6CDD1Dull;
unsigned next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return unsigned(state >> 33); }
u16 random_bf16() {
  const unsigned r = next();
  if (r % 61 == 0) return u16((r >> 8 & 1) << 15);
  return u16((r >> 8 & 1) << 15 | (127 - 9 + (r >> 9) % 12) << 7 | (r >> 16 & 0x7f));
}

template <bool KC>
void launch_old(const Problem& p) {
  using namespace lfm::gemm::detail;
  for_wide_tile<LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN, 4>(p.m, [&](auto ty) {
    constexpr int TY = decltype(ty)::value;
    using T = WideTile<TY, LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN>;
    const dim3 grid((p.n + T::BN - 1) / T::BN, (p.m + T::BM - 1) / T::BM, p.batch);
    wide_kernel<TY, LFM_GEMM_WIDE_RM, LFM_GEMM_WIDE_RN, KC><<<grid, T::THREADS>>>(p);
  });
}
template <bool KC>
void launch_new(const Problem& p) {
  using namespace lfm::gemm::detail;
  const dim3 grid((p.n + kW2N - 1) / kW2N, (p.m + kW2M - 1) / kW2M, p.batch);
  wide2_kernel<KC><<<grid, kW2Threads>>>(p);
}

std::vector<unsigned char> run(Problem p, size_t a_elems, size_t b_elems, size_t c_elems, bool f32,
                               const std::vector<u16>& a, const std::vector<u16>& b, bool use_new) {
  lfm::gemm::bf16 *da, *db; void* dc;
  const size_t bytes = c_elems * (f32 ? 4 : 2);
  check(cudaMalloc(&da, a_elems * 2)); check(cudaMalloc(&db, b_elems * 2)); check(cudaMalloc(&dc, bytes));
  check(cudaMemcpy(da, a.data(), a_elems * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), b_elems * 2, cudaMemcpyHostToDevice));
  check(cudaMemset(dc, 0x5a, bytes));
  p.a = da; p.b = db;
  if (f32) p.c_f32 = static_cast<float*>(dc); else p.c = static_cast<lfm::gemm::bf16*>(dc);
  lfm::gemm::check(p);
  if (use_new) { if (p.b_k_contiguous) launch_new<true>(p); else launch_new<false>(p); }
  else { if (p.b_k_contiguous) launch_old<true>(p); else launch_old<false>(p); }
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  std::vector<unsigned char> out(bytes);
  check(cudaMemcpy(out.data(), dc, bytes, cudaMemcpyDeviceToHost));
  check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
  return out;
}

int cases = 0, failures = 0;
// A plain product m x n x k; B N x K (kc) or K x N with b_rows rows; optional NaN/inf poison.
void plain(int m, int n, int k, bool kc, bool f32, int b_rows = -1, int poison = 0) {
  Problem p;
  p.m = m; p.n = n; p.k = k; p.lda = k; p.ldc = n + (f32 ? 0 : 8); p.wide = true;
  if (kc) p.ldb = k; else { p.ldb = n; p.b_k_contiguous = false; p.b_rows = b_rows < 0 ? k : b_rows; }
  std::vector<u16> a(size_t(m) * k), b(size_t(n) * k);
  for (auto& v : a) v = random_bf16();
  for (auto& v : b) v = random_bf16();
  if (poison & 1) a[size_t(m / 2) * k + k / 3] = 0x7fc0;           // NaN in A
  if (poison & 2) b[size_t(n / 3) * (kc ? k : 1) + 5] = 0x7f80;    // infinity in B
  if (poison & 4) a[3] = 0xff80;                                   // -infinity in row 0 of A
  const size_t c_elems = size_t(m) * p.ldc;
  const auto x = run(p, a.size(), b.size(), c_elems, f32, a, b, false), y = run(p, a.size(), b.size(), c_elems, f32, a, b, true);
  ++cases;
  size_t differ = 0;
  for (size_t i = 0; i < x.size(); ++i) differ += x[i] != y[i];
  const std::string name = std::to_string(m) + " x " + std::to_string(n) + " x " + std::to_string(k) + (kc ? " kc" : " kn") +
                           (f32 ? " f32" : " bf16") + (b_rows >= 0 ? " b_rows " + std::to_string(b_rows) : "") +
                           (poison ? " poison " + std::to_string(poison) : "");
  if (differ) { ++failures; std::printf("FAIL %s: %zu bytes differ\n", name.c_str(), differ); }
  else std::printf("ok   %s\n", name.c_str());
}
// The runner's attention products, several heads per call (lfm3_main.cu head_scores / head_values).
void attention(int rows, int end, int pitch, int heads, int group) {
  const int head_dim = 128, hidden = heads * head_dim, context = end + 64;
  std::vector<u16> keys(size_t(heads / group) * context * head_dim);
  for (auto& v : keys) v = random_bf16();
  {
    std::vector<u16> q(size_t(rows) * hidden);
    for (auto& v : q) v = random_bf16();
    Problem p;
    p.m = rows; p.n = end; p.k = head_dim; p.lda = hidden; p.ldb = head_dim; p.ldc = pitch;
    p.stride_a = head_dim; p.stride_b = (long long)context * head_dim; p.stride_c = (long long)rows * pitch;
    p.batch = heads; p.group = group; p.wide = true;
    const size_t c = size_t(heads) * rows * pitch;
    const auto x = run(p, q.size(), keys.size(), c, true, q, keys, false), y = run(p, q.size(), keys.size(), c, true, q, keys, true);
    ++cases;
    const bool same = x == y;
    failures += !same;
    std::printf("%s attention scores rows %d end %d heads %d\n", same ? "ok  " : "FAIL", rows, end, heads);
  }
  {
    std::vector<u16> probs(size_t(heads) * rows * pitch);
    for (auto& v : probs) v = random_bf16();
    Problem p;
    p.m = rows; p.n = head_dim; p.k = pitch; p.b_rows = end; p.b_k_contiguous = false;
    p.lda = pitch; p.ldb = head_dim; p.ldc = hidden;
    p.stride_a = (long long)rows * pitch; p.stride_b = (long long)context * head_dim; p.stride_c = head_dim;
    p.batch = heads; p.group = group; p.wide = true;
    const size_t c = size_t(rows) * hidden;
    for (int f32 = 0; f32 <= 1; ++f32) {
      const auto x = run(p, probs.size(), keys.size(), c, f32, probs, keys, false), y = run(p, probs.size(), keys.size(), c, f32, probs, keys, true);
      ++cases;
      const bool same = x == y;
      failures += !same;
      std::printf("%s attention P x V %s rows %d end %d pitch %d heads %d\n", same ? "ok  " : "FAIL", f32 ? "f32" : "bf16", rows, end, pitch, heads);
    }
  }
}
}  // namespace

int main() {
  // The engine's prefill shapes (rows x columns x K) at the bar's lengths and their edges.
  for (int m : {65, 128, 129, 256, 257, 300})
    for (int f32 = 0; f32 <= 1; ++f32) {
      plain(m, 6144, 2048, true, f32);
      plain(m, 2048, 7680, true, f32);
    }
  plain(4096, 7680, 2048, true, false);
  plain(4096, 2048, 2048, true, true);
  plain(1000, 128000 / 16, 2048, true, true);
  // K, N and M at every remainder the tiles can leave.
  for (int k : {8, 16, 24, 32, 40, 48, 56, 136, 2040, 2056})
    for (int kc = 0; kc <= 1; ++kc) {
      plain(97, 200, k, kc, true);
      plain(131, 136, k, kc, false);
    }
  for (int n : {8, 120, 128, 136, 264}) plain(70, n, 64, false, true, 40);
  plain(200, 256, 104, false, true, 100);
  plain(200, 256, 104, false, false, 1);
  for (int poison : {1, 2, 4, 7}) { plain(150, 264, 2048, true, true, -1, poison); plain(150, 264, 2048, true, false, -1, poison); }
  // Attention: the 128-row tiles above 4,096 of capacity, and whole short prompts.
  attention(128, 4100, 4104, 4, 2);
  attention(129, 129, 136, 16, 2);
  attention(257, 257, 264, 16, 2);
  attention(65, 300, 304, 8, 2);
  std::printf("wide2_check: %d cases, %d failed: %s\n", cases, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
