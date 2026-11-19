// The engine's rule for prompt position 0 (PLAN decision 5; lfm/gemm.cuh), checked on the card through
// lfm::gemm::run with the shipping switches. For each shape, layout (B stored N x K or K x N) and output format:
//   1. row0 = 0: row 0 equals the tensor-core model computed here on the host (tcm::mma_step_reference chained over
//      k ascending), bit for bit; every other row equals the plain kernel's.
//   2. row0 = -1 (not a prompt) and row0 = 5 (a later block of a prompt): the plain kernel's result in every row.
//   3. A NaN or an infinity among row 0's operands: the plain kernel's non-finite results are kept, and every finite
//      output of row 0 still equals the model. Non-finite operands elsewhere do not touch row 0.
//   4. The two products of attention, several heads per call (batch, group, a padded K x N B): rule 1 for every head.
// The output buffer is filled with NaN before every run, so a kernel that read stale output would fail 1.
//   first_row_check        exit 0 if every check holds
// Build: nvcc [the engine's flags] first_row_check.cu first_row_model.cpp (the host half of the comparison).
#include <lfm/gemm.cuh>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(LFM_GEMM_TENSOR_CORE_MODEL) || defined(LFM_GEMM_ROUND_ONCE) || !LFM_GEMM_FIRST_ROW || LFM_GEMM_PLAIN_ROUND
#error "first_row_check tests the shipping switches: build it without LFM_GEMM_* definitions"
#endif

// first_row_model.cpp
unsigned first_row_model(const unsigned short* a, const unsigned short* b, int k, int n, int column, bool b_k_contiguous, bool f32);

namespace {
using u16 = unsigned short;
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
unsigned long long state = 0x9E3779B97F4A7C15ull;
unsigned next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return unsigned(state >> 33); }
u16 random_bf16() {  // model-like magnitudes, with some zeros
  const unsigned r = next();
  if (r % 61 == 0) return u16((r >> 8 & 1) << 15);
  return u16((r >> 8 & 1) << 15 | (127 - 9 + (r >> 9) % 12) << 7 | (r >> 16 & 0x7f));
}
const u16 kNaN = 0x7fc0, kInf = 0x7f80;

struct Shape { int m, k, n; };
struct Run { std::vector<unsigned> bits; };  // one value per output: BF16 bits or FP32 bits

// One product on the card; `p` carries the geometry, `outputs` is the size of C in elements.
Run run_problem(lfm::gemm::Problem p, size_t outputs, bool f32, const std::vector<u16>& a, const std::vector<u16>& b, int row0) {
  lfm::gemm::bf16 *da, *db; void* dc;
  const size_t bytes = outputs * (f32 ? 4 : 2);
  check(cudaMalloc(&da, a.size() * 2)); check(cudaMalloc(&db, b.size() * 2)); check(cudaMalloc(&dc, bytes));
  check(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
  std::vector<unsigned char> fill(bytes);
  for (size_t i = 0; i < outputs; ++i) {
    if (f32) { const unsigned q = 0x7fc00000u; std::memcpy(&fill[i * 4], &q, 4); } else std::memcpy(&fill[i * 2], &kNaN, 2);
  }
  check(cudaMemcpy(dc, fill.data(), bytes, cudaMemcpyHostToDevice));
  p.a = da; p.b = db; p.wide = true; p.row0 = row0;
  if (f32) p.c_f32 = static_cast<float*>(dc); else p.c = static_cast<lfm::gemm::bf16*>(dc);
  lfm::gemm::run(p); check(cudaDeviceSynchronize());
  std::vector<unsigned char> raw(bytes);
  check(cudaMemcpy(raw.data(), dc, bytes, cudaMemcpyDeviceToHost));
  Run r; r.bits.resize(outputs);
  for (size_t i = 0; i < outputs; ++i) {
    if (f32) std::memcpy(&r.bits[i], &raw[i * 4], 4); else { u16 v; std::memcpy(&v, &raw[i * 2], 2); r.bits[i] = v; }
  }
  check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
  return r;
}
Run run(const Shape& s, bool kc, bool f32, const std::vector<u16>& a, const std::vector<u16>& b, int row0) {
  lfm::gemm::Problem p;
  p.m = s.m; p.n = s.n; p.k = s.k; p.lda = s.k; p.ldc = s.n;
  if (kc) p.ldb = s.k; else { p.ldb = s.n; p.b_k_contiguous = false; p.b_rows = s.k; }
  return run_problem(p, size_t(s.m) * s.n, f32, a, b, row0);
}

// Row 0, column c, by the tensor-core model on the host.
unsigned model(const Shape& s, bool kc, bool f32, const std::vector<u16>& a, const std::vector<u16>& b, int c) {
  return first_row_model(a.data(), b.data(), s.k, s.n, c, kc, f32);
}
bool finite_bits(unsigned bits, bool f32) { return f32 ? (bits >> 23 & 0xff) != 0xff : (bits >> 7 & 0xff) != 0xff; }

int failures = 0, checks = 0;
void expect(bool ok, const std::string& what) {
  ++checks;
  if (!ok) { ++failures; std::printf("  FAIL: %s\n", what.c_str()); }
}

// Rule 4, with the geometry of the runner's head_scores and head_values: 4 query heads over 2 key/value heads.
void attention_products() {
  const int heads = 4, group = 2, head_dim = 128, hidden = heads * head_dim, rows = 16, context = 64, end = 36, pitch = 40;
  std::vector<u16> keys(size_t(heads / group) * context * head_dim);
  for (auto& v : keys) v = random_bf16();
  {  // scores: FP32, one output row per query row; C keeps `pitch` columns per row, of which `end` are written
    std::vector<u16> queries(size_t(rows) * hidden);
    for (auto& v : queries) v = random_bf16();
    lfm::gemm::Problem p;
    p.m = rows; p.n = end; p.k = head_dim; p.lda = hidden; p.ldb = head_dim; p.ldc = pitch;
    p.stride_a = head_dim; p.stride_b = (long long)context * head_dim; p.stride_c = (long long)rows * pitch;
    p.batch = heads; p.group = group;
    const size_t outputs = size_t(heads) * rows * pitch;
    const Run plain = run_problem(p, outputs, true, queries, keys, -1), first = run_problem(p, outputs, true, queries, keys, 0);
    for (int z = 0; z < heads; ++z) {
      size_t model_wrong = 0, rest_wrong = 0, differs = 0;
      const size_t base = size_t(z) * rows * pitch;
      for (int c = 0; c < end; ++c) {
        model_wrong += first.bits[base + c] != first_row_model(&queries[size_t(z) * head_dim], &keys[size_t(z / group) * context * head_dim], head_dim, end, c, true, true);
        differs += first.bits[base + c] != plain.bits[base + c];
        for (int r = 1; r < rows; ++r) rest_wrong += first.bits[base + size_t(r) * pitch + c] != plain.bits[base + size_t(r) * pitch + c];
      }
      const std::string name = "attention scores, head " + std::to_string(z);
      expect(model_wrong == 0, name + ": row 0 differs from the tensor-core model in " + std::to_string(model_wrong) + " outputs");
      expect(rest_wrong == 0, name + ": rows after 0 differ from the plain kernel in " + std::to_string(rest_wrong) + " outputs");
      expect(differs > 0, name + ": row 0 never differs from the plain kernel (did the model run?)");
      std::printf("%-28s row 0 = model in %d/%d, differs from plain in %zu; other rows = plain\n", name.c_str(), int(end - model_wrong), end, differs);
    }
  }
  // P x V: the probability rows are `pitch` wide, the values exist for the first `end` of them. The engine stores BF16,
  // which seldom shows the difference between the two arithmetics in so short a sum; FP32 shows it.
  for (int f32 = 0; f32 <= 1; ++f32) {
    std::vector<u16> probabilities(size_t(heads) * rows * pitch);
    for (auto& v : probabilities) v = random_bf16();
    lfm::gemm::Problem p;
    p.m = rows; p.n = head_dim; p.k = pitch; p.b_rows = end; p.b_k_contiguous = false;
    p.lda = pitch; p.ldb = head_dim; p.ldc = hidden;
    p.stride_a = (long long)rows * pitch; p.stride_b = (long long)context * head_dim; p.stride_c = head_dim;
    p.batch = heads; p.group = group;
    const size_t outputs = size_t(rows) * hidden;
    const Run plain = run_problem(p, outputs, f32, probabilities, keys, -1), first = run_problem(p, outputs, f32, probabilities, keys, 0);
    for (int z = 0; z < heads; ++z) {
      std::vector<u16> padded(size_t(pitch) * head_dim, 0);  // the K x N B the kernel must read: zero from row `end`
      std::memcpy(padded.data(), &keys[size_t(z / group) * context * head_dim], size_t(end) * head_dim * 2);
      size_t model_wrong = 0, rest_wrong = 0, differs = 0;
      for (int c = 0; c < head_dim; ++c) {
        const size_t out = size_t(z) * head_dim + c;
        model_wrong += first.bits[out] != first_row_model(&probabilities[size_t(z) * rows * pitch], padded.data(), pitch, head_dim, c, false, f32);
        differs += first.bits[out] != plain.bits[out];
        for (int r = 1; r < rows; ++r) rest_wrong += first.bits[size_t(r) * hidden + out] != plain.bits[size_t(r) * hidden + out];
      }
      const std::string name = std::string("attention P x V ") + (f32 ? "f32" : "bf16") + ", head " + std::to_string(z);
      expect(model_wrong == 0, name + ": row 0 differs from the tensor-core model in " + std::to_string(model_wrong) + " outputs");
      expect(rest_wrong == 0, name + ": rows after 0 differ from the plain kernel in " + std::to_string(rest_wrong) + " outputs");
      if (f32) expect(differs > 0, name + ": row 0 never differs from the plain kernel (did the model run?)");
      std::printf("%-28s row 0 = model in %d/%d, differs from plain in %zu; other rows = plain\n", name.c_str(), int(head_dim - model_wrong), head_dim, differs);
    }
  }
}
}  // namespace

int main() {
  // K and N are multiples of 8, as lfm::gemm requires; 2,040 and 24 are not multiples of the model's step of 16.
  const Shape shapes[] = {{16, 2048, 96}, {9, 7680, 40}, {1, 2048, 32}, {8, 2048, 64}, {129, 128, 200}, {33, 2040, 16}, {40, 24, 8}};
  for (const Shape& s : shapes)
    for (int kc = 1; kc >= 0; --kc)
      for (int f32 = 0; f32 <= 1; ++f32) {
        const std::string name = std::to_string(s.m) + " x " + std::to_string(s.k) + " x " + std::to_string(s.n) +
                                 (kc ? " kc " : " kn ") + (f32 ? "f32" : "bf16");
        std::vector<u16> a(size_t(s.m) * s.k), b(size_t(s.n) * s.k);
        for (auto& v : a) v = random_bf16();
        for (auto& v : b) v = random_bf16();
        const Run plain = run(s, kc, f32, a, b, -1), first = run(s, kc, f32, a, b, 0), later = run(s, kc, f32, a, b, 5);
        size_t model_wrong = 0, rest_wrong = 0, differs = 0;
        for (int c = 0; c < s.n; ++c) {
          model_wrong += first.bits[c] != model(s, kc, f32, a, b, c);
          differs += first.bits[c] != plain.bits[c];
        }
        for (size_t i = s.n; i < plain.bits.size(); ++i) rest_wrong += first.bits[i] != plain.bits[i];
        expect(model_wrong == 0, name + ": row 0 differs from the tensor-core model in " + std::to_string(model_wrong) + " outputs");
        expect(rest_wrong == 0, name + ": rows after 0 differ from the plain kernel in " + std::to_string(rest_wrong) + " outputs");
        expect(later.bits == plain.bits, name + ": row0 = 5 differs from the plain kernel");
        bool all_finite = true;
        for (unsigned v : plain.bits) all_finite &= finite_bits(v, f32);
        expect(all_finite, name + ": the plain kernel left a non-finite output on finite operands");
        std::printf("%-28s row 0 = model in %d/%d, differs from plain in %zu; other rows = plain\n", name.c_str(),
                    int(s.n - model_wrong), s.n, differs);
        if (s.m == 16 && f32) expect(differs > 0, name + ": row 0 never differs from the plain kernel (did the model run?)");
        if (s.m != 16) continue;
        // Non-finite operands. What each run must give in row 0: the plain result where that is not finite, the
        // model elsewhere; and the plain result in every other row.
        struct Poison { const char* what; int row, k, column; u16 bits; };  // column >= 0: in B instead of A
        const Poison poisons[] = {{"NaN in row 0 of A", 0, 5, -1, kNaN}, {"infinity in row 0 of A", 0, 5, -1, kInf},
                                  {"NaN in column 3 of B", 0, 7, 3, kNaN}, {"NaN in row 1 of A", 1, 5, -1, kNaN}};
        for (const Poison& q : poisons) {
          auto a2 = a; auto b2 = b;
          if (q.column >= 0) (kc ? b2[size_t(q.column) * s.k + q.k] : b2[size_t(q.k) * s.n + q.column]) = q.bits;
          else a2[size_t(q.row) * s.k + q.k] = q.bits;
          const Run p2 = run(s, kc, f32, a2, b2, -1), f2 = run(s, kc, f32, a2, b2, 0);
          size_t kept_wrong = 0, finite_wrong = 0, others = 0, nonfinite = 0;
          for (int c = 0; c < s.n; ++c) {
            if (!finite_bits(p2.bits[c], f32)) { ++nonfinite; kept_wrong += f2.bits[c] != p2.bits[c]; }
            else finite_wrong += f2.bits[c] != model(s, kc, f32, a2, b2, c);
          }
          for (size_t i = s.n; i < p2.bits.size(); ++i) others += f2.bits[i] != p2.bits[i];
          const int want = q.column >= 0 ? 1 : (q.row == 0 ? s.n : 0);
          expect(int(nonfinite) == want, name + ", " + q.what + ": " + std::to_string(nonfinite) + " non-finite in the plain row 0, expected " + std::to_string(want));
          expect(kept_wrong == 0, name + ", " + q.what + ": " + std::to_string(kept_wrong) + " non-finite outputs of row 0 not kept");
          expect(finite_wrong == 0, name + ", " + q.what + ": " + std::to_string(finite_wrong) + " finite outputs of row 0 differ from the model");
          expect(others == 0, name + ", " + q.what + ": other rows differ from the plain kernel");
        }
      }
  attention_products();
  std::printf("first_row_check: %d checks, %d failed: %s\n", checks, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
