// validation/round-once-v1: runs one prefill GEMM through lfm::gemm::run with the round-once kernel on every row
// (LFM_ONCE_ROWS set by the caller) and writes the output, for tools/pascal/round_once_check.py to compare with
// exact integer arithmetic. Build with -DLFM_GEMM_ROUND_ONCE.
//   round_once_check X.bf16 ROWS K W.bf16 N kc|kn bf16|f32 OUT
// kc: W is N x K (K contiguous, a projection). kn: W is K x N (the layout of attention's P.V).
#include <lfm/gemm.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
std::vector<unsigned short> read(const char* path, size_t n) {
  std::vector<unsigned short> v(n);
  FILE* f = std::fopen(path, "rb");
  if (!f || std::fread(v.data(), 2, n, f) != n) { std::fprintf(stderr, "cannot read %zu values from %s\n", n, path); std::exit(1); }
  std::fclose(f);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) { std::fprintf(stderr, "usage: round_once_check X.bf16 ROWS K W.bf16 N kc|kn bf16|f32 OUT\n"); return 2; }
  const int rows = std::atoi(argv[2]), k = std::atoi(argv[3]), n = std::atoi(argv[5]);
  const bool kc = !std::strcmp(argv[6], "kc"), f32 = !std::strcmp(argv[7], "f32");
  const auto x = read(argv[1], size_t(rows) * k), w = read(argv[4], size_t(n) * k);
  lfm::gemm::bf16 *dx, *dw; void* dy;
  check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&dw, w.size() * 2)); check(cudaMalloc(&dy, size_t(rows) * n * 4));
  check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  lfm::gemm::Problem p;
  p.a = dx; p.b = dw; p.m = rows; p.n = n; p.k = k; p.lda = k; p.ldc = n; p.wide = true;
  // Rows as prompt positions from 0 when the round-once kernel is on (it goes by position); with LFM_ONCE_ROWS=0,
  // not a prompt at all, so that the output is the plain kernel's in every row.
  const char* once = std::getenv("LFM_ONCE_ROWS");
  p.row0 = once && std::atoi(once) > 0 ? 0 : -1;
  if (kc) p.ldb = k; else { p.ldb = n; p.b_k_contiguous = false; p.b_rows = k; }
  if (f32) p.c_f32 = static_cast<float*>(dy); else p.c = static_cast<lfm::gemm::bf16*>(dy);
  lfm::gemm::run(p); check(cudaDeviceSynchronize());
  const size_t bytes = size_t(rows) * n * (f32 ? 4 : 2);
  std::vector<char> out(bytes);
  check(cudaMemcpy(out.data(), dy, bytes, cudaMemcpyDeviceToHost));
  FILE* f = std::fopen(argv[8], "wb");
  if (!f || std::fwrite(out.data(), 1, bytes, f) != bytes) { std::fprintf(stderr, "cannot write %s\n", argv[8]); return 1; }
  std::fclose(f);
  return 0;
}
