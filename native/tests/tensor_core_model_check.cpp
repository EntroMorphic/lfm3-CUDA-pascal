// Host check of lfm/tensor_core_model.hpp: a GEMM (rows x K BF16 times columns x K BF16, K contiguous in both) with
// the model's steps chained over k ascending, the accumulator rounded to BF16 nearest even, as the Thor prefill
// kernel does. Driven by tools/pascal/tensor_core/host_kernel_check.py, which compares the output with a Thor pin.
//   tensor_core_model_check fast|reference X.bf16 ROWS W.bf16 COLUMNS K OUT.bf16
#include <lfm/tensor_core_model.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<unsigned short> read(const char* path, size_t n) {
  std::vector<unsigned short> v(n);
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), n * 2);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
  return v;
}
static float value(unsigned short u) { return lfm::tcm::float_of(unsigned(u) << 16); }
static unsigned short to_bf16(float x) {  // nearest even
  const unsigned u = lfm::tcm::bits_of(x);
  return static_cast<unsigned short>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main(int argc, char** argv) {
  if (argc != 8) { std::fprintf(stderr, "usage: tensor_core_model_check fast|reference X ROWS W COLUMNS K OUT\n"); return 2; }
  const bool fast = std::string(argv[1]) == "fast";
  const size_t rows = std::atol(argv[3]), columns = std::atol(argv[5]), k = std::atol(argv[6]);
  if (k % 16) { std::fprintf(stderr, "K must be a multiple of 16\n"); return 2; }
  const auto xu = read(argv[2], rows * k), wu = read(argv[4], columns * k);
  std::vector<float> x(xu.size()), w(wu.size());
  std::vector<short> xe(xu.size()), we(wu.size());
  for (size_t i = 0; i < xu.size(); ++i) { x[i] = value(xu[i]); xe[i] = lfm::tcm::exponent_of_bf16_bits(xu[i]); }
  for (size_t i = 0; i < wu.size(); ++i) { w[i] = value(wu[i]); we[i] = lfm::tcm::exponent_of_bf16_bits(wu[i]); }
  std::vector<unsigned short> out(rows * columns);
  #pragma omp parallel for schedule(static)
  for (long long r = 0; r < static_cast<long long>(rows); ++r)
    for (size_t c = 0; c < columns; ++c) {
      float acc = 0.f;
      for (size_t k0 = 0; k0 < k; k0 += 16)
        acc = fast ? lfm::tcm::mma_step(acc, &x[r * k + k0], &xe[r * k + k0], &w[c * k + k0], &we[c * k + k0], 1, 1)
                   : lfm::tcm::mma_step_reference(acc, &x[r * k + k0], &xe[r * k + k0], &w[c * k + k0], &we[c * k + k0], 1, 1);
      out[r * columns + c] = to_bf16(acc);
    }
  std::ofstream f(argv[7], std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), out.size() * 2);
  return f ? 0 : 1;
}
