// Host side of first_row_check.cu: one output of row 0 by the tensor-core model (lfm/tensor_core_model.hpp's
// mma_step_reference chained over k ascending in steps of 16, zero-padded), as BF16 bits rounded to nearest even or
// as FP32 bits. Compiled by the host compiler: inside a .cu file the model's functions are device functions.
#include <lfm/tensor_core_model.hpp>
#include <cstddef>

unsigned first_row_model(const unsigned short* a, const unsigned short* b, int k, int n, int column, bool b_k_contiguous, bool f32) {
  auto value = [](unsigned short u) { return lfm::tcm::float_of(unsigned(u) << 16); };
  float acc = 0.f;
  for (int k0 = 0; k0 < k; k0 += 16) {
    float x[16], w[16]; short xe[16], we[16];
    for (int i = 0; i < 16; ++i) {
      const int kk = k0 + i;
      const unsigned short xu = kk < k ? a[kk] : 0;
      const unsigned short wu = kk < k ? (b_k_contiguous ? b[size_t(column) * k + kk] : b[size_t(kk) * n + column]) : 0;
      x[i] = value(xu); w[i] = value(wu);
      xe[i] = lfm::tcm::exponent_of_bf16_bits(xu); we[i] = lfm::tcm::exponent_of_bf16_bits(wu);
    }
    acc = lfm::tcm::mma_step_reference(acc, x, xe, w, we, 1, 1);
  }
  const unsigned u = lfm::tcm::bits_of(acc);
  return f32 ? u : (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
