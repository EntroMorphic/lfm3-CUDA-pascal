#pragma once
// The arithmetic of Blackwell's mma.sync.aligned.m16n8k16 (BF16 in, FP32 accumulate), for the validation build
// (-DLFM_GEMM_TENSOR_CORE_MODEL) and for a host check of the same code (validation/tensor-core-model-v1).
//
// The published model (MMA-Sim's T-FDPA; L = 16, F = 25, round toward zero), one step of 16 k:
//   the accumulator and the 16 exact products are aligned to the largest exponent among them (a product's exponent
//   is the sum of its operands' exponents), each truncated toward zero to a multiple of q = 2^(emax - 25), summed
//   exactly, and the sum truncated toward zero to FP32.
//
// Operands arrive as FP32 values holding BF16 numbers, with each operand's exponent beside it (kNoExponent for
// zero, so that a product with a zero operand never sets the window). Two implementations of the same step:
//   mma_step_reference   integers throughout: the plain statement of the model
//   mma_step             floats where they are exact: a product of two BF16 values is exact in FP32, scaling by
//                        1/q is exact, and float -> integer conversion toward zero is the model's truncation
// Not modelled: infinities and NaN.
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define LFM_TCM_FN __device__ __forceinline__
#else
#include <cfenv>
#include <cmath>
#define LFM_TCM_FN inline
#endif

namespace lfm::tcm {

inline constexpr int kNoExponent = -4096;

LFM_TCM_FN unsigned bits_of(float x) {
#ifdef __CUDACC__
  return __float_as_uint(x);
#else
  unsigned u; std::memcpy(&u, &x, 4); return u;
#endif
}
LFM_TCM_FN float float_of(unsigned u) {
#ifdef __CUDACC__
  return __uint_as_float(u);
#else
  float x; std::memcpy(&x, &u, 4); return x;
#endif
}
LFM_TCM_FN int trunc_to_int(float x) {  // toward zero
#ifdef __CUDACC__
  return __float2int_rz(x);
#else
  return static_cast<int>(x);
#endif
}
LFM_TCM_FN float trunc_to_float(long long n) {  // toward zero
#ifdef __CUDACC__
  return __ll2float_rz(n);
#else
  const float r = static_cast<float>(n);  // nearest
  if (static_cast<long double>(r) > 0 ? static_cast<long double>(r) > static_cast<long double>(n)
                                      : static_cast<long double>(r) < static_cast<long double>(n))
    return std::nextafterf(r, 0.0f);
  return r;
#endif
}
// x * 2^n, exact while the result is a normal number.
LFM_TCM_FN float scale_pow2(float x, int n) {
  while (n > 127) { x *= float_of(254u << 23); n -= 127; }
  while (n < -126) { x *= float_of(1u << 23); n += 126; }
  return x * float_of(unsigned(n + 127) << 23);
}
// The exponent of a BF16 value given as its 16 bits: floor(log2 |v|) for a normal number, -126 for a subnormal.
LFM_TCM_FN short exponent_of_bf16_bits(unsigned u) {
  const int e = (u >> 7) & 0xff, m = u & 0x7f;
  return short((e | m) ? (e ? e - 127 : -126) : kNoExponent);
}
LFM_TCM_FN int exponent_of_float(float c) {
  const unsigned u = bits_of(c);
  const int e = (u >> 23) & 0xff, m = u & 0x7fffff;
  return (e | m) ? (e ? e - 127 : -126) : kNoExponent;
}

// a[i * sa], b[i * sb]: the operands of product i; ae, be: their exponents.
LFM_TCM_FN float mma_step_reference(float c, const float* a, const short* ae, const float* b, const short* be, int sa, int sb) {
  const unsigned cu = bits_of(c);
  const int cb = (cu >> 23) & 0xff, cm = cu & 0x7fffff;
  const int cmag = cb ? (cm | 0x800000) : cm;  // |c| = cmag * 2^(cexp - 23)
  const int cexp = exponent_of_float(c);
  int emax = cexp;
  for (int i = 0; i < 16; ++i) {
    const int e = int(ae[i * sa]) + int(be[i * sb]);
    if (e > emax) emax = e;
  }
  if (emax < kNoExponent / 2) return 0.f;
  long long sum = 0;  // in units of 2^(emax - 25)
  if (cmag) {
    const int d = emax - cexp;  // |c| in units: cmag * 2^(2 - d)
    const long long mag = d <= 2 ? (long long)cmag << (2 - d) : (d - 2 < 31 ? (long long)(cmag >> (d - 2)) : 0);
    sum = (cu >> 31) ? -mag : mag;
  }
  for (int i = 0; i < 16; ++i) {
    const int ea = ae[i * sa], eb = be[i * sb];
    if (ea == kNoExponent || eb == kNoExponent) continue;
    // 8-bit significands: |a| = siga * 2^(ea - 7)
    const int siga = trunc_to_int(scale_pow2(a[i * sa], 7 - ea)), sigb = trunc_to_int(scale_pow2(b[i * sb], 7 - eb));
    const int prod = siga * sigb;  // exact; value = prod * 2^(ea + eb - 14)
    const int d = emax - (ea + eb);
    const unsigned pmag = prod < 0 ? unsigned(-prod) : unsigned(prod);
    const long long mag = d <= 11 ? (long long)pmag << (11 - d) : (d - 11 < 31 ? (long long)(pmag >> (d - 11)) : 0);
    sum += prod < 0 ? -mag : mag;
  }
  return scale_pow2(trunc_to_float(sum), emax - 25);
}

LFM_TCM_FN float mma_step(float c, const float* a, const short* ae, const float* b, const short* be, int sa, int sb) {
  int emax = exponent_of_float(c);
#ifdef __CUDACC__
#pragma unroll
#endif
  for (int i = 0; i < 16; ++i) {
    const int e = int(ae[i * sa]) + int(be[i * sb]);
    emax = e > emax ? e : emax;
  }
  if (emax < kNoExponent / 2) return 0.f;
  // Outside this range a scale factor or a product could leave the normal numbers: the integer statement instead.
  if (emax < -90 || emax > 90) return mma_step_reference(c, a, ae, b, be, sa, sb);
  const float inverse = float_of(unsigned(25 - emax + 127) << 23);  // 1 / q
  long long sum = trunc_to_int(c * inverse);                        // |c| / q < 2^26
#ifdef __CUDACC__
#pragma unroll
#endif
  for (int i = 0; i < 16; ++i) sum += trunc_to_int((a[i * sa] * b[i * sb]) * inverse);  // each below 2^27 in magnitude
  return trunc_to_float(sum) * float_of(unsigned(emax - 25 + 127) << 23);
}

}  // namespace lfm::tcm
