#pragma once

// Isolated, correctness-first LFM3 primitives. Not yet wired into main.cu.
// Layout contracts and rounding boundaries are exercised by check_lfm3_ops.py.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

namespace lfm::lfm3 {

__global__ void rotary(__nv_bfloat16* x, const int* positions, int rows,
                       int heads, int dim, int pairs, float theta) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(rows) * heads * (dim / 2);
  if (index >= count) return;
  const int half = dim / 2, d = index % half;
  if (d >= pairs) return; // Zero-frequency tail: identity, not a shortened RoPE basis.
  const size_t head = index / half;
  const float inverse = 1.0f / powf(theta, static_cast<float>(2 * d) / dim);
  const float angle = positions[head / heads] * inverse;
  const float cosine = __bfloat162float(__float2bfloat16_rn(cosf(angle)));
  const float sine = __bfloat162float(__float2bfloat16_rn(sinf(angle)));
  const float a = __bfloat162float(x[head * dim + d]);
  const float b = __bfloat162float(x[head * dim + d + half]);
  // LFM3 does arithmetic in FP32 after BF16 trig values, then rounds once.
  x[head * dim + d] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(a, cosine), -__fmul_rn(b, sine)));
  x[head * dim + d + half] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, cosine), __fmul_rn(a, sine)));
}

__global__ void head_gate(const __nv_bfloat16* attended, const __nv_bfloat16* logits,
                          __nv_bfloat16* output, size_t count, int dim) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float value = __bfloat162float(logits[i / dim]);
  const auto sigmoid = __float2bfloat16_rn(1.0f / (1.0f + expf(-value)));
  output[i] = __hmul(attended[i], sigmoid);
}

__global__ void sandwich_residual(const __nv_bfloat16* update, const __nv_bfloat16* weight,
                                   const __nv_bfloat16* residual, __nv_bfloat16* output,
                                   int width, float epsilon) {
  const size_t base = static_cast<size_t>(blockIdx.x) * width;
  float sum = 0;
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    const float x = __bfloat162float(update[base + d]);
    sum = __fadd_rn(sum, __fmul_rn(x, x));
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = sum;
  for (int offset = 128; offset; offset /= 2) {
    __syncthreads();
    if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
  }
  __syncthreads();
  const float scale = rsqrtf(__fadd_rn(reduction[0] / width, epsilon));
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    const auto normalized = __float2bfloat16_rn(__fmul_rn(__bfloat162float(update[base + d]), scale));
    output[base + d] = __hadd(__hmul(normalized, weight[d]), residual[base + d]);
  }
}

// One decode token, BCx projections [B,3,D], depthwise weights [D,7],
// cache [B,D,7] including the discarded oldest value. All states are row-local.
__global__ void convolution7(const __nv_bfloat16* projected, const __nv_bfloat16* weights,
                             __nv_bfloat16* cache, __nv_bfloat16* output, int width, int batch) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(batch) * width) return;
  const size_t b = i / width, d = i % width, base = b * 3 * width + d;
  const auto input = __hmul(projected[base], projected[base + 2 * width]);
  float products[8];
  for (int k = 0; k < 7; ++k) {
    const auto value = k == 6 ? input : cache[i * 7 + k + 1];
    products[k] = __fmul_rn(__bfloat162float(value), __bfloat162float(weights[d * 7 + k]));
  }
  products[7] = 0.0f;
  // Match the pinned Triton decode's 8-wide butterfly reduction (offsets 4,2,1),
  // not prefill's sequential reduction. Cancellation makes the order observable.
  const float even = __fadd_rn(__fadd_rn(products[0], products[4]), __fadd_rn(products[2], products[6]));
  const float odd = __fadd_rn(__fadd_rn(products[1], products[5]), __fadd_rn(products[3], products[7]));
  const float sum = __fadd_rn(even, odd);
  for (int k = 0; k < 6; ++k) cache[i * 7 + k] = cache[i * 7 + k + 1];
  cache[i * 7 + 6] = input;
  output[i] = __hmul(projected[base + width], __float2bfloat16_rn(sum));
}

// Empty-cache prefill only: projected=[B,T,3,D], output=[B,T,D], state=[B,D,7].
// No padding/packed documents or prior cache are accepted by this prototype.
__global__ void convolution7_prefill(const __nv_bfloat16* projected, const __nv_bfloat16* weights,
                                     __nv_bfloat16* state, __nv_bfloat16* output,
                                     int batch, int length, int width) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(batch) * length * width) return;
  const int d = i % width;
  const int token = (i / width) % length;
  const size_t row = i / (static_cast<size_t>(length) * width);
  float sum = 0;
  for (int k = 0; k < 7; ++k) {
    const int previous = token + k - 6;
    const size_t offset = (row * length + static_cast<size_t>(max(previous, 0))) * 3 * width + d;
    const auto input = previous >= 0 ? __hmul(projected[offset], projected[offset + 2 * width]) : __float2bfloat16_rn(0.0f);
    sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input), __bfloat162float(weights[static_cast<size_t>(d) * 7 + k])));
  }
  const size_t current = (row * length + token) * 3 * width + d;
  output[i] = __hmul(projected[current + width], __float2bfloat16_rn(sum));
  if (token == length - 1) {
    for (int k = 0; k < 7; ++k) {
      const int previous = length + k - 7;
      const size_t offset = (row * length + static_cast<size_t>(max(previous, 0))) * 3 * width + d;
      state[(row * width + d) * 7 + k] = previous >= 0 ? __hmul(projected[offset], projected[offset + 2 * width]) : __float2bfloat16_rn(0.0f);
    }
  }
}

// Deliberately unoptimized eager-style decode attention, for numerical bring-up.
// Q=[B,Hq,D], KV=[B,Hkv,capacity,D], inclusive absolute positions=[B].
// Full causal storage (no ring buffer yet); window=0 is global attention.
__global__ void attention_decode(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                 const __nv_bfloat16* v, const int* positions,
                                 __nv_bfloat16* output, int heads, int kv_heads,
                                 int dim, int capacity, int window) {
  extern __shared__ float scores[];
  __shared__ float reduction[256];
  const int head = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  const int position = positions[row];
  const int end = min(position + 1, capacity);
  const int begin = window > 0 ? max(0, position - window + 1) : 0;
  const int kv_head = head / (heads / kv_heads);
  const size_t query_base = (static_cast<size_t>(row) * heads + head) * dim;
  const size_t cache_base = (static_cast<size_t>(row) * kv_heads + kv_head) * capacity * dim;
  float maximum = -CUDART_INF_F;
  for (int token = begin + tid; token < end; token += blockDim.x) {
    float dot = 0;
    for (int d = 0; d < dim; ++d)
      dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(q[query_base + d]),
                                    __bfloat162float(k[cache_base + static_cast<size_t>(token) * dim + d])));
    // Eager reference materializes QK in BF16 before applying scaling.
    const float rounded = __bfloat162float(__float2bfloat16_rn(dot));
    const float score = __bfloat162float(__float2bfloat16_rn(rounded / sqrtf(static_cast<float>(dim))));
    scores[token] = score;
    maximum = fmaxf(maximum, score);
  }
  reduction[tid] = maximum;
  for (int offset = 128; offset; offset /= 2) {
    __syncthreads();
    if (tid < offset) reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
  }
  __syncthreads();
  maximum = reduction[0];
  __syncthreads();
  float denominator = 0;
  for (int token = begin + tid; token < end; token += blockDim.x) {
    const float value = expf(scores[token] - maximum);
    scores[token] = value;
    denominator += value;
  }
  reduction[tid] = denominator;
  for (int offset = 128; offset; offset /= 2) {
    __syncthreads();
    if (tid < offset) reduction[tid] += reduction[tid + offset];
  }
  __syncthreads();
  denominator = reduction[0];
  for (int token = begin + tid; token < end; token += blockDim.x)
    scores[token] = __bfloat162float(__float2bfloat16_rn(scores[token] / denominator));
  __syncthreads();
  for (int d = tid; d < dim; d += blockDim.x) {
    float sum = 0;
    for (int token = begin; token < end; ++token)
      sum = __fadd_rn(sum, __fmul_rn(scores[token], __bfloat162float(v[cache_base + static_cast<size_t>(token) * dim + d])));
    output[query_base + d] = __float2bfloat16_rn(sum);
  }
}

} // namespace lfm::lfm3
