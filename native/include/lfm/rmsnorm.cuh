#pragma once

#include <cuda_bf16.h>

namespace lfm {

constexpr int kHiddenSize = 1024;
constexpr int kRmsThreads = 256;
constexpr float kRmsEpsilon = 1.0e-5f;

__device__ inline float pytorch_1024_sum_squares(const __nv_bfloat16* input) {
  const int base = threadIdx.x * 4;
  float values[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float value = __bfloat162float(input[base + i]);
    values[i] = value * value;
  }
  float sum = values[0];
#pragma unroll
  for (int i = 1; i < 4; ++i) sum += values[i];

  __shared__ float shared[kRmsThreads];
  shared[threadIdx.x] = sum;
  for (int offset = kRmsThreads / 2; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      sum += shared[threadIdx.x + offset];
      shared[threadIdx.x] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }
  return sum;
}

__global__ void rms_norm_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const auto* x = input + row * kHiddenSize;
  auto* y = output + row * kHiddenSize;
  const float sum = pytorch_1024_sum_squares(x);
  __shared__ float scale;
  if (threadIdx.x == 0) {
    scale = rsqrtf(sum / static_cast<float>(kHiddenSize) + kRmsEpsilon);
  }
  __syncthreads();
  for (int column = threadIdx.x; column < kHiddenSize; column += kRmsThreads) {
    const __nv_bfloat16 normalized =
        __float2bfloat16_rn(__bfloat162float(x[column]) * scale);
    y[column] = __hmul(normalized, weight[column]);
  }
}

// Generic width/epsilon fallback. The 1024-wide specialization above remains
// available for its established reduction contract, selected from metadata.
__global__ void rms_norm_dynamic(const __nv_bfloat16* input,
                                  const __nv_bfloat16* weight,
                                  __nv_bfloat16* output, int width, float epsilon) {
  const auto* x = input + static_cast<size_t>(blockIdx.x) * width;
  auto* y = output + static_cast<size_t>(blockIdx.x) * width;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = __bfloat162float(x[i]);
    sum += v * v;
  }
  __shared__ float partials[256];
  partials[threadIdx.x] = sum;
  for (int offset = 128; offset; offset /= 2) {
    __syncthreads();
    if (threadIdx.x < offset) partials[threadIdx.x] += partials[threadIdx.x + offset];
  }
  __syncthreads();
  const float scale = rsqrtf(partials[0] / width + epsilon);
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    y[i] = __hmul(__float2bfloat16_rn(__bfloat162float(x[i]) * scale), weight[i]);
}

// Contiguous vector-4, warp-per-row schedule used by Torch's reduction for
// 32-row inputs in this width range. MoE uses it consistently even when only
// one logical row is active, just as projections retain their canonical N.
// See the PyTorch-derived schedule attribution in third_party/PYTORCH_LICENSE.
__global__ void rms_norm_vectorized_warp(const __nv_bfloat16* input,
                                        const __nv_bfloat16* weight,
                                        __nv_bfloat16* output, int width, float epsilon) {
  const int lane = threadIdx.x;
  const auto* x = input + static_cast<size_t>(blockIdx.x) * width;
  auto* y = output + static_cast<size_t>(blockIdx.x) * width;
  float a[4] = {0, 0, 0, 0};
  for (int base = lane * 4; base < width; base += 128) {
#pragma unroll
    for (int j = 0; j < 4; ++j) if (base + j < width) {
      const float v = __bfloat162float(x[base + j]);
      a[j] = __fadd_rn(a[j], __fmul_rn(v, v));
    }
  }
  float sum = ((a[0] + a[1]) + a[2]) + a[3];
  for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
  sum = __shfl_sync(0xffffffff, sum, 0);
  const float variance = __fmul_rn(sum, 1.0f / width);
  const float scale = rsqrtf(__fadd_rn(variance, epsilon));
  for (int d = lane; d < width; d += 32)
    y[d] = __hmul(__float2bfloat16_rn(__bfloat162float(x[d]) * scale), weight[d]);
}

}  // namespace lfm
