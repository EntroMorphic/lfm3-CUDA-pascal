#pragma once

#include <cuda_bf16.h>
#include <cfloat>
#include <climits>
#include <cmath>

namespace lfm {

// Match torch.argmax: equal maxima select the lowest vocabulary index.
// Inference logits must be finite; NaN ordering is not defined here.
__global__ void argmax_bf16(const __nv_bfloat16* logits, int count, int* output) {
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -INFINITY;
  int best_index = INT_MAX;
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = __bfloat162float(logits[index]);
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  for (int offset = 128; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      const float other = values[threadIdx.x + offset];
      const int other_index = indices[threadIdx.x + offset];
      if (other > values[threadIdx.x] ||
          (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
  }
  if (threadIdx.x == 0) output[0] = indices[0];
}

// Match torch.argmax: equal maxima select the lowest vocabulary index.
// Inference logits must be finite; NaN ordering is not defined here.
//
// One block reduces one row of a [rows, count] logit matrix. The reduction
// order within a row is the same for every row, so a row selected from a
// batched matrix matches the same row reduced on its own.
__global__ void argmax_bf16_rows(
    const __nv_bfloat16* logits, int rows, int count, int* output) {
  const int row = blockIdx.x;
  const __nv_bfloat16* row_logits = logits + static_cast<size_t>(row) * count;
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -INFINITY;
  int best_index = INT_MAX;
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = __bfloat162float(row_logits[index]);
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  for (int offset = 128; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      const float other = values[threadIdx.x + offset];
      const int other_index = indices[threadIdx.x + offset];
      if (other > values[threadIdx.x] ||
          (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
  }
  if (threadIdx.x == 0) output[row] = indices[0];
}

}  // namespace lfm
