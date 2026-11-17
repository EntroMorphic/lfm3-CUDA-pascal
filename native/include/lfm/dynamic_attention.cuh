#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <lfm/model_config.hpp>

namespace lfm {
// General even-head-dimension fallback. Preserve the online 32-token tile
// schedule; use the staged/published/lookahead specialization for dimension 64.
__global__ void online_attention_dynamic(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, __nv_bfloat16* output,
    const int* positions, const int* policy, RuntimeShape s, const int* stopped = nullptr) {
  if (stopped && stopped[blockIdx.y]) {
    for (int d = threadIdx.x; d < s.head_dim; d += blockDim.x)
      output[static_cast<size_t>(blockIdx.y) * s.hidden + blockIdx.x * s.head_dim + d] =
          __float2bfloat16_rn(0.f);
    return;
  }
  const int row = blockIdx.y, head = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int kv_head = head / (s.query_heads / s.kv_heads);
  const size_t cache_row = static_cast<size_t>(row) * s.attention_layers * s.kv_heads * s.context * s.head_dim;
  const size_t head_base = cache_row + static_cast<size_t>(kv_head) * s.context * s.head_dim;
  const auto* q = query + static_cast<size_t>(row) * (s.hidden + 2 * s.kv_heads * s.head_dim) + head * s.head_dim;
  __shared__ float scores[32], probability[32], maximum, denominator, scale;
  if (threadIdx.x == 0) { maximum = -FLT_MAX; denominator = 0; }
  float attended = 0;
  int starts[2] = {0, 0}, ends[2] = {positions[row], -1}, ranges = 1;
  if (policy[0] > 0 && positions[row] >= policy[2]) {
    const int recent = max(0, positions[row] - policy[0] + 1);
    const int sink = min(positions[row], policy[1] - 1);
    if (recent > sink + 1) {
      ranges = 0;
      if (sink >= 0) { starts[ranges] = 0; ends[ranges++] = sink; }
      starts[ranges] = recent; ends[ranges++] = positions[row];
    }
  }
  __syncthreads();
  for (int range = 0; range < ranges; ++range) {
    for (int tile = starts[range]; tile <= ends[range]; tile += 32) {
      const int token = tile + warp;
      float score = 0;
      if (token <= ends[range]) {
        for (int d = lane; d < s.head_dim; d += 32)
          score += __bfloat162float(q[d]) * __bfloat162float(keys[head_base + static_cast<size_t>(token) * s.head_dim + d]);
        for (int offset = 16; offset; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
        score *= rsqrtf(static_cast<float>(s.head_dim));
      } else score = -FLT_MAX;
      if (lane == 0) scores[warp] = score;
      __syncthreads();
      const int count = min(32, ends[range] - tile + 1);
      if (threadIdx.x == 0) {
        float next = maximum;
        for (int i = 0; i < count; ++i) next = fmaxf(next, scores[i]);
        scale = maximum == -FLT_MAX ? 0 : expf(maximum - next);
        float sum = 0;
        for (int i = 0; i < count; ++i) { probability[i] = expf(scores[i] - next); sum += probability[i]; }
        denominator = denominator * scale + sum;
        maximum = next;
      }
      __syncthreads();
      if (threadIdx.x < s.head_dim) {
        attended *= scale;
        for (int i = 0; i < count; ++i)
          attended += probability[i] * __bfloat162float(values[head_base + static_cast<size_t>(tile + i) * s.head_dim + threadIdx.x]);
      }
      __syncthreads();
    }
  }
  if (threadIdx.x < s.head_dim)
    output[static_cast<size_t>(row) * s.hidden + head * s.head_dim + threadIdx.x] = __float2bfloat16_rn(attended / denominator);
}
}  // namespace lfm
