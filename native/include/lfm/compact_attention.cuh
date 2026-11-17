#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <lfm/model_config.hpp>

namespace lfm {

__global__ void zero_stopped_attention_rows(
    __nv_bfloat16* output, const int* stopped, int hidden) {
  if (!stopped[blockIdx.y]) return;
  for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden;
       d += gridDim.x * blockDim.x)
    output[static_cast<size_t>(blockIdx.y) * hidden + d] = __float2bfloat16_rn(0.f);
}

// Preserve the production 32-token softmax schedule and left-to-right PV sum.
// Eight warps each compute four QK scores, rather than dedicating 32 warps to
// one score each. The runtime selects this kernel only for batched dimension-64
// attention; other geometries retain their existing production specialization.
template<int Threads>
__global__ void online_attention_compact(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, __nv_bfloat16* output,
    const int* positions, const int* policy, RuntimeShape s, const int* stopped) {
  static_assert(Threads >= 32 && Threads <= 1024 && Threads % 32 == 0);
  const int row = blockIdx.y, head = blockIdx.x;
  const int thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
  constexpr int kWarps = Threads / 32;
  if (stopped && stopped[row]) {
    if (thread < s.head_dim)
      output[static_cast<size_t>(row) * s.hidden + head * s.head_dim + thread] =
          __float2bfloat16_rn(0.f);
    return;
  }
  const int kv_head = head / (s.query_heads / s.kv_heads);
  const size_t base = static_cast<size_t>(row) * s.attention_layers * s.kv_heads *
      s.context * s.head_dim + static_cast<size_t>(kv_head) * s.context * s.head_dim;
  const auto* q = query + static_cast<size_t>(row) *
      (s.hidden + 2 * s.kv_heads * s.head_dim) + head * s.head_dim;
  __shared__ float scores[32], probabilities[32];
  __shared__ float published_max, published_scale, published_sum;
  extern __shared__ __nv_bfloat16 staged_values[];
  float running_max = -FLT_MAX, running_sum = 0.f, attended = 0.f;
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
  for (int range = 0; range < ranges; ++range) {
    for (int tile = starts[range]; tile <= ends[range]; tile += 32) {
      const int count = min(32, ends[range] - tile + 1);
      for (int item = warp; item < 32; item += kWarps) {
        float score = -FLT_MAX;
        if (item < count) {
          const auto* k = keys + base + static_cast<size_t>(tile + item) * s.head_dim;
          if (s.head_dim == 64) {
            const float q0 = __bfloat162float(q[lane]);
            const float q1 = __bfloat162float(q[lane + 32]);
            const float k0 = __bfloat162float(k[lane]);
            const float k1 = __bfloat162float(k[lane + 32]);
            score = q0 * k0 + q1 * k1;
          } else {
            score = 0.f;
            for (int d = lane; d < s.head_dim; d += 32)
              score += __bfloat162float(q[d]) * __bfloat162float(k[d]);
          }
          for (int offset = 16; offset; offset /= 2)
            score += __shfl_down_sync(0xffffffff, score, offset);
          score *= s.head_dim == 64 ? .125f : rsqrtf(static_cast<float>(s.head_dim));
        }
        if (lane == 0) scores[item] = score;
      }
      for (int i = thread; i < 32 * s.head_dim; i += Threads) {
        const int item = i / s.head_dim, dimension = i % s.head_dim;
        staged_values[i] = item < count
            ? values[base + static_cast<size_t>(tile + item) * s.head_dim + dimension]
            : __float2bfloat16_rn(0.f);
      }
      __syncthreads();
      if (warp == 0) {
        float maximum = lane < count ? scores[lane] : -FLT_MAX;
        for (int offset = 16; offset; offset /= 2)
          maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        if (lane == 0) {
          const float next = fmaxf(running_max, maximum);
          published_scale = running_max == -FLT_MAX ? 0.f : expf(running_max - next);
          running_max = next;
          published_max = next;
        }
      }
      __syncthreads();
      if (thread < 32)
        probabilities[thread] = thread < count ? expf(scores[thread] - published_max) : 0.f;
      __syncthreads();
      if (thread == 0) {
        float total = 0.f;
        for (int item = 0; item < count; ++item) total += probabilities[item];
        running_sum = running_sum * published_scale + total;
      }
      if (thread < s.head_dim) {
        attended *= published_scale;
        for (int item = 0; item < count; ++item)
          attended += probabilities[item] *
              __bfloat162float(staged_values[item * s.head_dim + thread]);
      }
      __syncthreads();
    }
  }
  if (thread == 0) published_sum = running_sum;
  __syncthreads();
  if (thread < s.head_dim)
    output[static_cast<size_t>(row) * s.hidden + head * s.head_dim + thread] =
        __float2bfloat16_rn(attended / published_sum);
}
}  // namespace lfm
