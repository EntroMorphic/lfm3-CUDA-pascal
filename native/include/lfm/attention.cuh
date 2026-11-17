#pragma once

// The decode attention kernel.
//
// Decode attends one query position against the whole KV cache. The cache is
// laid out [batch][attention_slot][kv_head][position][dim], and the model has
// 16 query heads over 8 KV heads, so two query heads share each KV head.
//
// The kernel is one block per (query head, sequence): 16 blocks per sequence
// at batch 1, which leaves most of a 20-SM device idle, and each block reads
// the KV cache of its own sequence, so the cache traffic does not amortize
// across a batch either.
//
// The original kernel below uses an online softmax over 32-token tiles.
// Every tile is reduced by a single thread: maximum, then tile sum, in the
// original sequential order, with the running maximum and sum rescaled
// between tiles. The value accumulation then uses 64 of the block's 1024
// threads, one per head dimension, each looping over the tile's 32 tokens.
//
// The kernel lives in its own header so that it can be measured and
// replaced without touching the rest of the runtime, and so that a candidate
// can be compared against it on the same data.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include <lfm/rmsnorm.cuh>

namespace lfm {

constexpr int kAttentionLayers = 6;
constexpr int kKvHeads = 8;
constexpr int kHeadDimension = 64;
constexpr int kKvWidth = kKvHeads * kHeadDimension;
constexpr int kQkvWidth = kHiddenSize + 2 * kKvWidth;
constexpr int kQueryHeads = 16;
constexpr int kContextLength = 32768;
// The KV cache is laid out [batch][attention_slot][kv_head][position][dim].
constexpr size_t kKvLayerStride =
    static_cast<size_t>(kKvHeads) * kContextLength * kHeadDimension;
constexpr size_t kKvBatchStride = static_cast<size_t>(kAttentionLayers) * kKvLayerStride;

__global__ void online_softmax_gqa_attention(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
  const int window = attention_policy[0];
  const int sinks = attention_policy[1];
  const int sparse_after = attention_policy[2];
  const int query_head = blockIdx.x;
  const int kv_head = query_head / 2;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread % 32;
  constexpr int kTileTokens = 32;
  const size_t query_row = static_cast<size_t>(batch) * kQkvWidth;
  const size_t cache_row = static_cast<size_t>(batch) * kv_batch_stride;
  const size_t output_row = static_cast<size_t>(batch) * kHiddenSize;
  const int query_base = query_head * kHeadDimension;
  const float query0 = __bfloat162float(query[query_row + query_base + lane]);
  const float query1 = __bfloat162float(query[query_row + query_base + lane + 32]);
  __shared__ float scores[kTileTokens];
  __shared__ float probabilities[kTileTokens];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float previous_scale;
  if (thread == 0) {
    running_max = -FLT_MAX;
    running_sum = 0.0f;
  }
  float attended = 0.0f;
  __syncthreads();

  int range_starts[2] = {0, 0};
  int range_ends[2] = {position, -1};
  int range_count = 1;
  if (window > 0 && position >= sparse_after) {
    const int recent_start = max(0, position - window + 1);
    const int sink_end = min(position, sinks - 1);
    if (recent_start > sink_end + 1) {
      range_count = 0;
      if (sink_end >= 0) {
        range_starts[range_count] = 0;
        range_ends[range_count++] = sink_end;
      }
      range_starts[range_count] = recent_start;
      range_ends[range_count++] = position;
    }
  }

  for (int range = 0; range < range_count; ++range) {
    const int range_end = range_ends[range];
    for (int tile = range_starts[range]; tile <= range_end; tile += kTileTokens) {
      const int token = tile + warp;
      float score = -FLT_MAX;
      if (token <= range_end) {
      const size_t cache_base =
        cache_row + (static_cast<size_t>(kv_head) * kContextLength + token) * kHeadDimension;
      const float key0 = __bfloat162float(key_cache[cache_base + lane]);
      const float key1 = __bfloat162float(key_cache[cache_base + lane + 32]);
      score = query0 * key0 + query1 * key1;
      for (int offset = 16; offset > 0; offset >>= 1) {
        score += __shfl_down_sync(0xffffffff, score, offset);
      }
      score *= 0.125f;
      }
      if (lane == 0) scores[warp] = score;
      __syncthreads();

      if (thread == 0) {
        float tile_maximum = -FLT_MAX;
        const int tile_count = min(kTileTokens, range_end - tile + 1);
        for (int item = 0; item < tile_count; ++item) {
          tile_maximum = fmaxf(tile_maximum, scores[item]);
        }
        const float old_maximum = running_max;
        const float new_maximum = fmaxf(old_maximum, tile_maximum);
        const float scale = old_maximum == -FLT_MAX
            ? 0.0f
            : expf(old_maximum - new_maximum);
        float tile_sum = 0.0f;
        for (int item = 0; item < tile_count; ++item) {
          const float probability = expf(scores[item] - new_maximum);
          probabilities[item] = probability;
          tile_sum += probability;
        }
        previous_scale = scale;
        running_max = new_maximum;
        running_sum = running_sum * scale + tile_sum;
      }
      __syncthreads();

      if (thread < kHeadDimension) {
        attended *= previous_scale;
        const int tile_count = min(kTileTokens, range_end - tile + 1);
        for (int item = 0; item < tile_count; ++item) {
          const size_t value_index =
            cache_row +
            (static_cast<size_t>(kv_head) * kContextLength + tile + item) *
                kHeadDimension +
            thread;
          const float value = __bfloat162float(value_cache[value_index]);
          attended += probabilities[item] * value;
        }
      }
      __syncthreads();
    }
  }

  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / running_sum);
  }
}


// The same kernel with the value tile staged, the softmax published, and
// the loads pipelined.
//
// Three changes, and none of them touches the arithmetic.
//
// The value tile is staged in shared memory. The shipped kernel reads it with
// the 64 threads that own the head dimensions, and each of them loads the
// tile's 32 values one at a time. Two warps' worth of memory-level
// parallelism is not enough to cover the cache latency, so the block spends
// most of the tile waiting for memory it has already asked for. Here every
// warp stages its part of the tile, the loads overlap, and the accumulation
// reads from shared memory.
//
// The tile maximum and the tile sum are computed once and published. The
// shipped kernel has one thread compute the maximum, exponentiate the
// scores and sum them. Here warp 0 reduces the maximum, every warp
// exponentiates its own score, and one lane sums the exponentials in the
// original order. This removes redundant softmax reductions; source-level
// access counts are not measured shared-memory transactions (broadcasts and
// compiler dead-code elimination matter).
//
// The original loads keys before softmax and values after it. Here the
// keys for the next tile are held in registers and the
// values for the next tile are staged into the other shared buffer, both
// issued before the current tile's softmax runs. These are ordinary loads,
// not cp.async: the following block barrier waits for value staging. This
// is synchronous lookahead, not proof of load/softmax execution overlap.
//
// The output is the same bytes: the same scores, the same exponentials, the
// same left-to-right sum of the same floats, the same rescale of the running
// sum, and the same accumulation order over the tile's tokens.
__global__ void online_softmax_gqa_attention_staged(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride,
    int hidden = kHiddenSize, int query_heads = kQueryHeads,
    int kv_heads = kKvHeads, int context = kContextLength, const int* stopped = nullptr) {
  if (stopped && stopped[blockIdx.y]) {
    if (threadIdx.x < kHeadDimension)
      output[static_cast<size_t>(blockIdx.y) * hidden + blockIdx.x * kHeadDimension + threadIdx.x] =
          __float2bfloat16_rn(0.f);
    return;
  }
  // The optimized 64-dimensional tile schedule is a kernel specialization,
  // not a model architecture. All row/head/cache geometry comes from metadata.
  const int kQkvWidth = hidden + 2 * kv_heads * kHeadDimension;
  const int kHiddenSize = hidden;
  const int kContextLength = context;
  const int batch = blockIdx.y;
  const int position = positions[batch];
  const int window = attention_policy[0];
  const int sinks = attention_policy[1];
  const int sparse_after = attention_policy[2];
  const int query_head = blockIdx.x;
  const int kv_head = query_head / (query_heads / kv_heads);
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread % 32;
  constexpr int kTileTokens = 32;
  const size_t query_row = static_cast<size_t>(batch) * kQkvWidth;
  const size_t cache_row = static_cast<size_t>(batch) * kv_batch_stride;
  const size_t output_row = static_cast<size_t>(batch) * kHiddenSize;
  const int query_base = query_head * kHeadDimension;
  const float query0 = __bfloat162float(query[query_row + query_base + lane]);
  const float query1 = __bfloat162float(query[query_row + query_base + lane + 32]);
  __shared__ float scores[kTileTokens];
  __shared__ float probabilities[kTileTokens];
  __shared__ __nv_bfloat16 tile_values[2][kTileTokens * kHeadDimension];
  __shared__ float published_max;
  __shared__ float published_scale;
  __shared__ float published_sum;
  // The keys of the tile being processed and of the tile being prefetched.
  __nv_bfloat16 key0 = __float2bfloat16(0.0f);
  __nv_bfloat16 key1 = __float2bfloat16(0.0f);
  __nv_bfloat16 next_key0 = __float2bfloat16(0.0f);
  __nv_bfloat16 next_key1 = __float2bfloat16(0.0f);
  bool key_valid = false;
  bool next_key_valid = false;
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float attended = 0.0f;

  const size_t head_base =
    cache_row + static_cast<size_t>(kv_head) * kContextLength * kHeadDimension;

  int range_starts[2] = {0, 0};
  int range_ends[2] = {position, -1};
  int range_count = 1;
  if (window > 0 && position >= sparse_after) {
    const int recent_start = max(0, position - window + 1);
    const int sink_end = min(position, sinks - 1);
    if (recent_start > sink_end + 1) {
      range_count = 0;
      if (sink_end >= 0) {
        range_starts[range_count] = 0;
        range_ends[range_count++] = sink_end;
      }
      range_starts[range_count] = recent_start;
      range_ends[range_count++] = position;
    }
  }

  for (int range = 0; range < range_count; ++range) {
    const int range_start = range_starts[range];
    const int range_end = range_ends[range];
    if (range_start > range_end) continue;
    int buffer = 0;

    // The prologue: the range's first tile.
    {
      const int token = range_start + warp;
      key_valid = token <= range_end;
      if (token <= range_end) {
        const size_t index = head_base + static_cast<size_t>(token) * kHeadDimension;
        key0 = key_cache[index + lane];
        key1 = key_cache[index + lane + 32];
      }
      for (int index = thread; index < kTileTokens * kHeadDimension; index += blockDim.x) {
        const int row = index / kHeadDimension;
        const int dimension = index % kHeadDimension;
        tile_values[buffer][index] = range_start + row <= range_end
            ? value_cache[head_base + static_cast<size_t>(range_start + row) * kHeadDimension + dimension]
            : __float2bfloat16(0.0f);
      }
    }

    for (int tile = range_start; tile <= range_end; tile += kTileTokens) {
      float score = -FLT_MAX;
      if (key_valid) {
        const float k0 = __bfloat162float(key0);
        const float k1 = __bfloat162float(key1);
        score = query0 * k0 + query1 * k1;
        for (int offset = 16; offset > 0; offset >>= 1) {
          score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score *= 0.125f;
      }
      if (lane == 0) scores[warp] = score;

      // Stage into the alternate buffer. The previous iteration's final
      // barrier protects it from overwrite while PV still reads it. The
      // barrier below publishes both the scores and the staged values.
      next_key_valid = false;
      if (tile + kTileTokens <= range_end) {
        const int next_token = tile + kTileTokens + warp;
        if (next_token <= range_end) {
          const size_t index = head_base + static_cast<size_t>(next_token) * kHeadDimension;
          next_key0 = key_cache[index + lane];
          next_key1 = key_cache[index + lane + 32];
          next_key_valid = true;
        }
        const int next_buffer = buffer ^ 1;
        for (int index = thread; index < kTileTokens * kHeadDimension; index += blockDim.x) {
          const int row = index / kHeadDimension;
          const int dimension = index % kHeadDimension;
          tile_values[next_buffer][index] = tile + kTileTokens + row <= range_end
              ? value_cache[head_base +
                    static_cast<size_t>(tile + kTileTokens + row) * kHeadDimension + dimension]
              : __float2bfloat16(0.0f);
        }
      }
      __syncthreads();

      const int tile_count = min(kTileTokens, range_end - tile + 1);
      // Warp 0 reduces the tile maximum and publishes the new maximum and
      // the scale the tile's exponentials need.
      if (warp == 0) {
        float value = lane < tile_count ? scores[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
          value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (lane == 0) {
          const float old_maximum = running_max;
          const float new_maximum = fmaxf(old_maximum, value);
          const float scale = old_maximum == -FLT_MAX ? 0.0f : expf(old_maximum - new_maximum);
          running_max = new_maximum;
          published_max = new_maximum;
          published_scale = scale;
        }
      }
      __syncthreads();

      // Every warp exponentiates its own score.
      if (lane == 0) {
        probabilities[warp] = key_valid ? expf(scores[warp] - published_max) : 0.0f;
      }
      __syncthreads();

      // One lane sums the exponentials in the original order, so the total
      // is the same float.
      if (warp == 0 && lane == 0) {
        float tile_sum = 0.0f;
        for (int item = 0; item < tile_count; ++item) {
          tile_sum += probabilities[item];
        }
        running_sum = running_sum * published_scale + tile_sum;
      }

      if (thread < kHeadDimension) {
        attended *= published_scale;
        for (int item = 0; item < tile_count; ++item) {
          const float value =
            __bfloat162float(tile_values[buffer][item * kHeadDimension + thread]);
          attended += probabilities[item] * value;
        }
      }
      __syncthreads();

      key0 = next_key0;
      key1 = next_key1;
      key_valid = next_key_valid;
      buffer ^= 1;
    }
  }

  if (warp == 0 && lane == 0) published_sum = running_sum;
  __syncthreads();
  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / published_sum);
  }
}

}  // namespace lfm
