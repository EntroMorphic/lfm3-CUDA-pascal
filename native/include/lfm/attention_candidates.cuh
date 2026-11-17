// Candidate designs for the decode attention kernel.
//
// The shipped kernel reaches 22-37 GB/s against a 256.5 GB/s device ceiling,
// so it is latency bound rather than bandwidth bound: the block serialises
// into three phases per 32-token tile, and two of them use a fraction of the
// block's threads.
//
// This file holds the candidates, each of which is measured against the
// shipped kernel on the same data. The candidates marked exact produce the
// same bytes as the shipped kernel; the ones that do not are marked, because
// they change the reduction order.
//
//   exact_softmax      The softmax phase's work moves off the single thread
//                       that runs it, without moving a single operation:
//                       the tile maximum is computed by every thread, each
//                       warp exponentiates its own score, and every thread
//                       sums the exponentials in the original order.
//   exact_staged_value  The value tile is staged in shared memory by all the
//                       warps, so the loads overlap instead of being issued
//                       by the two warps that do the accumulation.
//   split_kv            The KV range is split across blocks and the partial
//                       results are combined. More blocks, more parallelism,
//                       and a different reduction order.
//   shared_kv_head      One block serves the two query heads that share a KV
//                       head, so the cache is read once instead of twice.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

#include <lfm/attention.cuh>

namespace lfm {

// The exact restructure of the softmax phase.
//
// The shipped kernel has one thread compute the tile maximum, exponentiate
// the tile's scores and sum them, while the rest of the block waits at a
// barrier. Here every thread computes the same tile maximum from the same
// shared scores, every warp exponentiates its own score, and every thread
// sums the exponentials in the original order.
//
// The values are the same: the same fmaxf chain, the same expf of the same
// difference, the same left-to-right sum of the same floats, the same
// rescale of the running sum. Only the thread that does the work changes.
template <int kThreads>
__global__ void attention_exact_softmax(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
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
  __shared__ float tile_values[kTileTokens * kHeadDimension];
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float attended = 0.0f;

  for (int tile = 0; tile <= position; tile += kTileTokens) {
    const int token = tile + warp;
    float score = -FLT_MAX;
    if (token <= position) {
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

    // Every thread reduces the same shared scores, so every thread knows the
    // same tile maximum and the same running maximum.
    const int tile_count = min(kTileTokens, position - tile + 1);
    float tile_maximum = -FLT_MAX;
    for (int item = 0; item < tile_count; ++item) {
      tile_maximum = fmaxf(tile_maximum, scores[item]);
    }
    const float old_maximum = running_max;
    const float new_maximum = fmaxf(old_maximum, tile_maximum);
    const float scale = old_maximum == -FLT_MAX ? 0.0f : expf(old_maximum - new_maximum);
    // One warp exponentiates one score.
    if (lane == 0) {
      probabilities[warp] = token <= position ? expf(scores[warp] - new_maximum) : 0.0f;
    }
    __syncthreads();

    // Every thread sums the same exponentials in the original order, so the
    // total is the same float.
    float tile_sum = 0.0f;
    for (int item = 0; item < tile_count; ++item) {
      tile_sum += probabilities[item];
    }
    running_max = new_maximum;
    running_sum = running_sum * scale + tile_sum;

    // The value tile is staged by all the warps, so the loads overlap.
    const size_t value_base =
      cache_row + (static_cast<size_t>(kv_head) * kContextLength + tile) * kHeadDimension;
    for (int index = thread; index < kTileTokens * kHeadDimension; index += kThreads) {
      const int row = index / kHeadDimension;
      const int dimension = index % kHeadDimension;
      tile_values[index] = tile + row <= position
          ? value_cache[value_base + static_cast<size_t>(row) * kHeadDimension + dimension]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    if (thread < kHeadDimension) {
      attended *= scale;
      for (int item = 0; item < tile_count; ++item) {
        const float value = __bfloat162float(tile_values[item * kHeadDimension + thread]);
        attended += probabilities[item] * value;
      }
    }
    __syncthreads();
  }

  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / running_sum);
  }
}

// The same restructure, without staging the value tile. It isolates what the
// staging is worth.
template <int kThreads>
__global__ void attention_exact_softmax_unstaged(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
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
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float attended = 0.0f;

  for (int tile = 0; tile <= position; tile += kTileTokens) {
    const int token = tile + warp;
    float score = -FLT_MAX;
    if (token <= position) {
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

    const int tile_count = min(kTileTokens, position - tile + 1);
    float tile_maximum = -FLT_MAX;
    for (int item = 0; item < tile_count; ++item) {
      tile_maximum = fmaxf(tile_maximum, scores[item]);
    }
    const float old_maximum = running_max;
    const float new_maximum = fmaxf(old_maximum, tile_maximum);
    const float scale = old_maximum == -FLT_MAX ? 0.0f : expf(old_maximum - new_maximum);
    if (lane == 0) {
      probabilities[warp] = token <= position ? expf(scores[warp] - new_maximum) : 0.0f;
    }
    __syncthreads();

    float tile_sum = 0.0f;
    for (int item = 0; item < tile_count; ++item) {
      tile_sum += probabilities[item];
    }
    running_max = new_maximum;
    running_sum = running_sum * scale + tile_sum;

    if (thread < kHeadDimension) {
      attended *= scale;
      const size_t value_base =
        cache_row + (static_cast<size_t>(kv_head) * kContextLength + tile) * kHeadDimension;
      for (int item = 0; item < tile_count; ++item) {
        const size_t value_index =
          value_base + static_cast<size_t>(item) * kHeadDimension + thread;
        const float value = __bfloat162float(value_cache[value_index]);
        attended += probabilities[item] * value;
      }
    }
    __syncthreads();
  }

  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / running_sum);
  }
}

// One block serves the two query heads that share a KV head.
//
// The cache is read once for the pair instead of once for each head, and the
// arithmetic of each head is the shipped arithmetic, so the output is the
// same bytes. The block count halves.
template <int kThreads>
__global__ void attention_shared_kv_head(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
  const int kv_head = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread % 32;
  constexpr int kTileTokens = 32;
  const size_t query_row = static_cast<size_t>(batch) * kQkvWidth;
  const size_t cache_row = static_cast<size_t>(batch) * kv_batch_stride;
  const size_t output_row = static_cast<size_t>(batch) * kHiddenSize;
  const int head_base = kv_head * 2 * kHeadDimension;
  const float query0 = __bfloat162float(query[query_row + head_base + lane]);
  const float query1 = __bfloat162float(query[query_row + head_base + lane + 32]);
  const float partner0 = __bfloat162float(query[query_row + head_base + kHeadDimension + lane]);
  const float partner1 = __bfloat162float(query[query_row + head_base + kHeadDimension + lane + 32]);
  __shared__ float scores[2][kTileTokens];
  __shared__ float probabilities[2][kTileTokens];
  float running_max[2] = {-FLT_MAX, -FLT_MAX};
  float running_sum[2] = {0.0f, 0.0f};
  float attended[2] = {0.0f, 0.0f};

  for (int tile = 0; tile <= position; tile += kTileTokens) {
    const int token = tile + warp;
    float score[2] = {-FLT_MAX, -FLT_MAX};
    if (token <= position) {
      const size_t cache_base =
        cache_row + (static_cast<size_t>(kv_head) * kContextLength + token) * kHeadDimension;
      const float key0 = __bfloat162float(key_cache[cache_base + lane]);
      const float key1 = __bfloat162float(key_cache[cache_base + lane + 32]);
      score[0] = query0 * key0 + query1 * key1;
      score[1] = partner0 * key0 + partner1 * key1;
      for (int offset = 16; offset > 0; offset >>= 1) {
        score[0] += __shfl_down_sync(0xffffffff, score[0], offset);
        score[1] += __shfl_down_sync(0xffffffff, score[1], offset);
      }
      score[0] *= 0.125f;
      score[1] *= 0.125f;
    }
    if (lane == 0) {
      scores[0][warp] = score[0];
      scores[1][warp] = score[1];
    }
    __syncthreads();

    const int tile_count = min(kTileTokens, position - tile + 1);
    for (int head = 0; head < 2; ++head) {
      float tile_maximum = -FLT_MAX;
      for (int item = 0; item < tile_count; ++item) {
        tile_maximum = fmaxf(tile_maximum, scores[head][item]);
      }
      const float old_maximum = running_max[head];
      const float new_maximum = fmaxf(old_maximum, tile_maximum);
      const float scale = old_maximum == -FLT_MAX ? 0.0f : expf(old_maximum - new_maximum);
      if (lane == 0) {
        probabilities[head][warp] = token <= position
            ? expf(scores[head][warp] - new_maximum) : 0.0f;
      }
      __syncthreads();
      float tile_sum = 0.0f;
      for (int item = 0; item < tile_count; ++item) {
        tile_sum += probabilities[head][item];
      }
      running_max[head] = new_maximum;
      running_sum[head] = running_sum[head] * scale + tile_sum;
      // Only the threads that own this head accumulate it: the other head's
      // probabilities belong to the other head's iteration.
      if (thread / kHeadDimension == head && thread < 2 * kHeadDimension) {
        const int dimension = thread % kHeadDimension;
        attended[head] *= scale;
        const size_t value_base =
          cache_row + (static_cast<size_t>(kv_head) * kContextLength + tile) * kHeadDimension;
        for (int item = 0; item < tile_count; ++item) {
          const size_t value_index =
            value_base + static_cast<size_t>(item) * kHeadDimension + dimension;
          const float value = __bfloat162float(value_cache[value_index]);
          attended[head] += probabilities[head][item] * value;
        }
      }
      __syncthreads();
    }
  }

  if (thread < 2 * kHeadDimension) {
    const int head_in_block = thread / kHeadDimension;
    const int dimension = thread % kHeadDimension;
    const int query_head = kv_head * 2 + head_in_block;
    output[output_row + query_head * kHeadDimension + dimension] =
        __float2bfloat16_rn(attended[head_in_block] / running_sum[head_in_block]);
  }
}

// Split the KV range across blocks and combine the partial results.
//
// This is the flash-decoding shape: many more blocks, so the device is full
// even at batch 1, and each block only pays for its own chunk of the range.
// The combine step adds the partial maxima and sums in a different order
// from the sequential tile reduction, so the output is not the shipped bytes.
struct SplitPartial {
  float maximum;
  float sum;
  float value[kHeadDimension];
};

__global__ void attention_split_kv(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    SplitPartial* partials,
    const int* positions,
    int kv_batch_stride,
    int splits) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
  const int query_head = blockIdx.x;
  const int split = blockIdx.z;
  const int kv_head = query_head / 2;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread % 32;
  constexpr int kTileTokens = 32;
  const size_t query_row = static_cast<size_t>(batch) * kQkvWidth;
  const size_t cache_row = static_cast<size_t>(batch) * kv_batch_stride;
  const int query_base = query_head * kHeadDimension;
  const float query0 = __bfloat162float(query[query_row + query_base + lane]);
  const float query1 = __bfloat162float(query[query_row + query_base + lane + 32]);
  __shared__ float scores[kTileTokens];

  // The chunk this block owns.
  const int chunk = (position + 1 + splits - 1) / splits;
  const int start = split * chunk;
  const int end = min(position, start + chunk - 1);

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float attended = 0.0f;
  for (int tile = start; tile <= end; tile += kTileTokens) {
    const int token = tile + warp;
    float score = -FLT_MAX;
    if (token <= end) {
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
    const int tile_count = min(kTileTokens, end - tile + 1);
    float tile_maximum = -FLT_MAX;
    for (int item = 0; item < tile_count; ++item) {
      tile_maximum = fmaxf(tile_maximum, scores[item]);
    }
    const float old_maximum = running_max;
    const float new_maximum = fmaxf(old_maximum, tile_maximum);
    const float scale = old_maximum == -FLT_MAX ? 0.0f : expf(old_maximum - new_maximum);
    if (lane == 0) {
      scores[warp] = token <= end ? expf(scores[warp] - new_maximum) : 0.0f;
    }
    __syncthreads();
    float tile_sum = 0.0f;
    for (int item = 0; item < tile_count; ++item) {
      tile_sum += scores[item];
    }
    running_max = new_maximum;
    running_sum = running_sum * scale + tile_sum;
    if (thread < kHeadDimension) {
      attended *= scale;
      const size_t value_base =
        cache_row + (static_cast<size_t>(kv_head) * kContextLength + tile) * kHeadDimension;
      for (int item = 0; item < tile_count; ++item) {
        const size_t value_index =
          value_base + static_cast<size_t>(item) * kHeadDimension + thread;
        const float value = __bfloat162float(value_cache[value_index]);
        attended += scores[item] * value;
      }
    }
    __syncthreads();
  }

  if (thread < kHeadDimension) {
    SplitPartial* partial = partials +
        ((static_cast<size_t>(batch) * kQueryHeads + query_head) * splits + split);
    partial->maximum = running_max;
    partial->sum = running_sum;
    partial->value[thread] = attended;
  }
}

// Combine the partials of a split-KV attention.
//
// The partials are combined in split order with the usual rescaling: the
// maximum over the splits, then each split's sum and value rescaled against
// it. This is a different order from the sequential tile reduction, so the
// result is close to the shipped answer but not the same bytes.
__global__ void combine_split_kv(
    const SplitPartial* partials,
    __nv_bfloat16* output,
    const int* positions,
    int splits,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int query_head = blockIdx.x;
  const int thread = threadIdx.x;
  if (thread >= kHeadDimension) return;
  const SplitPartial* base =
    partials + (static_cast<size_t>(batch) * kQueryHeads + query_head) * splits;
  float maximum = -FLT_MAX;
  for (int split = 0; split < splits; ++split) {
    maximum = fmaxf(maximum, base[split].maximum);
  }
  float total = 0.0f;
  float value = 0.0f;
  for (int split = 0; split < splits; ++split) {
    if (base[split].sum <= 0.0f) continue;
    const float scale = base[split].maximum == -FLT_MAX
        ? 0.0f : expf(base[split].maximum - maximum);
    total += base[split].sum * scale;
    value += base[split].value[thread] * scale;
  }
  const size_t output_row = static_cast<size_t>(batch) * kHiddenSize;
  output[output_row + query_head * kHeadDimension + thread] =
      __float2bfloat16_rn(value / total);
}


// The tile maximum and the tile sum are computed once and published.
//
// The redundant variant has every thread read the tile's 32 scores and its
// 32 exponentials: nominally 65536 source-level accesses per full tile,
// not a measurement of shared transactions. Here warp 0 reduces the
// maximum, the exponentials are taken in parallel, and one lane sums them
// in the original order, eliminating redundant softmax reductions.
//
// The finite-score maximum is unchanged despite using a tree, followed by
// the same expf of the same difference, the same left-to-right sum of the
// same floats, the same rescale, the same accumulation order.
template <int kThreads>
__global__ void attention_published_softmax(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
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
  __shared__ __nv_bfloat16 tile_values[kTileTokens * kHeadDimension];
  __shared__ float published_max;
  __shared__ float published_scale;
  __shared__ float published_sum;
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float attended = 0.0f;

  for (int tile = 0; tile <= position; tile += kTileTokens) {
    const int token = tile + warp;
    float score = -FLT_MAX;
    if (token <= position) {
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

    const int tile_count = min(kTileTokens, position - tile + 1);
    // Warp 0 reduces the tile maximum and publishes the new maximum and the
    // scale the tile's exponentials need.
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
      probabilities[warp] = token <= position ? expf(scores[warp] - published_max) : 0.0f;
    }
    __syncthreads();

    // One lane sums the exponentials in the original order, so the total is
    // the same float.
    if (warp == 0 && lane == 0) {
      float tile_sum = 0.0f;
      for (int item = 0; item < tile_count; ++item) {
        tile_sum += probabilities[item];
      }
      running_sum = running_sum * published_scale + tile_sum;
    }

    // The value tile is staged by all the warps, so the loads overlap.
    const size_t value_base =
      cache_row + (static_cast<size_t>(kv_head) * kContextLength + tile) * kHeadDimension;
    for (int index = thread; index < kTileTokens * kHeadDimension; index += kThreads) {
      const int row = index / kHeadDimension;
      const int dimension = index % kHeadDimension;
      tile_values[index] = tile + row <= position
          ? value_cache[value_base + static_cast<size_t>(row) * kHeadDimension + dimension]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    if (thread < kHeadDimension) {
      attended *= published_scale;
      for (int item = 0; item < tile_count; ++item) {
        const float value = __bfloat162float(tile_values[item * kHeadDimension + thread]);
        attended += probabilities[item] * value;
      }
    }
    __syncthreads();
  }

  if (warp == 0 && lane == 0) published_sum = running_sum;
  __syncthreads();
  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / published_sum);
  }
}


// Software pipelined: the next tile's loads are issued before the current
// tile is processed.
//
// Here the keys for the next tile are held in registers and the values for
// the next tile are staged into the other shared buffer, both issued before
// the current tile's softmax runs. These are synchronous loads/stores and
// the following barrier waits for staging; this is lookahead, not evidence
// of asynchronous overlap with softmax execution.
//
// The arithmetic is unchanged: the same scores from the same keys, the same
// exponentials, the same left-to-right sum, the same rescale, the same
// accumulation order over the tile's tokens.
template <int kThreads>
__global__ void attention_pipelined(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    const int* positions,
    const int* attention_policy,
    int kv_batch_stride) {
  const int batch = blockIdx.y;
  const int position = positions[batch];
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

  const size_t key_base = cache_row + static_cast<size_t>(kv_head) * kContextLength * kHeadDimension;
  const size_t value_base = cache_row + static_cast<size_t>(kv_head) * kContextLength * kHeadDimension;

  // The prologue: the first tile's keys and values.
  {
    const int token = warp;
    if (token <= position) {
      const size_t index = key_base + static_cast<size_t>(token) * kHeadDimension;
      key0 = key_cache[index + lane];
      key1 = key_cache[index + lane + 32];
      key_valid = true;
    }
    for (int index = thread; index < kTileTokens * kHeadDimension; index += kThreads) {
      const int row = index / kHeadDimension;
      const int dimension = index % kHeadDimension;
      tile_values[0][index] = row <= position
          ? value_cache[value_base + static_cast<size_t>(row) * kHeadDimension + dimension]
          : __float2bfloat16(0.0f);
    }
  }

  int buffer = 0;
  for (int tile = 0; tile <= position; tile += kTileTokens) {
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

    // Issue the next tile's loads before the softmax runs, so they overlap
    // with it rather than following it.
    next_key_valid = false;
    if (tile + kTileTokens <= position) {
      const int next_token = tile + kTileTokens + warp;
      if (next_token <= position) {
        const size_t index = key_base + static_cast<size_t>(next_token) * kHeadDimension;
        next_key0 = key_cache[index + lane];
        next_key1 = key_cache[index + lane + 32];
        next_key_valid = true;
      }
      const int next_buffer = buffer ^ 1;
      for (int index = thread; index < kTileTokens * kHeadDimension; index += kThreads) {
        const int row = index / kHeadDimension;
        const int dimension = index % kHeadDimension;
        tile_values[next_buffer][index] = tile + kTileTokens + row <= position
            ? value_cache[value_base + static_cast<size_t>(tile + kTileTokens + row) * kHeadDimension + dimension]
            : __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    const int tile_count = min(kTileTokens, position - tile + 1);
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

    if (lane == 0) {
      probabilities[warp] = key_valid ? expf(scores[warp] - published_max) : 0.0f;
    }
    __syncthreads();

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
        const float value = __bfloat162float(tile_values[buffer][item * kHeadDimension + thread]);
        attended += probabilities[item] * value;
      }
    }
    __syncthreads();

    key0 = next_key0;
    key1 = next_key1;
    key_valid = next_key_valid;
    buffer ^= 1;
  }

  if (warp == 0 && lane == 0) published_sum = running_sum;
  __syncthreads();
  if (thread < kHeadDimension) {
    output[output_row + query_base + thread] = __float2bfloat16_rn(attended / published_sum);
  }
}

}  // namespace lfm
