#pragma once

// Per-row sampler for the batched decode path.
//
// The batched path needs to draw one token per sequence from one set of
// logits. The single-sequence sampler is a serial pipeline over a single
// 65536-entry vocabulary: it adjusts the logits, sorts them descending,
// exponentiates them, sums the eligible prefix, and draws from the resulting
// distribution.
//
// The kernels below are the same pipeline with the vocabulary axis split into
// rows, one row per sequence. The arithmetic inside a row is unchanged:
//
//   * the temperature division happens per row, on the same float value;
//   * the sort is a stable descending sort, so equal logits keep the same
//     lowest-index-first order the single-sequence sort produced;
//   * the exponentials are the same expf of the same difference;
//   * the sum is still accumulated left to right by a single thread, so the
//     running total is the same float, bit for bit;
//   * the nucleus scan and the CDF draw use the same comparisons.
//
// A row therefore reproduces the single-sequence answer exactly when it is
// given the same seed, and rows differ only in the seed they draw from.

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace lfm {

// The same generator the single-sequence sampler draws from.
__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// Adjust one row of logits and write the matching vocabulary indices.
//
// The batched path rejects the penalty options, so the token counts and the
// banned-token mask that the single-sequence sampler applies are not part of
// the batched arithmetic: the only adjustment left is the temperature.
__global__ void prepare_logits_rows(
    const __nv_bfloat16* logits,
    float* adjusted,
    int* indices,
    int count,
    float temperature) {
  const int row = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const size_t offset = static_cast<size_t>(row) * count + index;
  float value = __bfloat162float(logits[offset]);
  if (temperature > 0.0f) value /= temperature;
  adjusted[offset] = value;
  indices[offset] = index;
}

// Exponentiate one row of descending-sorted logits against that row's
// maximum, which is its first element.
__global__ void exponentiate_sorted_logits_rows(
    const float* sorted_logits,
    float* exponentials,
    int count,
    int limit) {
  const int row = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= limit) return;
  const size_t row_base = static_cast<size_t>(row) * count;
  exponentials[row_base + index] = expf(sorted_logits[row_base + index] - sorted_logits[row_base]);
}

// Length of the leading prefix that satisfies the min_p floor, per row. The
// count is an integer, so the atomic accumulation order does not matter.
__global__ void count_eligible_exponentials_rows(
    const float* exponentials,
    int* eligible_out,
    int count,
    int limit,
    float min_p) {
  const int row = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  int eligible = 0;
  if (index < limit && !(exponentials[static_cast<size_t>(row) * count + index] < min_p)) {
    eligible = 1;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    eligible += __shfl_down_sync(0xffffffff, eligible, offset);
  }
  if (threadIdx.x % 32 == 0) atomicAdd(eligible_out + row, eligible);
}

// Sequential left-to-right sum of one row's eligible exponentials.
//
// One thread per row, accumulating in the original order, so the total is the
// same float the single-sequence sampler produced. The early exit is the
// same: a term below half an ulp of the running sum rounds away, and the
// exponentials are non-increasing, so the total is already final.
__global__ void sum_exponentials_rows(
    const float* exponentials,
    const int* eligible_value,
    float* total_out,
    int count) {
  const int row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* row_exponentials = exponentials + static_cast<size_t>(row) * count;
  const int eligible = eligible_value[row];
  float running = 0.0f;
  int index = 0;
  for (; index + 8 <= eligible; index += 8) {
    const float v0 = row_exponentials[index];
    const float v1 = row_exponentials[index + 1];
    const float v2 = row_exponentials[index + 2];
    const float v3 = row_exponentials[index + 3];
    const float v4 = row_exponentials[index + 4];
    const float v5 = row_exponentials[index + 5];
    const float v6 = row_exponentials[index + 6];
    const float v7 = row_exponentials[index + 7];
    running += v0;
    running += v1;
    running += v2;
    running += v3;
    running += v4;
    running += v5;
    running += v6;
    running += v7;
    if (index + 8 < eligible && running > 0.0f) {
      const float next = row_exponentials[index + 8];
      const float ulp = __int_as_float(__float_as_int(running) + 1) - running;
      if (next < 0.5f * ulp) {
        total_out[row] = running;
        return;
      }
    }
  }
  for (; index < eligible; ++index) running += row_exponentials[index];
  total_out[row] = running;
}

// Draw one token per row from the row's distribution.
//
// The nucleus scan and the CDF walk are the same comparisons the
// single-sequence sampler made, and the uniform comes from the row's own
// seed, so a row with the single-sequence seed reproduces that sequence's
// draw exactly.
__global__ void select_sorted_token_rows(
    const float* exponentials,
    const float* total_value,
    const int* eligible_value,
    const int* sorted_indices,
    const uint64_t* row_seeds,
    int* output,
    int count,
    float top_p,
    uint64_t step,
    bool sample) {
  const int row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* row_exponentials = exponentials + static_cast<size_t>(row) * count;
  const int* row_indices = sorted_indices + static_cast<size_t>(row) * count;
  if (!sample) {
    output[row] = row_indices[0];
    return;
  }
  const int eligible = eligible_value[row];
  const float total = total_value[row];
  if (eligible == 0 || !isfinite(total)) {
    output[row] = row_indices[0];
    return;
  }
  int nucleus = eligible;
  float nucleus_total = total;
  if (top_p < 1.0f) {
    float running = 0.0f;
    for (int index = 0; index < eligible; ++index) {
      running += row_exponentials[index];
      if (running / total >= top_p) {
        nucleus = index + 1;
        nucleus_total = running;
        break;
      }
    }
  }
  const uint64_t random_bits = splitmix64(row_seeds[row] + step);
  const float uniform = static_cast<float>((random_bits >> 40) & 0xffffff) / 16777216.0f;
  const float target = uniform * nucleus_total;
  float running = 0.0f;
  for (int index = 0; index < nucleus; ++index) {
    running += row_exponentials[index];
    if (running >= target) {
      output[row] = row_indices[index];
      return;
    }
  }
  output[row] = row_indices[nucleus - 1];
}

}  // namespace lfm
