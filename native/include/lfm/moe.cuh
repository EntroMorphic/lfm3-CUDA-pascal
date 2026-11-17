#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace lfm {
// Write the entire canonical N=32 input tile, including zero-padded rows, in
// one pass. The host validates 0 < count <= batch <= 32 before launch.
__global__ void moe_gather_canonical(
    const __nv_bfloat16* input, __nv_bfloat16* packed,
    const int* rows, int count, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 32 * hidden) return;
  const int row = i / hidden, col = i % hidden;
  packed[i] = row < count
      ? input[static_cast<size_t>(rows[row]) * hidden + col]
      : __float2bfloat16_rn(0.f);
}

__global__ void moe_swiglu_canonical(
    const __nv_bfloat16* gate, __nv_bfloat16* product, int count, int width) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col >= width) return;
  __nv_bfloat16 value = __float2bfloat16_rn(0.f);
  if (row < count) {
    const size_t base = static_cast<size_t>(row) * 2 * width;
    const float a = __bfloat162float(gate[base + col]);
    const auto activated = __float2bfloat16_rn(a / (1.f + expf(-a)));
    value = __hmul_rn(activated, gate[base + width + col]);
  }
  product[static_cast<size_t>(row) * width + col] = value;
}

// LFM2-MoE routing contract: BF16 sigmoid probabilities, optional FP32 bias
// for selection ONLY, BF16 normalization and scaling. Ties select the lower
// expert index deterministically; torch.topk does not promise a stable tie order.
__global__ void moe_sigmoid(const __nv_bfloat16* logits, __nv_bfloat16* probability,
                             int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) probability[i] = __float2bfloat16_rn(1.0f / (1.0f + expf(-__bfloat162float(logits[i]))));
}

__global__ void moe_select(const __nv_bfloat16* logits, const __nv_bfloat16* probability,
                            const float* bias, int* selected, __nv_bfloat16* weights,
                            int* counts_and_error, int experts, int top_k,
                            bool normalize, float scaling) {
  if (threadIdx.x != 0) return;
  const int row = blockIdx.x;
  const size_t base = static_cast<size_t>(row) * experts;
  const size_t out = static_cast<size_t>(row) * top_k;
  float sum = 0;
  for (int rank = 0; rank < top_k; ++rank) {
    float best = -INFINITY;
    int winner = -1;
    for (int e = 0; e < experts; ++e) {
      bool used = false;
      for (int j = 0; j < rank; ++j) used |= selected[out + j] == e;
      const float p = __bfloat162float(probability[base + e]);
      const float correction = bias ? bias[e] : 0.0f;
      if (!isfinite(__bfloat162float(logits[base + e])) || !isfinite(correction)) {
        atomicExch(counts_and_error + experts, 1);
        continue;
      }
      const float score = p + correction;
      if (!used && (score > best || (score == best && (winner < 0 || e < winner)))) {
        best = score; winner = e;
      }
    }
    if (winner < 0) { atomicExch(counts_and_error + experts, 1); winner = 0; }
    selected[out + rank] = winner;
    weights[out + rank] = probability[base + winner];
    sum += __bfloat162float(weights[out + rank]);
  }
  // Torch's reduction stores its result in BF16, then its scalar add stores
  // another BF16 result. Keep both roundings before the division.
  const float denominator = __bfloat162float(__float2bfloat16_rn(
      __bfloat162float(__float2bfloat16_rn(sum)) + 1.0e-6f));
  for (int rank = 0; rank < top_k; ++rank) {
    auto value = weights[out + rank];
    if (normalize) value = __float2bfloat16_rn(__bfloat162float(value) / denominator);
    value = __float2bfloat16_rn(__bfloat162float(value) * scaling);
    if (!isfinite(__bfloat162float(value))) atomicExch(counts_and_error + experts, 1);
    weights[out + rank] = value;
  }
}

// One thread owns an expert's compact list. No atomics, duplicate destinations
// or nondeterministic packing order: rows are kept in their original order.
__global__ void moe_groups(const int* selected, int* rows, int* ranks,
                            int* counts, int batch, int experts, int top_k) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  int count = 0;
  for (int row = 0; row < batch; ++row) {
    for (int rank = 0; rank < top_k; ++rank) {
      if (selected[static_cast<size_t>(row) * top_k + rank] == e) {
        const size_t index = static_cast<size_t>(e) * batch + count++;
        rows[index] = row; ranks[index] = rank;
        break;  // Also bound writes if a failed route temporarily duplicates IDs.
      }
    }
  }
  counts[e] = count;
}

__global__ void moe_gather(const __nv_bfloat16* source, __nv_bfloat16* packed,
                            const int* rows, int count, int hidden) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<size_t>(count) * hidden)
    packed[i] = source[static_cast<size_t>(rows[i / hidden]) * hidden + i % hidden];
}

__global__ void moe_scatter(const __nv_bfloat16* source, __nv_bfloat16* accumulated,
                             const int* rows, const int* ranks, const __nv_bfloat16* weights,
                             int count, int hidden, int top_k) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(count) * hidden) return;
  const int packed_row = i / hidden, row = rows[packed_row];
  const size_t destination = static_cast<size_t>(row) * hidden + i % hidden;
  const auto weight = weights[static_cast<size_t>(row) * top_k + ranks[packed_row]];
  // Called in ascending expert order on one stream, matching eager HF expert
  // accumulation. Products and additions each round back to BF16.
  // The _rn forms explicitly forbid contraction into one BF16 FMA. Ordinary
  // __hadd(__hmul(...)) can contract and lose the reference's intermediate
  // BF16 product rounding, which can later change selected experts.
  const auto contribution = __hmul_rn(source[i], weight);
  accumulated[destination] = __hadd_rn(accumulated[destination], contribution);
}

__global__ void moe_swiglu(const __nv_bfloat16* packed, __nv_bfloat16* output, int width) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= width) return;
  const size_t row = blockIdx.y;
  const float gate = __bfloat162float(packed[row * 2 * width + d]);
  const auto activated = __float2bfloat16_rn(gate / (1.0f + expf(-gate)));
  output[row * width + d] = __hmul(activated, packed[row * 2 * width + width + d]);
}
}  // namespace lfm
