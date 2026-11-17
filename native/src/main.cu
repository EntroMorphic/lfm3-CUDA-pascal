#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <lfm/decode_tuning.cuh>
#include <lfm/profiling.hpp>
#include <lfm/argmax.cuh>
#include <lfm/attention.cuh>
#include <lfm/compact_attention.cuh>
#include <lfm/batch_sampler.cuh>
#include <lfm/parity_attention.cuh>
#include <lfm/rmsnorm.cuh>
#include <lfm/safetensors.hpp>
#include <lfm/tokenizer.hpp>
#include <lfm/model_config.hpp>
#include <lfm/dynamic_attention.cuh>
#include <lfm/moe.cuh>
#include <lfm/math_attention.cuh>
#include <lfm/checkpoint_path.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int kHiddenSize = 0;
constexpr int kThreads = lfm::kRmsThreads;
int kAttentionLayers = 0, kKvHeads = 0, kHeadDimension = 0;
int kKvWidth = 0, kQkvWidth = 0, kContextLength = 0, kMlpWidth = 0;
constexpr uint64_t kLayerZeroTokenOneReferenceHash = 11630943916332811017ULL;
constexpr uint64_t kFullTokenOneReferenceLogitsHash = 7296020554711068888ULL;
int kQueryHeads = 0, kVocabularySize = 0;
constexpr int kLmHeadBlocks = lfm::kLmHeadBlocks;
constexpr int kTopKBlockSize = 256;
// The batched path derives a distinct seed per row from the base seed. The
// stride is large and odd so two different rows cannot collide.
constexpr uint64_t kRowSeedStride = 0x9e3779b97f4a7c15ULL;
int kTopKBlocks = 0;
constexpr int kMaximumHierarchicalTopK = 64;
// The KV cache is laid out [batch][attention_slot][kv_head][position][dim].
size_t kKvLayerStride = 0, kKvBatchStride = 0;
lfm::ModelConfig model_config;
bool legacy_fixtures = false;
// Opt-in dense math mode shares the existing MoE reference reduction schedule.
// The default dense schedule remains unchanged for legacy output compatibility.
bool reference_math_norms = false;
constexpr size_t kCublasWorkspaceBytes = 32 * 1024 * 1024;
constexpr int kControlPrompt[] = {
    1, 6, 6423, 708, 58393, 849, 4207, 1939, 521, 2724, 44210, 1460,
    43022, 874, 768, 1947, 803, 41158, 540, 7, 708, 6, 64015, 708};
constexpr const char* kControlPromptText =
    "Hypothetically, might reflective recursion be a function of cognition?";

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
  }
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("self-test failed: " + message);
}

void rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
              __nv_bfloat16* output, int rows) {
  LFM_PROFILE_RANGE(profile_norm, "norm/width=" + std::to_string(kHiddenSize) + "/rows=" + std::to_string(rows));
  const auto& s = lfm::runtime_shape;
  if ((model_config.moe || reference_math_norms) && s.hidden >= 128 && s.hidden < 8192 && s.hidden % 4 == 0)
    lfm::rms_norm_vectorized_warp<<<rows, 32>>>(input, weight, output, s.hidden, s.epsilon);
  else if (s.hidden == lfm::kHiddenSize && s.epsilon == lfm::kRmsEpsilon)
    lfm::rms_norm_kernel<<<rows, kThreads>>>(input, weight, output, rows);
  else lfm::rms_norm_dynamic<<<rows, kThreads>>>(input, weight, output, s.hidden, s.epsilon);
}

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t size = 0;

  explicit DeviceBuffer(size_t count) : size(count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::runtime_error("device allocation size overflow");
    check_cuda(cudaMalloc(&data, std::max<size_t>(1, count) * sizeof(T)), "cudaMalloc DeviceBuffer");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// A numerical-contract tile, not a model dimension. cuBLAS may change its
// reduction schedule with N even when benign synthetic vectors round to the
// same BF16 result. Always project fixed-width tiles so request coalescing and
// active-prefix shrinkage cannot select a different GEMM geometry.
constexpr int kProjectionColumns = 32;
struct LinearScratch {
  DeviceBuffer<__nv_bfloat16> input;
  DeviceBuffer<__nv_bfloat16> output;
  LinearScratch() : input(static_cast<size_t>(kProjectionColumns) * std::max({kHiddenSize, kMlpWidth, model_config.expert_width})),
                    output(static_cast<size_t>(kProjectionColumns) *
                           std::max({kVocabularySize, kMlpWidth, 3 * kHiddenSize, kQkvWidth, model_config.num_experts, 2 * model_config.expert_width})) {}
};
std::unique_ptr<LinearScratch> linear_scratch;

__global__ void fill_bf16(__nv_bfloat16* output, size_t count, float scale) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = scale * sinf(static_cast<float>(index % 4096) * 0.00390625f);
    output[index] = __float2bfloat16_rn(value);
  }
}

__global__ void apply_rope_device_position(
    __nv_bfloat16* states, int heads, const int* positions, int batch_stride,
    const float* frequencies, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kHeadDimension = shape.head_dim;
  const int half = kHeadDimension / 2;
  const int position = positions[blockIdx.y];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= heads * half) return;
  const size_t row = static_cast<size_t>(blockIdx.y) * batch_stride;
  const int head_offset = (index / half) * kHeadDimension;
  const int dimension = index % half;
  const float angle = static_cast<float>(position) * frequencies[dimension];
  const __nv_bfloat16 cosine = __float2bfloat16_rn(cosf(angle));
  const __nv_bfloat16 sine = __float2bfloat16_rn(sinf(angle));
  const __nv_bfloat16 first = states[row + head_offset + dimension];
  const __nv_bfloat16 second = states[row + head_offset + dimension + half];
  const __nv_bfloat16 first_cosine = __float2bfloat16_rn(
      __bfloat162float(first) * __bfloat162float(cosine));
  const __nv_bfloat16 second_sine = __float2bfloat16_rn(
      -__bfloat162float(second) * __bfloat162float(sine));
  const __nv_bfloat16 second_cosine = __float2bfloat16_rn(
      __bfloat162float(second) * __bfloat162float(cosine));
  const __nv_bfloat16 first_sine = __float2bfloat16_rn(
      __bfloat162float(first) * __bfloat162float(sine));
  states[row + head_offset + dimension] = __float2bfloat16_rn(
      __bfloat162float(first_cosine) + __bfloat162float(second_sine));
  states[row + head_offset + dimension + half] = __float2bfloat16_rn(
      __bfloat162float(second_cosine) + __bfloat162float(first_sine));
}

__global__ void rope_coefficients(__nv_bfloat16* output, int position,
                                  const float* frequencies, int half) {
  const int index = threadIdx.x;
  if (index >= half) return;
  const float angle = static_cast<float>(position) * frequencies[index];
  output[index] = __float2bfloat16_rn(cosf(angle));
  output[index + half] = __float2bfloat16_rn(sinf(angle));
}

__global__ void gather_embedding(
    const __nv_bfloat16* embedding,
    const int* tokens,
    __nv_bfloat16* output, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kHiddenSize = shape.hidden;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kHiddenSize) {
    const int token = tokens[blockIdx.y];
    output[static_cast<size_t>(blockIdx.y) * kHiddenSize + index] =
        embedding[static_cast<size_t>(token) * kHiddenSize + index];
  }
}

__global__ void add_residual(
    __nv_bfloat16* output,
    const __nv_bfloat16* update,
    const __nv_bfloat16* residual,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const size_t offset = static_cast<size_t>(blockIdx.y) * count + index;
    output[offset] = __hadd(update[offset], residual[offset]);
  }
}

__global__ void conv_decode(
    const __nv_bfloat16* projected,
    const __nv_bfloat16* kernel,
    __nv_bfloat16* state,
    __nv_bfloat16* output,
    int layer, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kHiddenSize = shape.hidden;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kHiddenSize) return;
  const size_t row = static_cast<size_t>(blockIdx.y);
  const __nv_bfloat16 b = projected[row * 3 * kHiddenSize + index];
  const __nv_bfloat16 c = projected[row * 3 * kHiddenSize + kHiddenSize + index];
  const __nv_bfloat16 x = projected[row * 3 * kHiddenSize + 2 * kHiddenSize + index];
  const __nv_bfloat16 gated = __hmul(b, x);
  const int width = shape.conv_width;
  auto* channel_state = state + ((row * shape.layers + layer) * kHiddenSize + index) * width;
  const auto* channel_kernel = kernel + static_cast<size_t>(index) * width;
  float convolved = width > 1 ? __bfloat162float(channel_state[1]) * __bfloat162float(channel_kernel[0])
                              : __bfloat162float(gated) * __bfloat162float(channel_kernel[0]);
  for (int item = 1; item < width; ++item) {
    const auto value = item + 1 == width ? gated : channel_state[item + 1];
    convolved += __bfloat162float(value) * __bfloat162float(channel_kernel[item]);
  }
  for (int item = 0; item + 1 < width; ++item) channel_state[item] = channel_state[item + 1];
  channel_state[width - 1] = gated;
  output[row * kHiddenSize + index] = __hmul(c, __float2bfloat16_rn(convolved));
}

__global__ void swiglu(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* value,
    __nv_bfloat16* output,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const size_t offset = static_cast<size_t>(blockIdx.y) * count + index;
  const float x = __bfloat162float(gate[offset]);
  const __nv_bfloat16 activated = __float2bfloat16_rn(x / (1.0f + expf(-x)));
  output[offset] = __hmul(activated, value[offset]);
}

__global__ void head_rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int heads,
    int batch_stride, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kHeadDimension = shape.head_dim;
  const int lane = threadIdx.x;
  const int head = blockIdx.y * blockDim.y + threadIdx.y;
  if (head >= heads) return;
  const size_t row = static_cast<size_t>(blockIdx.x) * batch_stride;
  const int base = head * kHeadDimension;
  float sum = 0.0f;
  // Keep the established 64-wide schedule, and use a generic warp reduction
  // for other even head dimensions.
  if (kHeadDimension == 64) {
    const float first = __bfloat162float(input[row + base + lane]);
    const float second = __bfloat162float(input[row + base + lane + 32]);
    sum = first * first + second * second;
  } else for (int d = lane; d < kHeadDimension; d += 32) {
    const float v = __bfloat162float(input[row + base + d]);
    sum += v * v;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float scale = __shfl_sync(0xffffffff, rsqrtf(sum / kHeadDimension + shape.epsilon), 0);
  for (int d = lane; d < kHeadDimension; d += 32) {
    const float v = __bfloat162float(input[row + base + d]);
    output[row + base + d] = __hmul(__float2bfloat16_rn(v * scale), weight[d]);
  }
}

__global__ void set_decode_input(int* token, int* position, int token_value, int position_value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *token = token_value;
    *position = position_value;
  }
}

__global__ void increment_position(int* position) {
  if (threadIdx.x == 0 && blockIdx.x == 0) ++*position;
}

__global__ void increment_position_batch(int* positions, int count, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kContextLength = shape.context;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  // The KV cache holds positions [0, 32768), so a position never runs past
  // the last slot. A sequence that reached the end of its context budget
  // simply stays where it is.
  if (index < count && positions[index] + 1 < kContextLength) ++positions[index];
}

__global__ void advance_selected_tokens(
    int* input_tokens, const int* selected, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) input_tokens[index] = selected[index];
}

// Records one generated token per sequence and reports how many sequences are
// still generating. Each sequence owns its generated buffer and its own stop
// state; the count lets the host leave the loop once every sequence has
// finished.
__global__ void record_token_and_check_stop_batch(
    const int* input_tokens,
    int* generated,
    int* generated_count,
    int* stop_flag,
    const int* stop_tokens,
    int stop_count,
    int* active_count,
    int count,
    const int* context_limits, lfm::RuntimeShape shape = lfm::runtime_shape,
    int* stop_snapshot = nullptr) {
  const int kContextLength = shape.context;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  if (stop_snapshot) stop_snapshot[index] = stop_flag[index];
  if (stop_flag[index]) return;
  const int token = input_tokens[index];
  const int slot = generated_count[index]++;
  generated[static_cast<size_t>(index) * kContextLength + slot] = token;
  int stop = 0;
  for (int i = 0; i < stop_count; ++i) stop |= token == stop_tokens[i];
  // A sequence stops once it has generated its whole context budget, which is
  // the same place the single-sequence path stops.
  if (generated_count[index] >= context_limits[index]) stop = 1;
  if (stop_snapshot) stop_snapshot[index] = stop;
  if (stop) {
    stop_flag[index] = 1;
  } else {
    atomicAdd(active_count, 1);
  }
}

__global__ void record_token_and_check_stop(
    const int* token,
    int* generated,
    int* generated_count,
    int* stop_flag,
    const int* stop_tokens,
    int stop_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const int index = (*generated_count)++;
  generated[index] = *token;
  int stop = 0;
  for (int i = 0; i < stop_count; ++i) stop |= *token == stop_tokens[i];
  *stop_flag = stop;
}

__global__ void record_token_and_set_condition(
    cudaGraphConditionalHandle handle,
    const int* token,
    const int* position,
    int* generated,
    int* generated_count,
    int* stop_flag,
    const int* stop_tokens,
    int stop_count, lfm::RuntimeShape shape = lfm::runtime_shape) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const int index = (*generated_count)++;
  generated[index] = *token;
  int stop = 0;
  for (int i = 0; i < stop_count; ++i) stop |= *token == stop_tokens[i];
  *stop_flag = stop;
  const unsigned int continue_loop = !stop && *position < shape.context - 1;
  cudaGraphSetConditional(handle, continue_loop);
}


__global__ void lm_head_argmax(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    float* block_values,
    int* block_indices, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kHiddenSize = shape.hidden;
  const int kVocabularySize = shape.vocabulary;
  constexpr int kWarpsPerBlock = 8;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int global_warp = blockIdx.x * kWarpsPerBlock + warp;
  const int warp_stride = gridDim.x * kWarpsPerBlock;
  float best_value = -FLT_MAX;
  int best_index = 0;
  for (int row = global_warp; row < kVocabularySize; row += warp_stride) {
    float sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * kHiddenSize;
    for (int column = lane; column < kHiddenSize; column += 32) {
      sum += __bfloat162float(input[column]) *
             __bfloat162float(weight[row_offset + column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      const __nv_bfloat16 logit = __float2bfloat16_rn(sum);
      const float value = __bfloat162float(logit);
      if (value > best_value || (value == best_value && row < best_index)) {
        best_value = value;
        best_index = row;
      }
    }
  }
  __shared__ float warp_values[kWarpsPerBlock];
  __shared__ int warp_indices[kWarpsPerBlock];
  if (lane == 0) {
    warp_values[warp] = best_value;
    warp_indices[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    float value = lane < kWarpsPerBlock ? warp_values[lane] : -FLT_MAX;
    int index = lane < kWarpsPerBlock ? warp_indices[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffff, value, offset);
      const int other_index = __shfl_down_sync(0xffffffff, index, offset);
      if (other_value > value || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
      }
    }
    if (lane == 0) {
      block_values[blockIdx.x] = value;
      block_indices[blockIdx.x] = index;
    }
  }
}

__global__ void block_topk(
    const float* values,
    float* candidates,
    int* candidate_indices,
    int top_k, lfm::RuntimeShape shape = lfm::runtime_shape) {
  using Sort = cub::BlockRadixSort<float, kTopKBlockSize, 1, int>;
  __shared__ typename Sort::TempStorage storage;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float value[1] = {index < shape.vocabulary ? values[index] : -INFINITY};
  int token[1] = {index};
  Sort(storage).SortDescending(value, token);
  if (threadIdx.x < top_k) {
    const int output = blockIdx.x * top_k + threadIdx.x;
    candidates[output] = value[0];
    candidate_indices[output] = token[0];
  }
}

__global__ void finalize_lm_head_argmax(
    const float* block_values,
    const int* block_indices,
    int* output) {
  __shared__ float values[256];
  __shared__ int indices[256];
  float best_value = -FLT_MAX;
  int best_index = 0;
  for (int index = threadIdx.x; index < kLmHeadBlocks; index += blockDim.x) {
    const float value = block_values[index];
    const int token = block_indices[index];
    if (value > best_value || (value == best_value && token < best_index)) {
      best_value = value;
      best_index = token;
    }
  }
  values[threadIdx.x] = best_value;
  indices[threadIdx.x] = best_index;
  for (int offset = 128; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      const float other_value = values[threadIdx.x + offset];
      const int other_index = indices[threadIdx.x + offset];
      if (other_value > values[threadIdx.x] ||
          (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_index;
      }
    }
  }
  if (threadIdx.x == 0) output[0] = indices[0];
}

__global__ void prepare_logits(
    const __nv_bfloat16* logits,
    const int* token_counts,
    const uint8_t* banned,
    float* adjusted,
    int* indices,
    float temperature,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kVocabularySize = shape.vocabulary;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kVocabularySize) return;
  float value = __bfloat162float(logits[index]);
  const int count = token_counts[index];
  if (count > 0 && repetition_penalty != 1.0f) {
    value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
  }
  if (count > 0) value -= presence_penalty;
  value -= frequency_penalty * static_cast<float>(count);
  if (temperature > 0.0f) value /= temperature;
  if (banned[index]) value = -FLT_MAX;
  adjusted[index] = value;
  indices[index] = index;
}

__global__ void mark_no_repeat_ngram(
    uint8_t* mask,
    const int* history,
    const int* history_length,
    int ngram_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || ngram_size <= 0) return;
  const int length = *history_length;
  if (ngram_size == 1) {
    for (int index = 0; index < length; ++index) mask[history[index]] = 1;
    return;
  }
  if (length + 1 < ngram_size) return;
  const int prefix = ngram_size - 1;
  const int suffix_start = length - prefix;
  for (int start = 0; start + ngram_size <= length; ++start) {
    bool match = true;
    for (int offset = 0; offset < prefix; ++offset) {
      match &= history[start + offset] == history[suffix_start + offset];
    }
    if (match) mask[history[start + prefix]] = 1;
  }
}

__global__ void increment_selected_count(int* counts, const int* token, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kVocabularySize = shape.vocabulary;
  if (threadIdx.x == 0 && blockIdx.x == 0 && *token >= 0 && *token < kVocabularySize) {
    ++counts[*token];
  }
}

__global__ void append_selected_token(
    int* history,
    int* history_length,
    const int* token) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    history[(*history_length)++] = *token;
  }
}


// Exponentiate a descending-sorted logit vector in parallel. The reference
// path recomputed expf(logit - max) inside a single-threaded scan, which cost
// about 8.6 ms per token at the full 65536-entry vocabulary. Computing the
// same exponentials in parallel keeps the sampling arithmetic bit-identical:
// every value has the same input, and the caller still accumulates them in the
// original sequential order.
__global__ void exponentiate_sorted_logits(
    const float* sorted_logits,
    float* exponentials,
    int limit) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= limit) return;
  const float maximum = sorted_logits[0];
  exponentials[index] = expf(sorted_logits[index] - maximum);
}

// Length of the leading prefix that satisfies the min_p floor. The
// exponentials are non-increasing because the logits are sorted descending, so
// counting the entries at or above the floor yields the same length the
// original break-on-first-below scan produced.
__global__ void count_eligible_exponentials(
    const float* exponentials,
    int limit,
    float min_p,
    int* eligible_out) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  // Negated comparison, matching the original break-on-below scan: a NaN
  // exponent compares false against the floor and is therefore retained.
  if (index < limit && !(exponentials[index] < min_p)) count = 1;
  for (int offset = 16; offset > 0; offset >>= 1) {
    count += __shfl_down_sync(0xffffffff, count, offset);
  }
  if (threadIdx.x % 32 == 0) atomicAdd(eligible_out, count);
}

// Sequential left-to-right sum of the eligible exponentials.
//
// The accumulation order is deliberately the original sequential order, so the
// total and every prefix the search compares against are the same
// floating-point values the previous implementation produced.
//
// Two properties keep this affordable without changing a single bit:
//   * the loads are issued eight at a time, so the serial cost is the
//     floating-point add chain rather than memory latency; and
//   * the scan stops once the next term is provably too small to change the
//     running sum. A term below half an ulp of the sum rounds away, and the
//     exponentials are non-increasing, so every later term is smaller still
//     and the total is already final.
__global__ void sum_exponentials(
    const float* exponentials,
    const int* eligible_value,
    float* total_out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const int eligible = *eligible_value;
  float running = 0.0f;
  int index = 0;
  for (; index + 8 <= eligible; index += 8) {
    const float v0 = exponentials[index];
    const float v1 = exponentials[index + 1];
    const float v2 = exponentials[index + 2];
    const float v3 = exponentials[index + 3];
    const float v4 = exponentials[index + 4];
    const float v5 = exponentials[index + 5];
    const float v6 = exponentials[index + 6];
    const float v7 = exponentials[index + 7];
    running += v0;
    running += v1;
    running += v2;
    running += v3;
    running += v4;
    running += v5;
    running += v6;
    running += v7;
    if (index + 8 < eligible && running > 0.0f) {
      const float next = exponentials[index + 8];
      const float ulp =
          __int_as_float(__float_as_int(running) + 1) - running;
      if (next < 0.5f * ulp) {
        *total_out = running;
        return;
      }
    }
  }
  for (; index < eligible; ++index) running += exponentials[index];
  *total_out = running;
}

__global__ void select_sorted_token(
    const float* exponentials,
    const float* total_value,
    const int* eligible_value,
    const int* sorted_indices,
    int* output,
    float top_p,
    uint64_t seed,
    uint64_t step,
    bool sample) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (!sample) {
    output[0] = sorted_indices[0];
    return;
  }
  const int eligible = *eligible_value;
  const float total = *total_value;
  if (eligible == 0 || !isfinite(total)) {
    output[0] = sorted_indices[0];
    return;
  }
  int nucleus = eligible;
  float nucleus_total = total;
  if (top_p < 1.0f) {
    float running = 0.0f;
    for (int index = 0; index < eligible; ++index) {
      running += exponentials[index];
      if (running / total >= top_p) {
        nucleus = index + 1;
        nucleus_total = running;
        break;
      }
    }
  }
  const uint64_t random_bits = lfm::splitmix64(seed + step);
  const float uniform = static_cast<float>((random_bits >> 40) & 0xffffff) / 16777216.0f;
  const float target = uniform * nucleus_total;
  float running = 0.0f;
  for (int index = 0; index < nucleus; ++index) {
    running += exponentials[index];
    if (running >= target) {
      output[0] = sorted_indices[index];
      return;
    }
  }
  output[0] = sorted_indices[nucleus - 1];
}

__global__ void write_kv(
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int position, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kKvWidth = shape.kv_heads * shape.head_dim;
  const int kHeadDimension = shape.head_dim;
  const int kContextLength = shape.context;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kKvWidth) return;
  const int head = index / kHeadDimension;
  const int dimension = index % kHeadDimension;
  const size_t cache_index =
      (static_cast<size_t>(head) * kContextLength + position) * kHeadDimension + dimension;
  key_cache[cache_index] = key[index];
  value_cache[cache_index] = value[index];
}

__global__ void write_kv_device_position(
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const int* positions,
    int kv_batch_stride, lfm::RuntimeShape shape = lfm::runtime_shape) {
  const int kKvWidth = shape.kv_heads * shape.head_dim;
  const int kHeadDimension = shape.head_dim;
  const int kContextLength = shape.context;
  const int kQkvWidth = shape.hidden + 2 * kKvWidth;
  const int batch = blockIdx.y;
  const int position = positions[batch];
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kKvWidth || position < 0 || position >= kContextLength) return;
  const int head = index / kHeadDimension;
  const int dimension = index % kHeadDimension;
  const size_t cache_index =
      static_cast<size_t>(batch) * kv_batch_stride +
      (static_cast<size_t>(head) * kContextLength + position) * kHeadDimension + dimension;
  key_cache[cache_index] = key[static_cast<size_t>(batch) * kQkvWidth + index];
  value_cache[cache_index] = value[static_cast<size_t>(batch) * kQkvWidth + index];
}

void write_kv_checked(
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int layer,
    int position) {
  if (layer < 0 || layer >= kAttentionLayers) {
    throw std::out_of_range("KV layer must be in [0, 6)");
  }
  if (position < 0 || position >= kContextLength) {
    throw std::out_of_range("KV position must be in [0, 32768)");
  }
  const size_t layer_stride = static_cast<size_t>(kKvHeads) * kContextLength * kHeadDimension;
  write_kv<<<(kKvWidth + 255) / 256, 256>>>(
      key_cache + layer * layer_stride,
      value_cache + layer * layer_stride,
      key,
      value,
      position);
  check_cuda(cudaGetLastError(), "write_kv launch");
}

void linear_bf16(
    cublasHandle_t handle,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int input_size,
    int output_size,
    int batch = 1, int output_stride = 0) {
  LFM_PROFILE_RANGE(profile_linear, "linear/in=" + std::to_string(input_size) +
      "/out=" + std::to_string(output_size) + "/rows=" + std::to_string(batch));
  if (!output_stride) output_stride = output_size;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // A row-major [output, input] weight is a column-major [input, output]
  // matrix. Transpose that view to produce y = W x.
  //
  // Logical batch rows ride on the n axis. Fixed-width padded tiles keep
  // the same GEMM geometry for an isolated request and a coalesced request;
  // each tile shares the weight read across its real rows.
  if (!linear_scratch) throw std::runtime_error("linear workspace not initialized");
  if (!input || !weight || !output || input_size <= 0 || output_size <= 0 ||
      batch <= 0 || batch > 65535 || output_stride < output_size ||
      static_cast<size_t>(input_size) > linear_scratch->input.size / kProjectionColumns ||
      static_cast<size_t>(output_size) > linear_scratch->output.size / kProjectionColumns)
    throw std::invalid_argument("invalid projection dimensions, pointers or workspace capacity");
  const size_t input_bytes = lfm::checked_product({static_cast<size_t>(batch), static_cast<size_t>(input_size), sizeof(__nv_bfloat16)});
  const size_t output_elements = lfm::checked_product({static_cast<size_t>(batch - 1), static_cast<size_t>(output_stride)}) + output_size;
  const size_t output_bytes = lfm::checked_product({output_elements, sizeof(__nv_bfloat16)});
  const size_t weight_bytes = lfm::checked_product({static_cast<size_t>(input_size), static_cast<size_t>(output_size), sizeof(__nv_bfloat16)});
  auto overlaps = [](const void* a, size_t an, const void* b, size_t bn) {
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    if (an > std::numeric_limits<uintptr_t>::max() - aa || bn > std::numeric_limits<uintptr_t>::max() - bb)
      throw std::invalid_argument("projection pointer span overflow");
    return aa < bb + bn && bb < aa + an;
  };
  // cuBLAS GEMM is not an in-place operator. In the direct path output poison
  // would otherwise destroy aliased inputs before GEMM; staged multi-tile calls
  // could overwrite future inputs too. Reject all overlap before any writes.
  if (overlaps(input, input_bytes, output, output_bytes) || overlaps(weight, weight_bytes, output, output_bytes))
    throw std::invalid_argument("projection output overlaps input or weights");
  for (int first = 0; first < batch; first += kProjectionColumns) {
    const int columns = std::min(kProjectionColumns, batch - first);
    auto* staged_input = linear_scratch->input.data;
    auto* staged_output = linear_scratch->output.data;
    // A full tile already has the canonical GEMM width: use its storage
    // directly instead of padding/copying it through the shared workspace.
    // Keep partial tiles staged so isolated and coalesced requests retain N=32.
    const auto* tile_input = input + static_cast<size_t>(first) * input_size;
    auto* tile_output = output + static_cast<size_t>(first) * output_size;
    // Match cudaMalloc workspace alignment as well as shape: vendor kernel
    // selection can depend on alignment (e.g. later tiles of an odd vocabulary).
    const bool direct = columns == kProjectionColumns && output_stride == output_size &&
        reinterpret_cast<uintptr_t>(tile_input) % 256 == 0 &&
        reinterpret_cast<uintptr_t>(tile_output) % 256 == 0;
    const auto* gemm_input = direct ? tile_input : staged_input;
    auto* gemm_output = direct ? tile_output : staged_output;
    if (!direct) check_cuda(cudaMemsetAsync(staged_input, 0, static_cast<size_t>(kProjectionColumns) * input_size * sizeof(__nv_bfloat16), cudaStreamPerThread), "pad projection input");
    // Own the initialization contract for library scratch. Poison rather than
    // zero so omitted output stores fail the finite/reference checks. This also
    // avoids initcheck's documented inability to track some Blackwell tensor
    // stores; it does not certify cuBLAS's internal write tracking.
    check_cuda(cudaMemsetAsync(gemm_output, 0xff, static_cast<size_t>(kProjectionColumns) * output_size * sizeof(__nv_bfloat16), cudaStreamPerThread), "poison projection output");
    if (!direct) check_cuda(cudaMemcpyAsync(staged_input, input + static_cast<size_t>(first) * input_size,
                               static_cast<size_t>(columns) * input_size * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread), "stage projection input");
    check_cublas(
      cublasGemmEx(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          output_size,
          kProjectionColumns,
          input_size,
          &alpha,
          weight,
          CUDA_R_16BF,
          input_size,
          gemm_input,
          CUDA_R_16BF,
          input_size,
          &beta,
          gemm_output,
          CUDA_R_16BF,
          output_size,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx BF16 linear");
    if (direct) continue;
    if (output_stride == output_size) check_cuda(cudaMemcpyAsync(output + static_cast<size_t>(first) * output_size, staged_output,
                               static_cast<size_t>(columns) * output_size * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread), "unstage projection output");
    else check_cuda(cudaMemcpy2DAsync(output + static_cast<size_t>(first) * output_stride,
          static_cast<size_t>(output_stride) * sizeof(__nv_bfloat16), staged_output,
          static_cast<size_t>(output_size) * sizeof(__nv_bfloat16),
          static_cast<size_t>(output_size) * sizeof(__nv_bfloat16), columns,
          cudaMemcpyDeviceToDevice, cudaStreamPerThread), "unstage strided projection output");
  }
}

void test_rmsnorm(
    const std::vector<__nv_bfloat16>& input,
    const std::vector<__nv_bfloat16>& weight,
    const __nv_bfloat16* device_output) {
  std::vector<__nv_bfloat16> actual(kHiddenSize);
  check_cuda(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost), "copy RMSNorm test output");

  if (kHiddenSize != lfm::kHiddenSize || lfm::runtime_shape.epsilon != lfm::kRmsEpsilon) {
    double sum = 0;
    for (auto value : input) { const double v = __bfloat162float(value); sum += v * v; }
    const float scale = 1.0 / std::sqrt(sum / kHiddenSize + lfm::runtime_shape.epsilon);
    for (int i = 0; i < kHiddenSize; ++i) {
      const auto normalized = __float2bfloat16_rn(__bfloat162float(input[i]) * scale);
      const float expected = __bfloat162float(__hmul(normalized, weight[i]));
      const float observed = __bfloat162float(actual[i]);
      require(std::isfinite(observed) && std::abs(observed - expected) <= 0.015625f,
              "generic RMSNorm differs from CPU reference");
    }
    return;
  }

  float partials[kThreads];
  for (int thread = 0; thread < kThreads; ++thread) {
    const int base = thread * 4;
    float values[4];
    for (int i = 0; i < 4; ++i) {
      const float value = __bfloat162float(input[base + i]);
      values[i] = value * value;
    }
    partials[thread] = ((values[0] + values[1]) + values[2]) + values[3];
  }
  for (int offset = 128; offset >= 32; offset >>= 1) {
    for (int thread = 0; thread < offset; ++thread) partials[thread] += partials[thread + offset];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    float previous[32];
    for (int lane = 0; lane < 32; ++lane) previous[lane] = partials[lane];
    for (int lane = 0; lane + offset < 32; ++lane) partials[lane] += previous[lane + offset];
  }
  const float scale = 1.0f / std::sqrt(partials[0] / kHiddenSize + lfm::kRmsEpsilon);
  for (int i = 0; i < kHiddenSize; ++i) {
    const auto normalized = __float2bfloat16_rn(__bfloat162float(input[i]) * scale);
    const auto expected = __float2bfloat16_rn(
        __bfloat162float(normalized) * __bfloat162float(weight[i]));
    require(__bfloat162float(actual[i]) == __bfloat162float(expected), "RMSNorm output mismatch at " + std::to_string(i));
  }
}

void test_linear(
    cublasHandle_t handle,
    const std::vector<__nv_bfloat16>& input,
    const __nv_bfloat16* device_input,
    __nv_bfloat16* device_weight,
    __nv_bfloat16* device_output) {
  std::vector<__nv_bfloat16> identity(static_cast<size_t>(kHiddenSize) * kHiddenSize);
  for (int i = 0; i < kHiddenSize; ++i) identity[static_cast<size_t>(i) * kHiddenSize + i] = __float2bfloat16(1.0f);
  check_cuda(cudaMemcpy(device_weight, identity.data(), identity.size() * sizeof(identity[0]), cudaMemcpyHostToDevice), "copy identity weight");
  linear_bf16(handle, device_input, device_weight, device_output, kHiddenSize, kHiddenSize);
  std::vector<__nv_bfloat16> output(kHiddenSize);
  check_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost), "copy identity output");
  for (int i = 0; i < kHiddenSize; ++i) {
    require(__bfloat162float(output[i]) == __bfloat162float(input[i]), "identity projection mismatch at " + std::to_string(i));
  }
}

float test_linear_reference(
    cublasHandle_t handle,
    const std::vector<__nv_bfloat16>& input,
    const __nv_bfloat16* device_input,
    const __nv_bfloat16* device_weight,
    __nv_bfloat16* device_output) {
  linear_bf16(handle, device_input, device_weight, device_output, kHiddenSize, kHiddenSize);
  std::vector<__nv_bfloat16> weights(static_cast<size_t>(kHiddenSize) * kHiddenSize);
  std::vector<__nv_bfloat16> output(kHiddenSize);
  check_cuda(cudaMemcpy(weights.data(), device_weight, weights.size() * sizeof(weights[0]), cudaMemcpyDeviceToHost), "copy projection weights");
  check_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost), "copy projection output");
  float max_absolute_error = 0.0f;
  for (int row = 0; row < kHiddenSize; ++row) {
    float expected = 0.0f;
    for (int column = 0; column < kHiddenSize; ++column) {
      expected += __bfloat162float(weights[static_cast<size_t>(row) * kHiddenSize + column]) *
                  __bfloat162float(input[column]);
    }
    const float expected_bf16 = __bfloat162float(__float2bfloat16_rn(expected));
    max_absolute_error = std::max(max_absolute_error, std::abs(expected_bf16 - __bfloat162float(output[row])));
  }
  require(max_absolute_error <= 0.015625f, "projection error exceeds one BF16 step");
  return max_absolute_error;
}

uint64_t fnv1a_bf16(const std::vector<__nv_bfloat16>& values) {
  uint64_t hash = 1469598103934665603ULL;
  for (const auto& value : values) {
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    hash ^= static_cast<uint8_t>(bits & 0xff);
    hash *= 1099511628211ULL;
    hash ^= static_cast<uint8_t>(bits >> 8);
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t fnv1a_tokens(const std::vector<int>& values) {
  uint64_t hash = 1469598103934665603ULL;
  for (const int value : values) {
    const uint32_t bits = static_cast<uint32_t>(value);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint8_t>(bits >> (byte * 8));
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

struct LayerBuffers {
  const int batch;
  // How many rows of the batch the kernels process. The batched path sorts
  // its sequences by prompt length, so the sequences still inside their prompt
  // are always a prefix of the batch, and the prefix shrinks as the shorter
  // prompts finish. Every kernel indexes its row from blockIdx, so a prefix
  // needs no row offset: only the grid size and the row count change.
  int active;
  // Only batched generation attaches this mask; prefill sees initialized zeros.
  const int* stopped_rows = nullptr;
  std::vector<int> host_stopped_rows;
  DeviceBuffer<int> input_token;
  DeviceBuffer<int> position;
  DeviceBuffer<int> attention_policy{3};
  DeviceBuffer<__nv_bfloat16> hidden;
  DeviceBuffer<__nv_bfloat16> normalized;
  DeviceBuffer<__nv_bfloat16> projection3;
  DeviceBuffer<__nv_bfloat16> operator_output;
  DeviceBuffer<__nv_bfloat16> convolution_states;
  DeviceBuffer<__nv_bfloat16> mlp_gate;
  DeviceBuffer<__nv_bfloat16> mlp_value;
  DeviceBuffer<__nv_bfloat16> mlp_product;
  DeviceBuffer<__nv_bfloat16> mlp_output;
  DeviceBuffer<__nv_bfloat16> qkv;
  DeviceBuffer<__nv_bfloat16> logits;
  DeviceBuffer<int> selected_token;
  DeviceBuffer<__nv_bfloat16> router_logits, router_probability, route_weights;
  DeviceBuffer<int> routes, expert_rows, expert_ranks, expert_counts;
  DeviceBuffer<__nv_bfloat16> expert_input, expert_product, expert_output;
  DeviceBuffer<float> lm_head_block_values{kLmHeadBlocks};
  DeviceBuffer<int> lm_head_block_indices{kLmHeadBlocks};

  explicit LayerBuffers(int batch_size = 1)
      : batch(batch_size),
        active(batch_size),
        host_stopped_rows(batch_size, 0),
        input_token(batch_size),
        position(batch_size),
        hidden(static_cast<size_t>(batch_size) * kHiddenSize),
        normalized(static_cast<size_t>(batch_size) * kHiddenSize),
        projection3(static_cast<size_t>(batch_size) * 3 * kHiddenSize),
        operator_output(static_cast<size_t>(batch_size) * kHiddenSize),
        convolution_states(static_cast<size_t>(batch_size) * model_config.shape.layers * model_config.shape.conv_width * kHiddenSize),
        mlp_gate(static_cast<size_t>(batch_size) * std::max(kMlpWidth, 2 * model_config.expert_width)),
        mlp_value(static_cast<size_t>(batch_size) * kMlpWidth),
        mlp_product(static_cast<size_t>(batch_size) * kMlpWidth),
        mlp_output(static_cast<size_t>(batch_size) * kHiddenSize),
        qkv(static_cast<size_t>(batch_size) * kQkvWidth),
        logits(static_cast<size_t>(batch_size) * kVocabularySize),
        selected_token(batch_size),
        router_logits(static_cast<size_t>(batch_size) * model_config.num_experts),
        router_probability(router_logits.size),
        route_weights(static_cast<size_t>(batch_size) * model_config.experts_per_token),
        routes(route_weights.size), expert_rows(router_logits.size), expert_ranks(router_logits.size),
        expert_counts(model_config.num_experts + 1),
        expert_input(model_config.moe ? static_cast<size_t>(batch_size) * kHiddenSize : 0),
        expert_product(static_cast<size_t>(batch_size) * model_config.expert_width),
        expert_output(expert_input.size) {
    check_cuda(cudaMemset(attention_policy.data, 0, attention_policy.size * sizeof(int)),
               "initialize attention policy");
  }
};

struct DecodeState {
  DeviceBuffer<int> generated{static_cast<size_t>(kContextLength)};
  DeviceBuffer<int> generated_count{1};
  DeviceBuffer<int> stop_flag{1};
  DeviceBuffer<int> stop_tokens{static_cast<size_t>(kVocabularySize)};
};

struct LayerWeights {
  const __nv_bfloat16* operator_norm = nullptr;
  const __nv_bfloat16* ffn_norm = nullptr;
  const __nv_bfloat16* w1 = nullptr;
  const __nv_bfloat16* w2 = nullptr;
  const __nv_bfloat16* w3 = nullptr;
  const __nv_bfloat16* conv_kernel = nullptr;
  const __nv_bfloat16* conv_in = nullptr;
  const __nv_bfloat16* conv_out = nullptr;
  const __nv_bfloat16* qkv = nullptr;
  const __nv_bfloat16* attention_out = nullptr;
  const __nv_bfloat16* query_norm = nullptr;
  const __nv_bfloat16* key_norm = nullptr;
  bool attention = false;
  bool moe = false;
  const __nv_bfloat16* router = nullptr;
  const float* expert_bias = nullptr;
  struct Expert {
    const __nv_bfloat16* w1;
    const __nv_bfloat16* w2;
    const __nv_bfloat16* w3;
  };
  std::vector<Expert> experts;
};

struct ModelWeights {
  const lfm::Safetensors* source_model = nullptr;
  const __nv_bfloat16* embedding = nullptr;
  const __nv_bfloat16* final_norm = nullptr;
  const __nv_bfloat16* lm_head = nullptr;
  DeviceBuffer<float> frequencies{static_cast<size_t>(kHeadDimension / 2)};
  DeviceBuffer<__nv_bfloat16> packed_qkv{
      static_cast<size_t>(kAttentionLayers) * kQkvWidth * kHiddenSize};
  std::vector<LayerWeights> layers;
  std::unique_ptr<lfm::MathAttentionDynamic> math_attention;
  std::unique_ptr<DeviceBuffer<__nv_bfloat16>> packed_experts;

  explicit ModelWeights(const lfm::Safetensors& model, int attention_horizon = 0, bool dynamic_math = false) : source_model(&model), layers(model_config.shape.layers) {
    if ((model_config.moe || dynamic_math) && kAttentionLayers)
      math_attention = std::make_unique<lfm::MathAttentionDynamic>(lfm::runtime_shape, attention_horizon);
    embedding = model.at("model.embed_tokens.weight").device_data;
    final_norm = model.at("model.embedding_norm.weight").device_data;
    lm_head = model_config.tied ? embedding : model.at("lm_head.weight").device_data;
    std::vector<float> inverse(kHeadDimension / 2);
    for (int i = 0; i < kHeadDimension / 2; ++i)
      inverse[i] = 1.0f / std::pow(model_config.shape.rope_theta, static_cast<float>(2 * i) / kHeadDimension);
    check_cuda(cudaMemcpy(frequencies.data, inverse.data(), inverse.size() * sizeof(float), cudaMemcpyHostToDevice), "copy config RoPE frequencies");
    int attention_slot = 0;
    for (int layer = 0; layer < model_config.shape.layers; ++layer) {
      LayerWeights& weights = layers[layer];
      weights.attention = model_config.attention[layer];
      const std::string prefix = "model.layers." + std::to_string(layer) + ".";
      weights.operator_norm = model.at(prefix + "operator_norm.weight").device_data;
      weights.ffn_norm = model.at(prefix + "ffn_norm.weight").device_data;
      weights.moe = model_config.moe && layer >= model_config.dense_layers;
      if (weights.moe) {
        const auto ff = prefix + "feed_forward.";
        weights.router = model.at(ff + "gate.weight").device_data;
        if (model_config.use_expert_bias) weights.expert_bias = reinterpret_cast<const float*>(model.at(ff + "expert_bias").device_data);
        const bool packed = model.has(ff + "experts.gate_up_proj");
        const size_t matrix = static_cast<size_t>(model_config.expert_width) * kHiddenSize;
        for (int e = 0; e < model_config.num_experts; ++e) {
          if (packed) {
            const auto* gate = model.at(ff + "experts.gate_up_proj").device_data + 2 * static_cast<size_t>(e) * matrix;
            weights.experts.push_back({gate, model.at(ff + "experts.down_proj").device_data + static_cast<size_t>(e) * matrix,
                                       gate + matrix});
          } else {
            const auto expert = ff + "experts." + std::to_string(e) + ".";
            const auto* original_gate = model.at(expert + "w1.weight").device_data;
            const auto* original_up = model.at(expert + "w3.weight").device_data;
            if (original_up == original_gate + matrix) {
              // Indexed checkpoints arrange gate/up adjacently during their
              // initial upload: reuse that storage, do not duplicate experts.
              weights.experts.push_back({original_gate, model.at(expert + "w2.weight").device_data, original_up});
              continue;
            }
            if (!packed_experts) packed_experts = std::make_unique<DeviceBuffer<__nv_bfloat16>>(
                lfm::checked_product({static_cast<size_t>(model_config.shape.layers - model_config.dense_layers),
                                     static_cast<size_t>(model_config.num_experts), 2, matrix}));
            auto* gate = packed_experts->data +
                (static_cast<size_t>(layer - model_config.dense_layers) * model_config.num_experts + e) * 2 * matrix;
            check_cuda(cudaMemcpy(gate, model.at(expert + "w1.weight").device_data,
                                  matrix * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice), "pack expert gate");
            check_cuda(cudaMemcpy(gate + matrix, model.at(expert + "w3.weight").device_data,
                                  matrix * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice), "pack expert up");
            weights.experts.push_back({gate, model.at(expert + "w2.weight").device_data, gate + matrix});
          }
        }
      } else {
        weights.w1 = model.at(prefix + "feed_forward.w1.weight").device_data;
        weights.w2 = model.at(prefix + "feed_forward.w2.weight").device_data;
        weights.w3 = model.at(prefix + "feed_forward.w3.weight").device_data;
      }
      if (weights.attention) {
        const std::string attention = prefix + "self_attn.";
        __nv_bfloat16* packed = packed_qkv.data +
            static_cast<size_t>(attention_slot++) * kQkvWidth * kHiddenSize;
        const size_t query_elements = static_cast<size_t>(kHiddenSize) * kHiddenSize;
        const size_t kv_elements = static_cast<size_t>(kKvWidth) * kHiddenSize;
        check_cuda(
            cudaMemcpy(packed,
                       model.at(attention + "q_proj.weight").device_data,
                       query_elements * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice),
            "pack query projection");
        check_cuda(
            cudaMemcpy(packed + query_elements,
                       model.at(attention + "k_proj.weight").device_data,
                       kv_elements * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice),
            "pack key projection");
        check_cuda(
            cudaMemcpy(packed + query_elements + kv_elements,
                       model.at(attention + "v_proj.weight").device_data,
                       kv_elements * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice),
            "pack value projection");
        weights.qkv = packed;
        weights.attention_out = model.at(attention + "out_proj.weight").device_data;
        weights.query_norm = model.at(attention + "q_layernorm.weight").device_data;
        weights.key_norm = model.at(attention + "k_layernorm.weight").device_data;
      } else {
        weights.conv_kernel = model.at(prefix + "conv.conv.weight").device_data;
        weights.conv_in = model.at(prefix + "conv.in_proj.weight").device_data;
        weights.conv_out = model.at(prefix + "conv.out_proj.weight").device_data;
      }
    }
  }
};

uint64_t hash_device_bf16(const __nv_bfloat16* data, size_t count) {
  std::vector<__nv_bfloat16> host(count);
  check_cuda(cudaMemcpy(host.data(), data, count * sizeof(host[0]), cudaMemcpyDeviceToHost),
             "copy device hash input");
  return fnv1a_bf16(host);
}

struct AttentionTrace {
  uint64_t query_norm = 0;
  uint64_t key_norm = 0;
  uint64_t query = 0;
  uint64_t key = 0;
  uint64_t value = 0;
  uint64_t attended = 0;
  uint64_t projected = 0;
  std::array<float, 4> query_norm_values{};
  std::array<float, 4> query_values{};
};

void copy_first_values(const __nv_bfloat16* data, std::array<float, 4>& output) {
  __nv_bfloat16 values[4];
  check_cuda(cudaMemcpy(values, data, sizeof(values), cudaMemcpyDeviceToHost), "copy trace values");
  for (int i = 0; i < 4; ++i) output[i] = __bfloat162float(values[i]);
}

void run_conv_layer_zero(
    cublasHandle_t handle,
    const ModelWeights& model,
    int token,
    LayerBuffers& buffers) {
  require(token >= 0 && token < kVocabularySize, "embedding token is out of range");
  check_cuda(
      cudaMemcpy(
          buffers.hidden.data,
          model.embedding + static_cast<size_t>(token) * kHiddenSize,
          kHiddenSize * sizeof(__nv_bfloat16),
          cudaMemcpyDeviceToDevice),
      "copy token embedding");
  check_cuda(cudaMemset(buffers.convolution_states.data, 0, buffers.convolution_states.size * sizeof(__nv_bfloat16)), "zero convolution state");
  const LayerWeights& weights = model.layers[0];
  const int batch = buffers.batch;
  rms_norm(
      buffers.hidden.data,
      weights.operator_norm,
      buffers.normalized.data,
      batch);
  linear_bf16(
      handle,
      buffers.normalized.data,
      weights.conv_in,
      buffers.projection3.data,
      kHiddenSize,
      3 * kHiddenSize,
      batch);
  conv_decode<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(
      buffers.projection3.data,
      weights.conv_kernel,
      buffers.convolution_states.data,
      buffers.operator_output.data,
      0);
  linear_bf16(
      handle,
      buffers.operator_output.data,
      weights.conv_out,
      buffers.mlp_output.data,
      kHiddenSize,
      kHiddenSize,
      batch);
  add_residual<<<(kHiddenSize + 255) / 256, 256>>>(
      buffers.hidden.data,
      buffers.mlp_output.data,
      buffers.hidden.data,
      kHiddenSize);
  rms_norm(
      buffers.hidden.data,
      weights.ffn_norm,
      buffers.normalized.data,
      1);
  linear_bf16(
      handle,
      buffers.normalized.data,
      weights.w1,
      buffers.mlp_gate.data,
      kHiddenSize,
      kMlpWidth);
  linear_bf16(
      handle,
      buffers.normalized.data,
      weights.w3,
      buffers.mlp_value.data,
      kHiddenSize,
      kMlpWidth);
  swiglu<<<(kMlpWidth + 255) / 256, 256>>>(
      buffers.mlp_gate.data,
      buffers.mlp_value.data,
      buffers.mlp_product.data,
      kMlpWidth);
  linear_bf16(
      handle,
      buffers.mlp_product.data,
      weights.w2,
      buffers.mlp_output.data,
      kMlpWidth,
      kHiddenSize);
  add_residual<<<(kHiddenSize + 255) / 256, 256>>>(
      buffers.hidden.data,
      buffers.mlp_output.data,
      buffers.hidden.data,
      kHiddenSize);
  check_cuda(cudaGetLastError(), "layer zero kernels");
}

std::ostream* parity_ops_stream = nullptr;
std::ostream* parity_conv_stream = nullptr;
std::ostream* parity_moe_stream = nullptr;
std::ostream* parity_math_stream = nullptr;
int parity_ops_layer = -1;
int parity_row = 0;  // Original batch slot; never compact or reorder diagnostic inputs.
bool parity_capture_enabled = true;
std::ostream* parity_expert_stream = nullptr;
std::ostream* parity_expert_index = nullptr;
int parity_expert_layer = -1, parity_expert_position = -1, parity_position = -1;
bool parity_expert_active(int layer) {
  return parity_capture_enabled && parity_expert_stream && layer == parity_expert_layer &&
      (parity_expert_position < 0 || parity_position == parity_expert_position);
}
template <typename T>
void parity_expert_tensor(const char* name, const T* device, int rows, int columns,
                          int layer, int expert = -1, int packed_row = -1) {
  if (!parity_expert_active(layer)) return;
  const size_t count = static_cast<size_t>(rows) * columns;
  std::vector<T> host(count);
  check_cuda(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "trace expert operation");
  const auto offset = parity_expert_stream->tellp();
  parity_expert_stream->write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
  *parity_expert_index << "{\"name\":\"" << name << "\",\"position\":" << parity_position
      << ",\"layer\":" << layer << ",\"expert\":" << expert << ",\"selected_row\":" << parity_row
      << ",\"packed_row\":" << packed_row << ",\"shape\":[" << rows << ',' << columns
      << "],\"dtype\":\"" << (sizeof(T) == 2 ? "bf16" : "i32") << "\",\"offset\":" << offset
      << ",\"bytes\":" << count * sizeof(T) << "}\n";
  if (!*parity_expert_stream || !*parity_expert_index) throw std::runtime_error("write expert operation trace failed");
}
void parity_write(std::ostream*, const __nv_bfloat16*, size_t);
void parity_write_row(std::ostream*, const __nv_bfloat16*, size_t, size_t = 0);

// Inputs already own a complete zero-padded N=32 tile. Keep the production
// GEMM geometry, precision and output poisoning, but consume its output directly
// instead of copying a partial expert result out of, then back into, scratch.
void linear_bf16_canonical(
    cublasHandle_t handle, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, int input_size, int output_size) {
  LFM_PROFILE_RANGE(profile_linear, "canonical_linear/in=" + std::to_string(input_size) +
      "/out=" + std::to_string(output_size) + "/rows=32");
  require(linear_scratch && input && weight && input_size > 0 && output_size > 0,
          "invalid canonical projection");
  require(static_cast<size_t>(input_size) <= linear_scratch->input.size / kProjectionColumns &&
          static_cast<size_t>(output_size) <= linear_scratch->output.size / kProjectionColumns,
          "canonical projection exceeds workspace");
  const float alpha = 1.f, beta = 0.f;
  check_cuda(cudaMemsetAsync(linear_scratch->output.data, 0xff,
      static_cast<size_t>(output_size) * kProjectionColumns * sizeof(__nv_bfloat16),
      cudaStreamPerThread), "poison canonical projection output");
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      output_size, kProjectionColumns, input_size, &alpha,
      weight, CUDA_R_16BF, input_size, input, CUDA_R_16BF, input_size,
      &beta, linear_scratch->output.data, CUDA_R_16BF, output_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "canonical expert projection");
}

void run_mlp(
    cublasHandle_t handle,
    const ModelWeights& model,
    int layer,
    LayerBuffers& buffers) {
  LFM_PROFILE_RANGE(profile_mlp, "mlp/layer=" + std::to_string(layer));
  const LayerWeights& weights = model.layers[layer];
  const int batch = buffers.active;
  auto dump = [&](const __nv_bfloat16* data, size_t count) {
    if (layer == parity_ops_layer) parity_write_row(parity_ops_stream, data, count);
  };
  dump(buffers.hidden.data, kHiddenSize);
  rms_norm(
      buffers.hidden.data,
      weights.ffn_norm,
      buffers.normalized.data,
      batch);
  dump(buffers.normalized.data, kHiddenSize);
  if (weights.moe) {
    LFM_PROFILE_RANGE(profile_route, "moe.routing");
    const int experts = model_config.num_experts, top_k = model_config.experts_per_token;
    const int width = model_config.expert_width;
    linear_bf16(handle, buffers.normalized.data, weights.router, buffers.router_logits.data,
                 kHiddenSize, experts, batch);
    lfm::moe_sigmoid<<<(batch * experts + 255) / 256, 256>>>(buffers.router_logits.data, buffers.router_probability.data, batch * experts);
    check_cuda(cudaMemsetAsync(buffers.expert_counts.data, 0, (experts + 1) * sizeof(int), cudaStreamPerThread), "reset expert counts");
    lfm::moe_select<<<batch, 1>>>(buffers.router_logits.data, buffers.router_probability.data, weights.expert_bias,
        buffers.routes.data, buffers.route_weights.data, buffers.expert_counts.data, experts, top_k,
        model_config.norm_topk_prob, model_config.routed_scaling);
    lfm::moe_groups<<<(experts + 255) / 256, 256>>>(buffers.routes.data, buffers.expert_rows.data,
        buffers.expert_ranks.data, buffers.expert_counts.data, batch, experts, top_k);
    std::vector<int> counts(experts + 1);
    check_cuda(cudaMemcpy(counts.data(), buffers.expert_counts.data, counts.size() * sizeof(int), cudaMemcpyDeviceToHost), "read active expert groups");
    if (counts[experts]) throw std::runtime_error("nonfinite MoE routing input or weight");
    parity_expert_tensor("moe.input", buffers.hidden.data, batch, kHiddenSize, layer);
    parity_expert_tensor("moe.normalized", buffers.normalized.data, batch, kHiddenSize, layer);
    parity_expert_tensor("moe.router_logits", buffers.router_logits.data, batch, experts, layer);
    parity_expert_tensor("moe.route_ids", buffers.routes.data, batch, top_k, layer);
    parity_expert_tensor("moe.route_weights", buffers.route_weights.data, batch, top_k, layer);
    if (parity_moe_stream) {
      parity_write_row(parity_moe_stream, buffers.router_logits.data, experts);
      std::vector<int> selected(top_k);
      check_cuda(cudaMemcpy(selected.data(), buffers.routes.data + static_cast<size_t>(parity_row) * top_k, top_k * sizeof(int), cudaMemcpyDeviceToHost), "trace expert IDs");
      parity_moe_stream->write(reinterpret_cast<const char*>(selected.data()), top_k * sizeof(int));
      parity_write_row(parity_moe_stream, buffers.route_weights.data, top_k);
    }
    check_cuda(cudaMemsetAsync(buffers.mlp_output.data, 0, static_cast<size_t>(batch) * kHiddenSize * sizeof(__nv_bfloat16), cudaStreamPerThread), "zero expert accumulation");
    LFM_PROFILE_END(profile_route);
    for (int expert = 0; expert < experts; ++expert) {
      const int count = counts[expert];
      require(count >= 0 && count <= batch, "invalid expert group size");
      if (count == 0) continue;
      LFM_PROFILE_RANGE(profile_expert, "expert/id=" + std::to_string(expert) + "/active=" + std::to_string(count));
      const int* rows = buffers.expert_rows.data + static_cast<size_t>(expert) * batch;
      const int* ranks = buffers.expert_ranks.data + static_cast<size_t>(expert) * batch;
      int trace_packed_row = -1;
      if (parity_expert_active(layer)) {
        std::vector<int> members(count);
        check_cuda(cudaMemcpy(members.data(), rows, count * sizeof(int), cudaMemcpyDeviceToHost), "trace expert membership");
        for (int item = 0; item < count; ++item) if (members[item] == parity_row) trace_packed_row = item;
        if (trace_packed_row >= 0) {
          parity_expert_tensor("expert.rows", rows, count, 1, layer, expert, trace_packed_row);
          parity_expert_tensor("expert.ranks", ranks, count, 1, layer, expert, trace_packed_row);
        }
      }
      if (batch <= kProjectionColumns) {
        const auto& e = weights.experts[expert];
        lfm::moe_gather_canonical<<<(kProjectionColumns * kHiddenSize + 255) / 256, 256>>>(
            buffers.normalized.data, linear_scratch->input.data, rows, count, kHiddenSize);
        if (trace_packed_row >= 0) parity_expert_tensor("expert.input", linear_scratch->input.data, kProjectionColumns, kHiddenSize, layer, expert, trace_packed_row);
        linear_bf16_canonical(handle, linear_scratch->input.data, e.w1, kHiddenSize, 2 * width);
        if (trace_packed_row >= 0) parity_expert_tensor("expert.gate_up", linear_scratch->output.data, kProjectionColumns, 2 * width, layer, expert, trace_packed_row);
        lfm::moe_swiglu_canonical<<<dim3((width + 255) / 256, kProjectionColumns), 256>>>(
            linear_scratch->output.data, linear_scratch->input.data, count, width);
        if (trace_packed_row >= 0) parity_expert_tensor("expert.product", linear_scratch->input.data, kProjectionColumns, width, layer, expert, trace_packed_row);
        linear_bf16_canonical(handle, linear_scratch->input.data, e.w2, width, kHiddenSize);
        if (trace_packed_row >= 0) parity_expert_tensor("expert.output", linear_scratch->output.data, kProjectionColumns, kHiddenSize, layer, expert, trace_packed_row);
        // The loop remains ascending by expert; retain both BF16 rounding points.
        lfm::moe_scatter<<<(static_cast<size_t>(count) * kHiddenSize + 255) / 256, 256>>>(
            linear_scratch->output.data, buffers.mlp_output.data, rows, ranks,
             buffers.route_weights.data, count, kHiddenSize, top_k);
        if (trace_packed_row >= 0) parity_expert_tensor("expert.accumulated", buffers.mlp_output.data + static_cast<size_t>(parity_row) * kHiddenSize, 1, kHiddenSize, layer, expert, 0);
        continue;
      }
      lfm::moe_gather<<<(static_cast<size_t>(count) * kHiddenSize + 255) / 256, 256>>>(
          buffers.normalized.data, buffers.expert_input.data, rows, count, kHiddenSize);
      const auto& e = weights.experts[expert];
      linear_bf16(handle, buffers.expert_input.data, e.w1, buffers.mlp_gate.data, kHiddenSize, 2 * width, count);
      lfm::moe_swiglu<<<dim3((width + 255) / 256, count), 256>>>(buffers.mlp_gate.data, buffers.expert_product.data, width);
      linear_bf16(handle, buffers.expert_product.data, e.w2, buffers.expert_output.data, width, kHiddenSize, count);
      lfm::moe_scatter<<<(static_cast<size_t>(count) * kHiddenSize + 255) / 256, 256>>>(
          buffers.expert_output.data, buffers.mlp_output.data, rows, ranks, buffers.route_weights.data, count, kHiddenSize, top_k);
    }
    dump(buffers.mlp_output.data, kHiddenSize);
    parity_expert_tensor("moe.output", buffers.mlp_output.data, batch, kHiddenSize, layer);
    add_residual<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(buffers.hidden.data, buffers.mlp_output.data, buffers.hidden.data, kHiddenSize);
    dump(buffers.hidden.data, kHiddenSize);
    parity_expert_tensor("layer.output", buffers.hidden.data, batch, kHiddenSize, layer);
    check_cuda(cudaGetLastError(), "MoE kernels");
    return;
  }
  linear_bf16(handle, buffers.normalized.data, weights.w1,
               buffers.mlp_gate.data, kHiddenSize, kMlpWidth, batch);
  dump(buffers.mlp_gate.data, kMlpWidth);
  linear_bf16(handle, buffers.normalized.data, weights.w3,
               buffers.mlp_value.data, kHiddenSize, kMlpWidth, batch);
  dump(buffers.mlp_value.data, kMlpWidth);
  swiglu<<<dim3((kMlpWidth + 255) / 256, batch), 256>>>(
      buffers.mlp_gate.data, buffers.mlp_value.data, buffers.mlp_product.data, kMlpWidth);
  dump(buffers.mlp_product.data, kMlpWidth);
  linear_bf16(handle, buffers.mlp_product.data, weights.w2,
               buffers.mlp_output.data, kMlpWidth, kHiddenSize, batch);
  dump(buffers.mlp_output.data, kHiddenSize);
  add_residual<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(
      buffers.hidden.data, buffers.mlp_output.data, buffers.hidden.data, kHiddenSize);
  dump(buffers.hidden.data, kHiddenSize);
}

void run_conv_layer(
    cublasHandle_t handle,
    const ModelWeights& model,
    int layer,
    LayerBuffers& buffers) {
  LFM_PROFILE_RANGE(profile_conv, "conv/layer=" + std::to_string(layer));
  const LayerWeights& weights = model.layers[layer];
  const int batch = buffers.active;
  auto dump = [&](const __nv_bfloat16* data, size_t count) {
    if (layer == parity_ops_layer) parity_write_row(parity_conv_stream, data, count);
  };
  parity_expert_tensor("layer.input", buffers.hidden.data, batch, kHiddenSize, layer);
  if (layer == parity_ops_layer) {
    parity_write(parity_conv_stream, weights.operator_norm, kHiddenSize);
    parity_write(parity_conv_stream, weights.conv_kernel, static_cast<size_t>(kHiddenSize) * model_config.shape.conv_width);
    parity_write(parity_conv_stream, weights.ffn_norm, kHiddenSize);
  }
  dump(buffers.hidden.data, kHiddenSize);
  rms_norm(
      buffers.hidden.data,
      weights.operator_norm,
      buffers.normalized.data,
      batch);
  dump(buffers.normalized.data, kHiddenSize);
  parity_expert_tensor("operator.normalized", buffers.normalized.data, batch, kHiddenSize, layer);
  linear_bf16(handle, buffers.normalized.data, weights.conv_in,
               buffers.projection3.data, kHiddenSize, 3 * kHiddenSize, batch);
  dump(buffers.projection3.data, 3 * kHiddenSize);
  parity_expert_tensor("conv.projected", buffers.projection3.data, batch, 3 * kHiddenSize, layer);
  conv_decode<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(
      buffers.projection3.data,
      weights.conv_kernel,
      buffers.convolution_states.data,
      buffers.operator_output.data,
      layer);
  dump(buffers.operator_output.data, kHiddenSize);
  parity_expert_tensor("conv.product", buffers.operator_output.data, batch, kHiddenSize, layer);
  linear_bf16(handle, buffers.operator_output.data, weights.conv_out,
               buffers.mlp_output.data, kHiddenSize, kHiddenSize, batch);
  dump(buffers.mlp_output.data, kHiddenSize);
  parity_expert_tensor("operator.projected", buffers.mlp_output.data, batch, kHiddenSize, layer);
  add_residual<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(
      buffers.hidden.data, buffers.mlp_output.data, buffers.hidden.data, kHiddenSize);
  dump(buffers.hidden.data, kHiddenSize);
  run_mlp(handle, model, layer, buffers);
}

// Diagnostic-only raw BF16 trace; disabled during production inference.
std::ostream* parity_stream = nullptr;
int parity_all_row_count = 0;  // Only the schema-2 teacher stream uses this.
std::ostream* parity_attention_stream = nullptr;
int parity_attention_layer = -1;
lfm::parity::MathAttention* pinned_attention = nullptr;
void parity_write(std::ostream* stream, const __nv_bfloat16* device, size_t count) {
  if (!stream || !parity_capture_enabled) return;
  std::vector<__nv_bfloat16> host(count);
  check_cuda(cudaMemcpy(host.data(), device, count * sizeof(host[0]), cudaMemcpyDeviceToHost),
             "copy parity tensor");
  stream->write(reinterpret_cast<const char*>(host.data()), count * sizeof(host[0]));
  if (!*stream) throw std::runtime_error("write parity tensor failed");
}
void parity_dump(const __nv_bfloat16* device, size_t count) {
  if (parity_all_row_count) parity_write(parity_stream, device, count * parity_all_row_count);
  else parity_write_row(parity_stream, device, count);
}
void parity_write_row(std::ostream* stream, const __nv_bfloat16* device, size_t count, size_t stride) {
  if (!stream) return;
  parity_write(stream, device + static_cast<size_t>(parity_row) * (stride ? stride : count), count);
}

void run_attention_layer(
    cublasHandle_t handle,
    const ModelWeights& model,
    int layer,
    int attention_slot,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    LayerBuffers& buffers,
    AttentionTrace* trace = nullptr) {
  LFM_PROFILE_RANGE(profile_attention, "attention/layer=" + std::to_string(layer));
  const LayerWeights& weights = model.layers[layer];
  const int batch = buffers.active;
  __nv_bfloat16* query = buffers.qkv.data;
  auto dump = [&](const __nv_bfloat16* data, size_t count) {
    // Q/K/V share an interleaved [batch, QKV] allocation, even when sliced.
    const bool qkv = data == query || data == query + kHiddenSize || data == query + kHiddenSize + kKvWidth;
    if (layer == parity_attention_layer) parity_write_row(parity_attention_stream, data, count, qkv ? kQkvWidth : count);
  };
  __nv_bfloat16* key = query + kHiddenSize;
  __nv_bfloat16* value = key + kKvWidth;
  parity_expert_tensor("layer.input", buffers.hidden.data, batch, kHiddenSize, layer);
  rms_norm(
      buffers.hidden.data,
      weights.operator_norm,
      buffers.normalized.data,
      batch);
  if (model_config.moe || model.math_attention) {
    linear_bf16(handle, buffers.normalized.data, weights.qkv, query,
                 kHiddenSize, kHiddenSize, batch, kQkvWidth);
    linear_bf16(handle, buffers.normalized.data, weights.qkv + static_cast<size_t>(kHiddenSize) * kHiddenSize,
                 key, kHiddenSize, kKvWidth, batch, kQkvWidth);
    linear_bf16(handle, buffers.normalized.data, weights.qkv + static_cast<size_t>(kHiddenSize + kKvWidth) * kHiddenSize,
                 value, kHiddenSize, kKvWidth, batch, kQkvWidth);
  } else linear_bf16(handle, buffers.normalized.data, weights.qkv,
                     buffers.qkv.data, kHiddenSize, kQkvWidth, batch);
  dump(buffers.normalized.data, kHiddenSize);
  parity_expert_tensor("operator.normalized", buffers.normalized.data, batch, kHiddenSize, layer);
  dump(buffers.qkv.data, kQkvWidth);
  parity_expert_tensor("attention.qkv", buffers.qkv.data, batch, kQkvWidth, layer);
  head_rms_norm<<<dim3(batch, (kQueryHeads + 31) / 32), dim3(32, std::min(kQueryHeads, 32))>>>(
      query, weights.query_norm,
      query, kQueryHeads, kQkvWidth);
  head_rms_norm<<<dim3(batch, (kKvHeads + 31) / 32), dim3(32, std::min(kKvHeads, 32))>>>(
      key, weights.key_norm,
      key, kKvHeads, kQkvWidth);
  dump(query, kHiddenSize);
  dump(key, kKvWidth);
  parity_expert_tensor("attention.qkv_norm", buffers.qkv.data, batch, kQkvWidth, layer);
  if (trace) {
    trace->query_norm = hash_device_bf16(query, kHiddenSize);
    trace->key_norm = hash_device_bf16(key, kKvWidth);
    copy_first_values(query, trace->query_norm_values);
  }
  apply_rope_device_position<<<dim3((kQueryHeads * (kHeadDimension / 2) + 255) / 256, batch), 256>>>(
      query, kQueryHeads, buffers.position.data, kQkvWidth, model.frequencies.data);
  apply_rope_device_position<<<dim3((kKvHeads * (kHeadDimension / 2) + 255) / 256, batch), 256>>>(
      key, kKvHeads, buffers.position.data, kQkvWidth, model.frequencies.data);
  dump(query, kHiddenSize);
  dump(key, kKvWidth);
  dump(value, kKvWidth);
  if (trace) {
    trace->query = hash_device_bf16(query, kHiddenSize);
    trace->key = hash_device_bf16(key, kKvWidth);
    trace->value = hash_device_bf16(value, kKvWidth);
    copy_first_values(query, trace->query_values);
  }
  const size_t layer_stride = kKvLayerStride;
  write_kv_device_position<<<dim3((kKvWidth + 255) / 256, batch), 256>>>(
      key_cache + static_cast<size_t>(attention_slot) * layer_stride,
      value_cache + static_cast<size_t>(attention_slot) * layer_stride,
      key,
      value,
      buffers.position.data,
      static_cast<int>(kKvBatchStride));
  LFM_PROFILE_RANGE(profile_attention_core, "attention.core");
  if (model.math_attention) {
    model.math_attention->run(query, key_cache + static_cast<size_t>(attention_slot) * layer_stride,
        value_cache + static_cast<size_t>(attention_slot) * layer_stride,
        buffers.operator_output.data, buffers.position.data, batch,
        buffers.stopped_rows, buffers.stopped_rows ? buffers.host_stopped_rows.data() : nullptr);
    if (parity_capture_enabled && parity_math_stream && layer == parity_attention_layer && batch <= 32) {
      auto write = [&](const float* data, size_t count) {
        std::vector<float> host(count);
        check_cuda(cudaMemcpy(host.data(), data + static_cast<size_t>(parity_row) * count, count * sizeof(float), cudaMemcpyDeviceToHost), "trace math attention");
        parity_math_stream->write(reinterpret_cast<const char*>(host.data()), count * sizeof(float));
        if (!*parity_math_stream) throw std::runtime_error("write math-attention trace failed");
      };
      const size_t scores = static_cast<size_t>(kQueryHeads) * model.math_attention->tokens();
      write(model.math_attention->scores(), scores);
      write(model.math_attention->probability(), scores);
      write(model.math_attention->attended(), kHiddenSize);
    }
  } else if (pinned_attention) {
    int position = 0;
    check_cuda(cudaMemcpy(&position, buffers.position.data, sizeof(position), cudaMemcpyDeviceToHost),
               "read math-attention position");
    pinned_attention->run(query,
        key_cache + static_cast<size_t>(attention_slot) * layer_stride,
        value_cache + static_cast<size_t>(attention_slot) * layer_stride,
        buffers.operator_output.data, position + 1, kContextLength);
  } else if (kHeadDimension == 64 && batch >= kProjectionColumns) {
    lfm::online_attention_compact<256><<<dim3(kQueryHeads, batch), 256,
        32 * kHeadDimension * sizeof(__nv_bfloat16)>>>(
        query, key_cache + static_cast<size_t>(attention_slot) * layer_stride,
        value_cache + static_cast<size_t>(attention_slot) * layer_stride,
        buffers.operator_output.data, buffers.position.data, buffers.attention_policy.data,
        lfm::runtime_shape, buffers.stopped_rows);
  } else if (kHeadDimension == 64) lfm::online_softmax_gqa_attention_staged<<<dim3(kQueryHeads, batch), 1024>>>(
      query,
      key_cache + static_cast<size_t>(attention_slot) * layer_stride,
      value_cache + static_cast<size_t>(attention_slot) * layer_stride,
      buffers.operator_output.data,
      buffers.position.data,
      buffers.attention_policy.data,
      static_cast<int>(kKvBatchStride), kHiddenSize, kQueryHeads, kKvHeads, kContextLength,
      buffers.stopped_rows);
  else lfm::online_attention_dynamic<<<dim3(kQueryHeads, batch), 1024>>>(
      query, key_cache + static_cast<size_t>(attention_slot) * layer_stride,
      value_cache + static_cast<size_t>(attention_slot) * layer_stride,
      buffers.operator_output.data, buffers.position.data, buffers.attention_policy.data,
      lfm::runtime_shape, buffers.stopped_rows);
  LFM_PROFILE_END(profile_attention_core);
  if (trace) trace->attended = hash_device_bf16(buffers.operator_output.data, kHiddenSize);
  dump(buffers.operator_output.data, kHiddenSize);
  parity_expert_tensor("attention.attended", buffers.operator_output.data, batch, kHiddenSize, layer);
  linear_bf16(handle, buffers.operator_output.data, weights.attention_out,
               buffers.mlp_output.data, kHiddenSize, kHiddenSize, batch);
  if (trace) trace->projected = hash_device_bf16(buffers.mlp_output.data, kHiddenSize);
  dump(buffers.mlp_output.data, kHiddenSize);
  parity_expert_tensor("operator.projected", buffers.mlp_output.data, batch, kHiddenSize, layer);
  add_residual<<<dim3((kHiddenSize + 255) / 256, batch), 256>>>(
      buffers.hidden.data, buffers.mlp_output.data, buffers.hidden.data, kHiddenSize);
  run_mlp(handle, model, layer, buffers);
}

void run_full_token_device(
    cublasHandle_t handle,
    const ModelWeights& model,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    LayerBuffers& buffers,
    bool reset_state,
    bool fused_lm_head = false,
    std::vector<uint64_t>* layer_trace = nullptr,
    AttentionTrace* attention_trace = nullptr,
    bool compute_head = true,
    bool compute_argmax = true) {
  LFM_PROFILE_RANGE(profile_forward, "forward/rows=" + std::to_string(buffers.active));
  if (reset_state) {
    check_cuda(cudaMemset(buffers.convolution_states.data, 0,
                           buffers.convolution_states.size * sizeof(__nv_bfloat16)), "reset convolution states");
    model.source_model->verify_small_tensors();
    model.source_model->debug_snapshot_tail("state-reset");
  }
  gather_embedding<<<dim3((kHiddenSize + 255) / 256, buffers.active), 256>>>(
      model.embedding, buffers.input_token.data, buffers.hidden.data);
  int attention_slot = 0;
  for (int layer = 0; layer < model_config.shape.layers; ++layer) {
    if (model.layers[layer].attention) {
      run_attention_layer(handle, model, layer, attention_slot++,
                          key_cache, value_cache, buffers,
                          layer == parity_attention_layer ? attention_trace : nullptr);
    } else {
      run_conv_layer(handle, model, layer, buffers);
    }
    parity_dump(buffers.hidden.data, kHiddenSize);
#ifdef LFM_BATCH_GRAPH_TEST
    if (graph_test_trace) check_cuda(cudaMemcpyAsync(
        graph_test_trace + static_cast<size_t>(layer) * buffers.active * kHiddenSize,
        buffers.hidden.data, static_cast<size_t>(buffers.active) * kHiddenSize * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, cudaStreamPerThread), "graph-test layer trace");
#endif
    if (reset_state) model.source_model->debug_snapshot_tail("layer-" + std::to_string(layer));
    if (layer_trace) {
      std::vector<__nv_bfloat16> state(kHiddenSize);
      check_cuda(cudaMemcpy(state.data(), buffers.hidden.data,
                            state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost),
                 "copy layer trace");
      layer_trace->push_back(fnv1a_bf16(state));
    }
  }
  LFM_PROFILE_RANGE(profile_head, "lm_head");
  rms_norm(
      buffers.hidden.data,
      model.final_norm,
      buffers.normalized.data,
      buffers.active);
  // The projection is the last stage and consumes nothing from the layers
  // below it, so a position whose logits are never read can skip it entirely.
  if (!compute_head) return;
  // The hand-reduced fused head has an established argmax contract only for
  // its 1024-wide specialization. Wider models use the same materialized
  // cuBLAS projection as batch serving; do not silently generalize a reduction
  // schedule and let batch size change the answer.
  if (fused_lm_head && buffers.batch == 1 && kHiddenSize == lfm::kHiddenSize) {
    lm_head_argmax<<<kLmHeadBlocks, 256>>>(
        buffers.normalized.data,
        model.lm_head,
        buffers.lm_head_block_values.data,
        buffers.lm_head_block_indices.data);
    finalize_lm_head_argmax<<<1, 256>>>(
        buffers.lm_head_block_values.data,
        buffers.lm_head_block_indices.data,
        buffers.selected_token.data);
  } else {
    linear_bf16(handle, buffers.normalized.data, model.lm_head,
                 buffers.logits.data, kHiddenSize, kVocabularySize, buffers.active);
    if (compute_argmax) {
      lfm::argmax_bf16_rows<<<buffers.active, 256>>>(
          buffers.logits.data, buffers.active, kVocabularySize, buffers.selected_token.data);
    }
  }
  check_cuda(cudaGetLastError(), "full token kernels");
}

// Fixed-shape dense forward only. Sampling, RNG step, token/position advancement
// and the host's per-row stop decisions remain outside capture.
class BatchDecodeGraph {
 public:
  BatchDecodeGraph(cublasHandle_t handle, const ModelWeights& model,
      __nv_bfloat16* keys, __nv_bfloat16* values, LayerBuffers& buffers,
      bool compute_argmax) {
    bool capturing = false;
    try {
      check_cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal),
                 "begin batch forward graph");
      capturing = true;
      run_full_token_device(handle, model, keys, values, buffers,
                            false, false, nullptr, nullptr, true, compute_argmax);
      const auto status = cudaStreamEndCapture(cudaStreamPerThread, &graph_);
      capturing = false;
      check_cuda(status, "end batch forward graph");
      check_cuda(cudaGraphInstantiate(&executable_, graph_, 0), "instantiate batch forward graph");
    } catch (...) {
      if (capturing) {
        cudaGraph_t aborted = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &aborted);
        if (aborted) cudaGraphDestroy(aborted);
      }
      if (executable_) cudaGraphExecDestroy(executable_);
      if (graph_) cudaGraphDestroy(graph_);
      throw;
    }
  }
  ~BatchDecodeGraph() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
  }
  BatchDecodeGraph(const BatchDecodeGraph&) = delete;
  BatchDecodeGraph& operator=(const BatchDecodeGraph&) = delete;
  void launch() const {
    check_cuda(cudaGraphLaunch(executable_, cudaStreamPerThread), "launch batch forward graph");
  }
 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

class DecodeGraph {
 public:
  DecodeGraph(
      cublasHandle_t handle,
      const ModelWeights& model,
      __nv_bfloat16* key_cache,
      __nv_bfloat16* value_cache,
      LayerBuffers& buffers,
      bool fused_greedy_lm_head) {
    check_cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal),
               "begin decode graph capture");
    run_full_token_device(
        handle, model, key_cache, value_cache, buffers, false, fused_greedy_lm_head);
    increment_position<<<1, 1>>>(buffers.position.data);
    check_cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph_),
               "end decode graph capture");
    check_cuda(cudaGraphInstantiate(&executable_, graph_, 0),
               "instantiate decode graph");
  }

  ~DecodeGraph() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
  }

  void launch() const {
    check_cuda(cudaGraphLaunch(executable_, cudaStreamPerThread), "launch decode graph");
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

class GreedyGenerationGraph {
 public:
  GreedyGenerationGraph(
      cublasHandle_t handle,
      const ModelWeights& model,
      __nv_bfloat16* key_cache,
      __nv_bfloat16* value_cache,
      LayerBuffers& buffers,
      DecodeState& state,
      int stop_count) {
    check_cuda(cudaGraphCreate(&graph_, 0), "create generation graph");
    check_cuda(
        cudaGraphConditionalHandleCreate(
            &condition_, graph_, 1, cudaGraphCondAssignDefault),
        "create generation condition");

    cudaGraphNode_t initialize_node = nullptr;
    cudaGraphNodeParams initialize_params{};
    initialize_params.type = cudaGraphNodeTypeKernel;
    initialize_params.kernel.func = reinterpret_cast<void*>(record_token_and_set_condition);
    initialize_params.kernel.gridDim = dim3(1);
    initialize_params.kernel.blockDim = dim3(1);
    void* initialize_arguments[] = {
        &condition_,
        &buffers.input_token.data,
        &buffers.position.data,
        &state.generated.data,
        &state.generated_count.data,
        &state.stop_flag.data,
        &state.stop_tokens.data,
        &stop_count,
        &lfm::runtime_shape};
    initialize_params.kernel.kernelParams = initialize_arguments;
    check_cuda(
        cudaGraphAddNode(&initialize_node, graph_, nullptr, nullptr, 0, &initialize_params),
        "add generation initializer");

    cudaGraphNode_t conditional_node = nullptr;
    cudaGraphNodeParams conditional_params{};
    conditional_params.type = cudaGraphNodeTypeConditional;
    conditional_params.conditional.handle = condition_;
    conditional_params.conditional.type = cudaGraphCondTypeWhile;
    conditional_params.conditional.size = 1;
    check_cuda(
        cudaGraphAddNode(
            &conditional_node,
            graph_,
            &initialize_node,
            nullptr,
            1,
            &conditional_params),
        "add generation while node");
    cudaGraph_t body = conditional_params.conditional.phGraph_out[0];
    check_cuda(
        cudaStreamBeginCaptureToGraph(
            cudaStreamPerThread,
            body,
            nullptr,
            nullptr,
            0,
            cudaStreamCaptureModeRelaxed),
        "begin generation body capture");
    run_full_token_device(
        handle, model, key_cache, value_cache, buffers, false, true);
    increment_position<<<1, 1>>>(buffers.position.data);
    check_cuda(
        cudaMemcpyAsync(
            buffers.input_token.data,
            buffers.selected_token.data,
            sizeof(int),
            cudaMemcpyDeviceToDevice,
            cudaStreamPerThread),
        "capture greedy token advance");
    record_token_and_set_condition<<<1, 1>>>(
        condition_,
        buffers.input_token.data,
        buffers.position.data,
        state.generated.data,
        state.generated_count.data,
        state.stop_flag.data,
        state.stop_tokens.data,
        stop_count);
    check_cuda(
        cudaStreamEndCapture(cudaStreamPerThread, nullptr),
        "end generation body capture");
    cudaGraphInstantiateParams instantiate_params{};
    const cudaError_t instantiate_status =
        cudaGraphInstantiateWithParams(&executable_, graph_, &instantiate_params);
    if (instantiate_status != cudaSuccess) {
      cudaGraphNodeType error_type = cudaGraphNodeTypeEmpty;
      if (instantiate_params.errNode_out) {
        cudaGraphNodeGetType(instantiate_params.errNode_out, &error_type);
      }
      throw std::runtime_error(
          "instantiate generation graph: " +
          std::string(cudaGetErrorString(instantiate_status)) +
          " result=" + std::to_string(instantiate_params.result_out) +
          " initial=" + std::to_string(instantiate_params.errNode_out == initialize_node) +
          " conditional=" + std::to_string(instantiate_params.errNode_out == conditional_node) +
          " node_type=" + std::to_string(error_type));
    }
  }

  ~GreedyGenerationGraph() {
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
  }

  void launch() const {
    check_cuda(cudaGraphLaunch(executable_, cudaStreamPerThread),
               "launch generation graph");
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaGraphConditionalHandle condition_ = 0;
};

int run_full_token(
    cublasHandle_t handle,
    const ModelWeights& model,
    int token,
    int position,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    LayerBuffers& buffers,
    bool reset_state,
    std::vector<uint64_t>* layer_trace = nullptr,
    AttentionTrace* attention_trace = nullptr,
    bool compute_head = true) {
  require(token >= 0 && token < kVocabularySize, "token is out of range");
  set_decode_input<<<1, 1>>>(buffers.input_token.data, buffers.position.data, token, position);
  run_full_token_device(
      handle, model, key_cache, value_cache, buffers, reset_state, false,
      layer_trace, attention_trace, compute_head);
  // Reading the selection back to the host synchronizes the stream. Positions
  // whose selection is not consumed skip that round trip so the prompt
  // evaluations can overlap instead of serializing on a device sync each.
  if (!compute_head) return 0;
  int selected = 0;
  check_cuda(cudaMemcpy(&selected, buffers.selected_token.data, sizeof(selected), cudaMemcpyDeviceToHost),
             "copy selected token");
  return selected;
}

void test_kv_cache(
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    size_t cache_elements_per_tensor,
    __nv_bfloat16* device_key,
    __nv_bfloat16* device_value) {
  if (kAttentionLayers == 0) return;
  check_cuda(cudaMemset(key_cache, 0, cache_elements_per_tensor * sizeof(__nv_bfloat16)), "zero key cache");
  check_cuda(cudaMemset(value_cache, 0, cache_elements_per_tensor * sizeof(__nv_bfloat16)), "zero value cache");
  std::vector<__nv_bfloat16> key(kKvWidth), value(kKvWidth);
  for (int i = 0; i < kKvWidth; ++i) {
    key[i] = __float2bfloat16_rn(static_cast<float>(i + 1) / 1024.0f);
    value[i] = __float2bfloat16_rn(-static_cast<float>(i + 1) / 2048.0f);
  }
  check_cuda(cudaMemcpy(device_key, key.data(), key.size() * sizeof(key[0]), cudaMemcpyHostToDevice), "copy test key");
  check_cuda(cudaMemcpy(device_value, value.data(), value.size() * sizeof(value[0]), cudaMemcpyHostToDevice), "copy test value");
  const int positions[] = {0, std::min(12345, kContextLength - 1), kContextLength - 1};
  for (int layer = 0; layer < kAttentionLayers; ++layer) {
    for (int position : positions) write_kv_checked(key_cache, value_cache, device_key, device_value, layer, position);
  }
  check_cuda(cudaDeviceSynchronize(), "KV self-test synchronize");

  const size_t layer_stride = static_cast<size_t>(kKvHeads) * kContextLength * kHeadDimension;
  std::vector<__nv_bfloat16> observed(kHeadDimension);
  for (int layer = 0; layer < kAttentionLayers; ++layer) {
    for (int position : positions) {
      for (int head = 0; head < kKvHeads; ++head) {
        const size_t offset = static_cast<size_t>(layer) * layer_stride +
            (static_cast<size_t>(head) * kContextLength + position) * kHeadDimension;
        check_cuda(cudaMemcpy(observed.data(), key_cache + offset, observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost), "read test key");
        for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
          const int index = head * kHeadDimension + dimension;
          require(__bfloat162float(observed[dimension]) == __bfloat162float(key[index]), "KV key mismatch");
        }
        check_cuda(cudaMemcpy(observed.data(), value_cache + offset, observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost), "read test value");
        for (int dimension = 0; dimension < kHeadDimension; ++dimension) {
          const int index = head * kHeadDimension + dimension;
          require(__bfloat162float(observed[dimension]) == __bfloat162float(value[index]), "KV value mismatch");
        }
      }
    }
  }
  if (kContextLength > 2) {
    check_cuda(cudaMemcpy(observed.data(), key_cache + kHeadDimension, observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost), "read unwritten key");
    for (const auto item : observed) require(__bfloat162float(item) == 0.0f, "unwritten KV slot is nonzero");
  }

  bool rejected = false;
  try { write_kv_checked(key_cache, value_cache, device_key, device_value, 0, kContextLength); }
  catch (const std::out_of_range&) { rejected = true; }
  require(rejected, "out-of-range KV position was accepted");
  rejected = false;
  try { write_kv_checked(key_cache, value_cache, device_key, device_value, kAttentionLayers, 0); }
  catch (const std::out_of_range&) { rejected = true; }
  require(rejected, "out-of-range KV layer was accepted");
}

float benchmark_events(const std::function<void()>& operation, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) operation();
  check_cuda(cudaDeviceSynchronize(), "benchmark warmup synchronize");
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  check_cuda(cudaEventRecord(start), "record start");
  for (int i = 0; i < iterations; ++i) operation();
  check_cuda(cudaEventRecord(stop), "record stop");
  check_cuda(cudaEventSynchronize(stop), "benchmark synchronize");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms;
}

struct Options {
  int iterations = 10000;
  bool benchmark = false;
  bool diagnostics = false;
  bool cuda_graph = true;
  std::optional<std::string> model;
  std::optional<std::string> config;
  int context_length = 0;  // Default allocation policy: min(32768, model limit).
  bool verify_fixtures = false;
  std::optional<std::string> expected_tokens;
  std::optional<std::string> tokens_output;
  std::optional<std::string> tokenizer;
  std::optional<std::string> prompt;
  std::vector<std::string> prompt_token_files;
  int batch_size = 1;
  bool print_prompt_tokens = false;
  std::optional<std::string> parity_output;
  bool parity_all_rows = false;
  bool parity_row_explicit = false;
  bool parity_layer_explicit = false;
  std::optional<std::string> parity_production_output;
  bool parity_experts = false;
  int parity_expert_position = -1;
  int parity_layer = -1;
  int parity_row = 0;
  int parity_mlp_layer = -1;
  int parity_workspace_tokens = 0;
  int max_new_tokens = 0;  // Zero preserves the context/EOS-limited default.
  bool attention_math = false;
  bool attention_math_dynamic = false;
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  int no_repeat_ngram_size = 0;
  int attention_window = 0;
  int attention_sinks = 0;
  int attention_sparse_after = 0;
  uint64_t seed = 42;
  // One seed per batched sequence, in the order the sequences were given.
  // Empty means the batch derives a seed per row from `seed`.
  std::vector<uint64_t> row_seeds;
  std::vector<int> banned_tokens;
  std::vector<int> stop_tokens;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
      if (options.iterations <= 0) throw std::runtime_error("iterations must be positive");
      options.benchmark = true;
    } else if (argument == "--benchmark") {
      options.benchmark = true;
    } else if (argument == "--diagnostics") {
      options.diagnostics = true;
    } else if (argument == "--no-cuda-graph") {
      options.cuda_graph = false;
    } else if (argument == "--attention-math") {
      options.attention_math = true;
    } else if (argument == "--attention-math-dynamic") {
      options.attention_math_dynamic = true;
    } else if (argument == "--model" && index + 1 < argc) {
      options.model = argv[++index];
    } else if (argument == "--config" && index + 1 < argc) {
      options.config = argv[++index];
    } else if (argument == "--context-length" && index + 1 < argc) {
      options.context_length = std::stoi(argv[++index]);
      if (options.context_length <= 0) throw std::runtime_error("context length must be positive");
    } else if (argument == "--verify-fixtures") {
      options.verify_fixtures = true;
    } else if (argument == "--expected-tokens" && index + 1 < argc) {
      options.expected_tokens = argv[++index];
    } else if (argument == "--tokens-output" && index + 1 < argc) {
      options.tokens_output = argv[++index];
    } else if (argument == "--tokenizer" && index + 1 < argc) {
      options.tokenizer = argv[++index];
    } else if (argument == "--prompt" && index + 1 < argc) {
      options.prompt = argv[++index];
    } else if (argument == "--prompt-tokens" && index + 1 < argc) {
      options.prompt_token_files.push_back(argv[++index]);
    } else if (argument == "--print-prompt-tokens") {
      options.print_prompt_tokens = true;
    } else if (argument == "--batch-size" && index + 1 < argc) {
      options.batch_size = std::stoi(argv[++index]);
      if (options.batch_size <= 0) throw std::runtime_error("batch size must be positive");
    } else if (argument == "--parity-output" && index + 1 < argc) {
      options.parity_output = argv[++index];
    } else if (argument == "--parity-all-rows") {
      options.parity_all_rows = true;
    } else if (argument == "--parity-production-output" && index + 1 < argc) {
      options.parity_production_output = argv[++index];
    } else if (argument == "--parity-experts") {
      options.parity_experts = true;
    } else if (argument == "--parity-expert-position" && index + 1 < argc) {
      options.parity_expert_position = std::stoi(argv[++index]);
      if (options.parity_expert_position < 0) throw std::runtime_error("invalid expert trace position");
    } else if (argument == "--parity-layer" && index + 1 < argc) {
      options.parity_layer_explicit = true;
      options.parity_layer = std::stoi(argv[++index]);
    } else if (argument == "--parity-row" && index + 1 < argc) {
      options.parity_row_explicit = true;
      options.parity_row = std::stoi(argv[++index]);
    } else if (argument == "--parity-mlp-layer" && index + 1 < argc) {
      options.parity_mlp_layer = std::stoi(argv[++index]);
      if (options.parity_mlp_layer < 0) throw std::runtime_error("invalid parity MLP layer");
    } else if (argument == "--parity-workspace-tokens" && index + 1 < argc) {
      options.parity_workspace_tokens = std::stoi(argv[++index]);
      if (options.parity_workspace_tokens < 1) throw std::runtime_error("invalid parity workspace length");
    } else if (argument == "--max-new-tokens" && index + 1 < argc) {
      options.max_new_tokens = std::stoi(argv[++index]);
      if (options.max_new_tokens <= 0) throw std::runtime_error("max-new-tokens must be positive");
    } else if (argument == "--temperature" && index + 1 < argc) {
      options.temperature = std::stof(argv[++index]);
    } else if (argument == "--top-k" && index + 1 < argc) {
      options.top_k = std::stoi(argv[++index]);
    } else if (argument == "--top-p" && index + 1 < argc) {
      options.top_p = std::stof(argv[++index]);
    } else if (argument == "--min-p" && index + 1 < argc) {
      options.min_p = std::stof(argv[++index]);
    } else if (argument == "--repetition-penalty" && index + 1 < argc) {
      options.repetition_penalty = std::stof(argv[++index]);
    } else if (argument == "--presence-penalty" && index + 1 < argc) {
      options.presence_penalty = std::stof(argv[++index]);
    } else if (argument == "--frequency-penalty" && index + 1 < argc) {
      options.frequency_penalty = std::stof(argv[++index]);
    } else if (argument == "--no-repeat-ngram-size" && index + 1 < argc) {
      options.no_repeat_ngram_size = std::stoi(argv[++index]);
    } else if (argument == "--attention-window" && index + 1 < argc) {
      options.attention_window = std::stoi(argv[++index]);
    } else if (argument == "--attention-sinks" && index + 1 < argc) {
      options.attention_sinks = std::stoi(argv[++index]);
    } else if (argument == "--attention-sparse-after" && index + 1 < argc) {
      options.attention_sparse_after = std::stoi(argv[++index]);
    } else if (argument == "--seed" && index + 1 < argc) {
      options.seed = std::stoull(argv[++index]);
    } else if (argument == "--row-seed" && index + 1 < argc) {
      options.row_seeds.push_back(std::stoull(argv[++index]));
    } else if (argument == "--ban-token" && index + 1 < argc) {
      options.banned_tokens.push_back(std::stoi(argv[++index]));
    } else if (argument == "--stop-token" && index + 1 < argc) {
      options.stop_tokens.push_back(std::stoi(argv[++index]));
    } else {
      throw std::runtime_error(
          "usage: liquid-native [--benchmark] [--diagnostics] [--no-cuda-graph] "
          "[--parity-output FILE | --parity-production-output FILE] [--parity-all-rows] [--parity-layer N] [--parity-row N] "
          "[--parity-experts --parity-mlp-layer N --parity-expert-position N] "
          "[--max-new-tokens N] "
           "[--attention-math | --attention-math-dynamic] [--parity-mlp-layer N] "
          "[--iterations N] [--model CHECKPOINT_FILE_OR_DIRECTORY] [--config config.json] "
          "[--context-length N] [--verify-fixtures] "
          "[--tokenizer tokenizer.json] [--prompt text] [--expected-tokens tokens.txt] "
          "[--tokens-output tokens.txt] [--prompt-tokens tokens.txt] [--print-prompt-tokens] "
          "[--batch-size N] [--row-seed N] [--temperature F] [--top-k N] [--top-p F] "
           "[--min-p F] [--repetition-penalty F] [--presence-penalty F] "
           "[--frequency-penalty F] [--no-repeat-ngram-size N] [--seed N] "
          "[--attention-window N] [--attention-sinks N] [--attention-sparse-after N] "
           "[--ban-token ID] [--stop-token ID]");
    }
  }
  if (options.model) {
    options.model = lfm::checkpoint_path(*options.model).string();
    const auto directory = std::filesystem::path(*options.model).parent_path();
    model_config = lfm::ModelConfig::load(options.config.value_or((directory / "config.json").string()),
                                         options.context_length);
    model_config.require_execution_support();
    if (!options.tokenizer && std::filesystem::exists(directory / "tokenizer.json"))
      options.tokenizer = (directory / "tokenizer.json").string();
  } else {
    if (options.config) throw std::runtime_error("--config requires --model");
    // Only the standalone component harness has a fixed synthetic fixture.
    model_config.shape = {1024, 4608, 65536, 16, 16, 8, 64, 3, 6, 32768, 1e-5f, 1e6f};
    model_config.attention = {false, false, true, false, false, true, false, false,
                              true, false, true, false, true, false, true, false};
    model_config.eos = {7};
  }
  lfm::runtime_shape = model_config.shape;
  if (model_config.moe) {
    options.cuda_graph = false;  // Active expert groups require host dispatch.
    options.attention_math = true;  // Routing is sensitive to online-attention rounding.
  }
  if (options.parity_layer == -1) {
    const auto first = std::find(model_config.attention.begin(), model_config.attention.end(), true);
    if (first != model_config.attention.end()) options.parity_layer = first - model_config.attention.begin();
  }
  const auto& shape = lfm::runtime_shape;
  if (options.parity_all_rows && (!options.parity_output || !options.model || options.parity_production_output ||
      options.parity_row_explicit || options.parity_layer_explicit || options.parity_experts || options.parity_expert_position >= 0 ||
      options.parity_mlp_layer >= 0 || (options.attention_math && !model_config.moe && !options.attention_math_dynamic)))
    throw std::runtime_error("--parity-all-rows requires model teacher parity output; excludes row/expert/MLP selectors, production output and legacy row-zero math");
  if (options.parity_row < 0 || options.parity_row >= options.batch_size ||
      (options.parity_row != 0 && !options.parity_output && !options.parity_production_output))
    throw std::runtime_error("--parity-row must select an original batch slot and requires parity output");
  if (options.parity_production_output && (options.parity_output || options.batch_size == 1 || !options.model))
    throw std::runtime_error("--parity-production-output requires a batched model run and excludes --parity-output");
  if (options.parity_production_output && options.cuda_graph)
    throw std::runtime_error("--parity-production-output requires explicit --no-cuda-graph");
  if (options.parity_workspace_tokens && (!options.parity_output || options.parity_workspace_tokens > shape.context))
    throw std::runtime_error("--parity-workspace-tokens requires parity output and must fit the context budget");
  if (options.batch_size > 65535 || static_cast<int64_t>(options.batch_size) * shape.vocabulary > std::numeric_limits<int>::max())
    throw std::runtime_error("batch/vocabulary exceeds CUDA grid or segmented-sort index limits");
  lfm::checked_product({2, static_cast<size_t>(options.batch_size), static_cast<size_t>(shape.hidden),
                         static_cast<size_t>(shape.layers), static_cast<size_t>(shape.conv_width)});
  lfm::checked_product({2, static_cast<size_t>(options.batch_size), shape.kv_batch_stride()});
  kHiddenSize = shape.hidden; kMlpWidth = shape.mlp; kVocabularySize = shape.vocabulary;
  kAttentionLayers = shape.attention_layers; kQueryHeads = shape.query_heads;
  kKvHeads = shape.kv_heads; kHeadDimension = shape.head_dim; kContextLength = shape.context;
  kKvWidth = shape.kv_width(); kQkvWidth = shape.qkv_width();
  kKvLayerStride = shape.kv_layer_stride(); kKvBatchStride = shape.kv_batch_stride();
  kTopKBlocks = (kVocabularySize + kTopKBlockSize - 1) / kTopKBlockSize;
  options.stop_tokens.insert(options.stop_tokens.end(), model_config.eos.begin(), model_config.eos.end());
  std::sort(options.stop_tokens.begin(), options.stop_tokens.end());
  options.stop_tokens.erase(std::unique(options.stop_tokens.begin(), options.stop_tokens.end()), options.stop_tokens.end());
  legacy_fixtures = options.verify_fixtures;
  reference_math_norms = options.attention_math_dynamic;
  if (legacy_fixtures && (!options.model || kHiddenSize != 1024 || kVocabularySize != 65536 ||
                          shape.layers != 16 || shape.conv_width != 3 || kAttentionLayers != 6))
    throw std::runtime_error("--verify-fixtures requires the original 350M fixture checkpoint");
  if (options.parity_mlp_layer >= shape.layers) throw std::runtime_error("invalid parity MLP layer");
  if ((options.parity_experts || options.parity_expert_position >= 0) &&
      (!options.parity_experts || !options.parity_output || !model_config.moe || options.batch_size > 32 ||
       options.parity_expert_position < 0 || options.parity_mlp_layer < model_config.dense_layers || options.parity_mlp_layer >= shape.layers))
    throw std::runtime_error("expert operation tracing requires --parity-output, --parity-experts, a MoE --parity-mlp-layer, and batch <= 32");
  if (options.attention_math && !model_config.moe && (kQueryHeads != 16 || kKvHeads != 8 || kHeadDimension != 64 || kContextLength > 32768))
    throw std::runtime_error("diagnostic math-attention supports only its 16/8/64 reference geometry");
  if (!std::isfinite(options.temperature) || !std::isfinite(options.top_p) || !std::isfinite(options.min_p) ||
      !std::isfinite(options.repetition_penalty) || !std::isfinite(options.presence_penalty) || !std::isfinite(options.frequency_penalty))
    throw std::runtime_error("invalid nonfinite sampling hyperparameter");
  if (options.temperature < 0.0f || options.top_k < 0 || options.top_k > kVocabularySize ||
      options.top_p <= 0.0f || options.top_p > 1.0f ||
      options.min_p < 0.0f || options.min_p > 1.0f ||
      options.repetition_penalty <= 0.0f || options.no_repeat_ngram_size < 0 ||
      options.attention_window < 0 || options.attention_window > kContextLength ||
      options.attention_sinks < 0 || options.attention_sinks > kContextLength ||
      options.attention_sparse_after < 0 || options.attention_sparse_after >= kContextLength) {
    throw std::runtime_error("invalid sampling hyperparameter");
  }
  auto validate_ids = [](const std::vector<int>& ids) {
    for (const int id : ids) if (id < 0 || id >= kVocabularySize) throw std::runtime_error("token ID option out of range");
  };
  validate_ids(options.banned_tokens);
  validate_ids(options.stop_tokens);
  if (options.attention_math && options.attention_math_dynamic && !model_config.moe)
    throw std::runtime_error("select either legacy or dynamic math attention, not both");
  if (options.attention_math || options.attention_math_dynamic) {
    options.cuda_graph = false;  // Host-length-driven BLAS is not graph-compatible.
    if (options.attention_window != 0) throw std::runtime_error("math attention requires full prefix attention");
  }
  if (!options.row_seeds.empty() &&
      options.row_seeds.size() != static_cast<size_t>(options.batch_size)) {
    throw std::runtime_error("give one --row-seed per batched sequence");
  }
  // A singleton is still one logical request. Honor its explicit row seed in
  // the scalar sampler rather than silently falling back to the base seed.
  if (options.batch_size == 1 && !options.row_seeds.empty())
    options.seed = options.row_seeds.front();
  if (options.batch_size > 1) {
    if (options.repetition_penalty != 1.0f || options.presence_penalty != 0.0f ||
        options.frequency_penalty != 0.0f || options.no_repeat_ngram_size > 0 ||
        !options.banned_tokens.empty()) {
      throw std::runtime_error(
          "the batched path does not support the penalty options; remove them");
    }
    if (options.attention_math && !model_config.moe) {
      throw std::runtime_error("the batched path does not support --attention-math");
    }
    if (options.expected_tokens) {
      throw std::runtime_error("--expected-tokens is not supported with --batch-size");
    }
    if (options.prompt_token_files.size() > 1 &&
        options.prompt_token_files.size() != static_cast<size_t>(options.batch_size)) {
      throw std::runtime_error(
          "give either one --prompt-tokens file or exactly one per batched sequence");
    }
  }
  return options;
}

class Sampler {
 public:
  Sampler(const Options& options, const std::vector<int>& prompt)
      : options_(options),
        counts_(kVocabularySize),
        base_banned_mask_(kVocabularySize),
        banned_mask_(kVocabularySize),
        history_(kContextLength),
        history_length_(1),
        adjusted_(kVocabularySize),
        sorted_(kVocabularySize),
        exponentials_(kVocabularySize),
        indices_(kVocabularySize),
        sorted_indices_(kVocabularySize),
        topk_candidates_(kTopKBlocks * kMaximumHierarchicalTopK),
        topk_sorted_(kTopKBlocks * kMaximumHierarchicalTopK),
        topk_candidate_indices_(kTopKBlocks * kMaximumHierarchicalTopK),
        topk_sorted_indices_(kTopKBlocks * kMaximumHierarchicalTopK) {
    std::vector<int> counts(kVocabularySize);
    for (const int token : prompt) ++counts[token];
    check_cuda(cudaMemcpy(counts_.data, counts.data(), counts.size() * sizeof(counts[0]), cudaMemcpyHostToDevice),
               "copy sampler token counts");
    std::vector<uint8_t> base_banned(kVocabularySize);
    for (const int token : options_.banned_tokens) base_banned[token] = 1;
    check_cuda(cudaMemcpy(base_banned_mask_.data, base_banned.data(), base_banned.size(), cudaMemcpyHostToDevice),
               "copy base banned-token mask");
    check_cuda(cudaMemcpy(history_.data, prompt.data(), prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice),
               "copy sampler prompt history");
    const int prompt_length = prompt.size();
    check_cuda(cudaMemcpy(history_length_.data, &prompt_length, sizeof(prompt_length), cudaMemcpyHostToDevice),
               "copy sampler history length");
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temporary_bytes_, adjusted_.data, sorted_.data,
        indices_.data, sorted_indices_.data, kVocabularySize), "query sampler sort storage");
    check_cuda(cudaMalloc(&temporary_, temporary_bytes_), "allocate sampler sort storage");
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        topk_temporary_bytes_,
        topk_candidates_.data,
        topk_sorted_.data,
        topk_candidate_indices_.data,
        topk_sorted_indices_.data,
        topk_candidates_.size),
        "query hierarchical top-k sort storage");
    check_cuda(cudaMalloc(&topk_temporary_, topk_temporary_bytes_),
               "allocate hierarchical top-k sort storage");
  }

  ~Sampler() {
    cudaFree(topk_temporary_);
    cudaFree(temporary_);
  }

  void select_device(
      const __nv_bfloat16* logits,
      uint64_t step,
      int* destination) {
    LFM_PROFILE_RANGE(profile_sampler, "sampler.singleton");
    check_cuda(cudaMemcpyAsync(
        banned_mask_.data,
        base_banned_mask_.data,
        banned_mask_.size * sizeof(uint8_t),
        cudaMemcpyDeviceToDevice),
        "reset banned-token mask");
    mark_no_repeat_ngram<<<1, 1>>>(
        banned_mask_.data,
        history_.data,
        history_length_.data,
        options_.no_repeat_ngram_size);
    prepare_logits<<<(kVocabularySize + 255) / 256, 256>>>(
        logits,
        counts_.data,
        banned_mask_.data,
        adjusted_.data,
        indices_.data,
        options_.temperature,
        options_.repetition_penalty,
        options_.presence_penalty,
        options_.frequency_penalty);
    const float* selected_logits = sorted_.data;
    const int* selected_indices = sorted_indices_.data;
    const int limit = options_.top_k > 0 && options_.top_k < kVocabularySize
        ? options_.top_k
        : kVocabularySize;
    if (options_.top_k > 0 && options_.top_k <= kMaximumHierarchicalTopK) {
      block_topk<<<kTopKBlocks, kTopKBlockSize>>>(
          adjusted_.data,
          topk_candidates_.data,
          topk_candidate_indices_.data,
          options_.top_k);
      const int candidate_count = kTopKBlocks * options_.top_k;
      check_cuda(cub::DeviceRadixSort::SortPairsDescending(
          topk_temporary_,
          topk_temporary_bytes_,
          topk_candidates_.data,
          topk_sorted_.data,
          topk_candidate_indices_.data,
          topk_sorted_indices_.data,
          candidate_count),
          "sort hierarchical top-k candidates");
      selected_logits = topk_sorted_.data;
      selected_indices = topk_sorted_indices_.data;
    } else {
      check_cuda(cub::DeviceRadixSort::SortPairsDescending(
          temporary_, temporary_bytes_, adjusted_.data, sorted_.data,
          indices_.data, sorted_indices_.data, kVocabularySize), "sort sampler logits");
    }
    exponentiate_sorted_logits<<<(limit + 255) / 256, 256>>>(
        selected_logits, exponentials_.data, limit);
    check_cuda(cudaMemset(eligible_.data, 0, sizeof(int)), "reset eligible count");
    count_eligible_exponentials<<<(limit + 255) / 256, 256>>>(
        exponentials_.data, limit, options_.min_p, eligible_.data);
    sum_exponentials<<<1, 1>>>(
        exponentials_.data, eligible_.data, total_.data);
    select_sorted_token<<<1, 1>>>(
        exponentials_.data,
        total_.data,
        eligible_.data,
        selected_indices,
        destination,
        options_.top_p,
        options_.seed,
        step,
        options_.temperature > 0.0f);
    check_cuda(cudaGetLastError(), "sampler kernels");
    increment_selected_count<<<1, 1>>>(counts_.data, destination);
    append_selected_token<<<1, 1>>>(history_.data, history_length_.data, destination);
  }

 private:
  const Options& options_;
  DeviceBuffer<int> counts_;
  DeviceBuffer<uint8_t> base_banned_mask_;
  DeviceBuffer<uint8_t> banned_mask_;
  DeviceBuffer<int> history_;
  DeviceBuffer<int> history_length_;
  DeviceBuffer<float> adjusted_;
  DeviceBuffer<float> sorted_;
  DeviceBuffer<float> exponentials_;
  DeviceBuffer<float> total_{1};
  DeviceBuffer<int> eligible_{1};
  DeviceBuffer<int> indices_;
  DeviceBuffer<int> sorted_indices_;
  DeviceBuffer<float> topk_candidates_;
  DeviceBuffer<float> topk_sorted_;
  DeviceBuffer<int> topk_candidate_indices_;
  DeviceBuffer<int> topk_sorted_indices_;
  void* temporary_ = nullptr;
  size_t temporary_bytes_ = 0;
  void* topk_temporary_ = nullptr;
  size_t topk_temporary_bytes_ = 0;
};

struct BatchGenerationResult {
  std::vector<std::vector<int>> generated;
  std::vector<uint64_t> hashes;
  std::vector<int> counts;
  std::vector<std::string> end_reasons;
  double seconds = 0.0;
  double tokens_per_second = 0.0;
  size_t total_tokens = 0;
  // The order the caller supplied the sequences in. The batch sorts them by
  // prompt length internally, so the results are reported back in the
  // caller's order.
  std::vector<int> order;
};

// Per-row sampler for the batched path.
//
// This is the single-sequence sampler with the vocabulary axis split into
// rows, one row per sequence. The arithmetic inside a row is unchanged, so a
// row given the single-sequence seed reproduces that sequence's answer
// exactly, and rows differ only in the seed they draw from.
class BatchSampler {
 public:
  BatchSampler(const Options& options, int capacity)
      : options_(options),
        capacity_(capacity),
        adjusted_(static_cast<size_t>(capacity) * kVocabularySize),
        sorted_(static_cast<size_t>(capacity) * kVocabularySize),
        indices_(static_cast<size_t>(capacity) * kVocabularySize),
        sorted_indices_(static_cast<size_t>(capacity) * kVocabularySize),
        exponentials_(static_cast<size_t>(capacity) * kVocabularySize),
        total_(capacity),
        eligible_(capacity),
        row_seeds_(capacity),
        segment_starts_(capacity),
        segment_ends_(capacity) {
    std::vector<int> starts(capacity);
    std::vector<int> ends(capacity);
    for (int row = 0; row < capacity; ++row) {
      starts[row] = row * kVocabularySize;
      ends[row] = (row + 1) * kVocabularySize;
    }
    check_cuda(cudaMemcpy(segment_starts_.data, starts.data(),
                          starts.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy batched sampler segment starts");
    check_cuda(cudaMemcpy(segment_ends_.data, ends.data(),
                          ends.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy batched sampler segment ends");
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(
                   nullptr, temporary_bytes_,
                   adjusted_.data, sorted_.data,
                   indices_.data, sorted_indices_.data,
                   capacity * kVocabularySize, capacity,
                   segment_starts_.data, segment_ends_.data,
                   0, sizeof(float) * 8, cudaStreamPerThread),
               "query batched sampler sort storage");
    check_cuda(cudaMalloc(&temporary_, temporary_bytes_),
               "allocate batched sampler sort storage");
  }

  ~BatchSampler() { cudaFree(temporary_); }

  void set_row_seeds(const std::vector<uint64_t>& seeds) {
    require(seeds.size() == static_cast<size_t>(capacity_),
            "batched sampler seed count does not match the batch size");
    check_cuda(cudaMemcpy(row_seeds_.data, seeds.data(),
                          seeds.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "copy batched sampler row seeds");
  }

  // Draw one token for each of the first `rows` rows.
  void select_device(const __nv_bfloat16* logits, int rows, uint64_t step, int* destination) {
    LFM_PROFILE_RANGE(profile_sampler, "sampler.batch");
    require(rows > 0 && rows <= capacity_, "batched sampler row count out of range");
    lfm::prepare_logits_rows<<<dim3((kVocabularySize + 255) / 256, rows), 256>>>(
        logits, adjusted_.data, indices_.data, kVocabularySize, options_.temperature);
    // The segmented sort is a stable descending sort, so equal logits keep
    // the same lowest-index-first order the single-sequence sort produced.
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(
                   temporary_, temporary_bytes_,
                   adjusted_.data, sorted_.data,
                   indices_.data, sorted_indices_.data,
                   capacity_ * kVocabularySize, rows,
                   segment_starts_.data, segment_ends_.data,
                   0, sizeof(float) * 8, cudaStreamPerThread),
               "sort batched sampler logits");
    const int limit = options_.top_k > 0 && options_.top_k < kVocabularySize
        ? options_.top_k
        : kVocabularySize;
    lfm::exponentiate_sorted_logits_rows<<<dim3((limit + 255) / 256, rows), 256>>>(
        sorted_.data, exponentials_.data, kVocabularySize, limit);
    check_cuda(cudaMemset(eligible_.data, 0, rows * sizeof(int)), "reset eligible count");
    lfm::count_eligible_exponentials_rows<<<dim3((limit + 255) / 256, rows), 256>>>(
        exponentials_.data, eligible_.data, kVocabularySize, limit, options_.min_p);
    lfm::sum_exponentials_rows<<<rows, 1>>>(
        exponentials_.data, eligible_.data, total_.data, kVocabularySize);
    lfm::select_sorted_token_rows<<<rows, 1>>>(
        exponentials_.data, total_.data, eligible_.data, sorted_indices_.data, row_seeds_.data,
        destination, kVocabularySize, options_.top_p, step, options_.temperature > 0.0f);
    check_cuda(cudaGetLastError(), "batched sampler kernels");
  }

 private:
  const Options& options_;
  const int capacity_;
  DeviceBuffer<float> adjusted_;
  DeviceBuffer<float> sorted_;
  DeviceBuffer<int> indices_;
  DeviceBuffer<int> sorted_indices_;
  DeviceBuffer<float> exponentials_;
  DeviceBuffer<float> total_;
  DeviceBuffer<int> eligible_;
  DeviceBuffer<uint64_t> row_seeds_;
  DeviceBuffer<int> segment_starts_;
  DeviceBuffer<int> segment_ends_;
  void* temporary_ = nullptr;
  size_t temporary_bytes_ = 0;
};

// Decode `prompts.size()` independent sequences in lockstep.
//
// Each sequence owns its KV cache slice, its convolution state, its position,
// its generated-token buffer and its stop state. The projections run once per
// batch, so the weight matrix is read once for the whole batch instead of once
// per sequence; that is the entire point of the batched path.
//
// The sequences are sorted by prompt length, longest first, so the sequences
// that are still inside their prompt are always a prefix of the batch. The
// prefix shrinks as the shorter prompts finish, which is what lets prompts of
// different lengths share one batch: a sequence that has finished its prompt
// is simply not part of the prefix any more, so its hidden state, its
// convolution state and its KV cache are left exactly where the prompt left
// them.
//
// The penalty options (repetition, presence, frequency, no-repeat-ngram and
// banned tokens) stay single-sequence only: they keep per-token histories and
// token counts that would need their own batched reduction. Temperature,
// top-k, top-p and min-p are per row and are sampled here.
BatchGenerationResult run_batched_generation(
    cublasHandle_t handle,
    const ModelWeights& model,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache,
    const std::vector<std::vector<int>>& prompts,
    const Options& options) {
  const int batch = static_cast<int>(prompts.size());
  require(batch > 0, "batch must contain at least one sequence");
  LayerBuffers buffers(batch);

  // Sort the sequences by prompt length, longest first. The sequences that are
  // still inside their prompt are then a prefix of the batch, so the batch can
  // shrink as the shorter prompts finish without any of the layer kernels
  // needing a row offset: every kernel indexes its row from blockIdx.
  std::vector<int> order(batch);
  for (int index = 0; index < batch; ++index) order[index] = index;
  std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
    return prompts[left].size() > prompts[right].size();
  });
  std::vector<size_t> lengths(batch);
  for (int index = 0; index < batch; ++index) lengths[index] = prompts[order[index]].size();
  const size_t longest = lengths[0];

  // Diagnostics only: follow the real sorted/shrinking production prefill.
  // Do not substitute an equal-length teacher run or change the sampled path.
  const int trace_slot = std::find(order.begin(), order.end(), options.parity_row) - order.begin();
  std::ofstream trace_states, trace_events, trace_rows;
  if (options.parity_production_output) {
    trace_states.open(*options.parity_production_output, std::ios::binary);
    trace_events.open(*options.parity_production_output + ".events.tsv");
    trace_rows.open(*options.parity_production_output + ".rows.json");
    if (!trace_states || !trace_events || !trace_rows) throw std::runtime_error("cannot open production parity trace");
    parity_stream = &trace_states;
    parity_row = trace_slot;
    trace_events << "phase\titeration\toriginal_row\tphysical_slot\tactive\tposition\tinput_token\tcompute_head\tstate_byte_offset\tselected_token\n";
    trace_rows << "{\"schema\":1,\"selected_row\":" << options.parity_row
               << ",\"physical_slot\":" << trace_slot << ",\"batch_size\":" << batch << ",\"order\":[";
    for (int slot = 0; slot < batch; ++slot) { if (slot) trace_rows << ','; trace_rows << order[slot]; }
    trace_rows << "],\"prompt_ids\":[";
    for (int original = 0; original < batch; ++original) {
      if (original) trace_rows << ',';
      trace_rows << '[';
      for (size_t pos = 0; pos < prompts[original].size(); ++pos) { if (pos) trace_rows << ','; trace_rows << prompts[original][pos]; }
      trace_rows << ']';
    }
    trace_rows << "]}\n";
    trace_rows.flush();
    if (!trace_rows) throw std::runtime_error("write production parity metadata failed");
  }
  auto trace_event = [&](const char* phase, size_t iteration, int active, bool compute_head,
                         std::streampos offset) {
    if (!options.parity_production_output || !parity_capture_enabled) return;
    int input, position, selected = -1;
    check_cuda(cudaMemcpy(&input, buffers.input_token.data + trace_slot, sizeof(int), cudaMemcpyDeviceToHost), "trace production input");
    check_cuda(cudaMemcpy(&position, buffers.position.data + trace_slot, sizeof(int), cudaMemcpyDeviceToHost), "trace production position");
    if (compute_head) {
      parity_dump(buffers.logits.data, kVocabularySize);
      check_cuda(cudaMemcpy(&selected, buffers.selected_token.data + trace_slot, sizeof(int), cudaMemcpyDeviceToHost), "trace production selection");
    }
    trace_events << phase << '\t' << iteration << '\t' << options.parity_row << '\t' << trace_slot
                 << '\t' << active << '\t' << position << '\t' << input << '\t' << compute_head
                 << '\t' << offset << '\t' << selected << '\n';
    trace_events.flush();
    if (!trace_events) throw std::runtime_error("write production parity event failed");
  };

  const int attention_policy[] = {
      options.attention_window,
      options.attention_sinks,
      options.attention_sparse_after};
  check_cuda(
      cudaMemcpy(buffers.attention_policy.data, attention_policy,
                 sizeof(attention_policy), cudaMemcpyHostToDevice),
      "copy attention policy");

  DeviceBuffer<int> stop_tokens(std::max<size_t>(1, options.stop_tokens.size()));
  if (!options.stop_tokens.empty()) {
    check_cuda(cudaMemcpy(stop_tokens.data, options.stop_tokens.data(),
                           options.stop_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy stop tokens");
  }
  DeviceBuffer<int> generated(static_cast<size_t>(batch) * kContextLength);
  DeviceBuffer<int> generated_count(batch);
  DeviceBuffer<int> stop_flag(batch);
  DeviceBuffer<int> active_count(batch + 1);
  std::vector<int> stop_snapshot(batch + 1);
  buffers.stopped_rows = stop_flag.data;
  // Every sequence has its own context budget: the number of tokens it can
  // generate before it reaches the end of the 32K context.
  DeviceBuffer<int> context_limits(batch);
  check_cuda(cudaMemset(generated_count.data, 0, batch * sizeof(int)), "reset generated count");
  check_cuda(cudaMemset(stop_flag.data, 0, batch * sizeof(int)), "reset stop flag");
  std::vector<int> host_limits(batch);
  for (int index = 0; index < batch; ++index) {
    host_limits[index] = static_cast<int>(kContextLength - lengths[index]);
  }
  check_cuda(cudaMemcpy(context_limits.data, host_limits.data(),
                        host_limits.size() * sizeof(int), cudaMemcpyHostToDevice),
             "copy batch context limits");

  const bool sampling = options.temperature > 0.0f || options.top_k > 0 ||
                        options.top_p < 1.0f || options.min_p > 0.0f;
  std::unique_ptr<BatchSampler> sampler;
  if (sampling) {
    sampler = std::make_unique<BatchSampler>(options, batch);
    // The row seeds follow the caller's order, so a caller that names a seed
    // per sequence gets the same draw whichever row the sequence ends up in.
    std::vector<uint64_t> row_seeds(batch);
    for (int index = 0; index < batch; ++index) {
      const int original = order[index];
      row_seeds[index] = options.row_seeds.empty()
          ? options.seed + static_cast<uint64_t>(index) * kRowSeedStride
          : options.row_seeds[original];
    }
    sampler->set_row_seeds(row_seeds);
  }

  // Prompt phase: every sequence advances through its own prompt. The
  // sequences still inside their prompt are a prefix, and the prefix shrinks
  // as the shorter prompts finish.
  std::vector<int> host_tokens(batch);
  std::vector<int> host_positions(batch);
  LFM_PROFILE_RANGE(profile_prefill, "phase.prefill");
  for (size_t position = 0; position < longest; ++position) {
    int active = batch;
    while (active > 0 && lengths[active - 1] <= position) --active;
    if (active == 0) break;
    for (int index = 0; index < active; ++index) {
      host_tokens[index] = prompts[order[index]][position];
      host_positions[index] = static_cast<int>(position);
    }
    check_cuda(cudaMemcpy(buffers.input_token.data, host_tokens.data(),
                          active * sizeof(int), cudaMemcpyHostToDevice), "copy batch tokens");
    check_cuda(cudaMemcpy(buffers.position.data, host_positions.data(),
                          active * sizeof(int), cudaMemcpyHostToDevice), "copy batch positions");
    // The projection result is consumed only by the sequences whose prompt
    // ends at this position; every other position can skip it, which removes
    // one 65536x1024 projection per prompt token.
    bool compute_head = false;
    for (int index = 0; index < active; ++index) {
      if (lengths[index] == position + 1) {
        compute_head = true;
        break;
      }
    }
    buffers.active = active;
    parity_capture_enabled = !options.parity_production_output || trace_slot < active;
    const auto trace_offset = options.parity_production_output ? trace_states.tellp() : std::streampos(0);
    run_full_token_device(handle, model, key_cache, value_cache, buffers,
                          position == 0, false, nullptr, nullptr, compute_head, !sampling);
    if (compute_head && sampling) {
      // The draw for a sequence that ends here is the one the single-sequence
      // path would make at step zero.
      sampler->select_device(buffers.logits.data, active, 0, buffers.selected_token.data);
    }
    trace_event("prefill", position, active, compute_head, trace_offset);
  }

  LFM_PROFILE_END(profile_prefill);
  // Each sequence starts decoding at the end of its own prompt.
  for (int index = 0; index < batch; ++index) {
    host_positions[index] = static_cast<int>(lengths[index]);
  }
  check_cuda(cudaMemcpy(buffers.position.data, host_positions.data(),
                        batch * sizeof(int), cudaMemcpyHostToDevice), "advance batch positions");
  advance_selected_tokens<<<(batch + 255) / 256, 256>>>(
      buffers.input_token.data, buffers.selected_token.data, batch);

  buffers.active = batch;
  const bool use_batch_graph = options.cuda_graph && !model_config.moe && !model.math_attention &&
      !options.attention_math && !options.attention_math_dynamic && options.max_new_tokens != 1;
  // Use the same synchronized prefill boundary for eager and graph controls.
  // Setup is outside decode timing but included in whole-process measurements.
  const auto graph_setup_start = std::chrono::steady_clock::now();
  check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "batch prefill boundary");
  std::unique_ptr<BatchDecodeGraph> batch_graph;
  if (use_batch_graph) batch_graph = std::make_unique<BatchDecodeGraph>(
      handle, model, key_cache, value_cache, buffers, !sampling);
  const double graph_setup_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - graph_setup_start).count();
  std::cout << "batch_graph_mode: " << (batch_graph ? "forward" : "eager") << "\n";
  std::cout << "batch_graph_setup_seconds: " << graph_setup_seconds << "\n";
  LFM_PROFILE_RANGE(profile_decode, "phase.decode");
  const auto generation_start = std::chrono::steady_clock::now();
  int step = 0;
  while (true) {
    check_cuda(cudaMemset(active_count.data, 0, sizeof(int)), "reset active count");
    record_token_and_check_stop_batch<<<(batch + 255) / 256, 256>>>(
        buffers.input_token.data,
        generated.data,
        generated_count.data,
        stop_flag.data,
        stop_tokens.data,
        static_cast<int>(options.stop_tokens.size()),
        active_count.data,
        batch,
        context_limits.data, lfm::runtime_shape, active_count.data + 1);
    // Piggyback row stop flags on the existing per-step synchronization. The
    // host snapshot lets math attention omit finished rows from its groups.
    check_cuda(cudaMemcpy(stop_snapshot.data(), active_count.data,
                         stop_snapshot.size() * sizeof(int), cudaMemcpyDeviceToHost),
               "copy active count and stopped rows");
    const int active = stop_snapshot[0];
    std::copy(stop_snapshot.begin() + 1, stop_snapshot.end(), buffers.host_stopped_rows.begin());
    ++step;
    if (active == 0) break;
    if (options.max_new_tokens > 0 && step >= options.max_new_tokens) break;
    parity_capture_enabled = !options.parity_production_output || !stop_snapshot[trace_slot + 1];
    const auto trace_offset = options.parity_production_output ? trace_states.tellp() : std::streampos(0);
    if (batch_graph) batch_graph->launch();
    else run_full_token_device(handle, model, key_cache, value_cache, buffers,
                               false, false, nullptr, nullptr, true, !sampling);
    if (sampling) {
      sampler->select_device(buffers.logits.data, batch, step, buffers.selected_token.data);
    }
    trace_event("decode", step, batch, true, trace_offset);
    increment_position_batch<<<(batch + 255) / 256, 256>>>(
        buffers.position.data, batch);
    advance_selected_tokens<<<(batch + 255) / 256, 256>>>(
        buffers.input_token.data, buffers.selected_token.data, batch);
  }
  const auto generation_end = std::chrono::steady_clock::now();
  LFM_PROFILE_END(profile_decode);
  if (options.parity_production_output) {
    parity_stream = nullptr;
    parity_row = 0;
    parity_capture_enabled = true;
    trace_states.flush();
    if (!trace_states) throw std::runtime_error("write production parity states failed");
  }

  BatchGenerationResult result;
  // Prefill's `order` maps sorted rows to caller rows. Consumers index this
  // vector by caller row, so publish its inverse. A two-row swap is its own
  // inverse and hid this bug; three-way length groups expose the mislabeling.
  result.order.resize(batch);
  for (int sorted = 0; sorted < batch; ++sorted) result.order[order[sorted]] = sorted;
  result.seconds = std::chrono::duration<double>(generation_end - generation_start).count();
  std::vector<int> counts(batch);
  check_cuda(cudaMemcpy(counts.data(), generated_count.data, batch * sizeof(int),
                        cudaMemcpyDeviceToHost), "copy batch generated counts");
  result.generated.resize(batch);
  result.hashes.resize(batch);
  result.counts = counts;
  result.end_reasons.resize(batch);
  for (int index = 0; index < batch; ++index) {
    require(counts[index] >= 0 && counts[index] <= kContextLength, "invalid generated row length");
    result.generated[index].resize(counts[index]);
    if (counts[index]) check_cuda(cudaMemcpy(result.generated[index].data(),
        generated.data + static_cast<size_t>(index) * kContextLength,
        static_cast<size_t>(counts[index]) * sizeof(int), cudaMemcpyDeviceToHost), "copy live generated row");
    result.hashes[index] = fnv1a_tokens(result.generated[index]);
    const auto& tokens = result.generated[index];
    const bool stopped = !tokens.empty() && std::binary_search(
        options.stop_tokens.begin(), options.stop_tokens.end(), tokens.back());
    result.end_reasons[index] = stopped ? "stop_token" :
        options.max_new_tokens > 0 && counts[index] >= options.max_new_tokens
            ? "token_limit" : "context_limit";
    result.total_tokens += result.generated[index].size();
  }
  if (result.seconds > 0.0) {
    result.tokens_per_second = result.total_tokens / result.seconds;
  }
  return result;
}

std::vector<int> load_expected_tokens(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open expected token fixture: " + path);
  std::vector<int> tokens;
  int token = 0;
  while (stream >> token) tokens.push_back(token);
  if (!stream.eof() || tokens.empty()) throw std::runtime_error("invalid expected token fixture");
  return tokens;
}

void require_shape(const lfm::Safetensors& model, const std::string& name, std::initializer_list<int64_t> shape,
                    const std::string& dtype = "BF16") {
  require(model.at(name).shape == std::vector<int64_t>(shape), "unexpected shape for " + name);
  require(model.at(name).dtype == dtype, "unexpected dtype for " + name);
}

void validate_model(const lfm::Safetensors& model) {
  require_shape(model, "model.embed_tokens.weight", {kVocabularySize, kHiddenSize});
  require_shape(model, "model.embedding_norm.weight", {kHiddenSize});
  if (!model_config.tied) require_shape(model, "lm_head.weight", {kVocabularySize, kHiddenSize});
  size_t tensor_count = 2 + !model_config.tied;
  for (int layer = 0; layer < model_config.shape.layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    require_shape(model, prefix + "operator_norm.weight", {kHiddenSize});
    require_shape(model, prefix + "ffn_norm.weight", {kHiddenSize});
    tensor_count += 2;
    if (model_config.moe && layer >= model_config.dense_layers) {
      const auto ff = prefix + "feed_forward.";
      const int experts = model_config.num_experts, width = model_config.expert_width;
      require_shape(model, ff + "gate.weight", {experts, kHiddenSize});
      ++tensor_count;
      if (model_config.use_expert_bias) {
        require_shape(model, ff + "expert_bias", {experts}, "F32");
        ++tensor_count;
      }
      if (model.has(ff + "experts.gate_up_proj")) {
        require_shape(model, ff + "experts.gate_up_proj", {experts, 2 * width, kHiddenSize});
        require_shape(model, ff + "experts.down_proj", {experts, kHiddenSize, width});
        tensor_count += 2;
      } else for (int e = 0; e < experts; ++e) {
        const auto expert = ff + "experts." + std::to_string(e) + ".";
        require_shape(model, expert + "w1.weight", {width, kHiddenSize});
        require_shape(model, expert + "w3.weight", {width, kHiddenSize});
        require_shape(model, expert + "w2.weight", {kHiddenSize, width});
        tensor_count += 3;
      }
    } else {
      require_shape(model, prefix + "feed_forward.w1.weight", {kMlpWidth, kHiddenSize});
      require_shape(model, prefix + "feed_forward.w3.weight", {kMlpWidth, kHiddenSize});
      require_shape(model, prefix + "feed_forward.w2.weight", {kHiddenSize, kMlpWidth});
      tensor_count += 3;
    }
    if (model_config.attention[layer]) {
      require_shape(model, prefix + "self_attn.q_proj.weight", {kHiddenSize, kHiddenSize});
      require_shape(model, prefix + "self_attn.k_proj.weight", {kKvWidth, kHiddenSize});
      require_shape(model, prefix + "self_attn.v_proj.weight", {kKvWidth, kHiddenSize});
      require_shape(model, prefix + "self_attn.out_proj.weight", {kHiddenSize, kHiddenSize});
      require_shape(model, prefix + "self_attn.q_layernorm.weight", {kHeadDimension});
      require_shape(model, prefix + "self_attn.k_layernorm.weight", {kHeadDimension});
      tensor_count += 6;
    } else {
      require_shape(model, prefix + "conv.conv.weight", {kHiddenSize, 1, model_config.shape.conv_width});
      require_shape(model, prefix + "conv.in_proj.weight", {3 * kHiddenSize, kHiddenSize});
      require_shape(model, prefix + "conv.out_proj.weight", {kHiddenSize, kHiddenSize});
      tensor_count += 3;
    }
  }
  require(model.tensor_count() == tensor_count, "checkpoint has unexpected tensors for the configured operators");
}

// Schema 2: [position][layer][original row][hidden], followed at each position
// by [original row][vocabulary]. No selected-row side streams or fused probe.
void run_all_rows_trace(cublasHandle_t handle, const ModelWeights& model,
                        __nv_bfloat16* keys, __nv_bfloat16* values,
                        const std::vector<std::vector<int>>& prompts, const Options& options) {
  std::ofstream stream(*options.parity_output, std::ios::binary);
  std::ofstream metadata(*options.parity_output + ".rows.json");
  if (!stream || !metadata) throw std::runtime_error("cannot open all-row trace");
  metadata << "{\"schema\":2,\"mode\":\"all_rows_teacher\",\"dtype\":\"bf16_le\","
           << "\"layout\":\"position,tensor,original_row,element\",\"batch_size\":" << prompts.size()
           << ",\"positions\":" << prompts[0].size() << ",\"layers\":" << model_config.shape.layers
           << ",\"hidden_size\":" << kHiddenSize << ",\"vocabulary_size\":" << kVocabularySize
           << ",\"original_slots\":[";
  for (size_t row = 0; row < prompts.size(); ++row) { if (row) metadata << ','; metadata << row; }
  metadata << "],\"teacher_ids\":[";
  for (size_t row = 0; row < prompts.size(); ++row) {
    if (row) metadata << ',';
    metadata << '[';
    for (size_t pos = 0; pos < prompts[row].size(); ++pos) { if (pos) metadata << ','; metadata << prompts[row][pos]; }
    metadata << ']';
  }
  metadata << "]}\n";
  metadata.flush();
  if (!metadata) throw std::runtime_error("write all-row metadata failed");
  LayerBuffers buffers(options.batch_size);
  const int policy[] = {options.attention_window, options.attention_sinks, options.attention_sparse_after};
  check_cuda(cudaMemcpy(buffers.attention_policy.data, policy, sizeof(policy), cudaMemcpyHostToDevice), "all-row policy");
  parity_stream = &stream;
  parity_all_row_count = options.batch_size;
  std::vector<int> tokens(options.batch_size), positions(options.batch_size);
  for (int position = 0; position < static_cast<int>(prompts[0].size()); ++position) {
    for (int row = 0; row < options.batch_size; ++row) { tokens[row] = prompts[row][position]; positions[row] = position; }
    check_cuda(cudaMemcpy(buffers.input_token.data, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice), "all-row tokens");
    check_cuda(cudaMemcpy(buffers.position.data, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice), "all-row positions");
    run_full_token_device(handle, model, keys, values, buffers, position == 0);
    parity_dump(buffers.logits.data, kVocabularySize);
  }
  parity_stream = nullptr;
  parity_all_row_count = 0;
  stream.flush();
  if (!stream) throw std::runtime_error("flush all-row trace failed");
  std::cout << "parity_all_rows: " << prompts.size() << "\nparity_positions: " << prompts[0].size() << '\n';
}

// Teacher-forced tracing of row zero at any batch width. This compares real
// activations under the exact production launch geometry, not just synthetic
// GEMM vectors. All rows must have the same length for this diagnostic.
void run_parity_trace(cublasHandle_t handle, const ModelWeights& model,
                      __nv_bfloat16* keys, __nv_bfloat16* values,
                      const std::vector<std::vector<int>>& prompts, const Options& options) {
  require(options.parity_layer == -1 || (options.parity_layer >= 0 && options.parity_layer < model_config.shape.layers &&
          model.layers[options.parity_layer].attention), "parity layer must be an attention layer");
  for (const auto& prompt : prompts) require(prompt.size() == prompts[0].size(), "parity tracing requires equal prompt lengths");
  if (options.parity_all_rows) {
    run_all_rows_trace(handle, model, keys, values, prompts, options);
    return;
  }
  require(options.parity_row == 0 || !options.attention_math || model_config.moe || options.attention_math_dynamic,
          "nonzero parity row requires dynamic or production attention, not the legacy row-zero math control");
  parity_row = options.parity_row;
  LayerBuffers buffers(options.batch_size);
  std::unique_ptr<lfm::parity::MathAttention> math_attention;
  if (options.attention_math && !model_config.moe) {
    math_attention = std::make_unique<lfm::parity::MathAttention>(kContextLength);
    pinned_attention = math_attention.get();
  }
  const int policy[] = {options.attention_window, options.attention_sinks, options.attention_sparse_after};
  check_cuda(cudaMemcpy(buffers.attention_policy.data, policy, sizeof(policy), cudaMemcpyHostToDevice), "parity policy");
  std::ofstream stream(*options.parity_output, std::ios::binary);
  std::ofstream attention_stream(*options.parity_output + ".attention", std::ios::binary);
  std::ofstream ops_stream(*options.parity_output + ".mlp", std::ios::binary);
  std::ofstream conv_stream(*options.parity_output + ".conv", std::ios::binary);
  std::ofstream moe_stream(*options.parity_output + ".moe", std::ios::binary);
  std::ofstream math_stream(*options.parity_output + ".math", std::ios::binary);
  std::ofstream selections(*options.parity_output + ".selection.tsv");
  std::ofstream expert_stream, expert_index;
  if (options.parity_experts) {
    require(options.parity_expert_position < static_cast<int>(prompts[0].size()), "expert trace position exceeds teacher length");
    expert_stream.open(*options.parity_output + ".experts.bin", std::ios::binary);
    expert_index.open(*options.parity_output + ".experts.jsonl");
    if (!expert_stream || !expert_index) throw std::runtime_error("cannot open expert operation trace");
    parity_expert_stream = &expert_stream; parity_expert_index = &expert_index;
    parity_expert_layer = options.parity_mlp_layer; parity_expert_position = options.parity_expert_position;
  }
  // Exact token matrix and original slot are recorded independently of Python
  // orchestration, so wrong-row/ordering mistakes cannot masquerade as parity.
  std::ofstream metadata(*options.parity_output + ".rows.json");
  if (!metadata) throw std::runtime_error("cannot open parity row metadata");
  metadata << "{\"schema\":1,\"selected_row\":" << parity_row
           << ",\"batch_size\":" << prompts.size() << ",\"teacher_ids\":[";
  for (size_t row = 0; row < prompts.size(); ++row) {
    if (row) metadata << ',';
    metadata << '[';
    for (size_t pos = 0; pos < prompts[row].size(); ++pos) {
      if (pos) metadata << ',';
      metadata << prompts[row][pos];
    }
    metadata << ']';
  }
  metadata << "]}\n";
  metadata.flush();
  if (!metadata) throw std::runtime_error("write parity row metadata failed");
  if (!stream || !attention_stream || !ops_stream || !conv_stream || !moe_stream || !math_stream || !selections) throw std::runtime_error("cannot open parity trace");
  parity_stream = &stream; parity_attention_stream = &attention_stream; parity_ops_stream = &ops_stream;
  parity_attention_layer = options.parity_layer; parity_ops_layer = options.parity_mlp_layer;
  parity_moe_stream = &moe_stream;
  parity_conv_stream = &conv_stream;
  parity_math_stream = &math_stream;
  selections << "position\tmaterialized_argmax\tfused_argmax\n";
  std::vector<int> tokens(options.batch_size), positions(options.batch_size);
  for (int position = 0; position < static_cast<int>(prompts[0].size()); ++position) {
    parity_position = position;
    for (int row = 0; row < options.batch_size; ++row) { tokens[row] = prompts[row][position]; positions[row] = position; }
    check_cuda(cudaMemcpy(buffers.input_token.data, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice), "parity tokens");
    check_cuda(cudaMemcpy(buffers.position.data, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice), "parity positions");
    run_full_token_device(handle, model, keys, values, buffers, position == 0);
    parity_dump(buffers.logits.data, kVocabularySize);
    int materialized, fused;
    check_cuda(cudaMemcpy(&materialized, buffers.selected_token.data + parity_row, sizeof(int), cudaMemcpyDeviceToHost), "parity selection");
    lm_head_argmax<<<kLmHeadBlocks, 256>>>(buffers.normalized.data + static_cast<size_t>(parity_row) * kHiddenSize, model.lm_head,
        buffers.lm_head_block_values.data, buffers.lm_head_block_indices.data);
    finalize_lm_head_argmax<<<1, 256>>>(buffers.lm_head_block_values.data,
        buffers.lm_head_block_indices.data, buffers.selected_token.data);
    check_cuda(cudaMemcpy(&fused, buffers.selected_token.data, sizeof(int), cudaMemcpyDeviceToHost), "parity fused selection");
    selections << position << '\t' << materialized << '\t' << fused << '\n';
  }
  parity_stream = nullptr; parity_attention_stream = nullptr; parity_ops_stream = nullptr;
  parity_moe_stream = nullptr;
  parity_conv_stream = nullptr;
  parity_math_stream = nullptr;
  pinned_attention = nullptr;
  if (options.parity_experts) {
    expert_stream.flush(); expert_index.flush();
    if (!expert_stream || !expert_index) throw std::runtime_error("flush expert operation trace failed");
  }
  parity_expert_stream = nullptr; parity_expert_index = nullptr;
  parity_expert_layer = -1; parity_expert_position = -1; parity_position = -1;
  parity_row = 0;
  std::cout << "parity_positions: " << prompts[0].size() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    parity_attention_layer = options.parity_layer;
    if (options.parity_output && !options.model) {
      throw std::runtime_error("--parity-output requires --model");
    }
    const int iterations = options.iterations;
    std::unique_ptr<lfm::Safetensors> model;
    std::unique_ptr<ModelWeights> model_weights;
    std::unique_ptr<lfm::Tokenizer> tokenizer;
    if (options.tokenizer) {
      tokenizer = std::make_unique<lfm::Tokenizer>(*options.tokenizer, kVocabularySize);
      require(tokenizer->vocab_size() <= static_cast<size_t>(kVocabularySize), "tokenizer vocabulary exceeds model output vocabulary");
      if (legacy_fixtures) {
      const std::string decoder_fixture = tokenizer->decode({9906, 1397, 975, 1460, 43022});
      require(
          decoder_fixture == "Reflective recursion",
          "native tokenizer decoder fixture mismatch: " + decoder_fixture);
      require(
          tokenizer->encode_chat(kControlPromptText) ==
              std::vector<int>(std::begin(kControlPrompt), std::end(kControlPrompt)),
           "native tokenizer encoder control fixture mismatch");
      }
    }
    const std::vector<int> expected_tokens = options.expected_tokens
        ? load_expected_tokens(*options.expected_tokens)
        : std::vector<int>{};
    if (options.prompt && !options.tokenizer) throw std::runtime_error("--prompt requires --tokenizer");
    if (options.model && !tokenizer && options.prompt_token_files.empty())
      throw std::runtime_error("model inference requires an adjacent tokenizer, --tokenizer, or --prompt-tokens");
    if (options.prompt && !options.prompt_token_files.empty()) {
      throw std::runtime_error("--prompt and --prompt-tokens are mutually exclusive");
    }
    if ((options.prompt || !options.prompt_token_files.empty()) && options.expected_tokens) {
      throw std::runtime_error("--expected-tokens is only valid for the built-in control prompt");
    }
    const bool custom_sampling =
        options.temperature > 0.0f || options.top_k > 0 || options.top_p < 1.0f ||
        options.min_p > 0.0f || options.repetition_penalty != 1.0f ||
        options.presence_penalty != 0.0f || options.frequency_penalty != 0.0f ||
        options.no_repeat_ngram_size > 0 || !options.banned_tokens.empty();
    if (options.expected_tokens && custom_sampling) {
      throw std::runtime_error("--expected-tokens requires default greedy decoding");
    }
    // The batch is a list of independent sequences. One prompt is replicated
    // across the batch; several prompts must supply exactly one sequence
    // each, and every prompt must have the same length so the sequences stay
    // in lockstep.
    std::vector<std::vector<int>> batch_prompts;
    if (!options.prompt_token_files.empty()) {
      for (const std::string& file : options.prompt_token_files) {
        batch_prompts.push_back(load_expected_tokens(file));
      }
    } else if (options.prompt) {
      batch_prompts.push_back(tokenizer->encode_chat(*options.prompt));
    } else if (tokenizer) {
      batch_prompts.push_back(tokenizer->encode_chat(kControlPromptText));
    } else {
      batch_prompts.push_back(
          std::vector<int>(std::begin(kControlPrompt), std::end(kControlPrompt)));
    }
    if (batch_prompts.size() == 1 && options.batch_size > 1) {
      batch_prompts.assign(static_cast<size_t>(options.batch_size), batch_prompts[0]);
    }
    require(batch_prompts.size() == static_cast<size_t>(options.batch_size),
            "batch prompt count does not match --batch-size");
    const std::vector<int>& prompt_tokens = batch_prompts[0];
    for (const auto& prompt : batch_prompts) {
      require(!prompt.empty(), "prompt token sequence must not be empty");
      require(options.parity_output ? prompt.size() <= kContextLength
                                    : prompt.size() < kContextLength,
              "prompt must leave at least one token available in the configured context");
      for (const int token : prompt) {
        require(token >= 0 && token < (tokenizer ? static_cast<int>(tokenizer->vocab_size()) : kVocabularySize),
                "prompt token ID is outside the vocabulary");
      }
    }
    if (options.print_prompt_tokens) {
      // Rendered prompt token IDs, one per line, for building --prompt-tokens
      // files without a second tokenizer implementation.
      for (const auto& prompt : batch_prompts) {
        for (const int token : prompt) std::cout << token << '\n';
      }
      return 0;
    }
    if (options.model) {
      std::cout << "model_architecture: " << (model_config.moe ? "lfm2_moe" : "lfm2") << '\n';
      std::cout << "model_dimensions: " << kHiddenSize << ' ' << kMlpWidth << ' '
                << kQueryHeads << ' ' << kKvHeads << ' ' << kHeadDimension << ' '
                << model_config.shape.layers << ' ' << kVocabularySize << '\n';
      model = std::make_unique<lfm::Safetensors>(*options.model);
      validate_model(*model);
      model->debug_snapshot_tail("validated");
      size_t horizon = 0;
      for (const auto& prompt : batch_prompts) horizon = std::max(horizon, prompt.size());
      if (options.parity_workspace_tokens) {
        if (static_cast<size_t>(options.parity_workspace_tokens) < horizon)
          throw std::runtime_error("parity workspace shorter than teacher trace");
        horizon = options.parity_workspace_tokens;
      }
      if (!options.parity_output) horizon = options.max_new_tokens > 0
          ? std::min(static_cast<size_t>(kContextLength), horizon + static_cast<size_t>(options.max_new_tokens))
          : static_cast<size_t>(kContextLength);
      model_weights = std::make_unique<ModelWeights>(*model, static_cast<int>(horizon), options.attention_math_dynamic);
      model->debug_snapshot_tail("operators-built");
      std::cout << "model_payload_bytes: " << model->payload_bytes() << '\n';
      std::cout << "model_device_bytes: " << model->device_bytes() << '\n';
      std::cout << "moe_extra_packed_bytes: " << (model_weights->packed_experts ?
          model_weights->packed_experts->size * sizeof(__nv_bfloat16) : 0) << '\n';
    }
    std::vector<__nv_bfloat16> input(kHiddenSize);
    std::vector<__nv_bfloat16> weight(kHiddenSize);
    for (int i = 0; i < kHiddenSize; ++i) {
      input[i] = __float2bfloat16(std::sin(static_cast<float>(i) * 0.03125f));
      weight[i] = __float2bfloat16(0.75f + 0.25f * std::cos(static_cast<float>(i) * 0.015625f));
    }

    DeviceBuffer<__nv_bfloat16> device_input(kHiddenSize);
    DeviceBuffer<__nv_bfloat16> device_norm_weight(kHiddenSize);
    DeviceBuffer<__nv_bfloat16> device_output(kHiddenSize);
    check_cuda(cudaMemcpy(device_input.data, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemcpy(device_norm_weight.data, weight.data(), weight.size() * sizeof(weight[0]), cudaMemcpyHostToDevice), "copy weight");

    float rms_ms = 0.0f;
    if (options.benchmark) {
      rms_ms = benchmark_events(
          [&] { rms_norm(device_input.data, device_norm_weight.data, device_output.data, 1); },
          100,
          iterations);
    } else {
      rms_norm(
          device_input.data, device_norm_weight.data, device_output.data, 1);
      check_cuda(cudaDeviceSynchronize(), "RMSNorm self-test synchronize");
    }
    test_rmsnorm(input, weight, device_output.data);
    if (model) model->debug_snapshot_tail("rms-tested");

    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");
    check_cublas(cublasSetStream(cublas, cudaStreamPerThread), "cublasSetStream");
    check_cublas(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH), "cublas default math");
    check_cublas(cublasSetAtomicsMode(cublas, CUBLAS_ATOMICS_NOT_ALLOWED), "cublas deterministic atomics policy");
    DeviceBuffer<std::byte> cublas_workspace(kCublasWorkspaceBytes);
    // Own workspace initialization, too. Initcheck cannot track some Blackwell
    // tensor stores before split-K reduction reads. Poisoning makes unwritten
    // floating-point scratch fail finite/output oracles rather than hiding it
    // with zeros; this does not certify the library's internal write tracking.
    check_cuda(cudaMemset(cublas_workspace.data, 0xff, cublas_workspace.size),
               "initialize cuBLAS workspace");
    check_cublas(
        cublasSetWorkspace(cublas, cublas_workspace.data, cublas_workspace.size),
        "cublasSetWorkspace");
    linear_scratch = std::make_unique<LinearScratch>();
    DeviceBuffer<__nv_bfloat16> q_weight(static_cast<size_t>(kHiddenSize) * kHiddenSize);
    DeviceBuffer<__nv_bfloat16> q_output(kHiddenSize);
    test_linear(cublas, input, device_input.data, q_weight.data, q_output.data);
    fill_bf16<<<(q_weight.size + 255) / 256, 256>>>(q_weight.data, q_weight.size, 0.01f);
    check_cuda(cudaGetLastError(), "fill q_weight");
    const float projection_max_error =
        test_linear_reference(cublas, input, device_input.data, q_weight.data, q_output.data);
    float q_projection_ms = 0.0f;
    if (options.benchmark) {
      q_projection_ms = benchmark_events(
          [&] { linear_bf16(cublas, device_input.data, q_weight.data, q_output.data, kHiddenSize, kHiddenSize); },
          20,
          iterations);
    }

    std::optional<float> layer_zero_ms;
    std::optional<uint64_t> layer_zero_hash;
    std::optional<double> layer_zero_checksum;
    if (model && legacy_fixtures) {
      LayerBuffers layer_buffers;
      run_conv_layer_zero(cublas, *model_weights, 1, layer_buffers);
      check_cuda(cudaDeviceSynchronize(), "layer zero synchronize");
      std::vector<__nv_bfloat16> layer_output(kHiddenSize);
      check_cuda(
          cudaMemcpy(
              layer_output.data(),
              layer_buffers.hidden.data,
              layer_output.size() * sizeof(layer_output[0]),
              cudaMemcpyDeviceToHost),
          "copy layer zero output");
      layer_zero_hash = fnv1a_bf16(layer_output);
      require(
          *layer_zero_hash == kLayerZeroTokenOneReferenceHash,
          "native layer 0 does not match the BF16 reference fixture");
      double checksum_value = 0.0;
      for (const auto value : layer_output) checksum_value += __bfloat162float(value);
      layer_zero_checksum = checksum_value;
      if (options.benchmark) {
        layer_zero_ms = benchmark_events(
            [&] { run_conv_layer_zero(cublas, *model_weights, 1, layer_buffers); },
            20,
            iterations);
      }
    }

    const size_t cache_elements_per_tensor =
        static_cast<size_t>(options.batch_size) * kAttentionLayers * kKvHeads *
        kContextLength * kHeadDimension;
    DeviceBuffer<__nv_bfloat16> key_cache(cache_elements_per_tensor);
    DeviceBuffer<__nv_bfloat16> value_cache(cache_elements_per_tensor);
    DeviceBuffer<__nv_bfloat16> key_vector(kKvWidth);
    DeviceBuffer<__nv_bfloat16> value_vector(kKvWidth);
    test_kv_cache(
        key_cache.data,
        value_cache.data,
        cache_elements_per_tensor,
        key_vector.data,
        value_vector.data);
    int cache_iteration = 0;
    float cache_write_ms = 0.0f;
    if (options.benchmark) {
      cache_write_ms = benchmark_events(
          [&] {
            const int position = cache_iteration++ % kContextLength;
            for (int layer = 0; layer < kAttentionLayers; ++layer) {
              write_kv_checked(
                  key_cache.data,
                  value_cache.data,
                  key_vector.data,
                  value_vector.data,
                  layer,
                  position);
            }
          },
          20,
          iterations);
    }

    if (options.parity_output) {
      model->debug_snapshot_tail("before-parity");
      run_parity_trace(cublas, *model_weights, key_cache.data, value_cache.data, batch_prompts, options);
      cublasDestroy(cublas);
      return 0;
    }

    // The batched path runs its own generation loop and reports its own
    // numbers; it does not share the single-sequence control fixtures.
    if (options.batch_size > 1) {
      require(model != nullptr, "--batch-size requires --model");
      const BatchGenerationResult result = run_batched_generation(
          cublas, *model_weights, key_cache.data, value_cache.data, batch_prompts, options);
      std::cout << "component_harness: native C++/CUDA\n";
      std::cout << "python: no\n";
      std::cout << "self_tests: pass\n";
      std::cout << "batch_size: " << options.batch_size << "\n";
      std::cout << "attention_mode: " << ((options.attention_math || options.attention_math_dynamic) ? "math_sdpa_fp32" : "native_online_fp32") << "\n";
      std::cout << "batch_prompt_tokens: " << prompt_tokens.size() << "\n";
      std::cout << "batch_generated_tokens: " << result.total_tokens << "\n";
      std::cout << "batch_tokens_per_second: " << result.tokens_per_second << "\n";
      // The batch sorts its sequences by prompt length internally, so the
      // results are reported back in the order the caller gave them.
      const int sequences = static_cast<int>(result.order.size());
      std::cout << "batch_generated_fnv1a:";
      for (int sequence = 0; sequence < sequences; ++sequence) {
        std::cout << ' ' << result.hashes[result.order[sequence]];
      }
      std::cout << "\n";
      std::cout << "batch_sequence_tokens:";
      for (int sequence = 0; sequence < sequences; ++sequence) {
        std::cout << ' ' << result.counts[result.order[sequence]];
      }
      std::cout << "\n";
      std::cout << "batch_sequence_end:";
      for (int sequence = 0; sequence < sequences; ++sequence) {
        std::cout << ' ' << result.end_reasons[result.order[sequence]];
      }
      std::cout << "\n";
      if (options.tokens_output) {
        // One file per sequence, so the streams stay separable: <path>.<seq>.txt
        for (int sequence = 0; sequence < sequences; ++sequence) {
          const int index = result.order[sequence];
          const std::string path =
              *options.tokens_output + "." + std::to_string(sequence) + ".txt";
          std::ofstream stream(path);
          if (!stream) throw std::runtime_error("cannot open token output: " + path);
          for (const int token : result.generated[index]) stream << token << '\n';
        }
      }
      if (tokenizer && !result.generated.empty()) {
        std::cout << "batch_text_begin\n" << tokenizer->decode(result.generated[result.order[0]])
                  << "\nbatch_text_end\n";
      }
      cublasDestroy(cublas);
      return 0;
    }

    std::optional<float> full_token_ms;
    std::optional<int> full_token_selected;
    std::optional<uint64_t> full_logits_hash;
    std::optional<double> full_logits_checksum;
    if (model && legacy_fixtures) {
      LayerBuffers full_buffers;
      full_token_selected = run_full_token(
          cublas,
          *model_weights,
          1,
          0,
          key_cache.data,
          value_cache.data,
          full_buffers,
          true);
      std::vector<__nv_bfloat16> logits(kVocabularySize);
      check_cuda(cudaMemcpy(logits.data(), full_buffers.logits.data,
                            logits.size() * sizeof(logits[0]), cudaMemcpyDeviceToHost),
                 "copy full token logits");
      full_logits_hash = fnv1a_bf16(logits);
      require(
          *full_logits_hash == kFullTokenOneReferenceLogitsHash,
          "native full first-token logits do not match the BF16 reference fixture");
      require(*full_token_selected == 1, "native full first-token argmax mismatch");
      double checksum_value = 0.0;
      for (const auto value : logits) checksum_value += __bfloat162float(value);
      full_logits_checksum = checksum_value;
      if (options.benchmark) {
        full_token_ms = benchmark_events(
            [&] {
              run_full_token(cublas, *model_weights, 1, 0, key_cache.data, value_cache.data, full_buffers, true);
            },
            5,
            iterations);
      }
    }

    std::optional<int> control_first_token;
    std::optional<uint64_t> control_logits_hash;
    std::optional<uint64_t> control_generated_hash;
    std::optional<size_t> control_generated_count;
    std::optional<double> control_tokens_per_second;
    std::string control_end_reason;
    std::optional<std::string> control_text;
    std::vector<uint64_t> control_position_hashes;
    std::vector<uint64_t> control_position1_layer_hashes;
    AttentionTrace control_position1_attention;
    if (model) {
      LayerBuffers control_buffers;
      std::unique_ptr<lfm::parity::MathAttention> math_attention;
      if (options.attention_math && !model_config.moe) {
        math_attention = std::make_unique<lfm::parity::MathAttention>(kContextLength);
        pinned_attention = math_attention.get();
      }
      const int attention_policy[] = {
          options.attention_window,
          options.attention_sinks,
          options.attention_sparse_after};
      check_cuda(
          cudaMemcpy(
              control_buffers.attention_policy.data,
              attention_policy,
              sizeof(attention_policy),
              cudaMemcpyHostToDevice),
          "copy attention policy");
      int selected = 0;
      // The projection result is consumed only by the final prompt position
      // (the greedy token seed) or by the sampler (which reads the last
      // position's logits). Every earlier position can skip it, which removes
      // one 65536x1024 projection and one 65536-entry reduction per prompt
      // token and lets the prompt evaluations overlap.
      const bool prompt_needs_every_projection = options.diagnostics;
      LFM_PROFILE_RANGE(profile_prefill, "phase.prefill");
      for (int position = 0; position < static_cast<int>(prompt_tokens.size()); ++position) {
        const bool final_position = position + 1 == static_cast<int>(prompt_tokens.size());
        selected = run_full_token(
            cublas,
            *model_weights,
            prompt_tokens[position],
            position,
            key_cache.data,
            value_cache.data,
            control_buffers,
            position == 0,
            options.diagnostics && position == 1 ? &control_position1_layer_hashes : nullptr,
            options.diagnostics && position == 1 ? &control_position1_attention : nullptr,
            final_position || prompt_needs_every_projection);
        if (options.diagnostics) {
          std::vector<__nv_bfloat16> position_logits(kVocabularySize);
          check_cuda(cudaMemcpy(position_logits.data(), control_buffers.logits.data,
                                position_logits.size() * sizeof(position_logits[0]), cudaMemcpyDeviceToHost),
                     "copy control position logits");
          control_position_hashes.push_back(fnv1a_bf16(position_logits));
        }
      }
      LFM_PROFILE_END(profile_prefill);
      control_first_token = selected;
      if (options.diagnostics) {
        std::vector<__nv_bfloat16> logits(kVocabularySize);
        check_cuda(cudaMemcpy(logits.data(), control_buffers.logits.data,
                              logits.size() * sizeof(logits[0]), cudaMemcpyDeviceToHost),
                   "copy control logits");
        control_logits_hash = fnv1a_bf16(logits);
      }
      std::vector<int> generated;
      std::unique_ptr<Sampler> sampler;
      if (custom_sampling) sampler = std::make_unique<Sampler>(options, prompt_tokens);
      DecodeState decode_state;
      check_cuda(cudaMemset(decode_state.generated_count.data, 0, sizeof(int)), "reset generated count");
      check_cuda(cudaMemset(decode_state.stop_flag.data, 0, sizeof(int)), "reset stop flag");
      check_cuda(cudaMemcpy(decode_state.stop_tokens.data, options.stop_tokens.data(),
                            options.stop_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
                 "copy stop tokens");
      set_decode_input<<<1, 1>>>(
          control_buffers.input_token.data,
          control_buffers.position.data,
          0,
          static_cast<int>(prompt_tokens.size()));
      if (sampler) {
        sampler->select_device(
            control_buffers.logits.data, 0, control_buffers.input_token.data);
      } else {
        check_cuda(cudaMemcpyAsync(
            control_buffers.input_token.data,
            control_buffers.selected_token.data,
            sizeof(int),
            cudaMemcpyDeviceToDevice),
            "seed greedy token");
      }
      const bool use_conditional_graph =
          options.cuda_graph && !sampler && expected_tokens.empty() && options.max_new_tokens == 0;
      // Observing expected tokens must not change the numerical decode path.
      const bool use_fused_greedy_lm_head = !sampler && !options.attention_math && !options.attention_math_dynamic;
      std::unique_ptr<DecodeGraph> decode_graph;
      if (options.cuda_graph && !use_conditional_graph) {
        decode_graph = std::make_unique<DecodeGraph>(
            cublas,
            *model_weights,
            key_cache.data,
            value_cache.data,
            control_buffers,
            use_fused_greedy_lm_head);
      }
      int generated_count = 0;
      LFM_PROFILE_RANGE(profile_decode, "phase.decode");
      const auto generation_start = std::chrono::steady_clock::now();
      if (use_conditional_graph) {
        GreedyGenerationGraph generation_graph(
            cublas,
            *model_weights,
            key_cache.data,
            value_cache.data,
            control_buffers,
            decode_state,
            options.stop_tokens.size());
        generation_graph.launch();
        check_cuda(cudaStreamSynchronize(cudaStreamPerThread),
                   "synchronize generation graph");
        check_cuda(cudaMemcpy(&generated_count, decode_state.generated_count.data,
                              sizeof(generated_count), cudaMemcpyDeviceToHost),
                   "copy generated count");
      } else while (static_cast<int>(prompt_tokens.size()) + generated_count < kContextLength) {
        record_token_and_check_stop<<<1, 1>>>(
            control_buffers.input_token.data,
            decode_state.generated.data,
            decode_state.generated_count.data,
            decode_state.stop_flag.data,
            decode_state.stop_tokens.data,
            options.stop_tokens.size());
        ++generated_count;
        int stop = 0;
        check_cuda(cudaMemcpy(&stop, decode_state.stop_flag.data, sizeof(stop), cudaMemcpyDeviceToHost),
                   "copy stop flag");
        int host_token = 0;
        const bool needs_host_token = !expected_tokens.empty();
        if (needs_host_token) {
          check_cuda(cudaMemcpy(&host_token, control_buffers.input_token.data,
                                sizeof(host_token), cudaMemcpyDeviceToHost),
                     "copy diagnostic token");
          if (!expected_tokens.empty()) {
            const size_t index = generated_count - 1;
            if (index >= expected_tokens.size() || host_token != expected_tokens[index]) {
              throw std::runtime_error(
                  "control token mismatch at " + std::to_string(index) +
                  ": native=" + std::to_string(host_token) +
                  " reference=" +
                  (index < expected_tokens.size() ? std::to_string(expected_tokens[index]) : std::string("<end>")));
            }
          }
        }
        if (stop || (options.max_new_tokens > 0 && generated_count >= options.max_new_tokens)) break;

        if (decode_graph) {
          decode_graph->launch();
        } else {
          run_full_token_device(
              cublas,
              *model_weights,
              key_cache.data,
              value_cache.data,
              control_buffers,
              false,
              use_fused_greedy_lm_head);
          increment_position<<<1, 1>>>(control_buffers.position.data);
        }
        if (sampler) {
          sampler->select_device(
              control_buffers.logits.data,
              generated_count,
              control_buffers.input_token.data);
        } else {
          check_cuda(cudaMemcpyAsync(
              control_buffers.input_token.data,
              control_buffers.selected_token.data,
              sizeof(int),
              cudaMemcpyDeviceToDevice),
              "advance greedy token");
        }
      }
      const auto generation_end = std::chrono::steady_clock::now();
      LFM_PROFILE_END(profile_decode);
      generated.resize(generated_count);
      check_cuda(cudaMemcpy(generated.data(), decode_state.generated.data,
                            generated.size() * sizeof(int), cudaMemcpyDeviceToHost),
                 "copy generated tokens");
      const double generation_seconds =
          std::chrono::duration<double>(generation_end - generation_start).count();
      const bool ended_with_stop = !generated.empty() &&
          std::find(options.stop_tokens.begin(), options.stop_tokens.end(), generated.back()) != options.stop_tokens.end();
      control_end_reason = ended_with_stop ? "stop_token" :
          options.max_new_tokens > 0 && generated_count >= options.max_new_tokens ? "token_limit" : "context_limit";
      control_generated_hash = fnv1a_tokens(generated);
      control_generated_count = generated.size();
      control_tokens_per_second = generated.size() / generation_seconds;
      if (!expected_tokens.empty()) {
        require(generated.size() == expected_tokens.size(), "control generation ended before reference fixture");
      }
      if (options.tokens_output) {
        std::ofstream stream(*options.tokens_output);
        if (!stream) throw std::runtime_error("cannot open token output: " + *options.tokens_output);
        for (const int token : generated) stream << token << '\n';
      }
      if (tokenizer) control_text = tokenizer->decode(generated);
      pinned_attention = nullptr;
    }
    std::optional<uint64_t> rope_position1_hash;
    if (options.diagnostics) {
      if (model_weights) {
        DeviceBuffer<__nv_bfloat16> coefficients(kHeadDimension);
        rope_coefficients<<<1, kHeadDimension / 2>>>(coefficients.data, 1, model_weights->frequencies.data, kHeadDimension / 2);
        rope_position1_hash = hash_device_bf16(coefficients.data, kHeadDimension);
      }
    }

    std::vector<__nv_bfloat16> output(kHiddenSize);
    check_cuda(cudaMemcpy(output.data(), device_output.data, output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost), "copy output");
    double checksum = 0.0;
    for (const auto value : output) checksum += __bfloat162float(value);

    std::cout << "component_harness: native C++/CUDA\n";
    std::cout << "python: no\n";
    std::cout << "self_tests: pass\n";
    std::cout << "model_inference: " << (model ? "native greedy" : "not requested") << "\n";
    if (model) {
      std::cout << "model_weights: validated\n";
      std::cout << "model_layer0_exact: " << (legacy_fixtures ? "pass" : "not requested") << "\n";
      std::cout << "model_tensors: " << model->tensor_count() << "\n";
      std::cout << "model_payload_mib: " << model->payload_bytes() / (1024.0 * 1024.0) << "\n";
    }
    if (tokenizer) std::cout << "tokenizer_decoder: validated\n";
    std::cout << "kernel: BF16 RMSNorm (" << (kHiddenSize == 1024 ? "1024-wide exact schedule" : "generic width schedule") << ")\n";
    std::cout << "temperature: " << options.temperature << "\n";
    std::cout << "top_k: " << options.top_k << "\n";
    std::cout << "top_p: " << options.top_p << "\n";
    std::cout << "min_p: " << options.min_p << "\n";
    std::cout << "repetition_penalty: " << options.repetition_penalty << "\n";
    std::cout << "presence_penalty: " << options.presence_penalty << "\n";
    std::cout << "frequency_penalty: " << options.frequency_penalty << "\n";
    std::cout << "no_repeat_ngram_size: " << options.no_repeat_ngram_size << "\n";
    std::cout << "attention_window: " << options.attention_window << "\n";
    std::cout << "attention_sinks: " << options.attention_sinks << "\n";
    std::cout << "attention_sparse_after: " << options.attention_sparse_after << "\n";
    std::cout << "seed: " << options.seed << "\n";
    if (options.benchmark) {
      std::cout << "iterations: " << iterations << "\n";
      std::cout << "rmsnorm_us: " << rms_ms * 1000.0 / iterations << "\n";
      std::cout << "q_projection_" << kHiddenSize << 'x' << kHiddenSize << "_us: " << q_projection_ms * 1000.0 / iterations << "\n";
    }
    std::cout << "q_projection_cpu_reference_max_abs: " << projection_max_error << "\n";
    std::cout << "attention_mode: " << ((options.attention_math || options.attention_math_dynamic) ? "math_sdpa_fp32" : "native_online_fp32") << "\n";
    std::cout << "cuda_graphs: " << (options.cuda_graph ? "enabled" : "disabled") << "\n";
    std::cout << "max_new_tokens: " << options.max_new_tokens << "\n";
    if (layer_zero_ms) {
      std::cout << "layer0_first_token_us: " << *layer_zero_ms * 1000.0 / iterations << "\n";
      std::cout << "layer0_first_token_fnv1a: " << *layer_zero_hash << "\n";
      std::cout << "layer0_first_token_checksum: " << std::setprecision(12) << *layer_zero_checksum << "\n";
    }
    if (full_token_ms) {
      std::cout << "full_first_token_us: " << *full_token_ms * 1000.0 / iterations << "\n";
      std::cout << "full_first_token_selected: " << *full_token_selected << "\n";
      std::cout << "full_first_token_logits_fnv1a: " << *full_logits_hash << "\n";
      std::cout << "full_first_token_logits_checksum: " << std::setprecision(12)
                << *full_logits_checksum << "\n";
    }
    if (control_first_token) {
      std::cout << "prompt_tokens: " << prompt_tokens.size() << "\n";
      std::cout << "control_prompt_first_token: " << *control_first_token << "\n";
      if (options.diagnostics) {
        std::cout << "control_prompt_logits_fnv1a: " << *control_logits_hash << "\n";
        std::cout << "control_position1_logits_fnv1a: " << control_position_hashes.at(1) << "\n";
        std::cout << "control_position1_layer_fnv1a:";
        for (const uint64_t hash : control_position1_layer_hashes) std::cout << ' ' << hash;
        std::cout << "\n";
        std::cout << "control_position1_attention_fnv1a: "
                  << control_position1_attention.query_norm << ' '
                  << control_position1_attention.key_norm << ' '
                  << control_position1_attention.query << ' '
                  << control_position1_attention.key << ' '
                  << control_position1_attention.value << ' '
                  << control_position1_attention.attended << ' '
                  << control_position1_attention.projected << "\n";
      }
      std::cout << "generated_tokens: " << *control_generated_count << "\n";
      std::cout << "generation_end: " << control_end_reason << "\n";
      std::cout << "generated_fnv1a: " << *control_generated_hash << "\n";
      std::cout << "generated_tokens_per_second: " << *control_tokens_per_second << "\n";
      if (control_text) {
        std::cout << "text_begin\n" << *control_text << "\ntext_end\n";
      }
    }
    if (rope_position1_hash) {
      std::cout << "rope_position1_coefficients_fnv1a: " << *rope_position1_hash << "\n";
    }
    if (options.benchmark) {
      std::cout << "kv_write_us: " << cache_write_ms * 1000.0 / iterations << "\n";
    }
    std::cout << "kv_cache_mib: "
              << (2.0 * cache_elements_per_tensor * sizeof(__nv_bfloat16) / (1024.0 * 1024.0)) << "\n";
    std::cout << "checksum: " << checksum << "\n";

    cublasDestroy(cublas);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
