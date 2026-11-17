#pragma once

// PyTorch-derived reduction schedules: see native/third_party/PYTORCH_LICENSE.

// Opt-in, single-query diagnostic attention; NOT a certified parity fix.
// Source contract inspected at PyTorch v2.10.0:
// aten/src/ATen/native/transformers/attention.cpp (_scaled_dot_product_attention_math)
// aten/src/ATen/native/cuda/{PersistentSoftmax.cuh,SoftMax.cu}
// https://github.com/pytorch/pytorch/tree/v2.10.0/aten/src/ATen/native
// Inspected SoftMax.cu SHA-256:
// dc9489795218bf526f98bdfe0ac1b462fee6ba4a777c1a29e699271a4e2c0c5a
// Inspected PersistentSoftmax.cuh SHA-256:
// 332061c98b3293713deb87debd495fced86b7d3a6bea3472dc8d09660270eb93
// Q/K/V are promoted to float; Q and K are separately multiplied by
// float(sqrt(0.125)); float BMM -> softmax -> float BMM -> BF16 RN.
// Requires allow_fp16_bf16_reduction_math_sdp=False and IEEE float32 matmul
// in the reference. Automatic/flash/efficient/cuDNN SDPA are different contracts.
// The short-row softmax follows upstream's lane order (XOR butterflies).
// Rows 2049..9216 follow the 1024-thread register-kernel reduction mapping.
// Rows 9217..32768 follow upstream's shared-memory/vectorized or alignment-
// sensitive global-ILP mapping, selected using default sharedMemPerBlock.
// Source-mapped does NOT mean GPU-measured bitwise parity at every length.
// cuBLAS algorithm/shape/version differences can still prevent bitwise parity.
// Compile WITHOUT --use_fast_math. No changes to the production path occur
// merely by including this header; construction explicitly allocates scratch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace lfm::parity {

inline constexpr int kMinAttentionTokens = 1;
inline constexpr int kMaxAttentionTokens = 32768;
inline constexpr const char* kSoftmaxSourceVersion = "pytorch/v2.10.0/CUDA/float32";
inline constexpr bool kBitwiseParityCertified = false;
inline constexpr bool kFiniteAttentionInputsRequired = true;

enum class AttentionScaling { MathSdpaSplit, PostQk };

enum class SoftmaxKernel { PersistentWarp, Register, SharedVector, GlobalIlp };

struct SoftmaxPlan {
  SoftmaxKernel kernel;
  int tokens;
  int threads;
  int registers_per_thread;
  int vector_width;
  size_t dynamic_shared_bytes;
  // The global-ILP prefix depends on each input row pointer modulo 16.
  bool input_alignment_sensitive;
};

inline const char* softmax_kernel_name(SoftmaxKernel kernel) {
  switch (kernel) {
    case SoftmaxKernel::PersistentWarp: return "persistent-warp";
    case SoftmaxKernel::Register: return "register";
    case SoftmaxKernel::SharedVector: return "shared-vector";
    case SoftmaxKernel::GlobalIlp: return "global-ilp";
  }
  throw std::invalid_argument("invalid parity softmax kernel");
}

// Metadata for CUDA float32 [16,S] contiguous last-dimension softmax only.
// Corresponds to host_softmax<...,is_log_softmax=false,use_fast_softmax=false>
// with half_to_float=false in v2.10.0 SoftMax.cu. This is NOT ROCm dispatch.
// shared_memory_per_block must be the DEFAULT device limit (not opt-in limit).
// aligned_bases means BOTH input and output base pointers are 16-byte aligned.
inline SoftmaxPlan softmax_plan(int tokens, int shared_memory_per_block,
                                bool aligned_bases = true) {
  if (tokens < 1 || tokens > 32768)
    throw std::invalid_argument("parity softmax tokens must be in [1,32768]");
  if (shared_memory_per_block < 128)
    throw std::invalid_argument("invalid default device shared-memory limit");
  if (tokens <= 2048) {
    int power = 1;
    while (power < tokens) power *= 2;
    return {SoftmaxKernel::PersistentWarp, tokens, power < 32 ? power : 32,
            0, 1, 0, false};
  }
  const int registers = (tokens + 1023) / 1024;
  if (registers < 10)
    return {SoftmaxKernel::Register, tokens, 1024, registers, 1, 0, false};
  // Upstream uses a STRICT less-than comparison for max_elements_per_smem,
  // plus global input/output alignment and dim_size % ILP == 0.
  const bool shared = aligned_bases && tokens % 4 == 0 &&
      tokens < (shared_memory_per_block - 128) / static_cast<int>(sizeof(float));
  return {shared ? SoftmaxKernel::SharedVector : SoftmaxKernel::GlobalIlp,
          tokens, 1024, 0, 4,
          shared ? static_cast<size_t>(tokens) * sizeof(float) + 128 : 128,
          !shared};
}

namespace detail {
inline void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
inline void blas_check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": cuBLAS " + std::to_string(status));
}

// Compact active context only, duplicating GQA heads exactly as repeat_interleave.
// Native caches are [8, cache_capacity, 64], not [token, head, dimension].
static __global__ void pack(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, float* q, float* k, float* v,
    int tokens, int cache_capacity, float scale) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < 16 * 64) q[i] = __bfloat162float(query[i]) * scale;
  if (i >= static_cast<size_t>(16) * tokens * 64) return;
  const int head = i / (static_cast<size_t>(tokens) * 64);
  const int within = i % (static_cast<size_t>(tokens) * 64);
  const size_t source = static_cast<size_t>(head / 2) * cache_capacity * 64 + within;
  k[i] = __bfloat162float(keys[source]) * scale;
  v[i] = __bfloat162float(values[source]);
}

static __global__ void scale_scores(float* scores, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) scores[i] *= 0.125f;
}

// Each logical subwarp handles a row. The upstream WARP_BATCH=2 for
// powers <=128 changes row assignment, not the arithmetic within a row.
template <int Log2>
static __global__ void softmax_short(const float* scores, float* probabilities, int tokens) {
  constexpr int power = 1 << Log2;
  constexpr int width = power < 32 ? power : 32;
  constexpr int iterations = power / width;
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  float elements[iterations];
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int j = 0; j < iterations; ++j) {
    const int t = lane + j * width;
    elements[j] = t < tokens ? scores[head * tokens + t] : -CUDART_INF_F;
    maximum = maximum > elements[j] ? maximum : elements[j];
  }
  const unsigned mask = __activemask();
#pragma unroll
  for (int delta = width / 2; delta > 0; delta /= 2) {
    const float other = __shfl_xor_sync(mask, maximum, delta, width);
    maximum = maximum < other ? other : maximum;
  }
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < iterations; ++j) {
    elements[j] = expf(elements[j] - maximum);
    sum += elements[j];
  }
#pragma unroll
  for (int delta = width / 2; delta > 0; delta /= 2)
    sum += __shfl_xor_sync(mask, sum, delta, width);
#pragma unroll
  for (int j = 0; j < iterations; ++j) {
    const int t = lane + j * width;
    if (t < tokens) probabilities[head * tokens + t] = elements[j] / sum;
  }
}

// PyTorch block_reduce.cuh: down-shuffle within warps, then warp zero
// down-shuffles the 32 warp results. All callers launch exactly 1024 threads.
template <bool Maximum>
__device__ inline float reduce_register_block(float value, float* shared) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
#pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, delta);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  __syncthreads();
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = threadIdx.x < 32 ? shared[lane] : (Maximum ? -FLT_MAX : 0.0f);
  if (warp == 0) {
#pragma unroll
    for (int delta = 16; delta; delta /= 2) {
      const float other = __shfl_down_sync(0xffffffff, value, delta);
      value = Maximum ? fmaxf(value, other) : value + other;
    }
  }
  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  return shared[0];
}

template <int Registers>
static __global__ void softmax_register(
    const float* scores, float* probabilities, int tokens) {
  __shared__ float reductions[32];
  float elements[Registers];
  float maximum = -FLT_MAX;
  const int base = blockIdx.x * tokens;
#pragma unroll
  for (int j = 0; j < Registers; ++j) {
    const int t = threadIdx.x + j * 1024;
    if (t < tokens) {
      elements[j] = scores[base + t];
      maximum = fmaxf(maximum, elements[j]);
    }
  }
  maximum = reduce_register_block<true>(maximum, reductions);
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < Registers; ++j)
    if (threadIdx.x + j * 1024 < tokens) sum += expf(elements[j] - maximum);
  sum = reduce_register_block<false>(sum, reductions);
#pragma unroll
  for (int j = 0; j < Registers; ++j) {
    const int t = threadIdx.x + j * 1024;
    if (t < tokens) probabilities[base + t] = expf(elements[j] - maximum) / sum;
  }
}

// cunn_SoftMaxForwardSmem<ILP=4,float,float,float,...> mapping. The aligned
// float4 per thread is consumed component-by-component, in increasing order.
static __global__ void softmax_shared_vector(
    const float* scores, float* probabilities, int tokens) {
  extern __shared__ __align__(16) float shared[];
  float* reductions = shared + tokens;
  const auto* input = reinterpret_cast<const float4*>(scores + blockIdx.x * tokens);
  auto* cached = reinterpret_cast<float4*>(shared);
  auto* output = reinterpret_cast<float4*>(probabilities + blockIdx.x * tokens);
  float maximum = -FLT_MAX;
  for (int offset = threadIdx.x; offset * 4 < tokens; offset += 1024) {
    const float4 value = input[offset];
    cached[offset] = value;
    maximum = fmaxf(maximum, value.x);
    maximum = fmaxf(maximum, value.y);
    maximum = fmaxf(maximum, value.z);
    maximum = fmaxf(maximum, value.w);
  }
  maximum = reduce_register_block<true>(maximum, reductions);
  float sum = 0.0f;
  for (int offset = threadIdx.x; offset * 4 < tokens; offset += 1024) {
    const float4 value = cached[offset];
    sum += expf(value.x - maximum);
    sum += expf(value.y - maximum);
    sum += expf(value.z - maximum);
    sum += expf(value.w - maximum);
  }
  sum = reduce_register_block<false>(sum, reductions);
  for (int offset = threadIdx.x; offset * 4 < tokens; offset += 1024) {
    const float4 value = cached[offset];
    output[offset] = make_float4(expf(value.x - maximum) / sum,
                                expf(value.y - maximum) / sum,
                                expf(value.z - maximum) / sum,
                                expf(value.w - maximum) / sum);
  }
}

// ilpReduce<...,ILP=4,float,float> mapping used by cunn_SoftMaxForward.
// The prefix is one scalar per thread after aligning the input BACKWARD to
// a 16-byte boundary. Only in-row values are read. Full vector tiles follow,
// then a scalar tail in strides of blockDim.x. Changing this order changes
// rounding. Separate max and exp-sum passes match upstream.
template <bool Maximum>
__device__ inline float reduce_global_ilp(const float* input, int tokens, float maximum) {
  float result = Maximum ? -FLT_MAX : 0.0f;
  const int shift = (reinterpret_cast<uintptr_t>(input) % 16) / sizeof(float);
  int offset = threadIdx.x;
  if (shift > 0) {
    // Express upstream data-=shift without forming an out-of-allocation
    // pointer for an offset input in the first row.
    if (offset >= shift && offset < tokens + shift) {
      const float value = input[offset - shift];
      result = Maximum ? fmaxf(result, value) : result + expf(value - maximum);
    }
    tokens += shift;
    tokens -= tokens < 1024 ? tokens : 1024;
    input += 1024 - shift;
  }
  const int tail = tokens % 4096;
  const auto* vectors = reinterpret_cast<const float4*>(input);
  for (; offset * 4 < tokens - tail; offset += 1024) {
    const float4 value = vectors[offset];
    if constexpr (Maximum) {
      result = fmaxf(result, value.x);
      result = fmaxf(result, value.y);
      result = fmaxf(result, value.z);
      result = fmaxf(result, value.w);
    } else {
      result += expf(value.x - maximum);
      result += expf(value.y - maximum);
      result += expf(value.z - maximum);
      result += expf(value.w - maximum);
    }
  }
  for (offset = tokens - tail + threadIdx.x; offset < tokens; offset += 1024) {
    const float value = input[offset];
    result = Maximum ? fmaxf(result, value) : result + expf(value - maximum);
  }
  return result;
}

static __global__ void softmax_global_ilp(
    const float* scores, float* probabilities, int tokens) {
  extern __shared__ float reductions[];
  scores += blockIdx.x * tokens;
  probabilities += blockIdx.x * tokens;
  float maximum = reduce_global_ilp<true>(scores, tokens, 0.0f);
  maximum = reduce_register_block<true>(maximum, reductions);
  float sum = reduce_global_ilp<false>(scores, tokens, maximum);
  sum = reduce_register_block<false>(sum, reductions);
  const int input_shift = (reinterpret_cast<uintptr_t>(scores) % 16) / sizeof(float);
  const int output_shift = (reinterpret_cast<uintptr_t>(probabilities) % 16) / sizeof(float);
  if (input_shift != output_shift) {
    // WriteFpropResults, mismatched input/output alignment.
    for (int t = threadIdx.x; t < tokens; t += 1024)
      probabilities[t] = expf(scores[t] - maximum) / sum;
    return;
  }
  // WriteFpropResultsVectorized: identical aligned-prefix/vector/tail
  // schedule to the reduction, now also reproducing output grouping.
  int offset = threadIdx.x;
  if (input_shift) {
    if (offset >= input_shift && offset < tokens + input_shift)
      probabilities[offset - input_shift] = expf(scores[offset - input_shift] - maximum) / sum;
    tokens += input_shift;
    tokens -= tokens < 1024 ? tokens : 1024;
    scores += 1024 - input_shift;
    probabilities += 1024 - input_shift;
  }
  const int tail = tokens % 4096;
  const auto* vectors = reinterpret_cast<const float4*>(scores);
  auto* output_vectors = reinterpret_cast<float4*>(probabilities);
  for (; offset * 4 < tokens - tail; offset += 1024) {
    const float4 value = vectors[offset];
    output_vectors[offset] = make_float4(expf(value.x - maximum) / sum,
                                        expf(value.y - maximum) / sum,
                                        expf(value.z - maximum) / sum,
                                        expf(value.w - maximum) / sum);
  }
  for (offset = tokens - tail + threadIdx.x; offset < tokens; offset += 1024)
    probabilities[offset] = expf(scores[offset] - maximum) / sum;
}

static __global__ void cast_output(const float* input, __nv_bfloat16* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 1024) output[i] = __float2bfloat16_rn(input[i]);
}

inline int default_shared_memory_per_block() {
  int device = 0, bytes = 0;
  cuda_check(cudaGetDevice(&device), "get parity device");
  cuda_check(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, device),
             "get default parity shared-memory limit");
  return bytes;
}
}  // namespace detail

// Standalone source-mapped softmax for boundary tests. Exactly 16 contiguous
// float32 rows, distinct non-overlapping input/output, float-aligned pointers.
// Only finite inputs are supported; NaN/Inf, masks and _safe_softmax's all--Inf
// special case are OUTSIDE this contract. No input scan/synchronization occurs.
// The caller may supply a cached DEFAULT shared-memory limit from this device.
// Passing zero queries it (outside graph capture). Returning a source dispatch
// plan is not a statement that this tensor or length passed a parity test.
inline SoftmaxPlan softmax_fp32(const float* scores, float* probabilities, int tokens,
                                cudaStream_t stream = cudaStreamPerThread,
                                int shared_memory_per_block = 0, int rows = 16) {
  if (rows < 1) throw std::invalid_argument("invalid softmax row count");
  if (!scores || !probabilities) throw std::invalid_argument("null parity softmax tensor");
  if (tokens < kMinAttentionTokens || tokens > kMaxAttentionTokens)
    throw std::invalid_argument("parity softmax tokens must be in [1,32768]");
  const uintptr_t input = reinterpret_cast<uintptr_t>(scores);
  const uintptr_t output = reinterpret_cast<uintptr_t>(probabilities);
  if (input % sizeof(float) || output % sizeof(float))
    throw std::invalid_argument("parity softmax tensors must be float-aligned");
  const size_t bytes = static_cast<size_t>(rows) * tokens * sizeof(float);
  if (input < output + bytes && output < input + bytes)
    throw std::invalid_argument("parity softmax input/output must not overlap");
  if (!shared_memory_per_block)
    shared_memory_per_block = detail::default_shared_memory_per_block();
  const auto plan = softmax_plan(tokens, shared_memory_per_block, input % 16 == 0 && output % 16 == 0);
  if (plan.kernel == SoftmaxKernel::PersistentWarp) {
    int log2 = 0;
    while ((1 << log2) < tokens) ++log2;
#define LFM_PARITY_SOFTMAX_CASE(L) case L: detail::softmax_short<L><<<rows, ((1 << L) < 32 ? (1 << L) : 32), 0, stream>>>(scores, probabilities, tokens); break
    switch (log2) {
      LFM_PARITY_SOFTMAX_CASE(0); LFM_PARITY_SOFTMAX_CASE(1);
      LFM_PARITY_SOFTMAX_CASE(2); LFM_PARITY_SOFTMAX_CASE(3);
      LFM_PARITY_SOFTMAX_CASE(4); LFM_PARITY_SOFTMAX_CASE(5);
      LFM_PARITY_SOFTMAX_CASE(6); LFM_PARITY_SOFTMAX_CASE(7);
      LFM_PARITY_SOFTMAX_CASE(8); LFM_PARITY_SOFTMAX_CASE(9);
      LFM_PARITY_SOFTMAX_CASE(10); LFM_PARITY_SOFTMAX_CASE(11);
    }
#undef LFM_PARITY_SOFTMAX_CASE
  } else if (plan.kernel == SoftmaxKernel::Register) {
#define LFM_PARITY_REG_CASE(R) case R: detail::softmax_register<R><<<rows, 1024, 0, stream>>>(scores, probabilities, tokens); break
    switch (plan.registers_per_thread) {
      LFM_PARITY_REG_CASE(3); LFM_PARITY_REG_CASE(4);
      LFM_PARITY_REG_CASE(5); LFM_PARITY_REG_CASE(6);
      LFM_PARITY_REG_CASE(7); LFM_PARITY_REG_CASE(8);
      LFM_PARITY_REG_CASE(9);
    }
#undef LFM_PARITY_REG_CASE
  } else if (plan.kernel == SoftmaxKernel::SharedVector) {
    detail::softmax_shared_vector<<<rows, 1024, plan.dynamic_shared_bytes, stream>>>(scores, probabilities, tokens);
  } else {
    detail::softmax_global_ilp<<<rows, 1024, plan.dynamic_shared_bytes, stream>>>(scores, probabilities, tokens);
  }
  detail::cuda_check(cudaGetLastError(), "parity softmax launch");
  return plan;
}

class MathAttention {
 public:
  // No default capacity: caller explicitly chooses memory cost. Allocates
  // (2080 * max_tokens + 2048) floats: ~200 KiB at 24, ~16.9 MiB at
  // 2125, ~260 MiB at 32768 (excluding cuBLAS internal workspace).
  // Allocate before inference/capture. The run() API is HOST-length-driven;
  // do not substitute it in a device-position-changing CUDA decode graph.
  explicit MathAttention(int max_tokens, cudaStream_t stream = cudaStreamPerThread)
      : capacity_(max_tokens), stream_(stream) {
    if (max_tokens < 1 || max_tokens > 32768)
      throw std::invalid_argument("MathAttention capacity must be in [1,32768]");
    shared_memory_per_block_ = detail::default_shared_memory_per_block();
    try {
      detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&scratch_), scratch_bytes(max_tokens)),
                         "allocate parity attention scratch");
      detail::blas_check(cublasCreate(&handle_), "create parity cuBLAS handle");
      detail::blas_check(cublasSetStream(handle_, stream_), "set parity cuBLAS stream");
      detail::blas_check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "set parity scalar mode");
      detail::blas_check(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "disable TF32 parity matmul");
      detail::blas_check(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED), "disable parity atomics");
    } catch (...) {
      if (handle_) cublasDestroy(handle_);
      if (scratch_) cudaFree(scratch_);
      throw;
    }
  }
  ~MathAttention() {
    if (handle_) cublasDestroy(handle_);
    if (scratch_) cudaFree(scratch_);
  }
  MathAttention(const MathAttention&) = delete;
  MathAttention& operator=(const MathAttention&) = delete;

  static size_t scratch_bytes(int max_tokens) {
    if (max_tokens < kMinAttentionTokens || max_tokens > kMaxAttentionTokens)
      throw std::invalid_argument("MathAttention capacity must be in [1,32768]");
    return (static_cast<size_t>(2080) * max_tokens + 2048) * sizeof(float);
  }

  // Full causal prefix of ONE latest query, no mask/dropout/sparse policy.
  // Caller must have already written cache[token_count-1] on this stream.
  // Cache pointers are for ONE attention layer/slot. q/output each have
  // 1024 BF16 elements. No host copy/sync occurs; errors may surface later.
  // Finite Q/K/V, scaled scores and FP32 PV intermediates required;
  // no nonfinite-input scan is performed.
  // Must not overlap calls on the same object or destroy before completion.
  // The final bool is retained for call-site compatibility but no longer
  // enables a fallback: ALL supported lengths now use source-mapped softmax.
  void run(const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
           const __nv_bfloat16* value_cache, __nv_bfloat16* output,
           int token_count, int cache_capacity = 32768,
           AttentionScaling scaling = AttentionScaling::MathSdpaSplit,
           bool allow_unpinned_long_softmax = false) {
    if (!q || !key_cache || !value_cache || !output)
      throw std::invalid_argument("null parity attention tensor");
    if (token_count < 1 || token_count > capacity_ || token_count > cache_capacity || cache_capacity > 32768)
      throw std::invalid_argument("parity attention token/cache capacity mismatch");
    if (scaling != AttentionScaling::MathSdpaSplit && scaling != AttentionScaling::PostQk)
      throw std::invalid_argument("invalid parity attention scaling");
    (void)allow_unpinned_long_softmax;
    float* q32 = scratch_;
    float* out32 = q32 + 1024;
    float* keys = out32 + 1024;
    const size_t kv_elements = static_cast<size_t>(16) * token_count * 64;
    float* values = keys + kv_elements;
    float* scores = values + kv_elements;
    float* probabilities = scores + 16 * token_count;
    const float input_scale = scaling == AttentionScaling::MathSdpaSplit
        ? static_cast<float>(std::sqrt(0.125)) : 1.0f;
    detail::pack<<<(kv_elements + 255) / 256, 256, 0, stream_>>>(
        q, key_cache, value_cache, q32, keys, values, token_count, cache_capacity, input_scale);
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major Q[1,64] @ K[S,64]^T => column-major K^T Q.
    detail::blas_check(cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_T, CUBLAS_OP_N, token_count, 1, 64,
        &alpha, keys, 64, static_cast<long long>(token_count) * 64,
        q32, 64, 64, &beta, scores, token_count, token_count, 16), "parity QK float BMM");
    if (scaling == AttentionScaling::PostQk)
      detail::scale_scores<<<(16 * token_count + 255) / 256, 256, 0, stream_>>>(scores, 16 * token_count);
    softmax_fp32(scores, probabilities, token_count, stream_, shared_memory_per_block_);
    // Row-major P[1,S] @ V[S,64] => column-major V P.
    detail::blas_check(cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, 64, 1, token_count,
        &alpha, values, 64, static_cast<long long>(token_count) * 64,
        probabilities, token_count, token_count, &beta, out32, 64, 64, 16), "parity PV float BMM");
    detail::cast_output<<<4, 256, 0, stream_>>>(out32, output);
    detail::cuda_check(cudaGetLastError(), "parity attention launches");
    last_tokens_ = token_count;
  }

  // Device diagnostic views, valid only until the next run. Copy on stream()
  // (or synchronize it) before inspection. Layouts [16,S] and [16,64].
  const float* scores() const { return scratch_ + 2048 + static_cast<size_t>(2048) * last_tokens_; }
  const float* probabilities() const { return scores() + 16 * last_tokens_; }
  const float* output_fp32() const { return scratch_ + 1024; }
  cudaStream_t stream() const { return stream_; }
  SoftmaxPlan plan(int tokens) const {
    if (tokens < 1 || tokens > capacity_)
      throw std::invalid_argument("parity attention plan exceeds allocated capacity");
    // Packed score and probability bases are always 16-byte aligned.
    return softmax_plan(tokens, shared_memory_per_block_);
  }
  int default_shared_memory_bytes() const { return shared_memory_per_block_; }

 private:
  int capacity_;
  int last_tokens_ = 0;
  int shared_memory_per_block_ = 0;
  cudaStream_t stream_;
  float* scratch_ = nullptr;
  cublasHandle_t handle_ = nullptr;
};

}  // namespace lfm::parity
