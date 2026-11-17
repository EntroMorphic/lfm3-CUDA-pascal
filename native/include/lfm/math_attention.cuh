#pragma once
#include <lfm/compact_attention.cuh>
#include <lfm/model_config.hpp>
#include <lfm/parity_attention.cuh>
#include <lfm/profiling.hpp>

namespace lfm {
namespace math_detail {
__global__ void pack(const __nv_bfloat16* query, const __nv_bfloat16* keys,
                      const __nv_bfloat16* values, float* q, float* k, float* v,
                      int heads, int kv_heads, int dim, int tokens, int capacity, float scale,
                      const int* mapping, size_t cache_stride, int query_stride) {
  // Grid axes own the sequence/head rather than recovering them with several
  // 64-bit divisions for every KV element. Each block copies a contiguous
  // segment of one head; the FP32 layout and arithmetic remain unchanged.
  int sequence = blockIdx.z, head = blockIdx.y;
  if (heads > 65535) {
    const int head_groups = (heads + 65534) / 65535;
    sequence = blockIdx.z / head_groups;
    head += (blockIdx.z % head_groups) * 65535;
    if (head >= heads) return;
  }
  const int row = mapping[sequence];
  const size_t within = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (blockIdx.x == 0) for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const size_t destination = (static_cast<size_t>(sequence) * heads + head) * dim + d;
    q[destination] = row < 0 ? 0 : __bfloat162float(query[static_cast<size_t>(row) * query_stride + head * dim + d]) * scale;
  }
  if (within >= static_cast<size_t>(tokens) * dim) return;
  const size_t i = (static_cast<size_t>(sequence) * heads + head) * tokens * dim + within;
  const size_t source = row < 0 ? 0 : static_cast<size_t>(row) * cache_stride +
      static_cast<size_t>(head / (heads / kv_heads)) * capacity * dim + within;
  k[i] = row < 0 ? 0 : __bfloat162float(keys[source]) * scale;
  v[i] = row < 0 ? 0 : __bfloat162float(values[source]);
}
__global__ void cast(const float* input, __nv_bfloat16* output, int hidden, const int* mapping, int sequences) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sequences * hidden) {
    const int row = mapping[i / hidden];
    if (row >= 0) output[static_cast<size_t>(row) * hidden + i % hidden] = __float2bfloat16_rn(input[i]);
  }
}
}  // namespace math_detail

// Metadata-driven FP32 math SDPA, using fixed sequence tiles to retain a
// fixed BMM geometry across request coalescing. Uses existing
// source-mapped softmax schedules; no claim of universal Torch byte parity.
class MathAttentionDynamic {
 public:
  explicit MathAttentionDynamic(RuntimeShape shape, int workspace_tokens = 0)
      : s_(shape), limit_(workspace_tokens ? workspace_tokens : shape.context) {
    using namespace parity::detail;
    if (limit_ < 1 || limit_ > s_.context || limit_ > parity::kMaxAttentionTokens)
      throw std::runtime_error("math attention capacity exceeds implemented softmax limit");
    shared_ = default_shared_memory_per_block();
    const size_t floats = kSequences * (2 * static_cast<size_t>(s_.hidden) +
        2 * static_cast<size_t>(s_.query_heads) * limit_ * (s_.head_dim + 1));
    try {
      cuda_check(cudaMalloc(&scratch_, floats * sizeof(float)), "math attention workspace");
      cuda_check(cudaMalloc(&mapping_, kSequences * sizeof(int)), "math attention mapping");
      cuda_check(cudaMalloc(&workspace_, kWorkspaceBytes), "math attention BLAS workspace");
      cuda_check(cudaMemset(scratch_, 0xff, floats * sizeof(float)), "initialize math attention workspace");
      cuda_check(cudaMemset(workspace_, 0xff, kWorkspaceBytes), "initialize math attention BLAS workspace");
      blas_check(cublasCreate(&handle_), "math attention handle");
      blas_check(cublasSetStream(handle_, cudaStreamPerThread), "math attention stream");
      blas_check(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "math attention precision");
      blas_check(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED), "math attention atomics");
      // Own reusable BLAS scratch explicitly, separate from attention tensors.
      // This resource safeguard is not a substitute for checkpoint integrity.
      // 4 MiB is the per-buffer size in NVIDIA's :4096:8 reproducibility config.
      blas_check(cublasSetWorkspace(handle_, workspace_, kWorkspaceBytes), "math attention fixed workspace");
    } catch (...) {
      if (handle_) cublasDestroy(handle_);
      if (scratch_) cudaFree(scratch_);
      if (mapping_) cudaFree(mapping_);
      if (workspace_) cudaFree(workspace_);
      throw;
    }
  }
  ~MathAttentionDynamic() { if (handle_) cublasDestroy(handle_); if (scratch_) cudaFree(scratch_); if (mapping_) cudaFree(mapping_); if (workspace_) cudaFree(workspace_); }
  MathAttentionDynamic(const MathAttentionDynamic&) = delete;
  MathAttentionDynamic& operator=(const MathAttentionDynamic&) = delete;
  void run(const __nv_bfloat16* query, const __nv_bfloat16* keys, const __nv_bfloat16* values,
            __nv_bfloat16* output, const int* device_positions, int batch,
            const int* stopped = nullptr, const int* host_stopped = nullptr) {
    using namespace parity::detail;
    if (batch < 1 || batch > 65535) throw std::invalid_argument("invalid math-attention batch size");
    if ((stopped == nullptr) != (host_stopped == nullptr))
      throw std::invalid_argument("math-attention stop masks must be paired");
    if (stopped)
      zero_stopped_attention_rows<<<dim3((s_.hidden + 255) / 256, batch), 256>>>(
          output, stopped, s_.hidden);
    std::vector<int> positions(batch);
    LFM_PROFILE_RANGE(profile_positions, "math.positions");
    cuda_check(cudaMemcpy(positions.data(), device_positions, batch * sizeof(int), cudaMemcpyDeviceToHost), "math attention positions");
    LFM_PROFILE_END(profile_positions);
    const float alpha = 1, beta = 0;
    const float scale = static_cast<float>(std::sqrt(1.0 / std::sqrt(static_cast<double>(s_.head_dim))));
    std::map<int, std::vector<int>> groups;
    for (int row = 0; row < batch; ++row) {
      // Check before +1: INT_MAX would otherwise overflow signed host arithmetic.
      if (positions[row] < 0 || positions[row] >= limit_)
        throw std::runtime_error("invalid math-attention position");
      if (host_stopped && host_stopped[row]) continue;
      groups[positions[row] + 1].push_back(row);
    }
    for (const auto& [tokens, rows] : groups) {
      if (tokens < 1 || tokens > limit_) throw std::runtime_error("invalid math-attention position");
      for (size_t first = 0; first < rows.size(); first += kSequences) {
      LFM_PROFILE_RANGE(profile_group, "math.group/tokens=" + std::to_string(tokens) +
          "/active=" + std::to_string(std::min<size_t>(kSequences, rows.size() - first)));
      std::vector<int> mapping(kSequences, -1);
      for (size_t i = 0; i < kSequences && first + i < rows.size(); ++i) mapping[i] = rows[first + i];
      cuda_check(cudaMemcpy(mapping_, mapping.data(), kSequences * sizeof(int), cudaMemcpyHostToDevice), "math attention group");
      float* q = scratch_;
      float* attended = q + kSequences * s_.hidden;
      float* k = attended + kSequences * s_.hidden;
      const int heads = kSequences * s_.query_heads;
      const size_t elements = static_cast<size_t>(heads) * tokens * s_.head_dim;
      float* v = k + elements;
      float* scores = v + elements;
      float* probability = scores + static_cast<size_t>(heads) * tokens;
      last_scores_ = scores; last_probability_ = probability; last_attended_ = attended; last_tokens_ = tokens;
      LFM_PROFILE_RANGE(profile_pack, "math.pack");
      math_detail::pack<<<dim3((static_cast<size_t>(tokens) * s_.head_dim + 255) / 256,
                              std::min(s_.query_heads, 65535),
                              kSequences * ((s_.query_heads + 65534) / 65535)), 256>>>(
          query, keys, values, q, k, v, s_.query_heads, s_.kv_heads, s_.head_dim, tokens, s_.context, scale,
          mapping_, s_.kv_batch_stride(), s_.qkv_width());
      LFM_PROFILE_END(profile_pack);
      LFM_PROFILE_RANGE(profile_qk, "math.qk");
      blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
          tokens, 1, s_.head_dim, &alpha, k, s_.head_dim, static_cast<long long>(tokens) * s_.head_dim,
           q, s_.head_dim, s_.head_dim, &beta, scores, tokens, tokens, heads), "math QK");
      LFM_PROFILE_END(profile_qk);
      LFM_PROFILE_RANGE(profile_softmax, "math.softmax");
      parity::softmax_fp32(scores, probability, tokens, cudaStreamPerThread, shared_, heads);
      LFM_PROFILE_END(profile_softmax);
      LFM_PROFILE_RANGE(profile_pv, "math.pv");
      blas_check(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          s_.head_dim, 1, tokens, &alpha, v, s_.head_dim, static_cast<long long>(tokens) * s_.head_dim,
           probability, tokens, tokens, &beta, attended, s_.head_dim, s_.head_dim, heads), "math PV");
      LFM_PROFILE_END(profile_pv);
      LFM_PROFILE_RANGE(profile_cast, "math.cast");
      math_detail::cast<<<(kSequences * s_.hidden + 255) / 256, 256>>>(attended, output, s_.hidden, mapping_, kSequences);
      LFM_PROFILE_END(profile_cast);
      cuda_check(cudaGetLastError(), "math attention kernels");
      }
    }
  }
  const float* scores() const { return last_scores_; }
  const float* probability() const { return last_probability_; }
  const float* attended() const { return last_attended_; }
  int tokens() const { return last_tokens_; }
 private:
  RuntimeShape s_;
  int limit_;
  static constexpr int kSequences = 32;
  static constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;
  int shared_;
  float* scratch_ = nullptr;
  int* mapping_ = nullptr;
  void* workspace_ = nullptr;
  cublasHandle_t handle_ = nullptr;
  const float* last_scores_ = nullptr;
  const float* last_probability_ = nullptr;
  const float* last_attended_ = nullptr;
  int last_tokens_ = 0;
};
}  // namespace lfm
