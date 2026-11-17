// Print the decode attention output of the shipped kernel and of a candidate,
// on a small fixture, so a mismatch can be read directly.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <lfm/attention.cuh>
#include <lfm/attention_candidates.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  const int batch = 1;
  const int position = argc > 1 ? atoi(argv[1]) : 8;
  const size_t elements = static_cast<size_t>(batch) * lfm::kKvLayerStride;

  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* key_cache = nullptr;
  __nv_bfloat16* value_cache = nullptr;
  __nv_bfloat16* shipped = nullptr;
  __nv_bfloat16* candidate = nullptr;
  int* positions = nullptr;
  int* policy = nullptr;
  cudaMalloc(&query, static_cast<size_t>(batch) * lfm::kQkvWidth * sizeof(__nv_bfloat16));
  cudaMalloc(&key_cache, elements * sizeof(__nv_bfloat16));
  cudaMalloc(&value_cache, elements * sizeof(__nv_bfloat16));
  cudaMalloc(&shipped, static_cast<size_t>(batch) * lfm::kHiddenSize * sizeof(__nv_bfloat16));
  cudaMalloc(&candidate, static_cast<size_t>(batch) * lfm::kHiddenSize * sizeof(__nv_bfloat16));
  cudaMalloc(&positions, batch * sizeof(int));
  cudaMalloc(&policy, 3 * sizeof(int));

  std::vector<__nv_bfloat16> host(static_cast<size_t>(lfm::kContextLength) * lfm::kHeadDimension);
  for (size_t index = 0; index < host.size(); ++index) {
    host[index] = __float2bfloat16(0.01f * sinf(static_cast<float>(index % 1021) * 0.017f));
  }
  cudaMemcpy(key_cache, host.data(), host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(value_cache, host.data(), host.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  std::vector<__nv_bfloat16> host_query(static_cast<size_t>(batch) * lfm::kQkvWidth);
  for (size_t index = 0; index < host_query.size(); ++index) {
    host_query[index] = __float2bfloat16(0.02f * sinf(static_cast<float>(index % 977) * 0.011f));
  }
  cudaMemcpy(query, host_query.data(), host_query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  const int host_positions[1] = {position};
  cudaMemcpy(positions, host_positions, sizeof(host_positions), cudaMemcpyHostToDevice);
  const int host_policy[3] = {0, 0, 0};
  cudaMemcpy(policy, host_policy, sizeof(host_policy), cudaMemcpyHostToDevice);

  const int stride = static_cast<int>(lfm::kKvLayerStride);
  lfm::online_softmax_gqa_attention<<<dim3(lfm::kQueryHeads, batch), 1024>>>(
      query, key_cache, value_cache, shipped, positions, policy, stride);
  lfm::attention_shared_kv_head<1024><<<dim3(lfm::kKvHeads, batch), 1024>>>(
      query, key_cache, value_cache, candidate, positions, policy, stride);
  cudaDeviceSynchronize();

  const size_t count = static_cast<size_t>(batch) * lfm::kHiddenSize;
  std::vector<__nv_bfloat16> a(count), b(count);
  cudaMemcpy(a.data(), shipped, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), candidate, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  int differing = 0;
  for (int head = 0; head < lfm::kQueryHeads; ++head) {
    int head_differences = 0;
    for (int dimension = 0; dimension < lfm::kHeadDimension; ++dimension) {
      const size_t index = static_cast<size_t>(head) * lfm::kHeadDimension + dimension;
      if (__bfloat162float(a[index]) != __bfloat162float(b[index])) ++head_differences;
    }
    differing += head_differences;
    std::printf("head %2d: %2d of %d dimensions differ\n", head, head_differences,
                lfm::kHeadDimension);
  }
  std::printf("total %d differing of %d\n", differing, static_cast<int>(count));
  if (differing > 0) {
    for (int head = 0; head < 4; ++head) {
      for (int dimension = 0; dimension < 2; ++dimension) {
        const size_t index = static_cast<size_t>(head) * lfm::kHeadDimension + dimension;
        std::printf("  head %d dim %d: shipped %12.6f candidate %12.6f\n", head, dimension,
                    __bfloat162float(a[index]), __bfloat162float(b[index]));
      }
    }
  }
  return 0;
}
