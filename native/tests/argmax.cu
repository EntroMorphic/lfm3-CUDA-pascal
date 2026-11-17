// GPU regression/replay and event benchmark of the actual production kernel.
#include <lfm/argmax.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// Frozen pre-fix reduction, retained only as the measured baseline.
__global__ void legacy_argmax(const __nv_bfloat16* logits, int count, int* output) {
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -FLT_MAX;
  int best_index = 0;
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    const float value = __bfloat162float(logits[index]);
    if (value > best) { best = value; best_index = index; }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  for (int offset = 128; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset && values[threadIdx.x + offset] > values[threadIdx.x]) {
      values[threadIdx.x] = values[threadIdx.x + offset];
      indices[threadIdx.x] = indices[threadIdx.x + offset];
    }
  }
  if (threadIdx.x == 0) output[0] = indices[0];
}

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: argmax-test native.bf16");
    __nv_bfloat16* device;
    int* result;
    check(cudaMalloc(&device, 65536 * sizeof(*device)));
    check(cudaMalloc(&result, sizeof(int)));
    int failures = 0, old_failures = 0, cases = 0;
    auto test = [&](const std::vector<__nv_bfloat16>& row) {
      int expected = 0;
      for (int i = 1; i < static_cast<int>(row.size()); ++i)
        if (__bfloat162float(row[i]) > __bfloat162float(row[expected])) expected = i;
      check(cudaMemcpy(device, row.data(), row.size() * sizeof(row[0]), cudaMemcpyHostToDevice));
      int actual = 0;
      lfm::argmax_bf16<<<1, 256>>>(device, row.size(), result);
      check(cudaMemcpy(&actual, result, sizeof(int), cudaMemcpyDeviceToHost));
      failures += actual != expected;
      legacy_argmax<<<1, 256>>>(device, row.size(), result);
      check(cudaMemcpy(&actual, result, sizeof(int), cudaMemcpyDeviceToHost));
      old_failures += actual != expected;
      ++cases;
    };
    for (int count : {1, 17, 255, 256, 257, 65536}) {
      std::vector<__nv_bfloat16> row(count, __float2bfloat16(-3));
      test(row);
      row.back() = __float2bfloat16(8); test(row);
      row.front() = __float2bfloat16(8); test(row);
    }
    for (auto pair : {std::pair{1927,40516}, {875,1593}, {1,256}, {255,256}, {0,65535}}) {
      std::vector<__nv_bfloat16> row(65536, __float2bfloat16(-5));
      row[pair.first] = row[pair.second] = __float2bfloat16(17.75f); test(row);
    }
    std::cout << "synthetic_cases: " << cases << "\nsynthetic_fixed_failures: " << failures << '\n';
    const int synthetic_old = old_failures;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot read captured logits");
    std::vector<__nv_bfloat16> capture(81920), row(65536);
    int captured = 0;
    while (input.read(reinterpret_cast<char*>(capture.data()), capture.size()*sizeof(capture[0]))) {
      std::copy(capture.begin()+16384, capture.end(), row.begin());
      test(row); ++captured;
    }
    if (input.gcount() || !input.eof() || !captured) throw std::runtime_error("truncated/empty capture");
    std::cout << "captured_cases: " << captured << "\ncaptured_old_failures: " << old_failures-synthetic_old
              << "\ntotal_fixed_failures: " << failures << '\n';
    if (failures) throw std::runtime_error("fixed argmax regression");
    // Both variants use identical data, 100 warmups and seven alternating
    // paired trials of 10000 launches. CUDA events exclude host I/O/model load.
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    for (int i = 0; i < 100; ++i) {
      legacy_argmax<<<1,256>>>(device,65536,result);
      lfm::argmax_bf16<<<1,256>>>(device,65536,result);
    }
    check(cudaDeviceSynchronize());
    for (int trial = 0; trial < 7; ++trial) {
      for (int order = 0; order < 2; ++order) {
        const bool fixed = (trial + order) % 2;
        check(cudaEventRecord(start));
        for (int i = 0; i < 10000; ++i) {
          if (fixed) lfm::argmax_bf16<<<1,256>>>(device,65536,result);
          else legacy_argmax<<<1,256>>>(device,65536,result);
        }
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        float ms = 0; check(cudaEventElapsedTime(&ms,start,stop));
        std::cout << "kernel_us " << trial << ' ' << (fixed ? "fixed" : "old") << ' ' << ms * 1000 / 10000 << '\n';
      }
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(device)); check(cudaFree(result));
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
