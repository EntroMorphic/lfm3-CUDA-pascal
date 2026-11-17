// Benchmark single-token BF16 projections (y = W x, M=1) for the shapes the
// LFM2.5-350M decode path uses.
//
// The decode path streams 676 MiB of distinct weights per token in a fixed
// cyclic order, so the reuse distance exceeds any last-level cache and every
// projection reads cold data. A benchmark that re-reads one matrix in a tight
// loop measures L2 bandwidth, not the behaviour the engine sees. This tool
// therefore rotates over a pool of distinct matrices sized to exceed the
// device cache, and compares:
//
//   * the production cuBLAS path (cublasGemmEx, OP_T, COMPUTE_32F)
//   * a streaming warp-per-row GEMV with 16-byte loads
//   * a pure streaming read of the same footprint (the achievable ceiling)
//
// It also reports the numerical deviation from the cuBLAS path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": cuBLAS status " + std::to_string(status));
  }
}

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t count = 0;
  explicit DeviceBuffer(size_t elements) : count(elements) {
    check_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Streaming warp-per-row GEMV: one warp per output row, 16-byte loads.
__global__ void gemv_warp_vector(
    const __nv_bfloat16* weight, const __nv_bfloat16* x, __nv_bfloat16* y,
    int rows, int cols) {
  extern __shared__ __nv_bfloat16 shared_x[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) shared_x[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int global_warp = blockIdx.x * warps_per_block + threadIdx.x / 32;
  const int warp_stride = gridDim.x * warps_per_block;
  const int vectors = cols / 8;
  for (int row = global_warp; row < rows; row += warp_stride) {
    const __nv_bfloat16* w = weight + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
      const uint4 raw = *reinterpret_cast<const uint4*>(w + v * 8);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
      const int base = v * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 xv = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&shared_x[base + 2 * j]));
        const float2 wv = __bfloat1622float2(pairs[j]);
        sum += xv.x * wv.x + xv.y * wv.y;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
  }
}

// Pure streaming read ceiling over the same footprint.
__global__ void stream_read(const uint4* input, uint4* sink, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint4 accumulator{};
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    const uint4 value = input[index];
    accumulator.x += value.x;
    accumulator.y += value.y;
    accumulator.z += value.z;
    accumulator.w += value.w;
  }
  if (accumulator.x == 0x7fffffffu) sink[0] = accumulator;
}

double time_ms(const std::function<void()>& op, int warmup, int iterations) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "event create");
  check_cuda(cudaEventCreate(&stop), "event create");
  for (int i = 0; i < warmup; ++i) op();
  check_cuda(cudaDeviceSynchronize(), "sync");
  check_cuda(cudaEventRecord(start), "event record");
  for (int i = 0; i < iterations; ++i) op();
  check_cuda(cudaEventRecord(stop), "event record");
  check_cuda(cudaEventSynchronize(stop), "event sync");
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  check_cuda(cudaEventDestroy(start), "event destroy");
  check_cuda(cudaEventDestroy(stop), "event destroy");
  return ms / iterations;
}

double max_abs_diff(const std::vector<__nv_bfloat16>& a,
                    const std::vector<__nv_bfloat16>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(static_cast<double>(__bfloat162float(a[i])) -
                              static_cast<double>(__bfloat162float(b[i])));
    if (d > worst) worst = d;
  }
  return worst;
}

}  // namespace

int main() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "device properties");
  int clock_khz = 0;
  int bus_width = 0;
  check_cuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0), "clock");
  check_cuda(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0), "bus");
  std::printf("device: %s  SMs=%d  L2=%.1f MiB  clock=%.2f GHz  bus=%d bit\n\n",
              properties.name, properties.multiProcessorCount,
              properties.l2CacheSize / 1048576.0, clock_khz / 1e6, bus_width);

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  size_t workspace_bytes = 32u * 1024 * 1024;
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, workspace_bytes), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, workspace_bytes), "set workspace");

  // Pool large enough that the cyclic access pattern cannot be served by cache.
  const size_t pool_bytes = 768u * 1024 * 1024;

  struct Shape { int rows; int cols; const char* name; };
  const Shape shapes[] = {
      {4608, 1024, "w1/w3     "},
      {1024, 4608, "w2        "},
      {3072, 1024, "conv_in   "},
      {2048, 1024, "qkv       "},
      {1024, 1024, "attn/conv_out"},
      {65536, 1024, "lm_head   "},
  };

  std::printf("%-14s %10s %10s %10s %10s %10s %10s\n", "shape", "GB/s cub",
               "GB/s gemv", "GB/s rd", "us cub", "us gemv", "max|d|");
  for (const Shape& shape : shapes) {
    const int rows = shape.rows, cols = shape.cols;
    const size_t matrix_elements = static_cast<size_t>(rows) * cols;
    const size_t matrix_bytes = matrix_elements * sizeof(__nv_bfloat16);
    const size_t pool_count = std::max<size_t>(1, pool_bytes / matrix_bytes);
    DeviceBuffer<__nv_bfloat16> pool(pool_count * matrix_elements);
    DeviceBuffer<__nv_bfloat16> x(cols);
    DeviceBuffer<__nv_bfloat16> reference(rows);
    DeviceBuffer<__nv_bfloat16> candidate(rows);
    DeviceBuffer<uint4> sink(4);

    std::vector<__nv_bfloat16> host_x(cols);
    for (int i = 0; i < cols; ++i) host_x[i] = __float2bfloat16(0.5f + 0.001f * i);
    check_cuda(cudaMemcpy(x.data, host_x.data(), cols * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemset(pool.data, 0x3c, pool.count * sizeof(__nv_bfloat16)),
               "fill pool");

    const float alpha = 1.0f, beta = 0.0f;
    size_t cursor = 0;
    auto cublas_gemv = [&] {
      const __nv_bfloat16* w = pool.data + cursor * matrix_elements;
      cursor = (cursor + 1) % pool_count;
      check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols,
                                &alpha, w, CUDA_R_16BF, cols,
                                x.data, CUDA_R_16BF, cols, &beta,
                                reference.data, CUDA_R_16BF, rows,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemmEx");
    };
    cublas_gemv();
    check_cuda(cudaDeviceSynchronize(), "sync");

    const double cublas_ms = time_ms(cublas_gemv, 5, 60);

    const int block = 256;
    const int blocks = std::max(1, (rows + block / 32 - 1) / (block / 32));
    cursor = 0;
    auto vec_gemv = [&] {
      const __nv_bfloat16* w = pool.data + cursor * matrix_elements;
      cursor = (cursor + 1) % pool_count;
      gemv_warp_vector<<<blocks, block, cols * sizeof(__nv_bfloat16)>>>(
          w, x.data, candidate.data, rows, cols);
    };
    vec_gemv();
    check_cuda(cudaDeviceSynchronize(), "sync");
    const double vec_ms = time_ms(vec_gemv, 5, 60);

    // Streaming read ceiling over the same pool.
    const size_t pool_uint4 = pool.count * sizeof(__nv_bfloat16) / sizeof(uint4);
    cursor = 0;
    auto read = [&] {
      const uint4* w = reinterpret_cast<const uint4*>(pool.data) +
                       (cursor * matrix_elements * sizeof(__nv_bfloat16) / sizeof(uint4));
      cursor = (cursor + 1) % pool_count;
      stream_read<<<1024, 256>>>(w, sink.data, matrix_elements * sizeof(__nv_bfloat16) / sizeof(uint4));
    };
    read();
    check_cuda(cudaDeviceSynchronize(), "sync");
    const double read_ms = time_ms(read, 5, 60);

    const double gbs = static_cast<double>(matrix_bytes) / 1e9;
    std::vector<__nv_bfloat16> host_reference(rows), host_candidate(rows);
    check_cuda(cudaMemcpy(host_reference.data(), reference.data,
                          rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy ref");
    check_cuda(cudaMemcpy(host_candidate.data(), candidate.data,
                          rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy cand");
    const double diff = max_abs_diff(host_reference, host_candidate);

    std::printf("%-14s %10.1f %10.1f %10.1f %10.1f %10.1f %10.2e\n", shape.name,
                gbs / (cublas_ms * 1e-3), gbs / (vec_ms * 1e-3), gbs / (read_ms * 1e-3),
                cublas_ms * 1000.0, vec_ms * 1000.0, diff);
    std::fflush(stdout);
  }
  return 0;
}
