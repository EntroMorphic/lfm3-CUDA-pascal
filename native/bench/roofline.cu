#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t kStreamBytes = 512ULL * 1024 * 1024;
constexpr int kStreamIterations = 40;
constexpr int kGemmDimension = 4096;
constexpr int kGemmIterations = 20;
constexpr size_t kWorkspaceBytes = 32ULL * 1024 * 1024;

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

__global__ void stream_copy(const uint4* input, uint4* output, size_t count) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    output[index] = input[index];
  }
}

__global__ void stream_read_checksum(
    const uint4* input,
    uint4* block_checksums,
    size_t count) {
  uint4 checksum{};
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 value = input[index];
    checksum.x += value.x;
    checksum.y += value.y;
    checksum.z += value.z;
    checksum.w += value.w;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    checksum.x += __shfl_down_sync(0xffffffff, checksum.x, offset);
    checksum.y += __shfl_down_sync(0xffffffff, checksum.y, offset);
    checksum.z += __shfl_down_sync(0xffffffff, checksum.z, offset);
    checksum.w += __shfl_down_sync(0xffffffff, checksum.w, offset);
  }
  __shared__ uint4 warp_checksums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_checksums[warp] = checksum;
  __syncthreads();
  if (warp == 0) {
    checksum = lane < blockDim.x / 32 ? warp_checksums[lane] : uint4{};
    for (int offset = 16; offset > 0; offset >>= 1) {
      checksum.x += __shfl_down_sync(0xffffffff, checksum.x, offset);
      checksum.y += __shfl_down_sync(0xffffffff, checksum.y, offset);
      checksum.z += __shfl_down_sync(0xffffffff, checksum.z, offset);
      checksum.w += __shfl_down_sync(0xffffffff, checksum.w, offset);
    }
    if (lane == 0) block_checksums[blockIdx.x] = checksum;
  }
}

template <typename Operation>
float measure_ms(Operation operation, int warmup, int iterations) {
  for (int iteration = 0; iteration < warmup; ++iteration) operation();
  check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  check_cuda(cudaEventRecord(start), "record start event");
  for (int iteration = 0; iteration < iterations; ++iteration) operation();
  check_cuda(cudaEventRecord(stop), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "measurement synchronize");
  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
}

}  // namespace

int main() {
  try {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    constexpr size_t vector_count = kStreamBytes / sizeof(uint4);
    DeviceBuffer<uint4> stream_input(vector_count);
    DeviceBuffer<uint4> stream_output(vector_count);
    const int stream_blocks = properties.multiProcessorCount * 16;
    DeviceBuffer<uint4> checksums(stream_blocks);
    check_cuda(
        cudaMemset(stream_input.data, 0x5a, kStreamBytes),
        "initialize stream input");

    const float read_ms = measure_ms(
        [&] {
          stream_read_checksum<<<stream_blocks, 256>>>(
              stream_input.data, checksums.data, vector_count);
        },
        5,
        kStreamIterations);
    check_cuda(cudaGetLastError(), "read benchmark kernel");
    const float copy_ms = measure_ms(
        [&] {
          stream_copy<<<stream_blocks, 256>>>(
              stream_input.data, stream_output.data, vector_count);
        },
        5,
        kStreamIterations);
    check_cuda(cudaGetLastError(), "copy benchmark kernel");
    uint4 checksum{};
    check_cuda(
        cudaMemcpy(&checksum, checksums.data, sizeof(checksum), cudaMemcpyDeviceToHost),
        "copy read checksum");

    const size_t matrix_elements =
        static_cast<size_t>(kGemmDimension) * kGemmDimension;
    DeviceBuffer<__nv_bfloat16> matrix_a(matrix_elements);
    DeviceBuffer<__nv_bfloat16> matrix_b(matrix_elements);
    DeviceBuffer<__nv_bfloat16> matrix_c(matrix_elements);
    DeviceBuffer<std::byte> workspace(kWorkspaceBytes);
    check_cuda(cudaMemset(matrix_a.data, 0x3f, matrix_elements * sizeof(__nv_bfloat16)),
               "initialize matrix A");
    check_cuda(cudaMemset(matrix_b.data, 0x3f, matrix_elements * sizeof(__nv_bfloat16)),
               "initialize matrix B");
    cublasHandle_t cublas = nullptr;
    check_cublas(cublasCreate(&cublas), "cublasCreate");
    check_cublas(
        cublasSetWorkspace(cublas, workspace.data, kWorkspaceBytes),
        "cublasSetWorkspace");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto gemm = [&] {
      check_cublas(
          cublasGemmEx(
              cublas,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              kGemmDimension,
              kGemmDimension,
              kGemmDimension,
              &alpha,
              matrix_a.data,
              CUDA_R_16BF,
              kGemmDimension,
              matrix_b.data,
              CUDA_R_16BF,
              kGemmDimension,
              &beta,
              matrix_c.data,
              CUDA_R_16BF,
              kGemmDimension,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "BF16 roofline GEMM");
    };
    const float gemm_ms = measure_ms(gemm, 5, kGemmIterations);
    cublasDestroy(cublas);

    const double read_seconds = read_ms / 1000.0;
    const double copy_seconds = copy_ms / 1000.0;
    const double read_gbps =
        static_cast<double>(kStreamBytes) * kStreamIterations / read_seconds / 1.0e9;
    const double copy_read_gbps =
        static_cast<double>(kStreamBytes) * kStreamIterations / copy_seconds / 1.0e9;
    const double copy_aggregate_gbps = 2.0 * copy_read_gbps;
    const double gemm_flops =
        2.0 * kGemmDimension * kGemmDimension * kGemmDimension * kGemmIterations;
    const double gemm_tflops = gemm_flops / (gemm_ms / 1000.0) / 1.0e12;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "device: " << properties.name << '\n';
    std::cout << "sm_count: " << properties.multiProcessorCount << '\n';
    std::cout << "stream_working_set_mib: "
              << kStreamBytes / (1024.0 * 1024.0) << '\n';
    std::cout << "stream_iterations: " << kStreamIterations << '\n';
    std::cout << "read_elapsed_ms: " << read_ms << '\n';
    std::cout << "read_gbps: " << read_gbps << '\n';
    std::cout << "read_checksum: " << checksum.x << ':' << checksum.y << ':'
              << checksum.z << ':' << checksum.w << '\n';
    std::cout << "copy_elapsed_ms: " << copy_ms << '\n';
    std::cout << "copy_read_gbps: " << copy_read_gbps << '\n';
    std::cout << "copy_aggregate_gbps: " << copy_aggregate_gbps << '\n';
    std::cout << "bf16_gemm_dimension: " << kGemmDimension << '\n';
    std::cout << "bf16_gemm_iterations: " << kGemmIterations << '\n';
    std::cout << "bf16_gemm_elapsed_ms: " << gemm_ms << '\n';
    std::cout << "bf16_tensor_tflops: " << gemm_tflops << '\n';
    std::cout << "ridge_point_flops_per_byte: "
              << gemm_tflops * 1000.0 / read_gbps << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
