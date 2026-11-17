// Does running two independent single-token projections concurrently recover
// the weight-streaming bandwidth that cuBLAS leaves on the table?
//
// cuBLAS picks a 256x8 output tile for the 4608x1024 projection, so a single
// GEMV occupies only 4608/256 = 18 thread blocks on a 20-SM device. Measured
// cold weight traffic was 140-160 GB/s against a 256.5 GB/s streaming
// ceiling, so the gap is parallelism, not bandwidth.
//
// The feed-forward gate and up projections are independent: both read the
// same normalised activation and write different buffers. This tool measures
// the same pair sequentially and concurrently on two streams, with cold
// weights, and reports achieved bandwidth and bit-exactness of the
// concurrent result against the sequential one.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <lfm/safetensors.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", operation, static_cast<int>(status));
    std::exit(1);
  }
}

double time_ms(const std::function<void()>& op, int warmup, int iterations) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "event");
  check_cuda(cudaEventCreate(&stop), "event");
  for (int i = 0; i < warmup; ++i) op();
  check_cuda(cudaDeviceSynchronize(), "sync");
  check_cuda(cudaEventRecord(start), "event");
  for (int i = 0; i < iterations; ++i) op();
  check_cuda(cudaEventRecord(stop), "event");
  check_cuda(cudaEventSynchronize(stop), "event");
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iterations;
}

int mismatches(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
  int count = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (__bfloat162float(a[i]) != __bfloat162float(b[i])) ++count;
  }
  return count;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: gemv_concurrent <model.safetensors>\n");
    return 2;
  }
  lfm::Safetensors model(argv[1]);
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, 32u * 1024 * 1024), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024), "set workspace");

  cudaStream_t side;
  check_cuda(cudaStreamCreate(&side), "stream");
  cublasHandle_t side_handle;
  check_cublas(cublasCreate(&side_handle), "cublasCreate side");
  check_cublas(cublasSetStream(side_handle, side), "set side stream");
  check_cublas(cublasSetWorkspace(side_handle, workspace, 32u * 1024 * 1024), "side workspace");

  const int rows = 4608, cols = 1024;
  const size_t elements = static_cast<size_t>(rows) * cols;
  const size_t pool_bytes = 768u * 1024 * 1024;
  const size_t pool_count = pool_bytes / (elements * 2);
  __nv_bfloat16* pool = nullptr;
  check_cuda(cudaMalloc(&pool, pool_count * elements * sizeof(__nv_bfloat16)), "pool");
  check_cuda(cudaMemset(pool, 0x3c, pool_count * elements * sizeof(__nv_bfloat16)), "fill");

  __nv_bfloat16* x = nullptr;
  check_cuda(cudaMalloc(&x, cols * sizeof(__nv_bfloat16)), "x");
  std::vector<__nv_bfloat16> host_x(cols);
  for (int i = 0; i < cols; ++i) host_x[i] = __float2bfloat16(0.05f * sinf(i * 0.37f) + 0.25f);
  check_cuda(cudaMemcpy(x, host_x.data(), cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "x");

  __nv_bfloat16* gate = nullptr;
  __nv_bfloat16* value = nullptr;
  check_cuda(cudaMalloc(&gate, rows * sizeof(__nv_bfloat16)), "gate");
  check_cuda(cudaMalloc(&value, rows * sizeof(__nv_bfloat16)), "value");

  const float alpha = 1.0f, beta = 0.0f;
  size_t cursor = 0;
  auto project = [&](cublasHandle_t h, __nv_bfloat16* out) {
    const __nv_bfloat16* w = pool + cursor * elements;
    cursor = (cursor + 1) % pool_count;
    check_cublas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols,
                              &alpha, w, CUDA_R_16BF, cols,
                              x, CUDA_R_16BF, cols, &beta,
                              out, CUDA_R_16BF, rows,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm");
  };

  // Sequential: two projections, one after the other.
  auto sequential = [&] {
    project(handle, gate);
    project(handle, value);
  };
  sequential();
  check_cuda(cudaDeviceSynchronize(), "sync");
  std::vector<__nv_bfloat16> host_gate(rows), host_value(rows);
  check_cuda(cudaMemcpy(host_gate.data(), gate, rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "g");
  check_cuda(cudaMemcpy(host_value.data(), value, rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "v");

  const double bytes = static_cast<double>(elements) * 2.0;
  const double sequential_ms = time_ms(sequential, 5, 40);

  // Concurrent: the second projection runs on a side stream, forked and
  // joined with events so the dependency structure is explicit.
  cudaEvent_t fork, join;
  check_cuda(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "fork");
  check_cuda(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "join");
  auto concurrent = [&] {
    check_cuda(cudaEventRecord(fork, cudaStreamPerThread), "record fork");
    check_cuda(cudaStreamWaitEvent(side, fork, 0), "wait fork");
    project(side_handle, value);
    check_cuda(cudaEventRecord(join, side), "record join");
    check_cuda(cudaStreamWaitEvent(cudaStreamPerThread, join, 0), "wait join");
    project(handle, gate);
  };
  concurrent();
  check_cuda(cudaDeviceSynchronize(), "sync");
  std::vector<__nv_bfloat16> concurrent_gate(rows), concurrent_value(rows);
  check_cuda(cudaMemcpy(concurrent_gate.data(), gate, rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "g");
  check_cuda(cudaMemcpy(concurrent_value.data(), value, rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "v");

  const double concurrent_ms = time_ms(concurrent, 5, 40);
  const double pair_bytes = 2.0 * bytes;

  std::printf("pair of 4608x1024 projections, cold weights\n");
  std::printf("  sequential   %8.3f ms   %8.1f GB/s\n", sequential_ms, pair_bytes / (sequential_ms * 1e-3) / 1e9);
  std::printf("  concurrent   %8.3f ms   %8.1f GB/s\n", concurrent_ms, pair_bytes / (concurrent_ms * 1e-3) / 1e9);
  std::printf("  speedup      %8.3fx\n", sequential_ms / concurrent_ms);
  std::printf("  mismatches   gate=%d value=%d of %d\n",
              mismatches(host_gate, concurrent_gate), mismatches(host_value, concurrent_value), rows);
  return 0;
}
