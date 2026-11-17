// Measure how single-token projections scale when B tokens share one weight
// read, and model the resulting batch decode throughput.
//
// A batch of B tokens does not need a batched GEMM: every token passes
// through the same weights, so the B activations form one [B, cols] matrix
// and each projection becomes a single GEMM with n=B. This tool measures
// exactly that call for every projection shape the decode path uses, with
// cold weights, and then models the whole token.
//
// Decode at batch 1 is bandwidth bound: every projection reads its whole
// weight matrix to produce one output vector, at roughly 1 FLOP per byte.
// A batch of B tokens reads the same matrix once and produces B output
// vectors, so the weight traffic per token falls by B and the projection
// becomes a real GEMM with M=B.
//
// This tool measures the actual projection shapes at each batch size with
// cold weights, and reports the effective tokens per second the projection
// alone would sustain, the weight bandwidth it achieves, and the arithmetic
// throughput it achieves.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
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

}  // namespace

int main() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "properties");
  std::printf("device: %s  SMs=%d  L2=%.1f MiB\n\n",
              properties.name, properties.multiProcessorCount,
              properties.l2CacheSize / 1048576.0);

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, 32u * 1024 * 1024), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024), "workspace");

  // Cold weight pool: the decode path streams 676 MiB per token in a fixed
  // cyclic order, so the reuse distance exceeds the cache and every weight
  // read misses.
  const size_t pool_bytes = 768u * 1024 * 1024;

  struct Shape { int rows; int cols; int per_token; const char* name; };
  const Shape shapes[] = {
      {4608, 1024, 32, "w1/w3     "},
      {1024, 4608, 16, "w2        "},
      {3072, 1024, 10, "conv_in   "},
      {2048, 1024, 6,  "qkv       "},
      {1024, 1024, 16, "attn/conv_out"},
      {65536, 1024, 1, "lm_head   "},
  };
  const int batches[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  const int max_batch = 1024;

  std::printf("%-14s %6s %10s %10s %10s %10s\n",
              "shape", "batch", "ms/batch", "tok/s", "GB/s", "TFLOP/s");
  // Measured per-projection batch time, kept for the aggregate model.
  std::vector<std::vector<double>> batch_ms(std::size(shapes));
  for (size_t s = 0; s < std::size(shapes); ++s) {
    const Shape& shape = shapes[s];
    const int rows = shape.rows, cols = shape.cols;
    const size_t matrix_elements = static_cast<size_t>(rows) * cols;
    const size_t matrix_bytes = matrix_elements * sizeof(__nv_bfloat16);
    const size_t pool_count = std::max<size_t>(1, pool_bytes / matrix_bytes);
    __nv_bfloat16* pool = nullptr;
    check_cuda(cudaMalloc(&pool, pool_count * matrix_elements * sizeof(__nv_bfloat16)), "pool");
    check_cuda(cudaMemset(pool, 0x3c, pool_count * matrix_elements * sizeof(__nv_bfloat16)), "fill");
    __nv_bfloat16* input = nullptr;
    __nv_bfloat16* output = nullptr;
    check_cuda(cudaMalloc(&input, static_cast<size_t>(max_batch) * cols * sizeof(__nv_bfloat16)), "input");
    check_cuda(cudaMalloc(&output, static_cast<size_t>(max_batch) * rows * sizeof(__nv_bfloat16)), "output");

    const float alpha = 1.0f, beta = 0.0f;
    size_t cursor = 0;
    for (const int batch : batches) {
      // Column-major view of a [batch, cols] row-major activation block.
      auto launch = [&] {
        const __nv_bfloat16* w = pool + cursor * matrix_elements;
        cursor = (cursor + 1) % pool_count;
        check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, batch, cols,
                                  &alpha, w, CUDA_R_16BF, cols,
                                  input, CUDA_R_16BF, cols, &beta,
                                  output, CUDA_R_16BF, rows,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                      "gemmEx");
      };
      launch();
      check_cuda(cudaDeviceSynchronize(), "sync");
      const double ms = time_ms(launch, 3, 30);
      const double seconds = ms * 1e-3;
      const double tokens_per_second = batch / seconds;
      const double weight_gbs = static_cast<double>(matrix_bytes) / seconds / 1e9;
      const double tflops = 2.0 * rows * cols * batch / seconds / 1e12;
      std::printf("%-14s %6d %10.4f %10.1f %10.1f %10.2f\n", shape.name, batch, ms,
                  tokens_per_second, weight_gbs, tflops);
      std::fflush(stdout);
      batch_ms[s].push_back(ms);
    }
    cudaFree(pool);
    cudaFree(input);
    cudaFree(output);
    std::printf("\n");
  }

  // Reference: a pure streaming read of the same 768 MiB pool, for the
  // bandwidth ceiling the projections are measured against.
  double ceiling_gbs = 0.0;
  {
    const size_t bytes = 768u * 1024 * 1024;
    const size_t count = bytes / sizeof(uint4);
    uint4* data = nullptr;
    uint4* sink = nullptr;
    check_cuda(cudaMalloc(&data, bytes), "data");
    check_cuda(cudaMalloc(&sink, sizeof(uint4)), "sink");
    check_cuda(cudaMemset(data, 0x3c, bytes), "fill");
    auto read = [&] { stream_read<<<1024, 256>>>(data, sink, count); };
    read();
    check_cuda(cudaDeviceSynchronize(), "sync");
    const double ms = time_ms(read, 3, 30);
    ceiling_gbs = bytes / (ms * 1e-3) / 1e9;
    std::printf("streaming read ceiling: %.1f GB/s\n\n", ceiling_gbs);
  }

  // Aggregate model: the decode path performs these projections per token.
  // A batch of B tokens reads each weight matrix once, so the per-batch time
  // is the sum of one projection per call site and the per-token cost falls
  // by the batch size.
  //
  // The per-token terms that do NOT amortize are added separately below:
  // attention reads a distinct KV cache per sequence, and the small
  // per-token kernels carry a fixed launch cost that a batched kernel would
  // amortize rather than scale.
  std::printf("%-7s %12s %12s %12s %12s %12s\n", "batch", "proj ms",
              "attn us/tok", "small us/tok", "total us/tok", "tok/s");
  for (size_t b = 0; b < std::size(batches); ++b) {
    const int batch = batches[b];
    double projection_ms = 0.0;
    for (size_t s = 0; s < std::size(shapes); ++s) {
      projection_ms += shapes[s].per_token * batch_ms[s][b];
    }
    // Attention: 6 layers, one kernel each, per token. Measured 18.8 us per
    // kernel at the control context with only 16 blocks on 20 SMs. The KV
    // read does not amortize across the batch, so the floor is the traffic
    // and the launch-bound cost is charged per token.
    const double attention_us_per_token = 6.0 * 18.8;
    // Small per-token kernels: 33 RMSNorm launches at 3.6 us and 48
    // elementwise launches at 1.5 us, measured at batch 1. A batched kernel
    // amortizes the launch cost, so the residual is the per-token work.
    const double small_us_per_token = 33.0 * 0.6 + 48.0 * 0.2;
    const double total_us_per_token =
        projection_ms * 1000.0 / batch + attention_us_per_token + small_us_per_token;
    std::printf("%-7d %12.3f %12.1f %12.1f %12.1f %12.1f\n", batch, projection_ms,
                attention_us_per_token, small_us_per_token, total_us_per_token,
                1e6 / total_us_per_token);
  }
  return 0;
}
