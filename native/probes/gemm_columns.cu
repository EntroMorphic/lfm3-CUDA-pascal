// Is a cuBLAS GEMM with n > 1 column-uniform?
//
// The batched decode path projects B sequences through the same weights with
// one GEMM of n=B. If cuBLAS produced the same result for every column, the
// B sequences of a batch over the same prompt would be identical. This tool
// projects one input vector replicated B times and compares the columns.
//
// It also compares the n=B result against the n=1 result, because that is
// the numerical difference the batched path introduces relative to the
// single-sequence path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

int argmax_of(const std::vector<__nv_bfloat16>& values) {
  int best = 0;
  float best_value = -INFINITY;
  for (size_t i = 0; i < values.size(); ++i) {
    const float value = __bfloat162float(values[i]);
    if (value > best_value) {
      best_value = value;
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace

int main() {
  const int rows = 65536;
  const int cols = 1024;
  const int batch = 8;
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, 32u * 1024 * 1024), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024), "workspace");

  std::vector<__nv_bfloat16> host_weight(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < host_weight.size(); ++i) {
    host_weight[i] = __float2bfloat16(0.02f * sinf(static_cast<float>(i % 977) * 0.031f));
  }
  __nv_bfloat16* weight = nullptr;
  check_cuda(cudaMalloc(&weight, host_weight.size() * sizeof(__nv_bfloat16)), "weight");
  check_cuda(cudaMemcpy(weight, host_weight.data(), host_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice), "copy weight");

  std::vector<__nv_bfloat16> host_input(cols);
  for (int i = 0; i < cols; ++i) host_input[i] = __float2bfloat16(0.4f * sinf(i * 0.11f) + 0.3f);
  __nv_bfloat16* input = nullptr;
  check_cuda(cudaMalloc(&input, static_cast<size_t>(batch) * cols * sizeof(__nv_bfloat16)), "input");
  for (int b = 0; b < batch; ++b) {
    check_cuda(cudaMemcpy(input + static_cast<size_t>(b) * cols, host_input.data(),
                          cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy input");
  }
  __nv_bfloat16* output = nullptr;
  check_cuda(cudaMalloc(&output, static_cast<size_t>(batch) * rows * sizeof(__nv_bfloat16)), "output");

  const float alpha = 1.0f, beta = 0.0f;
  auto gemm = [&](int n) {
    check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols,
                              &alpha, weight, CUDA_R_16BF, cols,
                              input, CUDA_R_16BF, cols, &beta,
                              output, CUDA_R_16BF, rows,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemmEx");
  };

  gemm(1);
  check_cuda(cudaDeviceSynchronize(), "sync");
  std::vector<__nv_bfloat16> single(rows);
  check_cuda(cudaMemcpy(single.data(), output, rows * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy single");

  gemm(batch);
  check_cuda(cudaDeviceSynchronize(), "sync");
  std::vector<__nv_bfloat16> many(static_cast<size_t>(batch) * rows);
  check_cuda(cudaMemcpy(many.data(), output, many.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy many");

  const int single_argmax = argmax_of(single);
  std::printf("single (n=1) argmax: %d\n", single_argmax);
  std::printf("n=%d per-column results:\n", batch);
  int identical_columns = 0;
  int matching_single = 0;
  for (int b = 0; b < batch; ++b) {
    const std::vector<__nv_bfloat16> column(
        many.begin() + static_cast<size_t>(b) * rows,
        many.begin() + static_cast<size_t>(b + 1) * rows);
    double worst = 0.0;
    for (int i = 0; i < rows; ++i) {
      const double d = std::abs(static_cast<double>(__bfloat162float(column[i])) -
                                static_cast<double>(__bfloat162float(single[i])));
      worst = std::max(worst, d);
    }
    const int column_argmax = argmax_of(column);
    const bool same_as_column0 =
        column == std::vector<__nv_bfloat16>(
                     many.begin(), many.begin() + rows);
    if (same_as_column0) ++identical_columns;
    if (column_argmax == single_argmax) ++matching_single;
    std::printf("  column %d: argmax=%-7d max|vs n=1|=%.3e  identical to column 0: %s\n",
                b, column_argmax, worst, same_as_column0 ? "yes" : "NO");
  }
  std::printf("columns identical to column 0: %d/%d\n", identical_columns, batch);
  std::printf("columns whose argmax matches n=1: %d/%d\n", matching_single, batch);
  return 0;
}
