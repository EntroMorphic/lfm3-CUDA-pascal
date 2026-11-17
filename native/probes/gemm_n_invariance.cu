// Is a cuBLAS GEMM's column bit-identical no matter how many columns ride along?
//
// The batched decode path projects B sequences through the same weights with one
// GEMM of n=B. The audit already showed that a GEMM with n>1 is column-uniform
// and that its n=1 result matches the single-sequence path, but it did not
// answer the question a length-sorted batch needs: if the batch shrinks as
// short prompts finish, the same row is computed with a different n at
// different steps. If the answer to a row changes with n, the batched path
// cannot change n mid-run without changing its output.
//
// This tool projects one input vector replicated across several column counts
// and compares every column, bit for bit, against the n=1 column, for each
// projection shape the model uses.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <lfm/model_config.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

struct Shape {
  int output_size;
  int input_size;
  const char* name;
};

// Every distinct (output, input) pair the model projects through.
const Shape kShapes[] = {
    {2048, 1024, "qkv"},
    {1024, 1024, "attention_out / conv_out_proj / q_proj"},
    {512, 1024, "k_proj / v_proj"},
    {4608, 1024, "feed_forward.w1 / w3"},
    {1024, 4608, "feed_forward.w2"},
    {3072, 1024, "conv.in_proj"},
    {65536, 1024, "lm_head"},
};

const int kColumnCounts[] = {1, 2, 3, 7, 8, 15, 16, 31, 32};

}  // namespace

int main(int argc, char** argv) {
  std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
  if (argc == 2) {
    const auto c = lfm::ModelConfig::load(argv[1]);
    const auto& s = c.shape;
    shapes = {{s.qkv_width(), s.hidden, "qkv"}, {s.hidden, s.hidden, "operator_out"},
              {s.kv_width(), s.hidden, "kv"}, {s.mlp, s.hidden, "mlp_in"},
              {s.hidden, s.mlp, "mlp_out"}, {3 * s.hidden, s.hidden, "conv_in"},
              {s.vocabulary, s.hidden, "lm_head"}};
  } else if (argc != 1) return 2;
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, 32u * 1024 * 1024), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024), "workspace");

  const float alpha = 1.0f, beta = 0.0f;
  int failures = 0;

  for (const Shape& shape : shapes) {
    const int rows = shape.output_size;
    const int cols = shape.input_size;
    const int maximum = 32;

    std::vector<__nv_bfloat16> host_weight(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < host_weight.size(); ++i) {
      host_weight[i] = __float2bfloat16(0.02f * sinf(static_cast<float>(i % 977) * 0.031f));
    }
    __nv_bfloat16* weight = nullptr;
    check_cuda(cudaMalloc(&weight, host_weight.size() * sizeof(__nv_bfloat16)), "weight");
    check_cuda(cudaMemcpy(weight, host_weight.data(), host_weight.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "copy weight");

    std::vector<__nv_bfloat16> host_input(cols);
    for (int i = 0; i < cols; ++i) {
      host_input[i] = __float2bfloat16(0.4f * sinf(i * 0.11f) + 0.3f);
    }
    __nv_bfloat16* input = nullptr;
    check_cuda(cudaMalloc(&input, static_cast<size_t>(maximum) * cols * sizeof(__nv_bfloat16)), "input");
    for (int b = 0; b < maximum; ++b) {
      check_cuda(cudaMemcpy(input + static_cast<size_t>(b) * cols, host_input.data(),
                            cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy input");
    }
    __nv_bfloat16* output = nullptr;
    check_cuda(cudaMalloc(&output, static_cast<size_t>(maximum) * rows * sizeof(__nv_bfloat16)), "output");

    auto gemm = [&](int n) {
      check_cuda(cudaMemset(output, 0xff, static_cast<size_t>(n) * rows * sizeof(__nv_bfloat16)), "poison output");
      check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols,
                                &alpha, weight, CUDA_R_16BF, cols,
                                input, CUDA_R_16BF, cols, &beta,
                                output, CUDA_R_16BF, rows,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemmEx");
      check_cuda(cudaDeviceSynchronize(), "sync");
    };

    // The reference column: the single-sequence projection.
    gemm(1);
    std::vector<__nv_bfloat16> reference(rows);
    check_cuda(cudaMemcpy(reference.data(), output, rows * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost), "copy reference");

    std::printf("%s (%d x %d):\n", shape.name, rows, cols);
    for (const int n : kColumnCounts) {
      gemm(n);
      std::vector<__nv_bfloat16> many(static_cast<size_t>(n) * rows);
      check_cuda(cudaMemcpy(many.data(), output, many.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost), "copy many");
      int mismatching_columns = 0;
      int worst_mismatches = 0;
      for (int b = 0; b < n; ++b) {
        const __nv_bfloat16* column = many.data() + static_cast<size_t>(b) * rows;
        int mismatches = 0;
        for (int i = 0; i < rows; ++i) {
          if (!std::isfinite(__bfloat162float(column[i])) || !std::isfinite(__bfloat162float(reference[i]))) {
            std::fprintf(stderr, "nonfinite projection output\n");
            return 1;
          }
          if (std::memcmp(column + i, reference.data() + i, sizeof(__nv_bfloat16)) != 0) ++mismatches;
        }
        if (mismatches > 0) {
          ++mismatching_columns;
          worst_mismatches = mismatches > worst_mismatches ? mismatches : worst_mismatches;
        }
      }
      std::printf("  n=%-3d columns bit-identical to n=1: %d/%d%s\n",
                  n, n - mismatching_columns, n,
                  mismatching_columns > 0 ? "  <-- MISMATCH" : "");
      if (mismatching_columns > 0) {
        std::printf("           worst column differs in %d of %d outputs\n",
                    worst_mismatches, rows);
        ++failures;
      }
    }

    check_cuda(cudaFree(weight), "free weight");
    check_cuda(cudaFree(input), "free input");
    check_cuda(cudaFree(output), "free output");
  }

  check_cublas(cublasDestroy(handle), "cublasDestroy");
  if (failures > 0) {
    std::printf("\nRESULT: %d column count(s) changed a row's bits\n", failures);
    return 1;
  }
  std::printf("\nRESULT: every column is bit-identical to the n=1 column\n");
  return 0;
}
