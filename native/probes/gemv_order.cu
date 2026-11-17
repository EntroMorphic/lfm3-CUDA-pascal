// Determine which BF16 GEMV accumulation order reproduces cuBLAS bit-exactly
// on the real LFM2.5-350M projection weights.
//
// The decode path is bandwidth bound, and cuBLAS's M=1 tensor-core GEMM
// reaches only 116-172 GB/s of weight traffic on Thor against a measured
// 242-257 GB/s streaming ceiling. A warp-per-row streaming GEMV reaches the
// ceiling but sums in a different order, which changes the last ulp of the
// FP32 accumulator and therefore the BF16 output.
//
// This tool compares candidate accumulation orders against cuBLAS on the
// model's own weights and reports the exact mismatch count, so a bit-exact
// replacement can be identified rather than guessed.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <lfm/safetensors.hpp>

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

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t count = 0;
  explicit DeviceBuffer(size_t elements) : count(elements) {
    check_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
};

// --- candidate kernels -----------------------------------------------------

// A: uint4 (8 BF16) loads, pair products summed before the accumulator.
__global__ void gemv_a(const __nv_bfloat16* w, const __nv_bfloat16* x,
                        __nv_bfloat16* y, int rows, int cols) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
  const int stride = gridDim.x * warps;
  const int vectors = cols >> 3;
  for (int row = gw; row < rows; row += stride) {
    const __nv_bfloat16* rw = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
      const uint4 raw = *reinterpret_cast<const uint4*>(rw + v * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
      const int base = v * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sx[base + 2 * j]));
        const float2 b = __bfloat1622float2(p[j]);
        sum += a.x * b.x + a.y * b.y;
      }
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
  }
}

// B: uint4 loads, products accumulated one at a time.
__global__ void gemv_b(const __nv_bfloat16* w, const __nv_bfloat16* x,
                        __nv_bfloat16* y, int rows, int cols) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
  const int stride = gridDim.x * warps;
  const int vectors = cols >> 3;
  for (int row = gw; row < rows; row += stride) {
    const __nv_bfloat16* rw = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
      const uint4 raw = *reinterpret_cast<const uint4*>(rw + v * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
      const int base = v * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sx[base + 2 * j]));
        const float2 b = __bfloat1622float2(p[j]);
        sum += a.x * b.x;
        sum += a.y * b.y;
      }
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
  }
}

// C: scalar loads, lane stride 32, products accumulated one at a time.
//    This is the order the accepted fused LM-head kernel already uses.
__global__ void gemv_c(const __nv_bfloat16* w, const __nv_bfloat16* x,
                        __nv_bfloat16* y, int rows, int cols) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
  const int stride = gridDim.x * warps;
  for (int row = gw; row < rows; row += stride) {
    const __nv_bfloat16* rw = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 32) {
      sum += __bfloat162float(rw[c]) * __bfloat162float(sx[c]);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
  }
}

// D: scalar loads, lane stride 32, pair products summed before the accumulator.
__global__ void gemv_d(const __nv_bfloat16* w, const __nv_bfloat16* x,
                        __nv_bfloat16* y, int rows, int cols) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
  const int stride = gridDim.x * warps;
  for (int row = gw; row < rows; row += stride) {
    const __nv_bfloat16* rw = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 64) {
      const float a0 = __bfloat162float(rw[c]) * __bfloat162float(sx[c]);
      const float a1 = __bfloat162float(rw[c + 32]) * __bfloat162float(sx[c + 32]);
      sum += a0 + a1;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
  }
}

// E: uint4 loads with the lane holding 8 contiguous elements, accumulated in
//    reverse within the vector (changes the intra-thread order).
__global__ void gemv_e(const __nv_bfloat16* w, const __nv_bfloat16* x,
                        __nv_bfloat16* y, int rows, int cols) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
  const int stride = gridDim.x * warps;
  const int vectors = cols >> 3;
  for (int row = gw; row < rows; row += stride) {
    const __nv_bfloat16* rw = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
      const uint4 raw = *reinterpret_cast<const uint4*>(rw + v * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
      const int base = v * 8;
#pragma unroll
      for (int j = 3; j >= 0; --j) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sx[base + 2 * j]));
        const float2 b = __bfloat1622float2(p[j]);
        sum += a.x * b.x + a.y * b.y;
      }
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(sum);
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

struct Mismatch { int different = 0; double max_abs = 0.0; };

Mismatch compare(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
  Mismatch result;
  for (size_t i = 0; i < a.size(); ++i) {
    const float left = __bfloat162float(a[i]);
    const float right = __bfloat162float(b[i]);
    if (left != right) {
      ++result.different;
      const double d = std::abs(static_cast<double>(left) - right);
      if (d > result.max_abs) result.max_abs = d;
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: gemv_order <model.safetensors>\n");
    return 2;
  }
  lfm::Safetensors model(argv[1]);
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetStream(handle, cudaStreamPerThread), "cublasSetStream");
  void* workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, 32u * 1024 * 1024), "workspace");
  check_cublas(cublasSetWorkspace(handle, workspace, 32u * 1024 * 1024), "set workspace");

  struct Shape { const char* tensor; int rows; int cols; };
  const Shape shapes[] = {
      {"model.layers.0.feed_forward.w1.weight", 4608, 1024},
      {"model.layers.0.feed_forward.w2.weight", 1024, 4608},
      {"model.layers.0.conv.in_proj.weight", 3072, 1024},
      {"model.layers.2.self_attn.q_proj.weight", 1024, 1024},
  };

  const float alpha = 1.0f, beta = 0.0f;
  std::printf("%-42s %-4s %8s %8s %10s %10s\n", "tensor", "cand", "differs", "of", "GB/s", "max|d|");
  for (const Shape& shape : shapes) {
    const lfm::TensorView& view = model.at(shape.tensor);
    const int rows = shape.rows, cols = shape.cols;
    DeviceBuffer<__nv_bfloat16> weight(view.elements());
    check_cuda(cudaMemcpy(weight.data, view.device_data,
                          view.elements() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice), "copy weight");
    DeviceBuffer<__nv_bfloat16> x(cols);
    std::vector<__nv_bfloat16> host_x(cols);
    for (int i = 0; i < cols; ++i) {
      host_x[i] = __float2bfloat16(0.05f * sinf(static_cast<float>(i) * 0.37f) + 0.25f);
    }
    check_cuda(cudaMemcpy(x.data, host_x.data(), cols * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "copy x");
    DeviceBuffer<__nv_bfloat16> reference(rows), candidate(rows);

    auto cublas_gemv = [&] {
      check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols,
                                &alpha, weight.data, CUDA_R_16BF, cols,
                                x.data, CUDA_R_16BF, cols, &beta,
                                reference.data, CUDA_R_16BF, rows,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemmEx");
    };
    cublas_gemv();
    check_cuda(cudaDeviceSynchronize(), "sync");
    std::vector<__nv_bfloat16> host_reference(rows);
    check_cuda(cudaMemcpy(host_reference.data(), reference.data,
                          rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy ref");

    const double bytes = static_cast<double>(view.elements()) * 2.0;
    const double cublas_ms = time_ms(cublas_gemv, 5, 40);

    struct Candidate { const char* name; void (*kernel)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int); bool vectorized; };
    const Candidate candidates[] = {
        {"A", gemv_a, true}, {"B", gemv_b, true},
        {"C", gemv_c, false}, {"D", gemv_d, false}, {"E", gemv_e, true},
    };
    for (const Candidate& entry : candidates) {
      const int threads = 256;
      const int blocks = (rows + threads / 32 - 1) / (threads / 32);
      auto launch = [&] {
        entry.kernel<<<blocks, threads, cols * sizeof(__nv_bfloat16)>>>(
            weight.data, x.data, candidate.data, rows, cols);
      };
      launch();
      check_cuda(cudaDeviceSynchronize(), "sync");
      std::vector<__nv_bfloat16> host_candidate(rows);
      check_cuda(cudaMemcpy(host_candidate.data(), candidate.data,
                            rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy cand");
      const Mismatch mismatch = compare(host_reference, host_candidate);
      const double ms = time_ms(launch, 5, 40);
      std::printf("%-42s %-4s %8d %8d %10.1f %10.3e\n", shape.tensor, entry.name,
                  mismatch.different, rows, bytes / (ms * 1e-3) / 1e9, mismatch.max_abs);
      std::fflush(stdout);
    }
    std::printf("%-42s %-4s %8s %8s %10.1f\n", shape.tensor, "cublas", "-", "-",
                bytes / (cublas_ms * 1e-3) / 1e9);
  }
  return 0;
}
