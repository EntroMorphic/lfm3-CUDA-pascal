// Private test ABI. Python callers must use tools/lfm3_bridge.py, which validates
// tensor shape, strides, dtype, allocation extents and aliasing. Raw pointers
// cannot communicate those contracts. This ABI is not a production interface.
#include <lfm/lfm3_ops.cuh>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <vector>

using bf16 = __nv_bfloat16;
static dim3 grid(size_t n) { return dim3(static_cast<unsigned>((n + 255) / 256)); }

static bool elements(std::initializer_list<int> factors) {
  size_t count = 1;
  for (int n : factors) {
    if (n <= 0 || count > static_cast<size_t>(INT_MAX) / n) return false;
    count *= n;
  }
  return true;
}

static int pointers(std::initializer_list<const void*> values) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  for (const void* p : values) {
    if (!p) return cudaErrorInvalidValue;
    cudaPointerAttributes attributes{};
    status = cudaPointerGetAttributes(&attributes, p);
    if (status != cudaSuccess) {
      cudaGetLastError(); // Consume only this failed validation's CUDA error.
      return status;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

// Synchronous correctness-only validation, on the caller's stream. The future
// runtime needs its own device-side error contract rather than these host copies.
static int positions_in_range(const int* positions, int rows, int capacity, cudaStream_t stream) {
  std::vector<int> host(rows);
  auto status = cudaMemcpyAsync(host.data(), positions, rows * sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) return status;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return status;
  for (int p : host) if (p < 0 || p >= capacity) return cudaErrorInvalidValue;
  return cudaSuccess;
}

extern "C" int lfm3_rotary(bf16* x, const int* positions, int rows, int heads,
                            int dim, int pairs, float theta, cudaStream_t stream) {
  if (rows > 65535 || heads > 65535 || dim > 1024 || dim % 2 || pairs < 0 ||
      pairs > dim / 2 || !std::isfinite(theta) || theta <= 0 || !elements({rows, heads, dim})) return cudaErrorInvalidValue;
  int status = pointers({x, positions});
  if (status) return status;
  status = positions_in_range(positions, rows, INT_MAX, stream);
  if (status) return status;
  lfm::lfm3::rotary<<<grid(static_cast<size_t>(rows) * heads * dim / 2), 256, 0, stream>>>(
      x, positions, rows, heads, dim, pairs, theta);
  return cudaGetLastError();
}

extern "C" int lfm3_gate(const bf16* input, const bf16* logits, bf16* output,
                          int rows, int heads, int dim, cudaStream_t stream) {
  if (rows > 65535 || heads > 65535 || dim > 1024 || !elements({rows, heads, dim})) return cudaErrorInvalidValue;
  const int status = pointers({input, logits, output});
  if (status) return status;
  const size_t n = static_cast<size_t>(rows) * heads * dim;
  lfm::lfm3::head_gate<<<grid(n), 256, 0, stream>>>(input, logits, output, n, dim);
  return cudaGetLastError();
}

extern "C" int lfm3_sandwich(const bf16* input, const bf16* weight, const bf16* residual,
                              bf16* output, int rows, int width, float epsilon, cudaStream_t stream) {
  if (rows > 65535 || width > 1048576 || !elements({rows, width}) ||
      !std::isfinite(epsilon) || epsilon <= 0) return cudaErrorInvalidValue;
  const int status = pointers({input, weight, residual, output});
  if (status) return status;
  lfm::lfm3::sandwich_residual<<<rows, 256, 0, stream>>>(input, weight, residual, output, width, epsilon);
  return cudaGetLastError();
}

extern "C" int lfm3_conv7(const bf16* projected, const bf16* weights, bf16* state,
                           bf16* output, int batch, int width, cudaStream_t stream) {
  if (batch > 65535 || width > 1048576 || !elements({batch, width, 7}) ||
      !elements({batch, width, 3})) return cudaErrorInvalidValue;
  const int status = pointers({projected, weights, state, output});
  if (status) return status;
  lfm::lfm3::convolution7<<<grid(static_cast<size_t>(batch) * width), 256, 0, stream>>>(
      projected, weights, state, output, width, batch);
  return cudaGetLastError();
}

extern "C" int lfm3_attention(const bf16* q, const bf16* k, const bf16* v,
                               const int* positions, bf16* output, int batch, int heads,
                               int kv_heads, int dim, int capacity, int window, cudaStream_t stream) {
  if (batch > 65535 || heads > 65535 || kv_heads <= 0 || heads % kv_heads || dim > 256 ||
      capacity > 4096 || window < 0 || !elements({batch, heads, dim}) ||
      !elements({batch, kv_heads, capacity, dim})) return cudaErrorInvalidValue;
  int status = pointers({q, k, v, positions, output});
  if (status) return status;
  status = positions_in_range(positions, batch, capacity, stream);
  if (status) return status;
  lfm::lfm3::attention_decode<<<dim3(heads, batch), 256, capacity * sizeof(float), stream>>>(
      q, k, v, positions, output, heads, kv_heads, dim, capacity, window);
  return cudaGetLastError();
}

extern "C" int lfm3_conv7_prefill(const bf16* projected, const bf16* weights, bf16* state,
                                   bf16* output, int batch, int length, int width, cudaStream_t stream) {
  if (batch > 65535 || width > 1048576 || length > 32768 ||
      !elements({batch, length, width, 3}) || !elements({batch, width, 7})) return cudaErrorInvalidValue;
  const int status = pointers({projected, weights, state, output});
  if (status) return status;
  lfm::lfm3::convolution7_prefill<<<grid(static_cast<size_t>(batch) * length * width), 256, 0, stream>>>(
      projected, weights, state, output, batch, length, width);
  return cudaGetLastError();
}
