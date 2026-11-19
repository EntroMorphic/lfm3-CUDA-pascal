// How fast a kernel can read the card's memory in the shape decode reads weights (speed only): one launch per
// matrix-sized chunk, chunks walking through a 3 GiB buffer as decode walks through the weights. Loads per thread in
// flight, cache hint and chunk size varied; the best is the ceiling for decode's projections.
//   read_rate
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); } }
template <int U, bool CS>
__global__ void __launch_bounds__(256) read_chunk(const int4* p, size_t n, int* sink) {
  int s = 0;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += U * stride) {
    int4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = i + u * stride < n ? (CS ? __ldcs(p + i + u * stride) : __ldg(p + i + u * stride)) : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; ++u) s ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (s == 0x12345678) *sink = s;
}
template <int U, bool CS>
void run(const int4* buffer, size_t total, size_t chunk, int blocks, int* sink) {
  const size_t chunks = total / chunk, n = chunk / 16;
  auto pass = [&] { for (size_t c = 0; c < chunks; ++c) read_chunk<U, CS><<<blocks, 256>>>(buffer + c * n, n, sink); };
  for (int i = 0; i < 3; ++i) pass();
  check(cudaDeviceSynchronize());
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    const auto t0 = std::chrono::steady_clock::now(); pass(); check(cudaDeviceSynchronize());
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("chunk %6.2f MB, %3d blocks, %d loads in flight, %s: %.1f GB/s\n", chunk / 1e6, blocks, U, CS ? "ldcs" : "ldg ", chunks * chunk / best * 1e-9);
}
}  // namespace
int main() {
  const size_t total = size_t(3) << 30;
  int4* buffer; int* sink;
  check(cudaMalloc(&buffer, total)); check(cudaMemset(buffer, 1, total)); check(cudaMalloc(&sink, 4));
  for (size_t chunk : {size_t(31457280), size_t(8388608), size_t(524288000)})
    for (int blocks : {160, 320, 640, 1280}) {
      run<4, true>(buffer, total, chunk, blocks, sink);
      run<8, true>(buffer, total, chunk, blocks, sink);
      run<8, false>(buffer, total, chunk, blocks, sink);
    }
}
