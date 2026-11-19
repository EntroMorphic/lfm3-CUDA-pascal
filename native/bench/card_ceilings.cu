// The GTX 1080's own ceilings, measured, for setting phase 2's speed bar: the rate at which a kernel can read device
// memory, and the rate of FP32 fused multiply-adds. Speed only; never acceptance.
//   card_ceilings [SECONDS_PER_TEST=3]
// Each test runs repeatedly for the given time after a 2 s warm-up (the clocks under load, as the engine sees them);
// the best pass is printed. Read: every byte of a 2 GiB buffer read once per pass with 16-byte loads, summed so the
// loads cannot be removed. FMA: 8 independent chains per thread, 2 FLOP per multiply-add.
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); } }

__global__ void read_all(const float4* p, size_t n, float* sink) {
  float s = 0.f;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {  // four loads in flight per thread
    const float4 v0 = __ldcs(p + i), v1 = __ldcs(p + i + stride), v2 = __ldcs(p + i + 2 * stride), v3 = __ldcs(p + i + 3 * stride);
    s += v0.x + v0.y + v0.z + v0.w + v1.x + v1.y + v1.z + v1.w + v2.x + v2.y + v2.z + v2.w + v3.x + v3.y + v3.z + v3.w;
  }
  for (; i < n; i += stride) { const float4 v = __ldcs(p + i); s += v.x + v.y + v.z + v.w; }
  if (s == 12345.678f) *sink = s;  // never true for the zero-filled buffer; keeps the loads
}

constexpr int kChains = 8, kSteps = 4096;
__global__ void fma_peak(float* out, float a, float b) {
  float x[kChains];
  for (int c = 0; c < kChains; ++c) x[c] = threadIdx.x * 1e-7f + c;
  for (int i = 0; i < kSteps; ++i)
#pragma unroll
    for (int c = 0; c < kChains; ++c) x[c] = fmaf(x[c], a, b);
  float s = 0.f;
  for (int c = 0; c < kChains; ++c) s += x[c];
  if (s == 12345.678f) out[threadIdx.x] = s;
}

template <class F>
double best_seconds(F launch, double budget) {
  using clock = std::chrono::steady_clock;
  const auto until = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
  auto start = clock::now();
  while (until(start) < 2.0) { launch(); check(cudaDeviceSynchronize()); }  // warm-up: clocks up
  double best = 1e30; start = clock::now();
  cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
  while (until(start) < budget) {
    check(cudaEventRecord(a)); launch(); check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
    float ms; check(cudaEventElapsedTime(&ms, a, b)); if (ms * 1e-3 < best) best = ms * 1e-3;
  }
  return best;
}
}  // namespace

int main(int argc, char** argv) {
  const double budget = argc > 1 ? std::atof(argv[1]) : 3.0;
  cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s, %d SMs\n", prop.name, prop.multiProcessorCount);
  const size_t bytes = size_t(2) << 30, n = bytes / sizeof(float4);
  float4* buffer; float* sink;
  check(cudaMalloc(&buffer, bytes)); check(cudaMemset(buffer, 0, bytes)); check(cudaMalloc(&sink, 4096));
  for (const int per_sm : {8, 16, 32, 64}) {  // the best of several launch sizes is the ceiling
    const double read = best_seconds([&] { read_all<<<prop.multiProcessorCount * per_sm, 256>>>(buffer, n, sink); }, budget);
    std::printf("read, %d blocks of 256 per SM: %.1f GB/s (2 GiB in %.3f ms)\n", per_sm, bytes / read * 1e-9, read * 1e3);
  }
  check(cudaFree(buffer));
  const int blocks = prop.multiProcessorCount * 16, threads = 256;
  const double fma = best_seconds([&] { fma_peak<<<blocks, threads>>>(sink, 0.999999f, 1e-6f); }, budget);
  const double flop = 2.0 * kChains * kSteps * double(blocks) * threads;
  std::printf("fma: %.0f GFLOP/s (%.3f ms)\n", flop / fma * 1e-9, fma * 1e3);
  return 0;
}
