// Does a trickle of work keep the GTX 1080's clocks up? (validation/decode-atoms-v1; measurement only.)
// Launches a small kernel every PERIOD_MS for SECONDS; the caller samples nvidia-smi alongside.
//   pascal_keepwarm PERIOD_MS BLOCKS ITERATIONS SECONDS
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

__global__ void spin(float* out, int iterations) {
  float x = threadIdx.x * 1e-3f + blockIdx.x;
  for (int i = 0; i < iterations; ++i) x = x * 1.0000001f + 1e-7f;
  if (threadIdx.x == 0) out[blockIdx.x] = x;
}

int main(int argc, char** argv) {
  if (argc != 5) { std::fprintf(stderr, "usage: pascal_keepwarm PERIOD_MS BLOCKS ITERATIONS SECONDS\n"); return 2; }
  const double period = std::atof(argv[1]) * 1e-3, seconds = std::atof(argv[4]);
  const int blocks = std::atoi(argv[2]), iterations = std::atoi(argv[3]);
  float* out;
  if (cudaMalloc(&out, blocks * sizeof(float)) != cudaSuccess) return 1;
  const auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  double busy = 0; long launches = 0;
  for (double next = 0; now() < seconds; next += period) {
    const double a = now();
    spin<<<blocks, 256>>>(out, iterations);
    cudaDeviceSynchronize();
    busy += now() - a; ++launches;
    const double wait = next + period - now();
    if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
  }
  std::printf("period %.0f ms, %d blocks x %d iterations: %ld launches, %.1f us each, duty %.2f%%\n", period * 1e3, blocks,
              iterations, launches, busy / launches * 1e6, 100 * busy / seconds);
  return 0;
}
