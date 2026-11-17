// Phase 0 (PLAN.md): what the GTX 1080's 8 GB holds. Uploads the checkpoint as the engine does (lfm::Safetensors,
// one device allocation per file), checks the small tensors' device bytes against the files, and reports the
// device's free memory before and after. Then finds the largest single allocation the remaining memory grants.
//   pascal_mem_budget MODEL_DIR
#include <lfm/safetensors.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
double mib(size_t b) { return b / 1048576.0; }
size_t free_now() {
  size_t f = 0, t = 0;
  if (cudaMemGetInfo(&f, &t) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo");
  return f;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 2) { std::fprintf(stderr, "usage: pascal_mem_budget MODEL_DIR\n"); return 2; }
  const std::string dir = argv[1];
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) throw std::runtime_error("no CUDA device");
  size_t f = 0, total = 0;
  if (cudaMemGetInfo(&f, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo");
  std::printf("device              %s sm_%d%d\n", p.name, p.major, p.minor);
  std::printf("total               %10.1f MiB\n", mib(total));
  const size_t free0 = free_now();
  std::printf("free, context only  %10.1f MiB\n", mib(free0));
  {
    lfm::Safetensors model(dir + "/model.safetensors");
    model.verify_small_tensors();
    const size_t free1 = free_now();
    std::printf("model.safetensors   %10.1f MiB payload, %zu tensors, free drops %.1f MiB\n",
                mib(model.device_bytes()), model.tensor_count(), mib(free0 - free1));
    lfm::Safetensors codec(dir + "/audio_tokenizer/model.safetensors");
    codec.verify_small_tensors();
    const size_t free2 = free_now();
    std::printf("audio_tokenizer     %10.1f MiB payload, %zu tensors, free drops %.1f MiB\n",
                mib(codec.device_bytes()), codec.tensor_count(), mib(free1 - free2));
    std::printf("free, all weights   %10.1f MiB\n", mib(free2));
    // Largest single allocation granted now, to 1 MiB.
    size_t lo = 0, hi = free2;
    while (hi - lo > (1u << 20)) {
      const size_t mid = lo + (hi - lo) / 2;
      void* q = nullptr;
      if (cudaMalloc(&q, mid) == cudaSuccess) { cudaFree(q); lo = mid; } else { cudaGetLastError(); hi = mid; }
    }
    std::printf("largest allocation  %10.1f MiB\n", mib(lo));
  }
  std::printf("free, after release %10.1f MiB\n", mib(free_now()));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "pascal_mem_budget: %s\n", e.what());
  return 1;
}
