// How fast is the batched sampler's sort?
//
// The batched sampler sorts one 65536-entry vocabulary per sequence. The
// single-sequence sampler sorts the same 65536 entries with one global radix
// sort; the batched one uses a segmented sort so every row comes out
// descending on its own.
//
// This compares the two, so the cost of the segmented sort is a measured
// number rather than an assumption.
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

float time_ms(void (*run)(void*), void* context, int iterations) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");
  run(context);  // warm up
  check_cuda(cudaDeviceSynchronize(), "sync");
  check_cuda(cudaEventRecord(start), "record start");
  for (int i = 0; i < iterations; ++i) run(context);
  check_cuda(cudaEventRecord(stop), "record stop");
  check_cuda(cudaDeviceSynchronize(), "sync");
  float elapsed = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
  check_cuda(cudaEventDestroy(start), "destroy start");
  check_cuda(cudaEventDestroy(stop), "destroy stop");
  return elapsed / iterations;
}

struct Global {
  float* keys_in;
  float* keys_out;
  int* values_in;
  int* values_out;
  void* temporary;
  size_t bytes;
  int count;
  static void run(void* context) {
    Global* self = static_cast<Global*>(context);
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                   self->temporary, self->bytes,
                   self->keys_in, self->keys_out,
                   self->values_in, self->values_out, self->count),
               "global sort");
  }
};

struct Segmented {
  float* keys_in;
  float* keys_out;
  int* values_in;
  int* values_out;
  void* temporary;
  size_t bytes;
  int* starts;
  int* ends;
  int count;
  int rows;
  static void run(void* context) {
    Segmented* self = static_cast<Segmented*>(context);
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(
                   self->temporary, self->bytes,
                   self->keys_in, self->keys_out,
                   self->values_in, self->values_out,
                   self->count * self->rows, self->rows,
                   self->starts, self->ends,
                   0, sizeof(float) * 8, cudaStreamPerThread),
               "segmented sort");
  }
};

}  // namespace

int main() {
  const int count = 65536;
  const int rows = 32;
  const int iterations = 20;

  float* keys_in = nullptr;
  float* keys_out = nullptr;
  int* values_in = nullptr;
  int* values_out = nullptr;
  check_cuda(cudaMalloc(&keys_in, static_cast<size_t>(count) * rows * sizeof(float)), "keys in");
  check_cuda(cudaMalloc(&keys_out, static_cast<size_t>(count) * rows * sizeof(float)), "keys out");
  check_cuda(cudaMalloc(&values_in, static_cast<size_t>(count) * rows * sizeof(int)), "values in");
  check_cuda(cudaMalloc(&values_out, static_cast<size_t>(count) * rows * sizeof(int)), "values out");
  std::vector<float> host(static_cast<size_t>(count) * rows);
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = 0.02f * sinf(static_cast<float>(i % 977) * 0.031f);
  }
  check_cuda(cudaMemcpy(keys_in, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice), "copy keys");

  int* starts = nullptr;
  int* ends = nullptr;
  check_cuda(cudaMalloc(&starts, rows * sizeof(int)), "starts");
  check_cuda(cudaMalloc(&ends, rows * sizeof(int)), "ends");
  std::vector<int> host_starts(rows), host_ends(rows);
  for (int row = 0; row < rows; ++row) {
    host_starts[row] = row * count;
    host_ends[row] = (row + 1) * count;
  }
  check_cuda(cudaMemcpy(starts, host_starts.data(), rows * sizeof(int),
                        cudaMemcpyHostToDevice), "copy starts");
  check_cuda(cudaMemcpy(ends, host_ends.data(), rows * sizeof(int),
                        cudaMemcpyHostToDevice), "copy ends");

  Global global{};
  global.keys_in = keys_in;
  global.keys_out = keys_out;
  global.values_in = values_in;
  global.values_out = values_out;
  global.count = count;
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                 nullptr, global.bytes, keys_in, keys_out, values_in, values_out, count),
             "query global sort");
  check_cuda(cudaMalloc(&global.temporary, global.bytes), "global temp");

  Segmented segmented{};
  segmented.keys_in = keys_in;
  segmented.keys_out = keys_out;
  segmented.values_in = values_in;
  segmented.values_out = values_out;
  segmented.starts = starts;
  segmented.ends = ends;
  segmented.count = count;
  segmented.rows = rows;
  check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(
                 nullptr, segmented.bytes, keys_in, keys_out, values_in, values_out,
                 count * rows, rows, starts, ends, 0, sizeof(float) * 8, cudaStreamPerThread),
             "query segmented sort");
  check_cuda(cudaMalloc(&segmented.temporary, segmented.bytes), "segmented temp");

  const float global_ms = time_ms(&Global::run, &global, iterations);
  const float segmented_ms = time_ms(&Segmented::run, &segmented, iterations);
  std::printf("one 65536-entry radix sort          : %8.3f ms\n", global_ms);
  std::printf("%d segmented 65536-entry radix sorts: %8.3f ms  (%.3f ms per row)\n",
              rows, segmented_ms, segmented_ms / rows);
  std::printf("segmented cost against %d global sorts : %8.1fx\n",
              rows, segmented_ms / (global_ms * rows));
  return 0;
}
