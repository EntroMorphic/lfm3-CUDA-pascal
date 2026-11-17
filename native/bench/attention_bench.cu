// Measure the decode attention kernel, and compare candidate designs.
//
// Decode attention is the one part of a decode step that does not amortize
// across a batch: every sequence reads its own KV cache, so the traffic grows
// with the context no matter how wide the batch is. The kernel also has a
// shape problem: it launches one block per (query head, sequence), so batch 1
// runs 16 blocks on a 20-SM device.
//
// Measures the original and production kernels and candidates on initialized
// data. Effective logical bandwidth is not measured DRAM traffic and does
// not by itself identify the bottleneck.
//
// Logical traffic is reported two ways: per-query-head bytes and unique
// KV bytes. Neither accounts for cache hits. The model has 16 query heads over 8
// KV heads, so two query heads share each KV head and the shipped kernel
// reads the cache twice.
//
// The candidates marked exact produce the same bytes as the shipped kernel;
// the others are reported with how far their output drifts.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <lfm/attention.cuh>
#include <lfm/attention_candidates.cuh>

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using lfm::kContextLength;
using lfm::kHeadDimension;
using lfm::kHiddenSize;
using lfm::kKvHeads;
using lfm::kKvLayerStride;
using lfm::kQkvWidth;
using lfm::kQueryHeads;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

enum Variant {
  kShipped,
  kExactSoftmax,
  kExactStaged,
  kSharedKvHead,
  kSplitKv,
  kPublishedSoftmax,
  kPipelined,
  kProduction,
};

const char* kVariantNames[] = {
    "shipped", "exact_softmax", "exact_staged_value", "shared_kv_head", "split_kv",
    "published_softmax", "pipelined", "production"};

// Initialize EVERY head and row independently. Poison the unused suffix so
// out-of-prefix contributions are more likely to fail the finite check.
// This is not a bounds detector for speculative reads whose values are unused.
__global__ void initialize_cache(__nv_bfloat16* keys, __nv_bfloat16* values,
                                 const int* positions, size_t count, unsigned seed,
                                 float key_scale) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int batch = i / kKvLayerStride;
    const int token = (i / kHeadDimension) % kContextLength;
    unsigned x = static_cast<unsigned>(i) ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    const float key = (static_cast<int>(x & 65535) - 32768) / 32768.0f;
    const float value = (static_cast<int>(x >> 16) - 32768) / 8192.0f;
    keys[i] = __float2bfloat16(token <= positions[batch] ? key * key_scale : nanf(""));
    values[i] = __float2bfloat16(token <= positions[batch] ? value : nanf(""));
  }
}

double moved_bytes(Variant variant, int batch, int position, int splits) {
  double heads = kQueryHeads;
  if (variant == kSharedKvHead) heads = kKvHeads;
  return heads * batch * 2.0 * (position + 1) * kHeadDimension * sizeof(__nv_bfloat16);
}

double useful_bytes(int batch, int position) {
  return static_cast<double>(kKvHeads) * batch * 2.0 *
         (position + 1) * kHeadDimension * sizeof(__nv_bfloat16);
}

struct Fixture {
  int batch;
  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* key_cache = nullptr;
  __nv_bfloat16* value_cache = nullptr;
  __nv_bfloat16* output = nullptr;
  __nv_bfloat16* reference = nullptr;
  lfm::SplitPartial* partials = nullptr;
  int* positions = nullptr;
  int* policy = nullptr;

  Fixture(int sequences, int position, unsigned seed = 1, float key_scale = 2.0f,
          bool mixed_positions = false) : batch(sequences) {
    const size_t elements = static_cast<size_t>(batch) * kKvLayerStride;
    check_cuda(cudaMalloc(&query, static_cast<size_t>(batch) * kQkvWidth * sizeof(__nv_bfloat16)), "query");
    check_cuda(cudaMalloc(&key_cache, elements * sizeof(__nv_bfloat16)), "key cache");
    check_cuda(cudaMalloc(&value_cache, elements * sizeof(__nv_bfloat16)), "value cache");
    check_cuda(cudaMalloc(&output, static_cast<size_t>(batch) * kHiddenSize * sizeof(__nv_bfloat16)), "output");
    check_cuda(cudaMalloc(&reference, static_cast<size_t>(batch) * kHiddenSize * sizeof(__nv_bfloat16)), "reference");
    check_cuda(cudaMalloc(&partials, static_cast<size_t>(batch) * kQueryHeads * 64 * sizeof(lfm::SplitPartial)), "partials");
    check_cuda(cudaMalloc(&positions, batch * sizeof(int)), "positions");
    check_cuda(cudaMalloc(&policy, 3 * sizeof(int)), "policy");
    std::vector<int> host_positions(batch, position);
    const int edges[] = {0, 1, 30, 31, 32, 33, 62, 63, 64, 65};
    if (mixed_positions) {
      for (int row = 1; row < batch; ++row) {
        host_positions[row] = std::min(position, edges[(row - 1) % 10]);
      }
    }
    check_cuda(cudaMemcpy(positions, host_positions.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice), "copy positions");
    const int host_policy[3] = {0, 0, 0};
    check_cuda(cudaMemcpy(policy, host_policy, sizeof(host_policy),
                          cudaMemcpyHostToDevice), "copy policy");
    initialize_cache<<<1024, 256>>>(key_cache, value_cache, positions, elements, seed, key_scale);
    check_cuda(cudaGetLastError(), "initialize cache");
    std::vector<__nv_bfloat16> host_query(static_cast<size_t>(batch) * kQkvWidth);
    for (size_t index = 0; index < host_query.size(); ++index) {
      host_query[index] = __float2bfloat16(2.0f * sinf(static_cast<float>((index + seed) % 977) * 0.011f));
    }
    check_cuda(cudaMemcpy(query, host_query.data(), host_query.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice), "copy query");
  }

  ~Fixture() {
    cudaFree(query);
    cudaFree(key_cache);
    cudaFree(value_cache);
    cudaFree(output);
    cudaFree(reference);
    cudaFree(partials);
    cudaFree(positions);
    cudaFree(policy);
  }
};

void launch(Variant variant, const Fixture& fixture, int splits) {
  const int stride = static_cast<int>(kKvLayerStride);
  switch (variant) {
    case kShipped:
      lfm::online_softmax_gqa_attention<<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kExactSoftmax:
      lfm::attention_exact_softmax_unstaged<1024><<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kExactStaged:
      lfm::attention_exact_softmax<1024><<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kSharedKvHead:
      lfm::attention_shared_kv_head<1024><<<dim3(kKvHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kPublishedSoftmax:
      lfm::attention_published_softmax<1024><<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kPipelined:
      lfm::attention_pipelined<1024><<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kProduction:
      lfm::online_softmax_gqa_attention_staged<<<dim3(kQueryHeads, fixture.batch), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.output,
          fixture.positions, fixture.policy, stride);
      break;
    case kSplitKv:
      lfm::attention_split_kv<<<dim3(kQueryHeads, fixture.batch, splits), 1024>>>(
          fixture.query, fixture.key_cache, fixture.value_cache, fixture.partials,
          fixture.positions, stride, splits);
      lfm::combine_split_kv<<<dim3(kQueryHeads, fixture.batch), 64>>>(
          fixture.partials, fixture.output, fixture.positions, splits, stride);
      break;
  }
  check_cuda(cudaGetLastError(), "launch attention");
}

// The largest difference between a candidate's output and the shipped
// kernel's, in units of the output's own scale.
double compare(const Fixture& fixture, bool exact, const char* context = "benchmark") {
  const size_t count = static_cast<size_t>(fixture.batch) * kHiddenSize;
  std::vector<__nv_bfloat16> produced(count);
  std::vector<__nv_bfloat16> expected(count);
  check_cuda(cudaMemcpy(produced.data(), fixture.output, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy produced");
  check_cuda(cudaMemcpy(expected.data(), fixture.reference, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy reference");
  double worst = 0.0;
  double scale = 0.0;
  size_t mismatches = 0;
  for (size_t index = 0; index < count; ++index) {
    if (!std::isfinite(__bfloat162float(produced[index])) ||
        !std::isfinite(__bfloat162float(expected[index]))) {
      std::fprintf(stderr, "%s: nonfinite output at element %zu\n", context, index);
      std::exit(1);
    }
    if (std::memcmp(&produced[index], &expected[index], sizeof(__nv_bfloat16)) != 0) {
      ++mismatches;
    }
    worst = std::max(worst, std::abs(static_cast<double>(__bfloat162float(produced[index])) -
                                     static_cast<double>(__bfloat162float(expected[index]))));
    scale = std::max(scale, std::abs(static_cast<double>(__bfloat162float(expected[index]))));
  }
  if (exact && mismatches) {
    std::fprintf(stderr, "%s: exactness failure: %zu / %zu BF16 elements differ\n", context, mismatches, count);
    std::exit(1);
  }
  return scale > 0.0 ? worst / scale : worst;
}

float time_variant(const Fixture& fixture, Variant variant, int splits, int iterations) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start");
  check_cuda(cudaEventCreate(&stop), "create stop");
  launch(variant, fixture, splits);
  check_cuda(cudaDeviceSynchronize(), "sync");
  check_cuda(cudaEventRecord(start), "record start");
  for (int index = 0; index < iterations; ++index) {
    launch(variant, fixture, splits);
  }
  check_cuda(cudaEventRecord(stop), "record stop");
  check_cuda(cudaDeviceSynchronize(), "sync");
  float elapsed = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
  check_cuda(cudaEventDestroy(start), "destroy start");
  check_cuda(cudaEventDestroy(stop), "destroy stop");
  return elapsed / iterations;
}

}  // namespace

int main(int argc, char** argv) {
  // Negative controls let the GPU test gate prove the checker fails closed.
  if (argc == 2 && (std::strcmp(argv[1], "--negative-nan") == 0 ||
                    std::strcmp(argv[1], "--negative-bit") == 0)) {
    Fixture fixture(1, 0);
    launch(kShipped, fixture, 1);
    check_cuda(cudaDeviceSynchronize(), "reference sync");
    check_cuda(cudaMemcpy(fixture.reference, fixture.output,
                          kHiddenSize * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice), "reference");
    uint16_t bits;
    check_cuda(cudaMemcpy(&bits, fixture.output, sizeof(bits), cudaMemcpyDeviceToHost), "read bits");
    bits = std::strcmp(argv[1], "--negative-nan") == 0 ? 0x7fc0u : bits ^ 1u;
    check_cuda(cudaMemcpy(fixture.output, &bits, sizeof(bits), cudaMemcpyHostToDevice), "corrupt bits");
    compare(fixture, true, "negative control");
    return 0;  // Tests require the checker to exit nonzero before reaching here.
  }
  const bool small = argc == 2 && std::strcmp(argv[1], "--verify-small") == 0;
  if (small || (argc == 2 && std::strcmp(argv[1], "--verify") == 0)) {
    const int edges[] = {0, 1, 30, 31, 32, 33, 62, 63, 64, 65, 127, 128,
                         255, 8192, kContextLength - 1};
    const int policies[][3] = {{0, 0, 0}, {1, 0, 0}, {31, 1, 0}, {32, 4, 0},
                               {33, 31, 0}, {65, 33, 64}, {64, 4, 128}, {32, 128, 0}};
    int cases = 0;
    for (const int batch : {1, 3, 8, 32}) {
      if (small && batch > 3) continue;
      for (const int position : edges) {
        if (small && position > 128) continue;
        for (const float key_scale : {0.0f, 2.0f, 32.0f}) {
          const unsigned seed = 123u + cases;
          Fixture fixture(batch, position, seed, key_scale, true);
          for (const auto& policy : policies) {
            check_cuda(cudaMemcpy(fixture.policy, policy, sizeof(policy), cudaMemcpyHostToDevice), "policy");
            launch(kShipped, fixture, 1);
            check_cuda(cudaDeviceSynchronize(), "reference sync");
            const size_t bytes = static_cast<size_t>(batch) * kHiddenSize * sizeof(__nv_bfloat16);
            check_cuda(cudaMemcpy(fixture.reference, fixture.output, bytes, cudaMemcpyDeviceToDevice), "reference");
            // An omitted output store must fail, rather than inherit the reference.
            check_cuda(cudaMemset(fixture.output, 0xff, bytes), "poison output");
            launch(kProduction, fixture, 1);
            check_cuda(cudaDeviceSynchronize(), "production sync");
            // Include the complete reproducer in the log if comparison fails.
            char context[256];
            std::snprintf(context, sizeof(context), "batch=%d position=%d scale=%g policy=%d,%d,%d seed=%u",
                          batch, position, key_scale, policy[0], policy[1], policy[2], seed);
            compare(fixture, true, context);
            ++cases;
          }
        }
      }
      std::printf("verified batch %d\n", batch);
      std::fflush(stdout);
    }
    std::printf("PASS: %d production/reference cases, finite and byte-identical\n", cases);
    return 0;
  }
  if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--verify|--verify-small|--negative-nan|--negative-bit]\n", argv[0]);
    return 2;
  }
  const int positions[] = {128, 512, 2048, 8192};
  const int batches[] = {1, 8, 32};
  const int iterations = 20;
  const Variant variants[] = {kShipped, kExactSoftmax, kExactStaged, kSharedKvHead,
                              kSplitKv, kPublishedSoftmax, kPipelined, kProduction};
  const int splits = 8;

  for (const int batch : batches) {
    for (const int position : positions) {
      Fixture fixture(batch, position);
      // The reference is the shipped kernel's own output.
      launch(kShipped, fixture, 1);
      check_cuda(cudaDeviceSynchronize(), "sync");
      const size_t count = static_cast<size_t>(batch) * kHiddenSize;
      check_cuda(cudaMemcpy(fixture.reference, fixture.output, count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice), "copy reference");
      std::printf("\n=== batch %d, position %d ===\n", batch, position);
      std::printf("%-20s %10s %12s %12s %10s %10s\n", "variant", "us/call",
                  "GB/s moved", "GB/s useful", "speedup", "drift");
      for (const Variant variant : variants) {
        // Paired trials alternate order to reduce clock/thermal order bias.
        std::vector<float> times, ratios;
        for (int trial = 0; trial < 5; ++trial) {
          float base, candidate;
          if (trial % 2 == 0) {
            base = time_variant(fixture, kShipped, 1, iterations);
            candidate = time_variant(fixture, variant, splits, iterations);
          } else {
            candidate = time_variant(fixture, variant, splits, iterations);
            base = time_variant(fixture, kShipped, 1, iterations);
          }
          times.push_back(candidate);
          ratios.push_back(base / candidate);
        }
        std::sort(times.begin(), times.end());
        std::sort(ratios.begin(), ratios.end());
        const float ms = times[2];
        check_cuda(cudaMemset(fixture.output, 0xff, count * sizeof(__nv_bfloat16)), "poison output");
        launch(variant, fixture, splits);
        check_cuda(cudaDeviceSynchronize(), "sync");
        const double drift = compare(fixture, variant != kSplitKv);
        const double moved = moved_bytes(variant, batch, position, splits);
        std::printf("%-20s %10.1f %12.1f %12.1f %9.2fx %10.2e\n",
                    kVariantNames[variant], ms * 1000.0,
                    moved / (ms / 1000.0) / 1e9,
                    useful_bytes(batch, position) / (ms / 1000.0) / 1e9,
                    ratios[2], drift);
      }
    }
  }
  return 0;
}
