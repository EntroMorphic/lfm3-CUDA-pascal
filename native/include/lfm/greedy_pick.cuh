#pragma once
// The greedy choice on the card (phase 2, validation/speed-v1), for a greedy text request with no tracing: the id
// the host's greedy pick gives over the logits, the highest logit among ids below `live` that are not blocked, the
// lowest id on equal values (-0 equals +0); and whether any logit of the whole output layer is not finite (the host
// then refuses the step, as it does when it reads the logits itself). Two stages: kGreedyBlocks blocks each reduce
// a share of the ids into part[block] (value, id, flag); greedy_finish reduces those into out[0], the id (-1 if every
// id is blocked), and out[1], the flag. The choice is the maximum of a total order (value descending, then id
// ascending) over finite values, so the order of reduction cannot change it. native/tests/greedy_pick_check.cu.
#include <cuda_bf16.h>

namespace lfm::greedy {
constexpr int kGreedyBlocks = 128, kGreedyThreads = 256;
struct GreedyPart { float value; int id, bad; };
__device__ __forceinline__ float logit_value(float x) { return x; }
__device__ __forceinline__ float logit_value(__nv_bfloat16 x) { return __bfloat162float(x); }
// Takes (value, id) when it comes first in the order: an id at all, a higher value, or an equal value and a lower id.
__device__ __forceinline__ void greedy_take(float value, int id, float& best, int& at) {
  if (id >= 0 && (at < 0 || value > best || (value == best && id < at))) { best = value; at = id; }
}
__device__ inline void greedy_reduce(float best, int at, int bad, GreedyPart* result) {
  __shared__ float values[kGreedyThreads];
  __shared__ int ids[kGreedyThreads], flags[kGreedyThreads];
  values[threadIdx.x] = best; ids[threadIdx.x] = at; flags[threadIdx.x] = bad;
  for (int half = blockDim.x / 2; half > 0; half /= 2) {
    __syncthreads();
    if (threadIdx.x < half) {
      greedy_take(values[threadIdx.x + half], ids[threadIdx.x + half], best, at);
      values[threadIdx.x] = best; ids[threadIdx.x] = at;
      bad |= flags[threadIdx.x + half]; flags[threadIdx.x] = bad;
    }
  }
  if (threadIdx.x == 0) *result = {best, at, bad};
}
template <class T>
__global__ void __launch_bounds__(kGreedyThreads) greedy_pick(const T* logits, int vocabulary, const unsigned char* blocked,
                                                              int live, GreedyPart* part) {
  float best = 0.f; int at = -1, bad = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocabulary; i += gridDim.x * blockDim.x) {
    const float v = logit_value(logits[i]);
    bad |= !isfinite(v);
    if (i < live && !blocked[i]) greedy_take(v, i, best, at);
  }
  greedy_reduce(best, at, bad, part + blockIdx.x);
}
__global__ void __launch_bounds__(kGreedyThreads) greedy_finish(const GreedyPart* part, int parts, int* out) {
  float best = 0.f; int at = -1, bad = 0;
  for (int b = threadIdx.x; b < parts; b += blockDim.x) {
    const GreedyPart q = part[b];
    bad |= q.bad;
    greedy_take(q.value, q.id, best, at);
  }
  __shared__ GreedyPart result;
  greedy_reduce(best, at, bad, &result);
  __syncthreads();
  if (threadIdx.x == 0) { out[0] = result.id; out[1] = result.bad; }
}
}  // namespace lfm::greedy
