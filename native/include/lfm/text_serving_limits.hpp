#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace lfm::text {
// The engine's ceiling: LFM3-1.7B's card, 256K input + output (262,144 positions). Each checkpoint's own limit is
// ModelConfig's (LFM3-Omni-2B: 32K, its card).
inline constexpr int max_context = 262144;
inline constexpr int attention_query_tile = 128;
inline int scratch_rows(int capacity) {
  if(capacity<1 || capacity>max_context) throw std::runtime_error("capacity must be 1..262144");
  // Preserve the previously validated <=4096 execution schedule.
  return capacity<=4096?capacity:attention_query_tile;
}
inline size_t attention_scratch_elements(int capacity,int heads) {
  if(heads<1 || heads>65535) throw std::runtime_error("invalid query head count");
  return size_t(scratch_rows(capacity))*capacity*heads;
}
inline void require_budget(size_t prompt,size_t output,int capacity) {
  scratch_rows(capacity);
  if(prompt==0 || prompt>size_t(capacity) || output>size_t(capacity)-prompt)
    throw std::runtime_error("input plus output exceeds context budget");
}
} // namespace lfm::text
