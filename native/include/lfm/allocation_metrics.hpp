#pragma once

// Explicit callsite wrappers only. Never #define cudaMalloc/cudaFree: that would
// rewrite vendor headers and risk counting internals or recursively calling us.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace lfm::allocation_metrics {
inline bool enabled() noexcept {
  static const bool value = [] {
    const char* p = std::getenv("LFM_ALLOCATION_METRICS");
    return p && std::strcmp(p, "1") == 0;
  }();
  return value;
}

struct Snapshot {
  size_t live = 0, peak = 0, allocations = 0, frees = 0;
  size_t failed_allocations = 0, failed_frees = 0, outstanding = 0;
  bool valid = true;
};
struct State {
  std::mutex mutex;
  std::unordered_map<void*, size_t> pointers;
  Snapshot metrics;
};
// An external-linkage inline function's local static is shared across TUs.
// Deliberately retained host bookkeeping permits late static-destructor frees.
// Allocation failure here is fatal rather than returning a fabricated CUDA error.
inline State& state() noexcept {
  static State* value = new State;
  return *value;
}
inline void increment(size_t& value, bool& valid) noexcept {
  if (value == std::numeric_limits<size_t>::max()) valid = false;
  else ++value;
}

template <class T>
inline cudaError_t malloc_requested(T** pointer, size_t bytes) {
  if (!enabled()) return ::cudaMalloc(pointer, bytes);
  auto& s = state();
  // Serialize API calls too: a freed address cannot be reused before its record
  // is removed, and snapshots cannot see a partially committed operation.
  std::lock_guard<std::mutex> lock(s.mutex);
  const cudaError_t result = ::cudaMalloc(pointer, bytes);
  auto& m = s.metrics;
  if (result != cudaSuccess) {
    increment(m.failed_allocations, m.valid);
    return result; // Do not inspect *pointer on failure (may be untouched).
  }
  increment(m.allocations, m.valid);
  if (!pointer || (!*pointer && bytes)) { m.valid = false; return result; }
  if (!*pointer) return result; // CUDA may return null for a zero-byte request.
  try {
    if (!s.pointers.emplace(static_cast<void*>(*pointer), bytes).second) {
      m.valid = false; // Never overwrite a still-live pointer's size.
      return result;
    }
    if (bytes > std::numeric_limits<size_t>::max() - m.live) m.valid = false;
    else { m.live += bytes; if (m.live > m.peak) m.peak = m.live; }
  } catch (...) { m.valid = false; } // Preserve the CUDA return and allocation.
  return result;
}

inline cudaError_t free_requested(void* pointer) {
  if (!enabled()) return ::cudaFree(pointer);
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const cudaError_t result = ::cudaFree(pointer);
  auto& m = s.metrics;
  if (result != cudaSuccess) {
    increment(m.failed_frees, m.valid);
    return result; // Retain record: only successful frees retire requests.
  }
  increment(m.frees, m.valid);
  if (!pointer) return result;
  const auto found = s.pointers.find(pointer);
  if (found == s.pointers.end()) { m.valid = false; return result; }
  if (found->second > m.live) m.valid = false;
  else m.live -= found->second;
  s.pointers.erase(found);
  return result;
}

inline Snapshot snapshot() {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  Snapshot result = s.metrics;
  result.outstanding = s.pointers.size();
  return result;
}
inline void emit(const char* phase) {
  if (!enabled()) return;
  const auto m = snapshot();
  std::fprintf(stderr,
      "LFM_ALLOCATION_METRICS {\"schema_version\":1,\"diagnostic_only\":true,"
      "\"scope\":\"application_cudaMalloc_requested_success_accounting\","
      "\"phase\":\"%s\",\"valid\":%s,\"app_requested_live_bytes\":%zu,"
      "\"app_requested_peak_bytes\":%zu,\"outstanding_allocations\":%zu,"
      "\"successful_allocations\":%zu,\"successful_frees\":%zu,"
      "\"failed_allocations\":%zu,\"failed_frees\":%zu,"
      "\"true_device_peak_bytes\":null,\"model_footprint_bytes\":null}\n",
      phase, m.valid ? "true" : "false", m.live, m.peak, m.outstanding,
      m.allocations, m.frees, m.failed_allocations, m.failed_frees);
  std::fflush(stderr);
}
} // namespace lfm::allocation_metrics
