#pragma once

// Isolated diagnostic overlay only. No calls are added to the production source.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm {
class InferenceMetrics {
  using Clock = std::chrono::steady_clock;
  bool enabled_ = false;
  Clock::time_point start_;
  std::vector<long long> completions_;
  size_t min_free_ = std::numeric_limits<size_t>::max();
  size_t total_ = 0;
  int batch_;

  void memory() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess || free > total ||
        (total_ && total_ != total))
      throw std::runtime_error("performance observer: invalid device memory observation");
    total_ = total;
    if (free < min_free_) min_free_ = free;
  }

 public:
  explicit InferenceMetrics(int batch) : batch_(batch) {
    const char* value = std::getenv("LFM_PERFORMANCE_OBSERVER");
    enabled_ = value && std::string(value) == "1";
    if (!enabled_) return;
    // Establish a GPU-completed boundary before starting the diagnostic clock.
    // This synchronization and memory sampling perturb diagnostic costs.
    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
      throw std::runtime_error("performance observer: initial synchronization failed");
    memory();
    start_ = Clock::now();
  }

  void require_stepwise(bool stepwise) const {
    if (enabled_ && !stepwise)
      throw std::runtime_error("performance observer: conditional graph has no token boundaries; use --no-cuda-graph");
  }

  // Invoke ONLY after an existing blocking D2H stop-state copy in this stream.
  // This is host time at a GPU-completed recording boundary, not GPU event time
  // and not user-visible streaming delivery. Batch samples are step barriers.
  void completed() {
    if (!enabled_) return;
    const auto now = Clock::now();
    completions_.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count());
    memory();
  }

  void emit() const {
    if (!enabled_) return;
    std::cerr << "LFM_PERFORMANCE_METRICS {\"schema_version\":1,\"diagnostic_only\":true,"
                 "\"clock\":\"host_steady_gpu_completed_record_boundary\",\"batch_size\":"
              << batch_ << ",\"completion_ns\":[";
    for (size_t i = 0; i < completions_.size(); ++i) {
      if (i) std::cerr << ',';
      std::cerr << completions_[i];
    }
    std::cerr << "],\"device_total_bytes\":" << total_
              << ",\"device_min_free_bytes\":" << min_free_
              << ",\"allocator_peak_bytes\":null}\n";
  }
};
}  // namespace lfm
