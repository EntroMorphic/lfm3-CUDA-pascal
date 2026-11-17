#pragma once

// Host-side ranges only. Disabled builds neither evaluate labels nor call NVTX.
#ifdef LFM_PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#include <string>

namespace lfm::profiling {
class Range {
 public:
  explicit Range(const std::string& label) { nvtxRangePushA(label.c_str()); }
  ~Range() { end(); }
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;
  void end() {
    if (active_) { nvtxRangePop(); active_ = false; }
  }
 private:
  bool active_ = true;
};
}  // namespace lfm::profiling

#define LFM_PROFILE_RANGE(name, label) ::lfm::profiling::Range name(label)
#define LFM_PROFILE_END(name) name.end()
#else
#define LFM_PROFILE_RANGE(name, label) ((void)0)
#define LFM_PROFILE_END(name) ((void)0)
#endif
