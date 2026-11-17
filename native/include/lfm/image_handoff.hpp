#pragma once
// validation/image-queue-v1 part 2: the in-process server's handler thread decodes and preprocesses an image part, and
// leaves the pixel values here under the part's base64 text for as long as its request lives; the engine-owner thread
// (WorkerCore::parse_speech) finds them there instead of decoding the same bytes again. A miss (an out-of-process
// worker, a front end that had the image's token count cached) decodes, as before. Exact by construction: the same
// bytes, the same decoder and preprocessing, the same pixel values.
#include <lfm/image_preprocess.hpp>
#include <lfm/leased_map.hpp>

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace lfm::image {

inline LeasedMap<PixelValues>& shared_pixel_handoff() {
  static LeasedMap<PixelValues> handoff;
  return handoff;
}

// validation/image-cache-v1 C3: which images the engine's own cache holds, published by the engine thread (its cache's
// observer) for the handler threads. The front end skips decoding an image whose token count it knows only when the
// engine holds its rows; otherwise it decodes and hands the pixels over, so the engine thread never decodes. Kept as
// counts of a 64-bit hash of the key, not the keys (each up to 16 MiB): a collision can only make the front end skip a
// decode the engine then does itself (slower, never wrong: the engine's cache compares the whole key).
class HeldImages {
 public:
  void kept(const std::string& key, bool kept) {
    const uint64_t h = std::hash<std::string>{}(key);
    std::lock_guard g(lock_);
    if (kept) {
      ++counts_[h];
    } else if (const auto it = counts_.find(h); it != counts_.end() && --it->second == 0) {
      counts_.erase(it);
    }
  }
  bool holds(const std::string& key) const {
    const uint64_t h = std::hash<std::string>{}(key);
    std::lock_guard g(lock_);
    return counts_.count(h) > 0;
  }
  size_t size() const {
    std::lock_guard g(lock_);
    return counts_.size();
  }

 private:
  mutable std::mutex lock_;
  std::unordered_map<uint64_t, int> counts_;
};

inline HeldImages& shared_held_images() {
  static HeldImages held;
  return held;
}

}  // namespace lfm::image
