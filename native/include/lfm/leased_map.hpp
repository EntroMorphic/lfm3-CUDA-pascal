#pragma once
// A value handed from one thread to another under a key, for as long as someone holds a lease on it
// (validation/image-queue-v1 part 2: the in-process server's handler thread decodes an image, the engine-owner thread
// finds the pixels instead of decoding them again). The pattern of lfm::audio_features::FeatureCache
// (validation/audio-frontend-v1), for any value type.
// The key is the full input (an image part's base64 text): std::map compares whole keys, so two different inputs never
// share an entry, whatever they have in common; the same input shares one, which is safe because the value is a
// deterministic function of it. An entry lives while a lease on it lives: a request holds its leases until it ends,
// however it ends, so nothing outlives the requests in flight. The map holds each key once; a lease refers to its
// entry, never copying the key (red-team, image-queue-v1 rule 5: a copy per lease took a request's images' size again).
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace lfm {

template <class T>
class LeasedMap {
 public:
  using Value = std::shared_ptr<const T>;

 private:
  struct Entry {
    Value value;
    int leases = 0;
  };
  using Entries = std::map<std::string, Entry>;

 public:
  class Lease {
   public:
    Lease() = default;
    Lease(Lease&& o) noexcept : map_(o.map_), at_(o.at_) { o.map_ = nullptr; }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) release(), map_ = o.map_, at_ = o.at_, o.map_ = nullptr;
      return *this;
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { release(); }

   private:
    friend class LeasedMap;
    Lease(LeasedMap* map, typename Entries::iterator at) : map_(map), at_(at) {}
    void release() noexcept {
      if (!map_) return;
      std::lock_guard g(map_->lock_);
      if (--at_->second.leases == 0) map_->entries_.erase(at_);  // valid: erased only here, at its last lease
      map_ = nullptr;
    }
    LeasedMap* map_ = nullptr;
    typename Entries::iterator at_{};
  };
  // Leaves `value` under `key` (the one already there kept: the same input, the same value).
  Lease put(const std::string& key, Value value) {
    std::lock_guard g(lock_);
    auto at = entries_.try_emplace(key).first;
    if (!at->second.value) at->second.value = std::move(value);
    ++at->second.leases;
    return Lease(this, at);
  }
  Value find(const std::string& key) const {
    std::lock_guard g(lock_);
    const auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.value;
  }
  size_t size() const {
    std::lock_guard g(lock_);
    return entries_.size();
  }

 private:
  mutable std::mutex lock_;
  Entries entries_;
};

}  // namespace lfm
