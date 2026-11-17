#pragma once
// A least-recently-used map under a byte budget (validation/image-cache-v1). The key is the full input (an image part's
// base64 text): std::map compares whole keys, never a hash, so two different inputs never share an entry. Each entry
// is charged its key's bytes plus the value's own (`cost`) and holds its key once (validation/image-cache-v1 red-team:
// a second copy in the order list let a cache of large keys hold twice its budget); inserting evicts the least recently used entries until the
// total fits; an entry larger than the whole budget is not kept; a budget of 0 keeps nothing. Values are shared
// pointers: an evicted entry stays alive for whoever still holds it (a request in flight), only the map forgets it.
// Not thread-safe: callers lock (LockedLruBytes) or keep it on one thread.
#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace lfm {

template <class T>
class LruBytes {
 public:
  using Value = std::shared_ptr<const T>;
  explicit LruBytes(size_t budget = 0) : budget_(budget) {}
  // Told each key the moment it is kept (true) and forgotten (false), in order (validation/image-cache-v1 C3: the
  // engine publishes which images it holds). Called under the caller's own locking; it must not throw.
  void set_observer(std::function<void(const std::string& key, bool kept)> observer) { observer_ = std::move(observer); }
  void set_budget(size_t budget) {
    budget_ = budget;
    trim();
  }
  Value find(const std::string& key) {
    const auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    order_.splice(order_.begin(), order_, it->second.at);  // most recent first
    return it->second.value;
  }
  void put(const std::string& key, Value value, size_t value_bytes) {
    // the cost compared without its sum overflowing (a wrapped sum would be a small cost, kept under any budget)
    if (!value || value_bytes > budget_ || key.size() > budget_ - value_bytes) return;
    const size_t cost = key.size() + value_bytes;
    const auto it = entries_.find(key);
    if (it != entries_.end()) {  // the same input: keep the one there, make it the most recent
      order_.splice(order_.begin(), order_, it->second.at);
      return;
    }
    // the key held once, by the map; the order list points at it (a std::map node's key never moves). If either
    // allocation throws, neither structure keeps a part of the entry.
    const auto at = entries_.emplace(key, Entry{std::move(value), cost, {}}).first;
    try {
      order_.push_front(&at->first);
    } catch (...) {
      entries_.erase(at);
      throw;
    }
    at->second.at = order_.begin();
    bytes_ += cost;
    if (observer_) observer_(at->first, true);
    trim();
  }
  size_t bytes() const { return bytes_; }
  size_t size() const { return entries_.size(); }
  size_t budget() const { return budget_; }

 private:
  struct Entry {
    Value value;
    size_t cost;
    std::list<const std::string*>::iterator at;
  };
  void trim() {
    while (bytes_ > budget_ && !order_.empty()) {
      const auto it = entries_.find(*order_.back());
      if (observer_) observer_(it->first, false);
      bytes_ -= it->second.cost;
      entries_.erase(it);
      order_.pop_back();
    }
  }
  size_t budget_ = 0, bytes_ = 0;
  std::list<const std::string*> order_;  // most recent first: each entry's key, as the map holds it
  std::map<std::string, Entry> entries_;
  std::function<void(const std::string&, bool)> observer_;
};

// The same, for several threads (the server's handler threads).
template <class T>
class LockedLruBytes {
 public:
  using Value = typename LruBytes<T>::Value;
  void set_budget(size_t budget) {
    std::lock_guard g(lock_);
    lru_.set_budget(budget);
  }
  Value find(const std::string& key) {
    std::lock_guard g(lock_);
    return lru_.find(key);
  }
  void put(const std::string& key, Value value, size_t value_bytes) {
    std::lock_guard g(lock_);
    lru_.put(key, std::move(value), value_bytes);
  }
  size_t bytes() {
    std::lock_guard g(lock_);
    return lru_.bytes();
  }
  size_t size() {
    std::lock_guard g(lock_);
    return lru_.size();
  }

 private:
  std::mutex lock_;
  LruBytes<T> lru_;
};

}  // namespace lfm
