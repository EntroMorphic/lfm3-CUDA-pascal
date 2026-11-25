#pragma once
// Host-only bookkeeping for the keys and values of a windowed attention layer kept in a buffer of 2 x window
// slots per head instead of one slot per context position (docs/long-context-memory.md).
//
// Slots [0, count) hold absolute positions [base, base + count), in order. Decode at `cursor` reads positions
// [max(0, cursor + 1 - window), cursor + 1) (lfm::text::attention_key_range), which are always present and
// contiguous here, so the keys enter every sum in the same order as in a full-length cache.
#include <algorithm>
#include <stdexcept>

namespace lfm::text {

struct WindowSlots {
  int window = 0, base = 0, count = 0;
  struct Copy { int from = 0, to = 0, n = 0; };

  explicit WindowSlots(int window_ = 0) : window(window_) {
    if (window < 0) throw std::runtime_error("invalid attention window");
  }
  int capacity() const { return 2 * window; }
  void reset() { base = count = 0; }

  // An empty-cache prefill of `rows` positions keeps the last min(rows, window): copy `n` positions starting at
  // absolute position `from` of the prefill's keys to slot `to`.
  Copy prefill(int rows) {
    if (rows < 1 || base || count) throw std::runtime_error("window prefill needs an empty cache");
    count = std::min(rows, window); base = rows - count;
    return {base, 0, count};
  }
  // Before position `cursor` is stored (decode). When the buffer is full the newest window - 1 positions move to
  // the front: copy `n` slots from slot `from` to slot `to` (the ranges never overlap). n == 0: nothing to move.
  Copy make_room(int cursor) {
    if (cursor != base + count) throw std::runtime_error("window cache is not contiguous");
    if (count < capacity()) return {};
    const int keep = window - 1;
    const Copy c{count - keep, 0, keep};
    base += count - keep; count = keep;
    return c;
  }
  // The slot position `cursor` is stored in; call after make_room.
  int push(int cursor) {
    if (cursor != base + count || count >= capacity()) throw std::runtime_error("window cache overflow");
    return count++;
  }
  // A continuation of `rows` positions after `cursor` (continuation-v1, Runner::extend), computed through the full-
  // length buffer of the prefill: the held positions and the new ones are there at their absolute positions, and the
  // last min(cursor + rows, window) are kept: copy `n` positions starting at absolute position `from` to slot `to`.
  // Every one of them is held or new: the held positions end at cursor and reach back at least window - 1 before it.
  Copy extend(int cursor, int rows) {
    if (cursor < 1 || rows < 1 || cursor != base + count) throw std::runtime_error("window cache is not contiguous");
    const int total = cursor + rows;
    count = std::min(total, window); base = total - count;
    return {base, 0, count};
  }
  // The slot of a held absolute position.
  int slot(int position) const {
    if (position < base || position >= base + count) throw std::runtime_error("position not in the window cache");
    return position - base;
  }
};

}  // namespace lfm::text
