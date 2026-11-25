"""CPU acceptance for the windowed key/value buffer bookkeeping (native/include/lfm/window_slots.hpp)."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_window_slots_always_hold_the_decode_range_in_order(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    include = Path(__file__).resolve().parents[1] / "native/include"
    source = tmp_path / "slots.cpp"
    source.write_text(r'''
#include <lfm/window_slots.hpp>
#include <lfm/text_attention_tiles.hpp>
#include <cassert>
#include <stdexcept>
#include <vector>
using namespace lfm::text;
// A model of the device buffer: slot -> the absolute position whose key it holds (-1: never written).
static void run(int window, int prompt, int steps, int capacity) {
  WindowSlots s(window);
  std::vector<int> buffer(s.capacity(), -1);
  const auto p = s.prefill(prompt);
  assert(p.to == 0 && p.n == std::min(prompt, window) && p.from == prompt - p.n);
  for (int i = 0; i < p.n; ++i) buffer[p.to + i] = p.from + i;
  for (int cursor = prompt; cursor < prompt + steps; ++cursor) {
    const auto c = s.make_room(cursor);
    assert(c.n == 0 || (c.to + c.n <= c.from && c.from + c.n <= s.capacity()));  // in range, never overlapping
    for (int i = 0; i < c.n; ++i) buffer[c.to + i] = buffer[c.from + i];
    const int slot = s.push(cursor);
    assert(slot >= 0 && slot < s.capacity());
    buffer[slot] = cursor;
    // What decode reads at this cursor, by the engine's own rule.
    const auto range = attention_key_range(capacity, cursor, 1, 0, 1, window);
    assert(range.begin == std::max(0, cursor + 1 - window) && range.end == cursor + 1);
    const int first = s.slot(range.begin);
    assert(first + range.span() <= s.capacity());
    for (int i = 0; i < range.span(); ++i) assert(buffer[first + i] == range.begin + i);  // present, contiguous, in order
  }
}
// continuation-v1: decode and extensions (Runner::extend) in any order. An extension copies the held slots to their
// absolute positions in the full-length buffer, stores its rows there, and keeps what extend() names.
static void mixed(int window, int prompt, const std::vector<int>& plan, int capacity) {
  WindowSlots s(window);
  std::vector<int> buffer(s.capacity(), -1), full(capacity, -1);
  const auto p = s.prefill(prompt);
  for (int i = 0; i < p.n; ++i) buffer[p.to + i] = p.from + i;
  int cursor = prompt;
  auto check_held = [&] {  // held: [base, cursor), in order, reaching back at least window - 1 before cursor
    assert(s.base + s.count == cursor);
    assert(s.base <= std::max(0, cursor - window + 1));
    for (int i = 0; i < s.count; ++i) assert(buffer[i] == s.base + i);
  };
  for (int step : plan) {
    check_held();
    if (step == 0) {  // one decode position
      const auto c = s.make_room(cursor);
      for (int i = 0; i < c.n; ++i) buffer[c.to + i] = buffer[c.from + i];
      buffer[s.push(cursor)] = cursor;
      ++cursor;
      continue;
    }
    std::fill(full.begin(), full.end(), -1);
    for (int i = 0; i < s.count; ++i) full[s.base + i] = buffer[i];
    for (int i = 0; i < step; ++i) full[cursor + i] = cursor + i;
    // every key a row of the extension can see is present: [max(0, q - window + 1), q]
    for (int q = cursor; q < cursor + step; ++q)
      for (int k = std::max(0, q - window + 1); k <= q; ++k) assert(full[k] == k);
    const auto keep = s.extend(cursor, step);
    cursor += step;
    assert(keep.to == 0 && keep.n == std::min(cursor, window) && keep.from == cursor - keep.n);
    for (int i = 0; i < keep.n; ++i) { assert(full[keep.from + i] == keep.from + i); buffer[keep.to + i] = full[keep.from + i]; }
  }
  check_held();
}
int main() {
  for (int window : {1, 2, 3, 7, 256})
    for (int prompt : {1, 2, window - 1, window, window + 1, 2 * window + 1})
      for (int rows : {1, 2, window - 1, window, window + 1, 3 * window + 2}) {
        if (prompt < 1 || rows < 1) continue;
        std::vector<int> plan;
        for (int i = 0; i < 2 * window + 3; ++i) plan.push_back(0);
        plan.push_back(rows);
        for (int i = 0; i < window + 1; ++i) plan.push_back(0);
        plan.push_back(rows); plan.push_back(1); plan.push_back(0);
        mixed(window, prompt, plan, prompt + 2 * rows + 4 * window + 8);
        mixed(window, prompt, {rows, rows, 0, rows}, prompt + 3 * rows + 1);
      }
  {
    WindowSlots s(4);
    bool rejected = false;
    try { s.extend(1, 1); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);  // nothing cached
    s.prefill(3);
    rejected = false;
    try { s.extend(4, 1); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);  // a gap
    rejected = false;
    try { s.extend(3, 0); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);  // no rows
  }
  for (int window : {1, 2, 3, 7, 256})
    for (int prompt : {1, 2, window - 1, window, window + 1, 2 * window, 2 * window + 1, 5 * window + 3}) {
      if (prompt < 1) continue;
      run(window, prompt, 6 * window + 5, 8192);
    }
  run(256, 4095, 1, 4096);
  run(256, 32000, 768, 32768);
  WindowSlots s(4);
  bool rejected = false;
  try { s.make_room(1); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);  // a gap
  s.prefill(9);
  rejected = false;
  try { s.prefill(1); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);  // prefill into a non-empty cache
  rejected = false;
  try { s.slot(4); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);  // a position that was dropped
  s.reset();
  assert(s.prefill(2).n == 2);
}
''')
    exe = tmp_path / "slots"
    subprocess.run([compiler, "-std=c++20", "-I", str(include), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
