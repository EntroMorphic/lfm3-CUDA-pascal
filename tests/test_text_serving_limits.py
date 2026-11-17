"""CPU acceptance for advertised budgets and bounded attention scratch."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_limits_and_memory_are_bounded(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    include = Path(__file__).resolve().parents[1] / "native/include"
    source = tmp_path / "limits.cpp"
    source.write_text(r'''
#include <lfm/text_serving_limits.hpp>
#include <cassert>
#include <stdexcept>
using namespace lfm::text;
int main() {
  assert(max_context==262144);  // the engine ceiling (LFM3-1.7B); checkpoints' own limits are ModelConfig's
  assert(scratch_rows(4096)==4096);
  assert(scratch_rows(4097)==128);
  assert(attention_scratch_elements(32768,16)*2*2==268435456);
  require_budget(32767,1,32768);
  require_budget(32768,0,32768);
  assert(scratch_rows(32769)==128 && scratch_rows(262144)==128);
  require_budget(262143,1,262144);
  for(int bad:{0,-1,262145,1<<20}) {
    bool rejected=false;
    try { scratch_rows(bad); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
  }
  bool rejected=false;
  try {require_budget(32768,1,32768);} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
}
''')
    exe = tmp_path / "limits"
    subprocess.run([compiler, "-std=c++17", "-I", str(include), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
