"""Initialization cancellation must be rechecked before launching forward."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_cancellation_during_construction_prevents_prefill(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "dispatch.cpp"
    source.write_text(r'''
#include <lfm/text_cancellation.hpp>
#include <cassert>
#include <stdexcept>
struct FakeRunner {
  int forwards=0;
  explicit FakeRunner(int writer) {
    const unsigned char frame[8]={7,0,0,0,0,0,0,0};
    assert(write(writer,frame,8)==8); // Cancellation arrives during initialization.
  }
  int forward(){++forwards;return 42;}
};
int main(){
  int fds[2];assert(pipe2(fds,O_NONBLOCK)==0);
  lfm::text::CancellationReader reader(fds[0]);
  auto check=[&]{if(reader.requested(7))throw std::runtime_error("cancelled");};
  check(); // The earlier poll cannot observe cancellation that hasn't arrived.
  FakeRunner runner(fds[1]);
  bool cancelled=false;
  try{lfm::text::at_cancellation_boundary(check,[&]{return runner.forward();});}
  catch(const std::runtime_error&){cancelled=true;}
  assert(cancelled && runner.forwards==0);
  assert(lfm::text::at_cancellation_boundary([]{},[]{return 19;})==19);
  close(fds[0]);close(fds[1]);
}
''')
    binary = tmp_path / "dispatch"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-I", str(root / "native/include"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
