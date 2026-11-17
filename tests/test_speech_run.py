"""Host check of the speech chunk loop (lfm3_speech::run_chunks, native/include/lfm/speech_run.hpp) against a scripted
fake session: seeds, codec continue/finalize, the step budget's bound and cap, and the truncated flag, in named cases
and 20,000 random ones (native/tests/speech_run_check.cpp), built with AddressSanitizer and UBSan."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_run_chunks(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path / "speech-run-check"
    subprocess.run(["g++", "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/speech_run_check.cpp"), "-o", str(binary)], check=True)
    r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("ok "), r.stdout + r.stderr
