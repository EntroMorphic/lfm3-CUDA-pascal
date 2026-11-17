"""lfm::Json::parse made faster (2026-11-17: a 13 MB image request was parsed twice at ~50 ms each): every value and
error identical to the frozen reference parser, native/tests/json_parse_check.cpp, plain and under ASan+UBSan."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flags", [["-O2"], ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]])
def test_json_parse_matches_reference(tmp_path, flags):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path / "json_parse_check"
    subprocess.run(["g++", "-std=c++20", *flags, "-I", str(ROOT / "native" / "include"),
                    str(ROOT / "native" / "tests" / "json_parse_check.cpp"), "-o", str(binary)], check=True)
    r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
