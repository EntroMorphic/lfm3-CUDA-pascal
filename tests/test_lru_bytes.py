"""validation/image-cache-v1 rule 3 (host): lfm::LruBytes (the image caches: least recently used under a byte budget,
keys compared in full) and lfm::LeasedMap (image-queue-v1 part 2's handoff), native/tests/lru_bytes_check.cpp under
ASan+UBSan and TSan."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("compiler, flags", [("g++", ["-DPLAIN"]),
                                             ("g++", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]),
                                             ("clang++", ["-fsanitize=thread"])])
def test_lru_bytes_and_leased_map(tmp_path, compiler, flags):
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} required")
    binary = tmp_path / "lru_bytes_check"
    subprocess.run([compiler, "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Werror", *flags, "-pthread",
                    "-I", str(ROOT / "native" / "include"), str(ROOT / "native" / "tests" / "lru_bytes_check.cpp"),
                    "-o", str(binary)], check=True)
    prefix = ["setarch", __import__("os").uname().machine, "-R"] if "thread" in flags[0] else []
    r = subprocess.run(prefix + [str(binary)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stderr
