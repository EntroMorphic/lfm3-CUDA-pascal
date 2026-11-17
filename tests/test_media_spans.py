"""lfm3_speech::check_audio_spans and check_image_spans (lfm/speech_run.hpp): accepted prompts, and each refusal with
its exact message (native/tests/media_spans_check.cpp), built plain and with ASan+UBSan."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flags", [[], ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]])
def test_media_spans(tmp_path, flags):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path / "media-spans-check"
    subprocess.run(["g++", "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Werror", *flags, "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/media_spans_check.cpp"), "-o", str(binary)], check=True)
    r = subprocess.run([str(binary)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "all passed", r.stdout + r.stderr
