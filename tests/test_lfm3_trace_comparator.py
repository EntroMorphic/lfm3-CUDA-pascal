"""Acceptance harness regressions, not additional GPU numerical tests."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="comparator needs PyTorch")
TOOL = Path(__file__).resolve().parents[1] / "tools/compare_lfm3_traces.py"


def fixture(tmp_path):
    for name in ("reference", "native"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "0.logits.bf16").write_bytes(struct.pack("HH", 0x3f80, 0x4000))
    (tmp_path / "reference.json").write_text(json.dumps(dict(generated_ids=[2])))
    (tmp_path / "native.json").write_text(json.dumps(dict(teacher_argmax=[2])))
    return tmp_path


def run(path, *extra):
    return subprocess.run([sys.executable, str(TOOL), str(path), *extra], capture_output=True, text=True)


def test_identical_traces_pass_strict_gate(tmp_path):
    path = fixture(tmp_path)
    assert run(path, "--require-exact").returncode == 0
    report = json.loads((path / "comparison.json").read_text())
    assert report["all_traces_bitwise_equal"] and report["teacher_argmax_exact"]
    assert report["results"][0]["native_sha256"] == report["results"][0]["reference_sha256"]


def test_tolerance_does_not_bypass_exact_gate(tmp_path):
    path = fixture(tmp_path)
    (path / "native/0.logits.bf16").write_bytes(struct.pack("HH", 0x3f81, 0x4000))
    assert run(path).returncode == 0
    assert run(path, "--require-exact").returncode == 1
    assert not json.loads((path / "comparison.json").read_text())["all_traces_bitwise_equal"]


@pytest.mark.parametrize("problem", ["empty", "missing", "shape", "nonfinite", "logits", "tokens"])
def test_bad_evidence_is_not_credited(tmp_path, problem):
    path = fixture(tmp_path)
    if problem == "empty":
        (path / "reference/0.logits.bf16").unlink()
    elif problem == "missing":
        (path / "native/0.logits.bf16").unlink()
    elif problem == "shape":
        (path / "native/0.logits.bf16").write_bytes(struct.pack("H", 0x3f80))
    elif problem == "nonfinite":
        (path / "native/0.logits.bf16").write_bytes(struct.pack("HH", 0x7f80, 0x4000))
    elif problem == "logits":
        (path / "native/0.logits.bf16").write_bytes(struct.pack("HH", 0x4200, 0x4000))
    else:
        (path / "native.json").write_text(json.dumps(dict(teacher_argmax=[3])))
    assert run(path).returncode != 0
