"""compare_trace_spread: common file set, prompt identity, token-agreement steps, exact logits."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/compare_trace_spread.py"


def bf16(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def make(root, sub, ids, prompt, files, exact_logits=None):
    d = root / "c"
    (d / sub).mkdir(parents=True, exist_ok=True)
    if sub == "reference":
        (d / "reference.json").write_text(json.dumps(dict(input_ids=prompt, generated_ids=ids)))
    for name, values in files.items():
        (d / sub / name).write_bytes(bf16(values))
    for name, values in (exact_logits or {}).items():
        (d / sub / name).write_bytes(np.asarray(values, np.float32).tobytes())


def run(tmp_path, *candidates):
    out = tmp_path / "out.json"
    cmd = [sys.executable, str(TOOL), str(tmp_path / "anchor"), str(out), "--cases", "c"]
    for name, traces, ids_root in candidates:
        cmd += ["--candidate", name, str(traces), str(ids_root)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, (json.loads(out.read_text()) if out.exists() else None)


BASE = np.linspace(-3, 3, 64)


def test_candidates_are_compared_on_the_common_file_set_and_agreeing_steps(tmp_path):
    files = {f"{s}.layers.0.{n}.bf16": BASE * (1 + s) for s in range(3) for n in ("a", "b")}
    make(tmp_path / "anchor", "reference", [5, 6, 7], [1, 2], files)
    # "x" traces everything; "y" lacks every ".b" file, and its third token diverges (steps 0..2 comparable).
    make(tmp_path / "x", "reference", [5, 6, 7], [1, 2], {k: v * 1.01 for k, v in files.items()})
    make(tmp_path / "y", "reference", [5, 6, 9], [1, 2], {k: v * 1.02 for k, v in files.items() if ".a." in k})
    result, report = run(tmp_path, ("x", tmp_path / "x/reference", tmp_path / "x"), ("y", tmp_path / "y/reference", tmp_path / "y"))
    assert result.returncode == 0, result.stderr
    row = report["cases"][0]
    assert row["compared_through_step"] == 2 and row["common_files"] == 3
    assert row["anchor_files_not_traced_by"] == {"x": 0, "y": 3}
    assert row["candidates"]["x"]["activation_files"] == row["candidates"]["y"]["activation_files"] == 3
    assert row["candidates"]["x"]["activation_rel_rms_median"] == pytest.approx(0.01, rel=0.2)


def test_a_candidate_with_a_different_prompt_is_rejected(tmp_path):
    files = {"0.layers.0.a.bf16": BASE}
    make(tmp_path / "anchor", "reference", [5], [1, 2], files)
    make(tmp_path / "x", "reference", [5], [1, 3], files)
    result, _ = run(tmp_path, ("x", tmp_path / "x/reference", tmp_path / "x"))
    assert result.returncode != 0 and "different prompt" in (result.stdout + result.stderr)


def test_exact_logits_are_used_when_present_and_paths_may_contain_at(tmp_path):
    root = tmp_path / "with@sign"
    logits = np.array([0.0, 1.0, 2.0, 3.0])
    make(tmp_path / "anchor", "reference", [5], [1], {"0.logits.bf16": logits}, {"0.logits.f32": logits})
    # The candidate's exact logits differ from the anchor's by 2^-10, which rounds away in BF16 (both .bf16 files
    # are identical and each is the rounding of its .f32): KL is nonzero only if the exact values are used.
    make(root, "reference", [5], [1], {"0.logits.bf16": logits}, {"0.logits.f32": logits + [0, 0, 0, 2.0**-10]})
    result, report = run(tmp_path, ("x", root / "reference", root))
    assert result.returncode == 0, result.stderr
    assert report["cases"][0]["exact_fp32_logits"] == {"anchor": True, "x": True}
    assert report["cases"][0]["candidates"]["x"]["logit_kl_mean"] > 0


def test_inconsistent_exact_logits_are_rejected(tmp_path):
    logits = np.array([0.0, 1.0, 2.0, 3.0])
    make(tmp_path / "anchor", "reference", [5], [1], {"0.logits.bf16": logits}, {"0.logits.f32": logits})
    make(tmp_path / "x", "reference", [5], [1], {"0.logits.bf16": logits}, {"0.logits.f32": logits + [0, 0, 0, 0.5]})
    result, _ = run(tmp_path, ("x", tmp_path / "x/reference", tmp_path / "x"))
    assert result.returncode != 0 and "not the rounding" in result.stderr
