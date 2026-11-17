"""Host checks for the perplexity bar (95% CI reading) in tools/ppl_compare.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT / "tools"))
from ppl_compare import EQUAL_EPS, bootstrap_ci, passes_bar, verdict
import numpy as np


def test_equal_when_ci_includes_zero():
    assert passes_bar(7.9e-5, (-6.1e-5, 2.2e-4)) and verdict(7.9e-5, (-6.1e-5, 2.2e-4)) == "equal"


def test_better_when_ci_entirely_below_zero():
    assert passes_bar(-1e-3, (-2e-3, -5e-4)) and verdict(-1e-3, (-2e-3, -5e-4)) == "better"


def test_fails_only_when_statistically_worse():
    assert not passes_bar(1e-3, (5e-4, 2e-3)) and verdict(1e-3, (5e-4, 2e-3)) == "worse"


def test_eps_band_absorbs_numerical_noise():
    assert passes_bar(2 * EQUAL_EPS, (0.5 * EQUAL_EPS, 3 * EQUAL_EPS))


def test_single_window_uses_point_estimate():
    assert passes_bar(0.0, None) and not passes_bar(1e-3, None)


def test_bootstrap_is_deterministic_and_brackets_mean():
    x = np.random.default_rng(1).normal(0.0, 1e-3, 500)
    lo, hi = bootstrap_ci(x)
    assert (lo, hi) == bootstrap_ci(x) and lo < x.mean() < hi


# ---- input verification (red-team remediation): bad inputs are errors, never a PASS ----
import hashlib
import json

import pytest

from ppl_compare import InputError, compare

BOS = 9


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _results(tmp_path, length=8, windows=6, engine_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d = tmp_path / "res"
    d.mkdir()
    w = rng.integers(0, BOS, size=(windows, length), dtype=np.int32)
    w[:, 0] = BOS
    w.tofile(d / f"windows-{length}.i32")
    summary = {"corpus_sha256": "x", "bos_id": BOS,
               "windows": {str(length): {"count": windows, "file": f"windows-{length}.i32", "sha256": _sha(d / f"windows-{length}.i32")}}}
    (d / "summary.json").write_text(json.dumps(summary))
    ref = rng.uniform(1.0, 4.0, size=(windows, length - 1)).astype(np.float32)
    # An independent engine never reproduces the reference bit for bit: add 1e-6 per-token noise.
    measured = ref + engine_shift + rng.uniform(-1e-6, 1e-6, size=ref.shape)
    for engine, values in (("hf-sdpa", ref), ("native", measured.astype(np.float32))):
        p = d / f"nll-{engine}-{length}.f32"
        values.tofile(p)
        meta = {"engine": engine, "length": length, "windows_sha256": summary["windows"][str(length)]["sha256"], "nll_sha256": _sha(p)}
        if engine == "hf-sdpa":
            meta["attn_implementation"] = "sdpa"
        else:
            meta["scorer_sha256"] = "a" * 64
        (d / f"nll-{engine}-{length}.meta.json").write_text(json.dumps(meta))
    return d, length


def test_matched_engine_passes(tmp_path):
    d, _ = _results(tmp_path)
    assert compare(d, "native")["passes_bar"]


def test_clearly_worse_engine_fails(tmp_path):
    d, _ = _results(tmp_path, engine_shift=0.01)
    r = compare(d, "native")
    assert not r["passes_bar"] and r["rows"][0]["verdict"] == "worse"


def test_empty_summary_is_an_error_not_a_pass(tmp_path):
    d = tmp_path / "res"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"corpus_sha256": "x", "bos_id": BOS, "windows": {}}))
    with pytest.raises(InputError, match="no windows"):
        compare(d, "native")


def test_missing_metadata_is_refused(tmp_path):
    d, L = _results(tmp_path)
    (d / f"nll-native-{L}.meta.json").unlink()
    with pytest.raises(InputError, match="cannot prove which windows"):
        compare(d, "native")


def test_metadata_for_other_windows_is_refused(tmp_path):
    d, L = _results(tmp_path)
    meta = d / f"nll-native-{L}.meta.json"
    m = json.loads(meta.read_text())
    m["windows_sha256"] = "0" * 64
    meta.write_text(json.dumps(m))
    with pytest.raises(InputError, match="different windows"):
        compare(d, "native")


def test_nll_changed_after_metadata_is_refused(tmp_path):
    d, L = _results(tmp_path)
    p = d / f"nll-native-{L}.f32"
    x = np.fromfile(p, dtype=np.float32)
    np.roll(x, L - 1).tofile(p)  # the reviewer's misalignment: same values, shifted one window
    with pytest.raises(InputError, match="contents differ"):
        compare(d, "native")


def test_non_finite_nll_is_refused(tmp_path):
    d, L = _results(tmp_path)
    p = d / f"nll-native-{L}.f32"
    x = np.fromfile(p, dtype=np.float32)
    x[3] = np.nan
    x.tofile(p)
    meta = d / f"nll-native-{L}.meta.json"
    m = json.loads(meta.read_text())
    m["nll_sha256"] = _sha(p)
    meta.write_text(json.dumps(m))
    with pytest.raises(InputError, match="non-finite"):
        compare(d, "native")


def test_windows_with_bos_inside_are_refused(tmp_path):
    # Shorter windows rescored as longer ones put BOS inside a window (72 x 4096 = 9 x 32768).
    d, L = _results(tmp_path)
    p = d / f"windows-{L}.i32"
    w = np.fromfile(p, dtype=np.int32).reshape(-1, L)
    w[1, 4] = BOS
    w.tofile(p)
    s = json.loads((d / "summary.json").read_text())
    s["windows"][str(L)]["sha256"] = _sha(p)
    (d / "summary.json").write_text(json.dumps(s))
    with pytest.raises(InputError, match="not \\[BOS\\] \\+ content"):
        compare(d, "native")


def test_missing_engine_results_are_an_error(tmp_path):
    d, L = _results(tmp_path)
    (d / f"nll-native-{L}.f32").unlink()
    with pytest.raises(InputError, match="missing"):
        compare(d, "native")


def test_non_finite_delta_never_passes():
    assert not passes_bar(float("nan"), None) and not passes_bar(0.0, (float("nan"), 0.0))
    assert verdict(float("inf"), None) == "invalid"


# ---- round-2 remediation: swapped, copied or mislabeled files are errors ----
import shutil


def test_reference_copied_over_the_engine_is_refused(tmp_path):
    d, L = _results(tmp_path, engine_shift=0.05)
    for ext in ("f32", "meta.json"):
        shutil.copyfile(d / f"nll-hf-sdpa-{L}.{ext}", d / f"nll-native-{L}.{ext}")
    with pytest.raises(InputError, match="records engine"):
        compare(d, "native")


def test_engine_copied_over_the_reference_is_refused(tmp_path):
    d, L = _results(tmp_path, engine_shift=0.05)
    for ext in ("f32", "meta.json"):
        shutil.copyfile(d / f"nll-native-{L}.{ext}", d / f"nll-hf-sdpa-{L}.{ext}")
    with pytest.raises(InputError, match="records engine"):
        compare(d, "native")


def test_relabeled_copy_with_matching_engine_field_is_refused(tmp_path):
    d, L = _results(tmp_path, engine_shift=0.05)
    shutil.copyfile(d / f"nll-hf-sdpa-{L}.f32", d / f"nll-native-{L}.f32")
    meta = json.loads((d / f"nll-native-{L}.meta.json").read_text())
    meta["nll_sha256"] = _sha(d / f"nll-native-{L}.f32")
    (d / f"nll-native-{L}.meta.json").write_text(json.dumps(meta))
    with pytest.raises(InputError, match="copy, not a measurement"):
        compare(d, "native")


def test_reference_cannot_be_compared_with_itself(tmp_path):
    d, _ = _results(tmp_path)
    with pytest.raises(InputError, match="reference configuration"):
        compare(d, "hf-sdpa")


def test_eager_file_presented_as_sdpa_is_refused(tmp_path):
    d, L = _results(tmp_path)
    meta = json.loads((d / f"nll-hf-sdpa-{L}.meta.json").read_text())
    meta["attn_implementation"] = "eager"
    (d / f"nll-hf-sdpa-{L}.meta.json").write_text(json.dumps(meta))
    with pytest.raises(InputError, match="ran attention"):
        compare(d, "native")


def test_input_error_overwrites_a_stale_passing_report(tmp_path):
    import subprocess, sys as _sys
    d, L = _results(tmp_path)
    (d / "compare-native.json").write_text(json.dumps({"passes_bar": True}))
    (d / f"nll-native-{L}.meta.json").unlink()
    r = subprocess.run([_sys.executable, str(ROOT_TOOLS / "ppl_compare.py"), str(d)], capture_output=True, text=True)
    assert r.returncode == 2 and json.loads((d / "compare-native.json").read_text())["passes_bar"] is False


# ---- round-3 remediation ----
def _run_cli(d):
    import subprocess, sys as _sys
    return subprocess.run([_sys.executable, str(ROOT_TOOLS / "ppl_compare.py"), str(d)], capture_output=True, text=True)


def test_any_failure_replaces_a_stale_passing_report(tmp_path):
    d, L = _results(tmp_path)
    (d / "compare-native.json").write_text(json.dumps({"passes_bar": True}))
    (d / f"windows-{L}.i32").unlink()  # FileNotFoundError, not an InputError
    r = _run_cli(d)
    report = json.loads((d / "compare-native.json").read_text())
    assert r.returncode == 2 and report["passes_bar"] is False and "FileNotFoundError" in report["input_error"]


def test_malformed_summary_replaces_a_stale_passing_report(tmp_path):
    d, _ = _results(tmp_path)
    (d / "compare-native.json").write_text(json.dumps({"passes_bar": True}))
    (d / "summary.json").write_text("{not json")
    assert _run_cli(d).returncode == 2 and json.loads((d / "compare-native.json").read_text())["passes_bar"] is False


def test_copy_of_the_eager_reference_is_refused(tmp_path):
    d, L = _results(tmp_path)
    eager = (np.fromfile(d / f"nll-hf-sdpa-{L}.f32", dtype=np.float32) - 1e-4).astype(np.float32)
    eager.tofile(d / f"nll-hf-eager-{L}.f32")
    (d / f"nll-hf-eager-{L}.meta.json").write_text(json.dumps({"engine": "hf-eager", "length": L, "attn_implementation": "eager",
        "windows_sha256": json.loads((d / "summary.json").read_text())["windows"][str(L)]["sha256"], "nll_sha256": _sha(d / f"nll-hf-eager-{L}.f32")}))
    shutil.copyfile(d / f"nll-hf-eager-{L}.f32", d / f"nll-native-{L}.f32")
    meta = json.loads((d / f"nll-native-{L}.meta.json").read_text())
    meta["nll_sha256"] = _sha(d / f"nll-native-{L}.f32")
    (d / f"nll-native-{L}.meta.json").write_text(json.dumps(meta))
    with pytest.raises(InputError, match="byte-identical to nll-hf-eager"):
        compare(d, "native")


def test_engine_file_without_a_scorer_binding_is_refused(tmp_path):
    d, L = _results(tmp_path)
    meta = json.loads((d / f"nll-native-{L}.meta.json").read_text())
    del meta["scorer_sha256"]
    (d / f"nll-native-{L}.meta.json").write_text(json.dumps(meta))
    with pytest.raises(InputError, match="does not name the scorer"):
        compare(d, "native")


# ---- round-4 remediation ----
def test_read_only_stale_passing_report_is_removed(tmp_path):
    import os
    d, L = _results(tmp_path)
    stale = d / "compare-native.json"
    stale.write_text(json.dumps({"passes_bar": True}))
    os.chmod(stale, 0o444)  # e.g. left by an earlier run as another user
    (d / f"nll-native-{L}.meta.json").unlink()
    r = _run_cli(d)
    report = json.loads(stale.read_text())
    assert r.returncode == 2 and report["passes_bar"] is False and "cannot prove which windows" in report["input_error"]


# ---- round-5 remediation ----
def test_leftover_report_that_is_not_a_json_object_still_fails_cleanly(tmp_path):
    d, L = _results(tmp_path)
    (d / "compare-native.json").mkdir()  # cannot be unlinked as a file nor parsed
    (d / f"nll-native-{L}.meta.json").unlink()
    r = _run_cli(d)
    assert r.returncode == 2 and "Traceback" not in r.stderr and "WARNING" in r.stdout
