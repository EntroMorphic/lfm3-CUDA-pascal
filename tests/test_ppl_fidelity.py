"""ppl_fidelity (G4 tier 2b): the verdict, the file checks, and that eager is optional."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools/ppl_fidelity.py"
LENGTHS = (256, 257, 4096, 32768)
BOS = 1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def windows(d, count=6, **extra):
    d.mkdir(parents=True, exist_ok=True)
    summary = {"bos_id": BOS, "windows": {}, **extra}
    for L in LENGTHS:
        w = np.full((count, L), 7, np.int32)
        w[:, 0] = BOS
        path = d / f"windows-{L}.i32"
        w.tofile(path)
        summary["windows"][str(L)] = {"count": count, "file": path.name, "sha256": sha(path)}
    (d / "summary.json").write_text(json.dumps(summary))
    return summary


def nll(d, engine, L, values, attn=None, dtype=None):
    path = d / f"nll-{engine}-{L}.f32"
    np.asarray(values, np.float32).tofile(path)
    meta = dict(engine=engine, length=L, windows_sha256=json.loads((d / "summary.json").read_text())["windows"][str(L)]["sha256"],
                nll_sha256=sha(path), attn_implementation=attn)
    if not engine.startswith("hf-"):
        meta["scorer_sha256"] = "a" * 64
    if dtype:
        meta["dtype"] = dtype
    (d / f"nll-{engine}-{L}.meta.json").write_text(json.dumps(meta))


def build(tmp_path, engine_noise, sdpa_noise, eager=True):
    rng = np.random.default_rng(0)
    fp32 = dict(dtype="float32", precision=dict(matmul_allow_tf32=False, cudnn_allow_tf32=False))
    for name in ("anchor", "refs", "engine"):
        windows(tmp_path / name, **(fp32 if name == "anchor" else {}))
    for L in LENGTHS:
        base = rng.uniform(1, 5, size=6 * (L - 1))
        nll(tmp_path / "anchor", "hf-sdpa-fp32", L, base, "sdpa", "float32")
        nll(tmp_path / "refs", "hf-sdpa", L, base + sdpa_noise * rng.standard_normal(base.size), "sdpa")
        if eager and L <= 4096:
            nll(tmp_path / "refs", "hf-eager", L, base + sdpa_noise * rng.standard_normal(base.size), "eager")
        nll(tmp_path / "engine", "native-gemm", L, base + engine_noise * rng.standard_normal(base.size))


def run(tmp_path, engine="native-gemm", engine_dir="engine", extra=()):
    return subprocess.run([sys.executable, str(TOOL), "--anchor", str(tmp_path / "anchor"), "--references", str(tmp_path / "refs"),
                           "--engine-dir", str(tmp_path / engine_dir), "--engine", engine, "--output", str(tmp_path / "out.json"),
                           "--anchor-engine", "hf-sdpa-fp32", *extra],
                          capture_output=True, text=True)


def test_as_close_as_sdpa_passes_and_eager_is_optional(tmp_path):
    build(tmp_path, 0.01, 0.01, eager=False)
    r = run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "out.json").read_text())["passes"] is True


def test_closer_than_sdpa_passes_and_reports_eager(tmp_path):
    build(tmp_path, 0.001, 0.01)
    r = run(tmp_path)
    report = json.loads((tmp_path / "out.json").read_text())
    assert r.returncode == 0 and report["rows"][0]["verdict"] == "closer than SDPA" and "hf-eager" in report["rows"][0]
    assert "hf-eager" not in report["rows"][3]  # no eager at 32,768


def test_farther_than_sdpa_fails(tmp_path):
    build(tmp_path, 0.02, 0.01)
    r = run(tmp_path)
    assert r.returncode == 1 and "TIER 2b: FAIL" in r.stdout


def test_an_anchor_not_recorded_as_fp32_is_rejected(tmp_path):
    build(tmp_path, 0.01, 0.01)
    meta = tmp_path / "anchor/nll-hf-sdpa-fp32-256.meta.json"
    m = json.loads(meta.read_text()); m["dtype"] = "bfloat16"; meta.write_text(json.dumps(m))
    r = run(tmp_path)
    assert r.returncode == 2 and "float32" in (r.stdout + r.stderr)
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "invalid input"


def test_tampered_nll_is_rejected(tmp_path):
    build(tmp_path, 0.01, 0.01)
    assert run(tmp_path).returncode == 0  # a passing report exists...
    p = tmp_path / "engine/nll-native-gemm-257.f32"
    data = bytearray(p.read_bytes()); data[0] ^= 1; p.write_bytes(bytes(data))
    r = run(tmp_path)  # ...and must not survive an invalid rerun
    assert r.returncode == 2 and "differ from the sha256" in (r.stdout + r.stderr)
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "invalid input"


def test_an_anchor_with_the_wrong_attention_is_rejected(tmp_path):
    build(tmp_path, 0.01, 0.01)
    meta = tmp_path / "anchor/nll-hf-sdpa-fp32-4096.meta.json"
    m = json.loads(meta.read_text()); m["attn_implementation"] = "eager"; meta.write_text(json.dumps(m))
    assert run(tmp_path).returncode == 2


def test_an_hf_reference_or_a_copy_is_not_an_engine(tmp_path):
    build(tmp_path, 0.01, 0.01)
    assert run(tmp_path, engine="hf-sdpa", engine_dir="refs").returncode == 2
    for L in LENGTHS:  # the engine files replaced by byte copies of the SDPA reference
        src = tmp_path / f"refs/nll-hf-sdpa-{L}.f32"
        nll(tmp_path / "engine", "native-gemm", L, np.fromfile(src, np.float32))
    r = run(tmp_path)
    assert r.returncode == 2 and "byte copy" in r.stderr


def test_expected_hashes_are_enforced(tmp_path):
    build(tmp_path, 0.01, 0.01)
    lines = [f"{sha(tmp_path / f'engine/nll-native-gemm-{L}.f32')}  x/nll-native-gemm-{L}.f32" for L in LENGTHS]
    (tmp_path / "ok.sha").write_text("\n".join(lines) + "\n")
    assert run(tmp_path, extra=("--expect-sha256", str(tmp_path / "ok.sha"))).returncode == 0
    (tmp_path / "bad.sha").write_text("\n".join(lines[:1] + ["0" * 64 + "  nll-native-gemm-257.f32"] + lines[2:]) + "\n")
    assert run(tmp_path, extra=("--expect-sha256", str(tmp_path / "bad.sha"))).returncode == 2


def test_anchor_tf32_or_empty_expectations_or_bad_lengths_give_no_verdict(tmp_path):
    build(tmp_path, 0.01, 0.01)
    assert run(tmp_path).returncode == 0
    (tmp_path / "empty.sha").write_text("")
    assert run(tmp_path, extra=("--expect-sha256", str(tmp_path / "empty.sha"))).returncode == 2
    assert run(tmp_path, extra=("--lengths", "512")).returncode == 2
    summary = tmp_path / "anchor/summary.json"
    for flag in ("cudnn_allow_tf32", "matmul_allow_tf32"):
        s = json.loads(summary.read_text())
        s["precision"] = {**dict(matmul_allow_tf32=False, cudnn_allow_tf32=False), flag: True}
        summary.write_text(json.dumps(s))
        r = run(tmp_path)
        assert r.returncode == 2 and "TF32" in r.stderr, flag


def test_a_subset_of_lengths_with_windows_elsewhere(tmp_path):
    # Robustness checks against an anchor that has no summary.json of its own (e.g. a partial run).
    build(tmp_path, 0.01, 0.01)
    (tmp_path / "anchor/summary.json").unlink()
    r = run(tmp_path, extra=("--lengths", "256", "4096", "--windows-dir", str(tmp_path / "refs")))
    assert r.returncode == 0, r.stderr
    assert [row["length"] for row in json.loads((tmp_path / "out.json").read_text())["rows"]] == [256, 4096]


def test_malformed_metadata_is_no_verdict(tmp_path):
    build(tmp_path, 0.01, 0.01)
    meta = tmp_path / "engine/nll-native-gemm-256.meta.json"
    meta.write_text("[]")  # not an object: an unexpected exception type
    assert run(tmp_path).returncode == 2


def test_insufficient_expectations_tf32_files_and_subset_labels(tmp_path):
    build(tmp_path, 0.01, 0.01)
    lines = [f"{sha(tmp_path / f'engine/nll-native-gemm-{L}.f32')}  nll-native-gemm-{L}.f32" for L in LENGTHS]
    (tmp_path / "three.sha").write_text("\n".join(lines[:3]) + "\n")
    r = run(tmp_path, extra=("--expect-sha256", str(tmp_path / "three.sha")))
    assert r.returncode == 2 and "lists no hash" in r.stderr
    r = run(tmp_path, extra=("--lengths", "256"))
    assert r.returncode == 0 and "not the gate" in r.stdout
    assert json.loads((tmp_path / "out.json").read_text())["full_gate"] is False
    meta = tmp_path / "anchor/nll-hf-sdpa-fp32-257.meta.json"
    m = json.loads(meta.read_text()); m["precision"] = dict(matmul_allow_tf32=True, cudnn_allow_tf32=False)
    meta.write_text(json.dumps(m))
    r = run(tmp_path)
    assert r.returncode == 2 and "TF32 enabled" in r.stderr
