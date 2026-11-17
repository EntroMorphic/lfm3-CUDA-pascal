"""Exact FP32 logit traces (G4): rounding, all-or-none presence, nonfinite rejection, and gate wiring."""
import hashlib
import importlib.util
from pathlib import Path
import random
import struct
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from lfm3_trace_exact import bf16_rounding_of, exact_logit_steps  # noqa: E402


def rne(word):
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


def test_rounding_matches_round_to_nearest_even_on_finite_values():
    rng = random.Random(0)
    words = [rng.getrandbits(32) for _ in range(20000)]
    words += [0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x3F808000, 0x3F818000, 0x3F80C000,
              0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x7F7F7FFF]  # zeros, subnormals, ties, near overflow
    finite = [w for w in words if (w >> 23) & 0xFF != 0xFF]
    got = struct.unpack(f"<{len(finite)}H", bf16_rounding_of(struct.pack(f"<{len(finite)}I", *finite)))
    assert list(got) == [rne(w) for w in finite]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_rounding_matches_torch():
    import torch
    rng = random.Random(1)
    finite = [w for w in (rng.getrandbits(32) for _ in range(20000)) if (w >> 23) & 0xFF != 0xFF]
    values = torch.tensor(finite, dtype=torch.int64).to(torch.int32).view(torch.float32)
    want = values.bfloat16().view(torch.int16).numpy().astype("<u2").tobytes()
    assert bf16_rounding_of(struct.pack(f"<{len(finite)}I", *finite)) == want


@pytest.mark.parametrize("word", [0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001])
def test_rounding_rejects_nonfinite(word):
    with pytest.raises(ValueError, match="nonfinite"):
        bf16_rounding_of(struct.pack("<If", word, 2.0))


def test_rounding_rejects_partial_words():
    with pytest.raises(ValueError, match="multiple of 4"):
        bf16_rounding_of(b"\0\0\0")


def write(directory, step, exact, rounded=None):
    raw = struct.pack(f"<{len(exact)}f", *exact)
    (directory / f"{step}.logits.f32").write_bytes(raw)
    (directory / f"{step}.logits.bf16").write_bytes(bf16_rounding_of(raw) if rounded is None else rounded)


def test_exact_steps_accepts_consistent_traces_and_absence(tmp_path):
    for step in range(3):
        write(tmp_path, step, [1.0, 16.0 - 0.03, -2.5])
    assert exact_logit_steps(tmp_path, range(3)) is True
    other = tmp_path / "bf16-only"
    other.mkdir()
    (other / "0.logits.bf16").write_bytes(b"\0\0")
    assert exact_logit_steps(other, [0]) is False


@pytest.mark.parametrize("damage", ["partial", "not-rounding", "nan", "short-bf16"])
def test_exact_steps_rejects_inconsistent_traces(tmp_path, damage):
    for step in range(2):
        write(tmp_path, step, [1.0, 16.0 - 0.03])
    if damage == "partial":
        (tmp_path / "1.logits.f32").unlink()
    if damage == "not-rounding":
        (tmp_path / "1.logits.bf16").write_bytes(struct.pack("<HH", 0x3F80, 0x417F))
    if damage == "nan":
        (tmp_path / "1.logits.f32").write_bytes(struct.pack("<ff", 1.0, float("nan")))
    if damage == "short-bf16":
        (tmp_path / "1.logits.bf16").write_bytes(b"\x80\x3f")
    with pytest.raises(ValueError):
        exact_logit_steps(tmp_path, range(2))


@pytest.mark.parametrize("tool", ["check_lfm3_context.py", "check_lfm3_forward.py"])
def test_gates_check_exact_logit_traces(tool):
    # Both gates that read native .bf16 logit traces must run the all-or-none rounding check.
    text = (TOOLS / tool).read_text()
    assert "from lfm3_trace_exact import exact_logit_steps" in text and "exact_logit_steps(" in text.split("import exact_logit_steps")[1]


def test_forward_tolerance_mode_requires_exact_logit_traces(tmp_path):
    spec = importlib.util.spec_from_file_location("forward_gate", TOOLS / "check_lfm3_forward.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "0.logits.bf16").write_bytes(struct.pack("<HH", 0x3F80, 0x4000))
    assert gate.check_exact_logits(tmp_path, False, "case") is False  # bitwise mode: BF16-only traces are fine
    with pytest.raises(AssertionError, match="requires the engine's exact"):
        gate.check_exact_logits(tmp_path, True, "case")
    (tmp_path / "0.logits.f32").write_bytes(struct.pack("<ff", 1.0, 2.0))
    assert gate.check_exact_logits(tmp_path, True, "case") is True
    (tmp_path / "0.logits.f32").write_bytes(struct.pack("<ff", 1.0, 2.5))
    with pytest.raises(ValueError, match="not the rounding"):
        gate.check_exact_logits(tmp_path, True, "case")


def load_forward():
    spec = importlib.util.spec_from_file_location("forward_gate_tiers", TOOLS / "check_lfm3_forward.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def test_forward_tier1_manifest_is_exact(tmp_path):
    gate = load_forward()
    (tmp_path / "0.logits.bf16").write_bytes(b"\x80\x3f")
    (tmp_path / "0.layers.0.x.bf16").write_bytes(b"\x00\x40")
    pinned = {"baseline": gate.trace_manifest(tmp_path, [5, 6])}
    gate.compare_manifest(pinned, {"baseline": gate.trace_manifest(tmp_path, [5, 6])})
    for damage in ("ids", "bytes", "extra", "missing", "fixtures"):
        actual = {"baseline": gate.trace_manifest(tmp_path, [5, 6])}
        if damage == "ids":
            actual["baseline"]["generated_ids"] = [5, 7]
        if damage == "bytes":
            actual["baseline"]["files"]["0.logits.bf16"] = "0" * 64
        if damage == "extra":
            actual["baseline"]["files"]["1.logits.bf16"] = "0" * 64
        if damage == "missing":
            del actual["baseline"]["files"]["0.layers.0.x.bf16"]
        if damage == "fixtures":
            actual["single"] = actual["baseline"]
        with pytest.raises(AssertionError, match="tier 1"):
            gate.compare_manifest(pinned, actual)


def _tier2a_tree(root, native_noise, anchor_dtype="float32"):
    import json
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.standard_normal(256)
    bf = lambda x: (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    ref = dict(input_ids=[1, 2], generated_ids=[5])
    # The layout check_lfm3_forward writes: OUT/fp32/CASE, OUT/CASE/{reference,native}, OUT/sdpa/CASE.
    for d, noise, sub in ((root / "fp32/case", 0.0, "reference"), (root / "case", 0.02, "reference"),
                          (root / "sdpa/case", 0.015, "reference"), (root / "case", native_noise, "native")):
        (d / sub).mkdir(parents=True, exist_ok=True)
        meta = dict(ref)
        if d.parent.name == "fp32":
            meta.update(dtype=anchor_dtype, precision=dict(matmul_allow_tf32=False, cudnn_allow_tf32=False))
        (d / "reference.json").write_text(json.dumps(meta))
        for step in range(2):
            (d / sub / f"{step}.layers.0.x.bf16").write_bytes(bf(base * (1 + noise * rng.standard_normal(256))))


@pytest.mark.parametrize("native_noise,passed", [(0.01, True), (0.05, False)])
def test_forward_tier2a_on_the_gate_layout(tmp_path, native_noise, passed):
    import subprocess
    gate = load_forward()
    _tier2a_tree(tmp_path, native_noise)
    run = lambda command, log: subprocess.run([str(x) for x in command], check=True, capture_output=True)
    row = gate.tier2a(TOOLS, tmp_path, "case", run)
    assert row["common_files"] == 2 and row["passed"] is passed


def test_forward_tier2a_requires_an_fp32_anchor(tmp_path):
    gate = load_forward()
    _tier2a_tree(tmp_path, 0.01, anchor_dtype="bfloat16")
    with pytest.raises(AssertionError, match="not recorded as FP32"):
        gate.tier2a(TOOLS, tmp_path, "case", lambda *a: None)


def test_forward_golden_ignores_pin_provenance(tmp_path):
    gate = load_forward()
    (tmp_path / "0.logits.bf16").write_bytes(b"\x80\x3f")
    m = {"baseline": gate.trace_manifest(tmp_path, [5])}
    gate.compare_manifest(dict(m, _provenance={"binary_sha256": "x"}), m)


def test_forward_pins_only_after_tier2a_passes(tmp_path):
    gate = load_forward()
    binary = tmp_path / "liquid-lfm3"
    binary.write_bytes(b"binary")
    pin = tmp_path / "pin.json"
    manifest = {"baseline": dict(generated_ids=[5], files={"0.logits.bf16": "0" * 64})}
    with pytest.raises(AssertionError, match="tier 2a"):
        gate.finish_tiers(pin, None, binary, manifest, [dict(case="baseline", passed=False)])
    assert not pin.exists()
    gate.finish_tiers(pin, None, binary, manifest, [dict(case="baseline", passed=True)])
    import json
    pinned = json.loads(pin.read_text())
    assert pinned["_provenance"]["binary_sha256"] == hashlib.sha256(b"binary").hexdigest() and "baseline" in pinned
    gate.finish_tiers(None, pin, binary, manifest, [dict(case="baseline", passed=True)])  # golden round-trip


def main_calls(tool, names):
    """Names called anywhere inside the tool's main(): tier code that main never calls does not gate anything."""
    import ast
    tree = ast.parse((TOOLS / tool).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", None)
              for c in ast.walk(main) if isinstance(c, ast.Call)}
    return {n for n in names if n in called}


def test_forward_main_runs_every_tier():
    assert main_calls("check_lfm3_forward.py", {"finish_tiers", "tier2a", "trace_manifest", "check_exact_logits",
                                                 "comparator_args"}) == {"finish_tiers", "tier2a", "trace_manifest",
                                                                        "check_exact_logits", "comparator_args"}


def test_context_and_http_batch_mains_use_their_tier_checks():
    assert main_calls("check_lfm3_context.py", {"compare_context_manifest", "exact_logit_steps"}) == {
        "compare_context_manifest", "exact_logit_steps"}
    assert main_calls("check_lfm3_http_batch.py", {"post_close_recovered", "resource_gate", "exact_accounting"}) == {
        "post_close_recovered", "resource_gate", "exact_accounting"}


def test_forward_comparator_args():
    import types
    gate = load_forward()
    tiers = gate.comparator_args(types.SimpleNamespace(native_tiers=True, activation_tolerance=True))
    assert tiers == ["--max-relative-rms", "inf", "--max-logit-error", "inf"]
    assert gate.comparator_args(types.SimpleNamespace(native_tiers=False, activation_tolerance=True)) == []
    assert gate.comparator_args(types.SimpleNamespace(native_tiers=False, activation_tolerance=False)) == ["--require-exact"]


def test_context_tier1_manifest_comparison():
    spec = importlib.util.spec_from_file_location("context_gate_tiers", TOOLS / "check_lfm3_context.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    pinned = {"a": dict(generated_ids=[1, 2], files={"0.logits.bf16": "x"}), "b": dict(generated_ids=None, files={"0.logits.bf16": "y"}),
              "_provenance": {"binary_sha256": "z"}}
    actual = {k: dict(v) for k, v in pinned.items() if not k.startswith("_")}
    gate.compare_context_manifest(pinned, actual)
    for damage in ("cases", "ids", "files"):
        changed = {k: dict(v) for k, v in actual.items()}
        if damage == "cases":
            del changed["b"]
        if damage == "ids":
            changed["a"]["generated_ids"] = [1, 3]
        if damage == "files":
            changed["b"]["files"] = {"0.logits.bf16": "w"}
        with pytest.raises(RuntimeError, match="tier 1"):
            gate.compare_context_manifest(pinned, changed)


def test_tier_code_is_wired_where_it_gates():
    # Calls that exist but are bypassed or lose their argument gate nothing (mutations found by review).
    forward = (TOOLS / "check_lfm3_forward.py").read_text()
    assert "    if args.native_tiers:\n        finish_tiers(args.pin, args.golden, args.binary, manifest, fidelity)" in forward
    http = (TOOLS / "check_lfm3_http_batch.py").read_text()
    assert "resource_gate(baseline,observed,args.require_exact_accounting)" in http
    assert "post_close_recovered(empty_gpu,post_close,args.require_exact_accounting)" in http
    runner = (TOOLS / "thor/run_native_gates.sh").read_text()
    assert "${metrics:+--require-exact-accounting}" in runner
    assert 'grep -c "lfm::gemm::"' in runner and '[ $cublas = 0 ] && [ $gemm = 3 ]' in runner
    context = (TOOLS / "check_lfm3_context.py").read_text()
    assert 'require(exact or not args.native_tiers' in context
