"""CPU-only contract tests for the context gate; not GPU numerical evidence."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_lfm3_context", ROOT / "tools/check_lfm3_context.py")
context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context)


def bf16(*bits):
    return struct.pack("<" + "H" * len(bits), *bits)


def test_suite_covers_exact_capacity_and_windows():
    full = context.cases("full")
    assert {length for _, length, _, _ in full} >= {1, 255, 256, 257, 1024, 4095, 4096}
    assert all(1 <= length + decodes <= 4096 for _, length, decodes, _ in full)
    assert any(length == 255 and decodes == 2 for _, length, decodes, _ in full)
    assert any(length == 4095 and decodes == 1 and free for _, length, decodes, free in full)
    assert max(length + decodes for _, length, decodes, _ in context.cases("smoke")) < 4096


def test_exact_logits_default():
    assert context.compare_logits(bf16(0x3f80, 0x4000), bf16(0x3f80, 0x4000), 2) == {
        "bitwise_equal": True, "max_abs": 0.0, "values": 2}


@pytest.mark.parametrize("reference,native,vocab", [
    (bf16(0x3f80), bf16(0x3f81), 1),
    (bf16(0), bf16(0x8000), 1),  # strict means bits, including signed zero
    (b"", b"", 1),
    (bf16(0x3f80), bf16(0x3f80, 0x3f80), 1),
    (bf16(0x7f80), bf16(0x7f80), 1),
    (bf16(0x7fc0), bf16(0x7fc0), 1),
    (bf16(0x3f80), bf16(0xff80), 1),
])
def test_bad_logits_fail_closed(reference, native, vocab):
    with pytest.raises(RuntimeError):
        context.compare_logits(reference, native, vocab)


def test_tolerance_is_explicit_and_does_not_change_exact_flag():
    reference, native = bf16(0x3f80), bf16(0x3f81)
    assert not context.compare_logits(reference, native, 1, atol=0.008)["bitwise_equal"]
    assert not context.compare_logits(reference, native, 1, rtol=0.008)["bitwise_equal"]
    with pytest.raises(RuntimeError):
        context.compare_logits(reference, native, 1, atol=0.007)


def test_rejection_requires_config_only_and_precise_diagnostics(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    calls = []
    diagnostics = {name: diagnostic for name, _, diagnostic in context.invalid_requests()}

    def run(command, **kwargs):
        stub, request, output = map(Path, command[2:])
        assert (stub / "config.json").exists()
        assert not (stub / "model.safetensors").exists()
        assert not output.exists()
        assert kwargs["timeout"] == 30
        calls.append(json.loads(request.read_text()))
        return SimpleNamespace(returncode=1, stdout="", stderr=diagnostics[request.name.removesuffix(".request.json")])

    accepted = context.reject_before_weights(Path("runner"), checkpoint, tmp_path, {}, 30, run=run)
    assert len(accepted) == len(calls) == 12


def test_preupload_option_validation_diagnostics():
    requests = {name: (request, diagnostic) for name, request, diagnostic in context.invalid_requests()}
    assert requests['capacity0'] == (dict(input_ids=[1], capacity=0), 'capacity must be 1..32768')
    assert requests['capacity4097'][0] == dict(input_ids=[1], capacity=4097)
    assert '1..4096' in requests['capacity4097'][1]
    assert requests["warmup_bool"] == (dict(input_ids=[1], warmup_runs=True),
                                       "configuration field must be a number")
    assert requests["warmup11"] == (dict(input_ids=[1], warmup_runs=11), "at most 10 warmup runs")
    assert requests["reset_number"] == (dict(input_ids=[1], check_reset=1),
                                        "configuration field must be boolean")
    assert requests["trace_logits_number"] == (dict(input_ids=[1], trace_logits_only=1),
                                              "configuration field must be boolean")


@pytest.mark.parametrize("returncode,stderr,write_output", [
    (1, "cannot open model.safetensors", False),
    (0, "capacity must be 1..32768", False),
    (1, "capacity must be 1..32768", True),
])
def test_missing_weights_failure_is_not_request_validation(tmp_path, returncode, stderr, write_output):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")

    def run(command, **kwargs):
        if write_output:
            Path(command[-1]).write_text("{}")
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    with pytest.raises(RuntimeError, match="did not reject before weights"):
        context.reject_before_weights(Path("runner"), checkpoint, tmp_path, {}, 30, run=run)


@pytest.mark.parametrize("reset_checked,reset_exact", [(False, True), (True, False), (None, None)])
def test_run_native_requires_reset_evidence(tmp_path, monkeypatch, reset_checked, reset_exact):
    def run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(dict(input_ids=[1], reset_checked=reset_checked,
                                                   reset_exact_logits_ok=reset_exact)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(context.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="reset not verified"):
        context.run_native(Path("runner"), Path("checkpoint"), tmp_path,
                           dict(input_ids=[1], check_reset=True), "teacher", {}, 30)


def test_help_needs_no_gpu_or_model_packages():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/check_lfm3_context.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--suite" in result.stdout


@pytest.mark.parametrize("damage", [None, "tokens", "logits", "unexpected_activation",
                                    "exact", "exact-partial", "exact-not-rounding"])
def test_main_mocked_end_to_end(tmp_path, monkeypatch, damage):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ["config.json", "modeling_lfm3.py", "lfm3_conv.py", "model.safetensors"]:
        (checkpoint / name).write_text("{}")
    binary = tmp_path / "runner"
    binary.write_text("mock")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["check", "--checkpoint", str(checkpoint), "--binary", str(binary),
                                     "--output", str(output), "--suite", "smoke"])
    monkeypatch.setattr(context, "reject_before_weights", lambda *a: ["mocked"])
    monkeypatch.setattr(context, "cases", lambda suite: [("mock", 1, 1, True)])

    def reference(checkpoint, output, selected):
        directory = output / "mock/reference"
        directory.mkdir(parents=True)
        for step in range(2):
            (directory / f"{step}.logits.bf16").write_bytes(bf16(0x3f80, 0x4000))
        return {"mock": (dict(input_ids=[1], teacher_ids=[1], capacity=2, check_reset=True),
                          dict(teacher_argmax=[1, 1], generated_ids=[1, 1]))}, dict(
                              layer_types=["conv"], vocab=2, attention="eager", dtype="bfloat16")

    def native(binary, checkpoint, directory, request, label, env, timeout):
        if label == "teacher":
            assert request["trace_logits_only"] is True
            trace = Path(request["trace_directory"])
            assert list(trace.iterdir()) == []
            for step in range(2):
                (trace / f"{step}.logits.bf16").write_bytes(bf16(0x3f80, 0x4001 if damage == "logits" else 0x4000))
            if damage == "unexpected_activation":
                (trace / "unanticipated.bf16").write_bytes(b"activation")
            # G4: an FP32-logit engine adds STEP.logits.f32; [1.0, 2.0] rounds exactly to the BF16 above, while
            # [1.0, 2.5] does not. Exact traces are accepted only for every step and only as the .bf16's source.
            if damage and damage.startswith("exact"):
                for step in range(1 if damage == "exact-partial" else 2):
                    value = 2.5 if damage == "exact-not-rounding" and step == 1 else 2.0
                    (trace / f"{step}.logits.f32").write_bytes(struct.pack("<ff", 1.0, value))
        return dict(teacher_argmax=[1, 0] if damage == "tokens" else [1, 1],
                    generated_ids=[] if label == "teacher" else [1, 1], decode_forward_calls=1)

    monkeypatch.setattr(context, "make_references", reference)
    monkeypatch.setattr(context, "run_native", native)
    if damage and damage != "exact":
        with pytest.raises(RuntimeError):
            context.main()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "failed"
        assert summary["failure"]["case"] == "mock"
        assert summary["cases"][0]["status"] == "failed"
        assert (output / "reference-manifest.json").exists()
        if damage in ("tokens", "logits"):
            assert summary["failure"]["phase"] == "compare_logits"
            assert summary["failure"]["step"] == (1 if damage == "tokens" else 0)
            assert "mock step=" in summary["failure"]["message"]
    else:
        context.main()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "passed"
        assert summary["atol"] == summary["rtol"] == 0
        assert not summary["maximum4096_checked"]
        assert summary["cases"][0]["greedy_final_slot_consumed"]
        original_summary = (output / "summary.json").read_bytes()
        binary.write_text("rebuilt mock binary")
        rerun = tmp_path / "rerun"
        monkeypatch.setattr(sys, "argv", ["check", "--checkpoint", str(checkpoint), "--binary", str(binary),
                                         "--output", str(rerun), "--suite", "smoke", "--reuse-reference", str(output)])

        def must_not_load_oracle(*args):
            raise AssertionError("reusing references must not load the model")

        monkeypatch.setattr(context, "make_references", must_not_load_oracle)
        context.main()
        resumed = json.loads((rerun / "summary.json").read_text())
        assert resumed["status"] == "passed"
        assert resumed["binary_sha256"] != summary["binary_sha256"]
        assert resumed["reuse_reference"] == str(output.resolve())
        assert (output / "summary.json").read_bytes() == original_summary


def reference_bundle(tmp_path):
    checkpoint, source, output = (tmp_path / name for name in ("checkpoint", "previous", "rerun"))
    checkpoint.mkdir()
    output.mkdir()
    directory = source / "case/reference"
    directory.mkdir(parents=True)
    for name in context.SOURCE_FILES + ["model.safetensors"]:
        (checkpoint / name).write_text("unchanged")
    for step in range(2):
        (directory / f"{step}.logits.bf16").write_bytes(bf16(0x3f80, 0x4000))
    request = dict(input_ids=[1], teacher_ids=[1], capacity=2, check_reset=True)
    expected = dict(teacher_argmax=[1, 1], generated_ids=[1, 1])
    metadata = dict(attention="eager", dtype="bfloat16", vocab=2, layer_types=["conv"])
    references = {"case": (request, expected)}
    context.save_reference_manifest(checkpoint, source, references, metadata)
    return checkpoint, source, output, references, metadata


def test_reuse_reference_is_verified_without_model_loading(tmp_path):
    checkpoint, source, output, references, metadata = reference_bundle(tmp_path)
    actual, actual_metadata = context.reuse_references(checkpoint, source, output, [("case", 1, 1, True)])
    assert actual == references and actual_metadata == metadata
    assert (output / "case/reference/1.logits.bf16").read_bytes() == bf16(0x3f80, 0x4000)
    assert (source / "reference-manifest.json").exists()


@pytest.mark.parametrize("damage", ["source", "weights", "logits", "missing", "shape", "oracle", "mode"])
def test_reuse_reference_fails_closed(tmp_path, damage):
    checkpoint, source, output, _, _ = reference_bundle(tmp_path)
    manifest_path = source / "reference-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "source":
        (checkpoint / "modeling_lfm3.py").write_text("different source")
    elif damage == "weights":
        (checkpoint / "model.safetensors").write_text("different weights")
    elif damage == "logits":
        (source / "case/reference/1.logits.bf16").write_bytes(bf16(0x3f81, 0x4000))
    elif damage == "missing":
        (source / "case/reference/1.logits.bf16").unlink()
    elif damage == "shape":
        manifest["cases"]["case"]["request"]["capacity"] = 3
    elif damage == "oracle":
        manifest["cases"]["case"]["request"]["teacher_ids"] = [0]
    elif damage == "mode":
        manifest["reference"]["attention"] = "sdpa"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        context.reuse_references(checkpoint, source, output, [("case", 1, 1, True)])


def test_legacy_references_require_explicit_trust(tmp_path):
    checkpoint, source, output, _, _ = reference_bundle(tmp_path)
    (source / "reference-manifest.json").unlink()
    with pytest.raises(RuntimeError, match="--allow-legacy-reference"):
        context.reuse_references(checkpoint, source, output, [("case", 1, 1, True)])


def test_legacy_adoption_uses_only_cpu_processor_and_checks_saved_request(tmp_path, monkeypatch):
    checkpoint, source, output, _, _ = reference_bundle(tmp_path)
    (source / "reference-manifest.json").unlink()
    (checkpoint / "config.json").write_text(json.dumps(dict(text_config=dict(layer_types=["conv"], vocab_size=2))))
    (source / "case/reference.json").write_text(json.dumps(dict(teacher_argmax=[1, 1], generated_ids=[1, 1])))
    request = dict(input_ids=[1], teacher_ids=[1], capacity=2, check_reset=True, suppress_tokens=[], stop_tokens=[])
    (source / "case/teacher-request.json").write_text(json.dumps(request))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1]))
    fake = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    actual, metadata = context.reuse_references(checkpoint, source, output, [("case", 1, 1, True)], allow_legacy=True)
    assert actual["case"][0] == request
    assert "legacy-adopted" in metadata["provenance"]


def test_native_tiers_pin_then_golden(tmp_path, monkeypatch):
    # G4 tier 1: --pin writes a manifest of the teacher logit traces and ids; --golden must match it bit for bit.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ["config.json", "modeling_lfm3.py", "lfm3_conv.py", "model.safetensors"]:
        (checkpoint / name).write_text("{}")
    binary = tmp_path / "runner"
    binary.write_text("mock")
    state = {"logit": 0x4000, "exact": 2.0}
    monkeypatch.setattr(context, "reject_before_weights", lambda *a: ["mocked"])
    monkeypatch.setattr(context, "cases", lambda suite: [("mock", 1, 1, True)])

    def reference(checkpoint, output, selected):
        directory = output / "mock/reference"
        directory.mkdir(parents=True)
        for step in range(2):
            (directory / f"{step}.logits.bf16").write_bytes(bf16(0x3f80, 0x4000))
        return {"mock": (dict(input_ids=[1], teacher_ids=[1], capacity=2, check_reset=True),
                          dict(teacher_argmax=[1, 1], generated_ids=[1, 1]))}, dict(
                              layer_types=["conv"], vocab=2, attention="eager", dtype="bfloat16")

    def native(binary, checkpoint, directory, request, label, env, timeout):
        if label == "teacher":
            trace = Path(request["trace_directory"])
            for step in range(2):
                # A native engine differs from eager in its logits (here 0x4001 vs 0x4000) but keeps the argmax, and
                # writes its exact FP32 logits too (bits 0x40008001 and 0x40008002 both round to 0x4001).
                (trace / f"{step}.logits.bf16").write_bytes(bf16(0x3f80, state["logit"]))
                if not state.get("no_f32"):
                    (trace / f"{step}.logits.f32").write_bytes(struct.pack("<ff", 1.0, state["exact"]))
        return dict(teacher_argmax=[1, 1], generated_ids=[] if label == "teacher" else [1, 1], decode_forward_calls=1)

    monkeypatch.setattr(context, "make_references", reference)
    monkeypatch.setattr(context, "run_native", native)
    pin = tmp_path / "pin.json"
    as_float = lambda bits: struct.unpack("<f", struct.pack("<I", bits))[0]
    state.update(logit=0x4001, exact=as_float(0x40008001))
    for run, flag in (("pin", ["--pin", str(pin)]), ("same", ["--golden", str(pin)])):
        monkeypatch.setattr(sys, "argv", ["check", "--checkpoint", str(checkpoint), "--binary", str(binary),
                                         "--output", str(tmp_path / run), "--suite", "smoke", "--native-tiers", *flag])
        context.main()
        assert json.loads((tmp_path / run / "summary.json").read_text())["status"] == "passed"
    pinned = json.loads(pin.read_text())
    assert set(pinned["mock"]["files"]) == {"0.logits.bf16", "1.logits.bf16", "0.logits.f32", "1.logits.f32"}
    assert "binary_sha256" in pinned["_provenance"]
    # A regression only in the exact logits (the .bf16 rounding is unchanged) must still fail tier 1.
    state.update(exact=as_float(0x40008002))
    monkeypatch.setattr(sys, "argv", ["check", "--checkpoint", str(checkpoint), "--binary", str(binary),
                                     "--output", str(tmp_path / "changed"), "--suite", "smoke", "--native-tiers", "--golden", str(pin)])
    with pytest.raises(RuntimeError, match="tier 1"):
        context.main()

    # Tier mode requires the exact FP32 logit traces: a run without them fails before tier 1.
    state.update(exact=as_float(0x40008001), no_f32=True)
    monkeypatch.setattr(sys, "argv", ["check", "--checkpoint", str(checkpoint), "--binary", str(binary),
                                     "--output", str(tmp_path / "no-f32"), "--suite", "smoke", "--native-tiers", "--golden", str(pin)])
    with pytest.raises(RuntimeError, match="requires the engine's exact"):
        context.main()
