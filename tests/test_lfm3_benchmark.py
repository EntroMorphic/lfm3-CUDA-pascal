"""Benchmark acceptance contracts, with no GPU/SSH/model workloads."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools/benchmark_lfm3.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lfm3", TOOL)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def gate(binary_hash="a" * 64):
    return dict(binary_sha256=binary_hash, stop_check=True, suppression_check=True,
                final_cache_slot_checked=True, invalid_requests_rejected=9,
                cases=[dict(case="baseline", activation_checks=100, tokens=4,
                            max_relative_rms=0, max_logit_abs=0,
                            greedy_exact=True, teacher_argmax_exact=True, reset_exact_logits=True)])


@pytest.fixture
def native_request():
    return dict(input_ids=[1, 2], capacity=8, max_new_tokens=4,
                suppress_tokens=[11, 12], stop_tokens=[9, 10], warmup_runs=1, check_reset=False)


def sample(tokens=None, native=False):
    tokens = [3, 4, 5, 10] if tokens is None else tokens
    result = dict(generated_ids=tokens, prefill_seconds=0.2, decode_seconds=0.3,
                  decode_forward_calls=max(0, len(tokens) - 1))
    if native:
        result.update(backend="native-cuda-experimental", attention="eager-bf16",
                      input_ids=[1, 2], warmup_runs=1, tracing=False,
                      reset_checked=False, reset_exact_logits_ok=None)
    return result


@pytest.fixture
def reference(native_request):
    return dict(backend="Transformers", attention="eager", input_ids=native_request["input_ids"],
                suppress_tokens=native_request["suppress_tokens"], stop_tokens=native_request["stop_tokens"],
                generated_ids=[3, 4, 5, 10], text="mock, not inference evidence",
                benchmark=dict(warmup_runs=1, samples=[sample() for _ in range(3)]))


def summary(reference, request, samples=None):
    return bench.summarize(reference, [sample(native=True) for _ in range(3)] if samples is None else samples,
                           request, 100, "eager", "a" * 64)


def test_valid_gate():
    bench.validate_gate(gate(), "a" * 64)


@pytest.mark.parametrize("field,value", [
    ("cases", []), ("cases", None), ("cases", [None]),
    ("binary_sha256", "wrong"), ("stop_check", False), ("stop_check", 1),
    ("suppression_check", "true"), ("final_cache_slot_checked", None),
    ("invalid_requests_rejected", 0), ("invalid_requests_rejected", True),
])
def test_bad_gate_top_level(field, value):
    value_gate = gate()
    value_gate[field] = value
    with pytest.raises(ValueError):
        bench.validate_gate(value_gate, "a" * 64)


@pytest.mark.parametrize("field,value", [
    ("greedy_exact", False), ("teacher_argmax_exact", "false"), ("reset_exact_logits", 1),
    ("case", ""), ("tokens", 0), ("activation_checks", 0), ("tokens", True),
    ("max_relative_rms", float("nan")), ("max_logit_abs", float("inf")),
    ("max_relative_rms", -1), ("max_logit_abs", 0.001),
])
def test_bad_gate_case(field, value):
    value_gate = gate()
    value_gate["cases"][0][field] = value
    with pytest.raises(ValueError):
        bench.validate_gate(value_gate, "a" * 64)


def test_duplicate_gate_case():
    value_gate = gate()
    value_gate["cases"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        bench.validate_gate(value_gate, "a" * 64)


def test_empty_gates_rejected_under_optimized_python(tmp_path):
    binary = tmp_path / "binary"
    binary.write_text("#!/bin/false\n")
    binary.chmod(0o700)
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    path = tmp_path / "gates.json"
    value_gate = gate(bench.sha256(binary))
    value_gate["cases"] = []
    path.write_text(json.dumps(value_gate))
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-O", str(TOOL), "--checkpoint", str(checkpoint),
                             "--binary", str(binary), "--gates", str(path), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "nonempty" in result.stderr and "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("prefill_seconds", 0), ("prefill_seconds", -1), ("prefill_seconds", True),
    ("decode_seconds", 0), ("decode_seconds", -1), ("decode_seconds", "0.1"),
    ("decode_seconds", float("nan")), ("decode_seconds", float("inf")),
    ("decode_seconds", 5e-324), ("prefill_seconds", 10**1000),
    ("decode_forward_calls", -1), ("decode_forward_calls", True),
    ("decode_forward_calls", 3.0), ("decode_forward_calls", 2),
    ("generated_ids", []), ("generated_ids", [True]),
    ("generated_ids", [100, 10]), ("generated_ids", [11, 10]),
    ("generated_ids", [10, 3, 4, 10]), ("generated_ids", [3, 4]),
])
def test_bad_sample(native_request, field, value):
    row = sample()
    row[field] = value
    with pytest.raises(ValueError):
        bench.validate_sample(row, native_request, 100)


@pytest.mark.parametrize("field,value", [
    ("backend", "Transformers"), ("attention", "sdpa"), ("input_ids", [2]),
    ("input_ids", [True, 2]), ("warmup_runs", True), ("warmup_runs", 0),
    ("tracing", True), ("tracing", 0), ("reset_checked", True),
    ("reset_exact_logits_ok", True),
])
def test_bad_native_metadata(native_request, field, value):
    row = sample(native=True)
    row[field] = value
    with pytest.raises(ValueError):
        bench.validate_sample(row, native_request, 100, native=True)


def test_zero_decode_calls_no_division_or_invented_throughput(reference, native_request):
    reference["generated_ids"] = [10]
    reference["benchmark"]["samples"] = [sample([10]) for _ in range(3)]
    native = [sample([10], native=True) for _ in range(3)]
    for row in reference["benchmark"]["samples"] + native:
        row["decode_seconds"] = 0
    report = summary(reference, native_request, native)
    assert report["accepted_parity_benchmark"]
    assert report["native"]["decode_tok_s_median"] is None
    assert report["transformers"]["decode_tok_s_median"] is None


def test_exact_report_and_truthful_warmup_descriptions(reference, native_request):
    report = summary(reference, native_request)
    assert report["greedy_exact"] and report["accepted_parity_benchmark"]
    assert report["mismatches"] == [] and report["first_mismatch"] is None
    assert report["native"]["decode_tok_s_median"] == 10
    assert report["native_warmup_runs_per_process"] == 1
    assert report["reference_warmup_runs_total"] == 1
    assert report["reference_initial_oracle_generation"] is True
    assert report["reference_samples_same_process"] is True
    assert "warmup_runs_per_sample" not in report


@pytest.mark.parametrize("backend,index", [("native", 0), ("native", 1), ("native", 2),
                                           ("transformers", 0), ("transformers", 1), ("transformers", 2)])
def test_mismatch_in_any_sample_reported(reference, native_request, backend, index):
    native = [sample(native=True) for _ in range(3)]
    rows = native if backend == "native" else reference["benchmark"]["samples"]
    rows[index]["generated_ids"][1] = 6
    report = summary(reference, native_request, native)
    assert not report["greedy_exact"] and not report["accepted_parity_benchmark"]
    assert report["first_mismatch"] == 1
    assert report["mismatches"] == [dict(backend=backend, sample=index, first_mismatch=1,
                                          expected_length=4, actual_length=4, expected_token=4, actual_token=6)]


@pytest.mark.parametrize("actual,expected,index", [([3], [3, 4], 1), ([3, 4], [3], 1),
                                                  ([3, 5], [3, 4], 1), ([], [3], 0)])
def test_prefix_length_mismatch_is_not_lost(actual, expected, index):
    result = bench.mismatch(actual, expected)
    assert result["first_mismatch"] == index
    assert result["actual_length"] == len(actual)
    assert result["expected_length"] == len(expected)


def test_report_length_only_mismatch(reference, native_request):
    # Different legal stopping lengths are rejected with explicit length evidence.
    native_request["stop_tokens"] = [9, 10]
    reference["generated_ids"] = [3, 9]
    reference["benchmark"]["samples"] = [sample([3, 9]) for _ in range(3)]
    # A genuine prefix cannot continue after a stop; compare different permitted
    # policies directly at the reporting helper, never bypass stop validation.
    assert bench.mismatch([3, 9, 5], [3, 9])["first_mismatch"] == 2
    native = [sample([3, 4, 5, 10], native=True) for _ in range(3)]
    report = summary(reference, native_request, native)
    assert not report["accepted_parity_benchmark"]
    assert all(row["actual_length"] == 4 and row["expected_length"] == 2 for row in report["mismatches"])


@pytest.mark.parametrize("field,value", [("backend", "native"), ("attention", "sdpa"),
                                        ("text", None), ("benchmark", {}), ("generated_ids", [])])
def test_bad_reference_top_level(reference, native_request, field, value):
    reference[field] = value
    with pytest.raises(ValueError):
        summary(reference, native_request)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_exactly_three_samples_required(reference, native_request, count):
    with pytest.raises(ValueError):
        summary(reference, native_request, [sample(native=True) for _ in range(count)])
    reference["benchmark"]["samples"] = [sample() for _ in range(count)]
    with pytest.raises(ValueError):
        summary(reference, native_request)


def test_request_whitelist(reference, native_request):
    source = dict(native_request, teacher_ids=[3, 4, 5], trace_directory="do-not-forward", unknown="bad")
    source.pop("max_new_tokens")
    result = bench.make_request(source, reference, 4, 100)
    assert result == native_request


@pytest.mark.parametrize("field,value", [("teacher_ids", []), ("input_ids", []), ("input_ids", [True]),
                                        ("capacity", 4), ("suppress_tokens", [9]), ("stop_tokens", [])])
def test_bad_reference_request(reference, native_request, field, value):
    source = dict(native_request, teacher_ids=[3, 4, 5])
    source[field] = value
    with pytest.raises(ValueError):
        bench.make_request(source, reference, 4, 100)


@pytest.fixture
def args(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(text_config=dict(vocab_size=100))))
    binary = tmp_path / "native binary"
    binary.write_text("#!/bin/false\n")
    binary.chmod(0o700)
    gates = tmp_path / "gate.json"
    gates.write_text(json.dumps(gate(bench.sha256(binary))))
    return argparse.Namespace(checkpoint=checkpoint, binary=binary, gates=gates,
                              output=tmp_path / "new output", tokens=4, attention="eager")


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_output_overwrite_rejected(args, kind):
    if kind == "directory":
        args.output.mkdir()
    elif kind == "file":
        args.output.write_text("user work")
    else:
        args.output.symlink_to(args.checkpoint if kind == "symlink" else args.output.parent / "missing")
    with pytest.raises(ValueError, match="fresh"):
        bench.run(args)


@pytest.mark.parametrize("value", [0, -1, True, 4097])
def test_invalid_generation_limits_no_output(args, value):
    args.tokens = value
    with pytest.raises(ValueError):
        bench.run(args)
    assert not args.output.exists()


def fake_commands(monkeypatch, reference, request, mismatch_index=None, broken_reference=False):
    commands = []

    def fake_run(command, **kwargs):
        assert kwargs["check"] is True and kwargs["timeout"] == 3600
        commands.append(command)
        if "--benchmark" in command:
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            data = copy.deepcopy(reference)
            if broken_reference:
                data["benchmark"]["samples"] = []
            (directory / "reference.json").write_text(json.dumps(data))
            source = dict(request, teacher_ids=[3, 4, 5], trace_directory="unused")
            (directory / "request.json").write_text(json.dumps(source))
        else:
            assert command[1] == "--experimental"
            result = sample(native=True)
            if len(commands) - 2 == mismatch_index:
                result["generated_ids"][1] = 7
            Path(command[-1]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    return commands


def test_full_host_orchestration(args, reference, native_request, monkeypatch):
    commands = fake_commands(monkeypatch, reference, native_request)

    class Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs == dict(trust_remote_code=True, local_files_only=True)
            return types.SimpleNamespace(decode=lambda *a, **k: "mock text")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer))
    report = bench.run(args)
    assert report["accepted_parity_benchmark"]
    assert len(commands) == 4
    assert all(Path(command[0]).is_absolute() for command in commands)
    assert json.loads((args.output / "request.json").read_text()) == native_request
    assert (args.output / "summary.json").exists()


def test_later_native_mismatch_retains_rejected_summary(args, reference, native_request, monkeypatch):
    fake_commands(monkeypatch, reference, native_request, mismatch_index=2)
    with pytest.raises(ValueError, match="Greedy sequence mismatch"):
        bench.run(args)
    report = json.loads((args.output / "summary.json").read_text())
    assert not report["accepted_parity_benchmark"]
    assert report["mismatches"][0]["sample"] == 2
    assert not (args.output / "native-response.txt").exists()


def test_bad_oracle_stops_before_native(args, reference, native_request, monkeypatch):
    commands = fake_commands(monkeypatch, reference, native_request, broken_reference=True)
    with pytest.raises(ValueError):
        bench.run(args)
    assert len(commands) == 1
    assert not (args.output / "summary.json").exists()


@pytest.mark.parametrize("raw", [b"[]", b"{", b'{"a":1,"a":2}', b'{"a":NaN}',
                                  b'{"a":Infinity}', b"\xff"])
def test_bad_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        bench.read_json(path)


def test_oversized_json(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MAX_JSON_BYTES", 10)
    path = tmp_path / "big.json"
    path.write_text(" " * 11)
    with pytest.raises(ValueError):
        bench.read_json(path)


def test_json_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("{}")
    path = tmp_path / "link"
    path.symlink_to(target)
    with pytest.raises(OSError):
        bench.read_json(path)


@pytest.mark.parametrize("labels,claim", [(["eager-bf16"] * 3, "applies"),
                                          (["eager-fp32-scores"] * 3, "does not apply"),
                                          (["eager-bf16", "eager-fp32-scores", "eager-bf16"], None),
                                          (["eager-int8"] * 3, None)])
def test_native_engine_labels(reference, native_request, labels, claim):
    # G4: the native-GEMM engine reports eager-fp32-scores; its eager-parity claim does not apply (greedy agreement is
    # still measured). All samples must come from one engine, and unknown labels are rejected.
    samples = [dict(sample(native=True), attention=label) for label in labels]
    if claim is None:
        with pytest.raises(Exception):
            summary(reference, native_request, samples)
        return
    report = summary(reference, native_request, samples)
    assert report["native_attention"] == labels[0] and report["eager_parity_claim"].startswith(claim)
