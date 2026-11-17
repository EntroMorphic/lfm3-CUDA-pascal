"""Host-only benchmark contract tests: not GPU correctness/performance evidence."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SPEC = importlib.util.spec_from_file_location("group_benchmark", Path(__file__).resolve().parents[1] / "tools/check_lfm3_group_benchmark.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
REFERENCE = list(range(32))


def response(mode="group"):
    grouped = mode == "group"
    lanes = [dict(generated_ids=REFERENCE.copy(), cancelled=False, cache_valid=True,
                  gpu_batching=grouped, finish_reason="length", do_sample=bool(i), seed=42,
                  consumed_tokens=80, reused_tokens=0, text="answer") for i in range(2)]
    result = dict(results=lanes)
    if grouped:
        result.update(gpu_batching=True, paired_decode_calls=31)
        for lane in lanes:
            lane["paired_decode_calls"] = 31
    return result


@pytest.mark.parametrize("mode", ["serial-pair", "group"])
def test_valid_exact_policy(mode):
    gate.validate_response(response(mode), mode, REFERENCE, 80, "length")


@pytest.mark.parametrize("field,value", [
    ("generated_ids", [True] + REFERENCE[1:]), ("generated_ids", REFERENCE[:-1]),
    ("generated_ids", REFERENCE + [32]), ("generated_ids", [99] * 32),
    ("consumed_tokens", True), ("consumed_tokens", 79), ("reused_tokens", False),
    ("reused_tokens", 1), ("cancelled", 0), ("cache_valid", 1), ("gpu_batching", 1),
    ("paired_decode_calls", True), ("paired_decode_calls", 30), ("finish_reason", "stop"),
    ("do_sample", 1), ("seed", True), ("seed", 43), ("text", None),
])
def test_malformed_lane_rejected(field, value):
    result = response()
    result["results"][1][field] = value
    with pytest.raises(AssertionError):
        gate.validate_response(result, "group", REFERENCE, 80, "length")


@pytest.mark.parametrize("field,value", [("paired_decode_calls", True), ("paired_decode_calls", 0),
                                         ("paired_decode_calls", 32), ("gpu_batching", False),
                                         ("gpu_batching", 1), ("results", []), ("results", [None, None])])
def test_malformed_group_metadata(field, value):
    result = response()
    result[field] = value
    with pytest.raises(AssertionError):
        gate.validate_response(result, "group", REFERENCE, 80, "length")


def test_serial_cannot_claim_batching():
    result = response("serial-pair")
    result["results"][0]["gpu_batching"] = True
    with pytest.raises(AssertionError):
        gate.validate_response(result, "serial-pair", REFERENCE, 80, "length")


@pytest.mark.parametrize("reference", [[], REFERENCE[:-1], [False] + REFERENCE[1:], [-1] + REFERENCE[1:]])
def test_invalid_reference(reference):
    with pytest.raises(AssertionError):
        gate.validate_reference(reference)


def records(samples=20):
    return [dict(phase=phase, round=r, mode=mode, seconds=1 if mode == "group" else 2)
            for phase, count in (("warmup", 3), ("measured", samples))
            for r in range(count) for mode in gate.mode_order(r)]


@pytest.mark.parametrize("samples", [5, 20, 100])
def test_order_balance_and_real_coverage(samples):
    rows = records(samples)
    gate.validate_coverage(rows, samples)
    orders = [gate.mode_order(i)[0] for i in range(samples)]
    assert abs(orders.count("group") - orders.count("serial-pair")) <= 1
    assert [x["mode"] for x in rows[:6]].count("group") == 3
    assert [x["mode"] for x in rows[:6]].count("serial-pair") == 3


@pytest.mark.parametrize("damage", ["no-warmup", "missing-warmup-group", "missing-measured", "empty", "reorder", "duplicate"])
def test_missing_or_fake_coverage_fails(damage):
    rows = records()
    if damage == "no-warmup":
        rows = rows[6:]
    elif damage == "missing-warmup-group":
        rows.pop(1)
    elif damage == "missing-measured":
        rows.pop()
    elif damage == "empty":
        rows = []
    elif damage == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(AssertionError):
        gate.validate_coverage(rows, 20)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), None])
def test_non_observed_latencies_rejected(value):
    rows = records()
    rows[-1]["seconds"] = value
    with pytest.raises(AssertionError):
        gate.summarize(rows, 20)


@pytest.mark.parametrize("samples", [0, 4, 101, True])
def test_bounded_nonempty_samples(samples):
    with pytest.raises(AssertionError):
        gate.validate_coverage([], samples)


@pytest.mark.parametrize("damage", ["worker", "source", "missing", "empty", "malformed"])
def test_hash_drift_and_missing_hashes_fail(damage):
    start = {"worker": "a" * 64, "tools/lfm3_server.py": "b" * 64,
             "control_result": "c" * 64, "numerical_evidence": "d" * 64}
    end = start.copy()
    if damage in ("worker", "source"):
        end["worker" if damage == "worker" else "tools/lfm3_server.py"] = "c" * 64
    elif damage == "missing":
        end.pop("worker")
    elif damage == "empty":
        start = end = {}
    else:
        start["worker"] = end["worker"] = "invalid"
    with pytest.raises(AssertionError):
        gate.validate_hashes(start, end)


def test_summary_fixed_statistics_no_adaptive_speed_gate():
    rows = records()
    result = gate.summarize(rows, 20)
    assert result["serial-pair"] == dict(samples=20, median_seconds=2, p95_seconds=2,
                                         aggregate_output_tokens_per_second=32)
    assert result["group"]["aggregate_output_tokens_per_second"] == 64
    assert len(result["paired_ratios"]) == 20
    assert result["median_paired_serial_over_group_wall"] == 2
    for row in rows:
        if row["mode"] == "group":
            row["seconds"] = 1000  # A slower group still satisfies the measurement contract.
    assert gate.summarize(rows, 20)["median_paired_serial_over_group_wall"] == .002


def test_matching_hashes_pass():
    hashes = dict(worker="a" * 64, source="b" * 64, control_result="c" * 64, numerical_evidence="d" * 64)
    gate.validate_hashes(hashes, hashes.copy())


def test_hash_manifest_covers_wrappers_and_native_headers(tmp_path):
    worker = tmp_path / "worker"
    worker.write_bytes(b"host-only manifest fixture, never executed")
    control, evidence = tmp_path / "control.json", tmp_path / "evidence.json"
    control.write_text("{}")
    evidence.write_text("{}")
    hashes = gate.hash_inputs(worker, control, evidence)
    for path in ("tools/check_lfm3_group_benchmark.py", "tools/lfm3_server.py",
                 "tools/lfm3_worker_client.py", "tools/lfm3_text_frontend.py",
                 "tools/lfm3_owned_process.py", "tools/lfm3_chat.py",
                 "native/include/lfm/text_sessions.hpp", "native/src/lfm3_worker.cu"):
        assert path in hashes
    gate.validate_hashes(hashes, gate.hash_inputs(worker, control, evidence))
    worker.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        gate.validate_hashes(hashes, gate.hash_inputs(worker, control, evidence))


def numerical_evidence():
    return dict(status="passed", worker_sha256="a" * 64, bitwise_logit_steps=73,
                independent_non_greedy_logit_replay=True, aggregate32k_checked=True)


@pytest.mark.parametrize("steps", [73, 74, 100])
def test_hash_matched_numerical_evidence(steps):
    evidence = numerical_evidence()
    evidence["bitwise_logit_steps"] = steps
    gate.validate_numerical_evidence(evidence, "a" * 64)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("status", True), ("worker_sha256", "b" * 64),
    ("worker_sha256", None), ("bitwise_logit_steps", True), ("bitwise_logit_steps", 72),
    ("bitwise_logit_steps", 73.0), ("bitwise_logit_steps", "73"),
    ("independent_non_greedy_logit_replay", False), ("independent_non_greedy_logit_replay", 1),
    ("aggregate32k_checked", False), ("aggregate32k_checked", 1),
])
def test_unaccepted_numerical_evidence_rejected(field, value):
    evidence = numerical_evidence()
    evidence[field] = value
    with pytest.raises(AssertionError):
        gate.validate_numerical_evidence(evidence, "a" * 64)


@pytest.mark.parametrize("field", list(numerical_evidence()))
def test_missing_numerical_evidence_fields_rejected(field):
    evidence = numerical_evidence()
    del evidence[field]
    with pytest.raises(AssertionError):
        gate.validate_numerical_evidence(evidence, "a" * 64)


@pytest.mark.parametrize("evidence", [None, [], True, "passed"])
def test_malformed_numerical_evidence_rejected(evidence):
    with pytest.raises(AssertionError):
        gate.validate_numerical_evidence(evidence, "a" * 64)


def test_matching_ids_do_not_accept_numerical_proof_for_different_binary():
    gate.validate_response(response(), "group", REFERENCE, 80, "length")
    with pytest.raises(AssertionError):
        gate.validate_numerical_evidence(numerical_evidence(), "b" * 64)


@pytest.mark.parametrize("artifact", ["control_result", "numerical_evidence"])
def test_artifact_mutation_fails_end_hashes(tmp_path, artifact):
    paths = {name: tmp_path / name for name in ("worker", "control_result", "numerical_evidence")}
    for path in paths.values():
        path.write_text("original")
    start = gate.hash_inputs(**paths)
    paths[artifact].write_text("changed")
    with pytest.raises(AssertionError):
        gate.validate_hashes(start, gate.hash_inputs(**paths))


@pytest.mark.parametrize("artifact", ["control_result", "numerical_evidence"])
def test_missing_artifact_hash_not_accepted(artifact):
    hashes = dict(worker="a" * 64, source="b" * 64, control_result="c" * 64, numerical_evidence="d" * 64)
    del hashes[artifact]
    with pytest.raises(AssertionError):
        gate.validate_hashes(hashes, hashes.copy())
