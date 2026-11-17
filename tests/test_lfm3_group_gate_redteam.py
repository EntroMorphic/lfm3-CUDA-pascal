"""Independent host-only counterexamples, now required passing regressions."""
import importlib.util
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location("redteam_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load("check_lfm3_batch_worker")
http = load("check_lfm3_http_batch")


def group_fixture():
    lanes = [dict(session_id="A"), dict(session_id="B", do_sample=True, seed=42)]
    refs = [dict(generated_ids=[1, 2]), dict(generated_ids=[3, 4])]
    result = dict(ok=True, paired_decode_calls=1, gpu_batching=True, results=[
        dict(ok=True, session_id=lane["session_id"], generated_ids=ref["generated_ids"][:],
             do_sample=lane.get("do_sample", False), seed=lane.get("seed", 0),
             cancelled=False, cache_valid=True, paired_decode_calls=1, gpu_batching=True)
        for lane, ref in zip(lanes, refs)])
    return lanes, refs, result


def test_serial_group_trace_must_be_independent(tmp_path):
    trace = tmp_path / "serial"
    trace.mkdir()
    (trace / "0.logits.bf16").write_bytes(b"\0\0" * 128000)
    with pytest.raises(AssertionError):
        native.compare_traces(tmp_path, "serial", "serial", 1)


def test_zero_trace_steps_not_evidence(tmp_path):
    with pytest.raises(AssertionError):
        native.compare_traces(tmp_path, "absent-serial", "absent-group", 0)


def test_empty_serial_references_cannot_validate_native_group():
    lanes, _, result = group_fixture()
    result["results"] = [{}, {}]
    with pytest.raises(AssertionError):
        native.require_group(result, lanes, [])


def test_impossible_paired_count_not_real_work_proof():
    lanes, refs, result = group_fixture()
    result["paired_decode_calls"] = 999999
    for lane in result["results"]:
        lane["paired_decode_calls"] = 999999
    with pytest.raises(AssertionError):
        native.require_group(result, lanes, refs)


def http_fixture(reference):
    return dict(paired_decode_calls=1, gpu_batching=True, results=[
        dict(generated_ids=reference[:], cancelled=False, paired_decode_calls=1,
             gpu_batching=True, do_sample=False, seed=42, cache_valid=True)
        for _ in range(2)])


def test_http_boolean_participation_rejected():
    reference = list(range(32))
    response = http_fixture(reference)
    response["results"][1]["paired_decode_calls"] = True
    with pytest.raises(AssertionError):
        http.validate_response(response, reference)


def test_http_successful_group_requires_valid_cache():
    reference = list(range(32))
    response = http_fixture(reference)
    response["results"][1]["cache_valid"] = False
    with pytest.raises(AssertionError):
        http.validate_response(response, reference)


def test_http_empty_reference_not_a_group_proof():
    with pytest.raises(AssertionError):
        http.validate_response(http_fixture([]), [])
