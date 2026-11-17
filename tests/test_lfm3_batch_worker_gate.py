"""Host counterexamples for grouped native acceptance; no GPU proof."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
PATH=Path(__file__).resolve().parents[1]/"tools/check_lfm3_batch_worker.py"
SPEC=importlib.util.spec_from_file_location("batch_worker_gate",PATH)
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("damage",[None,"no-pairs","fake-true","missing-lane","id","bool-id","policy","seed","cancelled","cache","participation"])
def test_group_gate_rejects_serial_or_mismatched_outputs(damage):
    lanes=[dict(session_id="A"),dict(session_id="B",do_sample=True,seed=42)]
    expected=[dict(generated_ids=[1,2]),dict(generated_ids=[3,4])]
    results=[dict(ok=True,session_id=lane['session_id'],generated_ids=ref['generated_ids'][:],
                  do_sample=lane.get('do_sample',False),seed=lane.get('seed',0),cancelled=False,
                  cache_valid=True,paired_decode_calls=1,gpu_batching=True) for lane,ref in zip(lanes,expected)]
    result=dict(ok=True,paired_decode_calls=1,gpu_batching=True,results=copy.deepcopy(results))
    if damage=="no-pairs":result.update(paired_decode_calls=0,gpu_batching=False)
    elif damage=="fake-true":result['paired_decode_calls']=0
    elif damage=="missing-lane":result['results'].pop()
    elif damage=="id":result['results'][0]['generated_ids']=[8]
    elif damage=="bool-id":result['results'][0]['generated_ids']=[True,2]
    elif damage=="policy":result['results'][1]['do_sample']=False
    elif damage=="seed":result['results'][1]['seed']=43
    elif damage=="cancelled":result['results'][0]['cancelled']=True
    elif damage=="cache":result['results'][0]['cache_valid']=False
    elif damage=="participation":result['results'][0]['paired_decode_calls']=0
    if damage:
        with pytest.raises(AssertionError):gate.require_group(result,lanes,expected)
    else:
        assert gate.require_group(result,lanes,expected)==results


def test_trace_comparison_is_full_bitwise_and_rejects_escape(tmp_path):
    for name in ('1','2/lane0'):
        p=tmp_path/name;p.mkdir(parents=True)
        (p/'0.logits.bf16').write_bytes(b'\0\0'*128000)
    assert gate.compare_traces(tmp_path,1,'2/lane0',1)==1
    # Signed zero must not compare equal merely because numeric values do.
    (tmp_path/'2/lane0/0.logits.bf16').write_bytes(b'\0\x80'+b'\0\0'*127999)
    with pytest.raises(AssertionError):gate.compare_traces(tmp_path,1,'2/lane0',1)
    with pytest.raises(AssertionError):gate.compare_traces(tmp_path,1,'../elsewhere',1)


@pytest.mark.parametrize('damage',[None,'empty-before','lost-transcript','lost-committed','retained-cache','inflight'])
def test_cancelled_group_preserves_nonempty_committed_history(damage):
    before=dict(transcript_tokens=30,committed_consumed_tokens=29)
    after=dict(before,cache_valid=False,consumed_tokens=0,inflight=False)
    if damage=='empty-before':before.update(transcript_tokens=0,committed_consumed_tokens=0);after.update(before)
    elif damage=='lost-transcript':after['transcript_tokens']=0
    elif damage=='lost-committed':after['committed_consumed_tokens']=0
    elif damage=='retained-cache':after['consumed_tokens']=3
    elif damage=='inflight':after['inflight']=True
    if damage:
        with pytest.raises(AssertionError):gate.validate_cancelled_history(before,after)
    else:gate.validate_cancelled_history(before,after)


# G6 runtime evidence: the worker's reported bound must equal the oracle's count, on every lane.
@pytest.mark.parametrize("results,ok",[
    ([dict(live_token_count=125033,generated_ids=[1]),dict(live_token_count=125033,generated_ids=[125032])],True),
    ([dict(live_token_count=125033,generated_ids=[1]),dict(live_token_count=128000,generated_ids=[2])],False),
    ([dict(generated_ids=[1]),dict(live_token_count=125033,generated_ids=[2])],False),
    ([dict(live_token_count=125033,generated_ids=[125033])],False),
    ([],False),
])
def test_live_agreement_is_required_on_every_lane(results,ok):
    if ok:
        gate.require_live_agreement(results,125033)
    else:
        with pytest.raises(AssertionError):gate.require_live_agreement(results,125033)


def test_trace_comparison_covers_exact_fp32_traces(tmp_path):
    # G4: the native-GEMM build also writes STEP.logits.f32; serial and grouped must agree on those bits
    # too, and an exact trace on one side only is a mismatch.
    for p in (tmp_path/'1', tmp_path/'2/lane0'):
        p.mkdir(parents=True)
        (p/'0.logits.bf16').write_bytes(b'\0\0'*128000)
        (p/'0.logits.f32').write_bytes(b'\0\0\0\0'*128000)
    assert gate.compare_traces(tmp_path,1,'2/lane0',1)==1
    (tmp_path/'2/lane0/0.logits.f32').write_bytes(b'\1\0\0\0'+b'\0\0\0\0'*127999)
    with pytest.raises(AssertionError,match="exact FP32"):gate.compare_traces(tmp_path,1,'2/lane0',1)
    (tmp_path/'2/lane0/0.logits.f32').unlink()
    with pytest.raises(AssertionError,match="presence"):gate.compare_traces(tmp_path,1,'2/lane0',1)
