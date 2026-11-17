import copy
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
PATH=Path(__file__).resolve().parents[1]/'tools/check_lfm3_http_batch.py'
SPEC=importlib.util.spec_from_file_location('http_group_gate',PATH)
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


@pytest.mark.parametrize('damage',[None,'serial','wrong-lane','bool-id','cancelled','fake-pairs','missing-lane'])
def test_http_group_requires_exact_two_lanes_and_real_pairs(damage):
    lane=dict(generated_ids=[1,2],cancelled=False,cache_valid=True,gpu_batching=True,paired_decode_calls=1)
    response=dict(paired_decode_calls=1,gpu_batching=True,results=[copy.deepcopy(lane),copy.deepcopy(lane)])
    if damage=='serial':response.update(paired_decode_calls=0,gpu_batching=False)
    elif damage=='wrong-lane':response['results'][1]['generated_ids']=[4]
    elif damage=='bool-id':response['results'][1]['generated_ids']=[True,2]
    elif damage=='cancelled':response['results'][1]['cancelled']=True
    elif damage=='fake-pairs':response['results'][1]['paired_decode_calls']=0
    elif damage=='missing-lane':response['results'].pop()
    if damage:
        with pytest.raises(AssertionError):gate.validate_response(response,[1,2])
    else:gate.validate_response(response,[1,2])


@pytest.mark.parametrize('damage',[None,'server-rss','worker-rss','server-fds','worker-fds','gpu-free','gpu-total','gpu-bool'])
def test_fixed_host_and_device_resource_budgets(damage):
    baseline=dict(server=dict(rss_bytes=1000,descriptors=5),worker=dict(rss_bytes=2000,descriptors=6),
                  gpu=dict(cuda_free_bytes=1024**3,cuda_total_bytes=2*1024**3))
    sample=copy.deepcopy(baseline)
    if damage and damage.endswith('-rss'):sample[damage.split('-')[0]]['rss_bytes']+=32*1024**2+1
    elif damage and damage.endswith('-fds'):sample[damage.split('-')[0]]['descriptors']+=3
    elif damage=='gpu-free':sample['gpu']['cuda_free_bytes']-=128*1024**2+1
    elif damage=='gpu-total':sample['gpu']['cuda_total_bytes']+=1
    elif damage=='gpu-bool':sample['gpu']['cuda_free_bytes']=True
    if damage:
        with pytest.raises(AssertionError):gate.resource_gate(baseline,sample)
    else:gate.resource_gate(baseline,sample)


@pytest.mark.parametrize('damage',[None,'host-noise','growth','accounting-lost','invalid'])
def test_exact_engine_buffer_accounting_decides_when_present(damage):
    # G4: with LFM_ALLOCATION_METRICS=1 the worker reports its engine-buffer bytes; growth fails, device-wide noise does not.
    gpu=dict(cuda_free_bytes=70<<30,cuda_total_bytes=128<<30,app_live_bytes=128956672,app_metrics_valid=True)
    baseline={'server':dict(rss_bytes=1<<28,descriptors=8),'worker':dict(rss_bytes=1<<32,descriptors=43),'gpu':dict(gpu)}
    sample=copy.deepcopy(baseline)
    if damage=='host-noise':sample['gpu']['cuda_free_bytes']-=512<<20  # far beyond the 128 MiB device-wide budget
    if damage=='growth':sample['gpu']['app_live_bytes']+=2
    if damage=='accounting-lost':del sample['gpu']['app_live_bytes']
    if damage=='invalid':sample['gpu']['app_metrics_valid']=False
    if damage in ('growth','accounting-lost','invalid'):
        with pytest.raises(AssertionError,match='engine-buffer'):gate.resource_gate(baseline,sample)
    else:gate.resource_gate(baseline,sample)


@pytest.mark.parametrize('case,ok',[('exact-equal',True),('exact-leak',False),('exact-lost',False),
                                    ('device-within',True),('device-beyond',False)])
def test_post_close_recovery(case,ok):
    exact=dict(cuda_free_bytes=70<<30,app_live_bytes=0,app_metrics_valid=True)
    device=dict(cuda_free_bytes=70<<30)
    empty,post={'exact-equal':(exact,dict(exact,cuda_free_bytes=60<<30)),  # device-wide noise ignored
                'exact-leak':(exact,dict(exact,app_live_bytes=4096)),
                'exact-lost':(exact,dict(cuda_free_bytes=70<<30)),
                'device-within':(device,dict(cuda_free_bytes=(70<<30)-(16<<20))),
                'device-beyond':(device,dict(cuda_free_bytes=(70<<30)-(16<<20)-1))}[case]
    assert gate.post_close_recovered(empty,post)[0] is ok


def test_required_exact_accounting_never_falls_back():
    # G4: with --require-exact-accounting (native builds), missing or invalid accounting fails; it never degrades to the
    # device-wide check, even when the device-wide reading would pass.
    device=dict(cuda_free_bytes=70<<30,cuda_total_bytes=128<<30)
    baseline={'server':dict(rss_bytes=1<<28,descriptors=8),'worker':dict(rss_bytes=1<<32,descriptors=43),'gpu':dict(device)}
    with pytest.raises(AssertionError,match='required'):gate.resource_gate(baseline,copy.deepcopy(baseline),True)
    gate.resource_gate(baseline,copy.deepcopy(baseline))  # without the requirement: device-wide, passes
    invalid=dict(device,app_live_bytes=0,app_metrics_valid=False)
    assert gate.post_close_recovered(invalid,dict(device,app_live_bytes=4096,app_metrics_valid=False),True)[0] is False
    assert gate.post_close_recovered(device,dict(device),True)[0] is False
    assert gate.post_close_recovered(device,dict(device))[0] is True
