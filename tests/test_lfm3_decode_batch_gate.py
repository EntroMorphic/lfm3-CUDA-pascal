"""Fixture/timeout counterexamples, not GPU batching evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
PATH=Path(__file__).resolve().parents[1]/"tools/check_lfm3_decode_batch.py"
SPEC=importlib.util.spec_from_file_location("paired_gate",PATH);gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def fixture(name,mode="normal"):
    counts=dict(scratch=(59,10,35,2),sliding=(47,17,9,2),softmax=(49,17,11,2),tiled=(49,17,11,2),upper=(5,1,1,0))
    n,p,r,s=counts[name]
    summary=dict(status="passed",fixture=name,reference_mode=mode,gpu_runner_owners=2 if mode=="compact" else 4,worker_integration=False,comparisons=n,expected_comparisons=n,paired_calls=p,
        expected_paired_calls=p,retired_checks=r,expected_retired_checks=r,reset_checks=s,rejections=11,dense_batch_rows=2)
    rows=[dict(exact_logits=True,logit_comparison="float_bits",exact_full_cache=True,cache_bytes=64,id=7,expected_id=7,cursor=8,expected_cursor=8,vocabulary=128000) for _ in range(n)]
    rows += [dict(rejected=True,metadata_unchanged=True,full_caches_unchanged=True) for _ in range(11)]
    rows += [dict(retired_lane="A",full_cache_unchanged_and_exact=True) for _ in range(r)]
    rows += [dict(case="reset",exact_full_cache=True) for _ in range(s)]
    geometry=dict(scratch=([12,300],[3,257]),sliding=([270,280],[254,256]),softmax=([2062,2074],[2046,2048]),tiled=([4112,4124],[4096,4098]),upper=([2,32766],[1,32764]))
    capacities,lengths=geometry[name]
    rows.append(dict(fixture=name,reference_mode=mode,gpu_runner_owners=2 if mode=="compact" else 4,capacities=capacities,prompt_lengths=lengths,
                     expected_comparisons=n,expected_paired_calls=p,expected_retired_checks=r,expected_rejections=11,expected_reset_checks=s,standard_six_step_sequence=name!="upper"))
    rows.append(dict(case="fixture-coverage",shifted_window=True,small_softmax=True,large_softmax=name in ("softmax","tiled","upper"),
                     B_exceeded_A_serial_scratch=name in ("scratch","upper"),tiled_prefill=name=="tiled"))
    return summary,rows


@pytest.mark.parametrize("name",["scratch","sliding","softmax","tiled","upper"])
@pytest.mark.parametrize("mode",["normal","compact"])
def test_complete_predeclared_fixture_records(name,mode):
    gate.validate_case(name,*fixture(name,mode),reference_mode=mode)


@pytest.mark.parametrize("damage",["summary-count","empty","missing-row","cache","cache-size","id","bool-id","cursor","rejection","retired","reset"])
def test_incomplete_or_incorrect_paired_records_are_rejected(damage):
    summary,rows=copy.deepcopy(fixture("scratch"))
    if damage=="summary-count":summary["comparisons"]=0;summary["expected_comparisons"]=0
    elif damage=="empty":rows=[]
    elif damage=="missing-row":rows.pop(0)
    elif damage=="cache":rows[0]["exact_full_cache"]=False
    elif damage=="cache-size":rows[0]["cache_bytes"]=0
    elif damage=="id":rows[0]["id"]=9
    elif damage=="bool-id":rows[0]["id"]=True;rows[0]["expected_id"]=True
    elif damage=="cursor":rows[0]["cursor"]=9
    elif damage=="rejection":next(r for r in rows if "rejected" in r)["full_caches_unchanged"]=False
    elif damage=="retired":next(r for r in rows if "retired_lane" in r)["full_cache_unchanged_and_exact"]=False
    elif damage=="reset":next(r for r in rows if r.get("case")=="reset")["exact_full_cache"]=False
    with pytest.raises(AssertionError):gate.validate_case("scratch",summary,rows)


def test_fixture_timeout_reaps_helper_and_does_not_advance(tmp_path,monkeypatch):
    from lfm3_owned_process import live_group
    probe=tmp_path/"fake-probe"
    probe.write_text(f'''#!{sys.executable}
import pathlib,subprocess,sys,time
directory=pathlib.Path(sys.argv[3]);directory.mkdir()
child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
(directory/"child.pid").write_text(str(child.pid))
time.sleep(30)
''')
    probe.chmod(0o700)
    output=tmp_path/"evidence"
    monkeypatch.setattr(sys,"argv",[str(PATH),"--binary",str(probe),"--checkpoint",str(tmp_path),"--output",str(output),
        "--fixture","scratch","--fixture","sliding","--timeout","0.3"])
    group=None
    try:
        assert gate.main()==86
        report=json.loads((output/"summary.json").read_text())
        group=report["records"][0]["owned_process_group"]
        assert report["status"]=="failed" and report["records"][0]["status"]=="timeout"
        assert report["records"][0]["group_cleanup_complete"] is True
        assert report["not_run"]==["sliding"] and not (output/"sliding").exists()
        assert live_group(group)==[]
    finally:
        if group is not None and live_group(group):os.killpg(group,9)
