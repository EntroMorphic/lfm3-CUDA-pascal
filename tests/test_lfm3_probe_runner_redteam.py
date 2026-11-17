"""Adversarial wrapper lifecycle/metadata regressions, no GPU."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lfm3_owned_process import live_group
from test_lfm3_decode_batch_gate import fixture

PATH=Path(__file__).resolve().parents[1]/"tools/check_lfm3_decode_batch.py"
SPEC=importlib.util.spec_from_file_location("probe_runner",PATH);gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def sleeper(tmp_path):
    path=tmp_path/"probe"
    path.write_text(f'''#!{sys.executable}
import os,pathlib,sys,time
d=pathlib.Path(sys.argv[3]);d.mkdir()
(d/"group").write_text(str(os.getpgrp()))
time.sleep(30)
''')
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("signals",[1,5])
def test_sigterm_wrapper_does_not_orphan_gpu_group(tmp_path,signals):
    probe=sleeper(tmp_path);output=tmp_path/"evidence";group=None
    command=[sys.executable,str(PATH),"--binary",str(probe),"--checkpoint",str(tmp_path),"--output",str(output),"--fixture","scratch"]
    controller=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        deadline=time.monotonic()+3
        marker=output/"scratch/group"
        while time.monotonic()<deadline:
            try:group=int(marker.read_text());break
            except (OSError,ValueError):time.sleep(0.005)
        assert group is not None
        for _ in range(signals):
            try:controller.terminate()
            except ProcessLookupError:break
        controller.wait(timeout=3)
        assert live_group(group)==[]
    finally:
        if group is not None and live_group(group):os.killpg(group,signal.SIGKILL)
        if controller.poll() is None:os.killpg(controller.pid,signal.SIGKILL)
        controller.wait(timeout=3)


def test_post_spawn_report_failure_still_cleans_owned_group(tmp_path,monkeypatch):
    probe=sleeper(tmp_path);output=tmp_path/"evidence";groups=[]
    monkeypatch.setattr(sys,"argv",[str(PATH),"--binary",str(probe),"--checkpoint",str(tmp_path),"--output",str(output),"--fixture","scratch"])
    original=Path.write_text
    def write(path,text,*args,**kwargs):
        if path==output/"summary.json" and not groups:
            record=json.loads(text)
            if record["records"] and "owned_process_group" in record["records"][0]:
                groups.append(record["records"][0]["owned_process_group"])
                raise OSError("injected post-spawn report failure")
        return original(path,text,*args,**kwargs)
    monkeypatch.setattr(Path,"write_text",write)
    try:
        with pytest.raises(OSError,match="post-spawn"):
            gate.main()
        assert groups and live_group(groups[0])==[]
    finally:
        for group in groups:
            if live_group(group):os.killpg(group,signal.SIGKILL)


@pytest.mark.parametrize("damage",["wrong-mode","wrong-owner-count","value-not-bits","worker-integration","bad-geometry","bad-coverage"])
def test_probe_metadata_cannot_claim_wrong_mode_or_coverage(damage):
    summary,rows=copy.deepcopy(fixture("scratch"))
    # Add the actual schema if an older fake fixture does not supply it yet.
    summary.update(reference_mode="normal",gpu_runner_owners=4,worker_integration=False)
    for row in rows:
        if "exact_logits" in row:row["logit_comparison"]="float_bits"
    if not any("fixture" in row for row in rows):
        rows.insert(0,dict(fixture="scratch",reference_mode="normal",gpu_runner_owners=4,capacities=[12,300],prompt_lengths=[3,257],
                          expected_comparisons=59,expected_paired_calls=10,expected_retired_checks=35,expected_rejections=11,expected_reset_checks=2,standard_six_step_sequence=True))
        rows.append(dict(case="fixture-coverage",shifted_window=True,small_softmax=True,large_softmax=False,B_exceeded_A_serial_scratch=True,tiled_prefill=False))
    header=next(row for row in rows if "fixture" in row)
    coverage=next(row for row in rows if row.get("case")=="fixture-coverage")
    if damage=="wrong-mode":summary["reference_mode"]="compact";header["reference_mode"]="compact"
    elif damage=="wrong-owner-count":summary["gpu_runner_owners"]=2;header["gpu_runner_owners"]=2
    elif damage=="value-not-bits":next(row for row in rows if "exact_logits" in row)["logit_comparison"]="float_value"
    elif damage=="worker-integration":summary["worker_integration"]=True
    elif damage=="bad-geometry":header["capacities"]=[2,300]
    elif damage=="bad-coverage":coverage.update(shifted_window=False,small_softmax=False,B_exceeded_A_serial_scratch=False)
    with pytest.raises(AssertionError):gate.validate_case("scratch",summary,rows)
