"""GPU gate termination runs transport cleanup; fake host worker only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from lfm3_owned_process import live_group


def test_native_group_gate_sigterm_cleans_resident_worker(tmp_path):
    tool=Path(__file__).resolve().parents[1]/'tools/check_lfm3_batch_worker.py'
    marker=tmp_path/'worker.group'
    ready=dict(ready=True,protocol=2,max_sessions=2,aggregate_capacity=32768,gpu_batching=False,
               cancellation=True,decode_batching=True)
    worker=tmp_path/'worker'
    worker.write_text(f'''#!{sys.executable}
import json,os,pathlib,sys,time
print({json.dumps(json.dumps(ready))},flush=True)
sys.stdin.readline()
pathlib.Path({str(marker)!r}).write_text(str(os.getpgrp()))
time.sleep(30)
''')
    worker.chmod(0o700)
    oracle=tmp_path/'oracle';oracle.write_text('fixture')
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(input_ids=[1])))
    output=tmp_path/'evidence';group=None
    gate=subprocess.Popen([sys.executable,str(tool),'--checkpoint',str(tmp_path),'--worker',str(worker),
        '--oracle',str(oracle),'--request',str(request),'--output',str(output)],
        start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            try:group=int(marker.read_text());break
            except (OSError,ValueError):time.sleep(0.005)
        assert group is not None
        gate.terminate();gate.wait(timeout=5)
        assert live_group(group)==[]
        assert json.loads((output/'summary.json').read_text())['status']=='failed'
    finally:
        if group is not None and live_group(group):os.killpg(group,signal.SIGKILL)
        if gate.poll() is None:os.killpg(gate.pid,signal.SIGKILL)
        gate.wait(timeout=3)


@pytest.mark.parametrize('name',['check_lfm3_batch_worker','check_lfm3_http_batch'])
def test_group_gate_entrypoint_has_signal_guard(name):
    import importlib
    assert hasattr(importlib.import_module(name).main,'__wrapped__')
