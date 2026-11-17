"""Bounded control-pipe reader tests with a fake worker; no GPU coverage."""
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[1]/"tools/check_lfm3_worker.py"
SPEC=importlib.util.spec_from_file_location("worker_gate",PATH)
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def binary(tmp_path,body):
    path=tmp_path/"worker"
    path.write_text("#!/usr/bin/env python3\nimport json,sys,time\n"+body)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("body",[
    "print('{}',flush=True)",
    "print('{\"ready\":true,\"protocol\":true}',flush=True)",
    "print('[]',flush=True)",
    "sys.stdout.write('x'*5000000);sys.stdout.flush()",
    "time.sleep(10)",
])
def test_bad_handshake_is_bounded_and_cleaned_up(tmp_path,body):
    import os
    with pytest.raises((RuntimeError,ValueError,TimeoutError)):
        gate.Worker(binary(tmp_path,body),tmp_path,tmp_path/"stderr.log",dict(os.environ),timeout=0.3)


def test_valid_rpc_and_failure_acknowledgment(tmp_path):
    import os
    exe=binary(tmp_path,'''
print('{"ready":true,"protocol":2}',flush=True)
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({"ok":r.get("op")=="good","rpc_id":r["rpc_id"]}),flush=True)
''')
    worker=gate.Worker(exe,tmp_path,tmp_path/"stderr.log",dict(os.environ),timeout=1)
    try:
        assert worker.call({"op":"good"})=={"ok":True}
        assert worker.call({"op":"bad"},False)=={"ok":False}
        with pytest.raises(AssertionError):worker.call({"op":"bad"})
        with pytest.raises(ValueError):worker.call({"op":"good","data":"x"*5000000})
    finally:worker.close()
    assert worker.process.poll() is not None


def test_request_write_deadline_kills_nonreading_worker(tmp_path):
    import os
    exe=binary(tmp_path,"print('{\"ready\":true,\"protocol\":2}',flush=True)\ntime.sleep(20)")
    worker=gate.Worker(exe,tmp_path,tmp_path/"stderr.log",dict(os.environ),timeout=0.3)
    with pytest.raises(TimeoutError,match="write timeout"):
        worker.call({"op":"generate","data":"x"*1000000})
    assert worker.process.poll() is not None
    worker.close() # Cleanup is idempotent after timeout.
