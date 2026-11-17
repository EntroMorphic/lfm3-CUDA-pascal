"""Host-only subprocess/pipe tests, with tiny fake workers and no GPU."""
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lfm3_worker_client import MAX_FRAME, WorkerClient, WorkerFailure, strict_json


READY = dict(ready=True, protocol=2, max_sessions=2, aggregate_capacity=32768, gpu_batching=False)


def executable(tmp_path, code, ready=READY):
    path = tmp_path / "fake-worker"
    path.write_text(f"#!{sys.executable}\nimport sys,json,time,os\nprint({json.dumps(json.dumps(ready))},flush=True)\n" + code)
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                                  b'{"a":1e9999}', b'[]', b'\xff', b'{"a":' + b'[' * 65 + b'0' + b']' * 65 + b'}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_calls_and_cleanup(tmp_path):
    worker = WorkerClient(executable(tmp_path, 'for line in sys.stdin:\n req=json.loads(line);rid=req.pop("rpc_id");print(json.dumps(dict(ok=True,rpc_id=rid,echo=req)),flush=True)\n'), tmp_path, timeout=2)
    try:
        assert worker.call({"op": "stats"}) == {"ok": True, "echo": {"op": "stats"}}
        with pytest.raises(ValueError, match="owned by the transport"):
            worker.call({"op":"stats", "rpc_id": 123})
        with pytest.raises(ValueError):
            worker.call({"large": "x" * MAX_FRAME})
        assert not worker.closed
        assert worker.call({"op": "stats"})["ok"]
        worker.lock.acquire()
        try:
            with pytest.raises(WorkerFailure):
                worker.call({"op": "stats"})
        finally:
            worker.lock.release()
    finally:
        worker.close()
        worker.close()
    assert worker.process.poll() is not None
    assert worker.process.stdin.closed and worker.process.stdout.closed


def test_child_restores_signal_mask_and_responds_to_sigterm(tmp_path):
    import os
    import signal
    parent_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    worker = WorkerClient(executable(tmp_path,
        'import signal\nfor line in sys.stdin:\n print(json.dumps(dict(ok=True,rpc_id=json.loads(line)["rpc_id"],mask=sorted(int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK,set())))),flush=True)\n'),
        tmp_path, timeout=2)
    try:
        assert worker.call({"op": "mask"})["mask"] == sorted(int(s) for s in parent_mask)
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == parent_mask
        os.killpg(worker.process.pid, signal.SIGTERM)
        assert worker.process.wait(timeout=1) == -signal.SIGTERM
    finally:
        worker.close()


def test_final_reap_timeout_still_closes_owned_pipes(monkeypatch):
    import io
    import signal
    import subprocess
    import lfm3_worker_client as client
    from types import SimpleNamespace
    worker = WorkerClient.__new__(WorkerClient)
    worker.closed = False
    waits, signals = [], []
    def wait(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired("uninterruptible fake worker", timeout)
    worker.process = SimpleNamespace(pid=999999, poll=lambda: None, wait=wait,
                                     stdin=io.BytesIO(), stdout=io.BytesIO())
    monkeypatch.setattr(client.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(subprocess.TimeoutExpired):
        worker.close()
    assert worker.closed and worker.process.stdin.closed and worker.process.stdout.closed
    assert waits == [2, 2] and signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("code", [
    'sys.stdin.readline()\nprint("not JSON",flush=True)\n',
    'sys.stdin.readline()\nprint("{}",flush=True)\n',
    'sys.stdin.readline()\nsys.exit(0)\n',
    'sys.stdin.readline()\nprint("x"*(4*1024*1024+1),flush=True)\n',
    'sys.stdin.readline()\ntime.sleep(10)\n',
])
def test_rpc_failure_invalidates_and_reaps(tmp_path, code):
    worker = WorkerClient(executable(tmp_path, code), tmp_path, timeout=0.5)
    with pytest.raises(WorkerFailure):
        worker.call({"op": "stats"})
    assert worker.closed and worker.process.poll() is not None


def test_write_deadline(tmp_path):
    worker = WorkerClient(executable(tmp_path, 'time.sleep(10)\n'), tmp_path, timeout=0.5)
    start = time.monotonic()
    with pytest.raises(WorkerFailure):
        worker.call({"op": "unused", "payload": "x" * (MAX_FRAME - 100)})
    assert time.monotonic() - start < 4
    assert worker.closed


def test_handshake_contract(tmp_path):
    with pytest.raises(WorkerFailure):
        WorkerClient(executable(tmp_path, 'time.sleep(10)\n', {**READY, "gpu_batching": True}), tmp_path, timeout=1)


@pytest.mark.parametrize("bad_id", [None, True, 0, 2, "1"])
def test_mismatched_rpc_identity_fails_closed(tmp_path, bad_id):
    reply = {"ok": True, "rpc_id": bad_id}
    worker = WorkerClient(executable(tmp_path, 'sys.stdin.readline()\nprint('+repr(json.dumps(reply))+',flush=True)\ntime.sleep(10)\n'), tmp_path, timeout=1)
    with pytest.raises(WorkerFailure):
        worker.call({"op": "reset"})
    assert worker.closed and worker.process.poll() is not None


def test_delayed_duplicate_cannot_acknowledge_next_rpc(tmp_path):
    worker = WorkerClient(executable(tmp_path, '''
request=json.loads(sys.stdin.readline())
reply=dict(ok=True,rpc_id=request["rpc_id"],reset=True)
print(json.dumps(reply),flush=True)
sys.stdin.readline()
print(json.dumps(reply),flush=True)
time.sleep(10)
'''), tmp_path, timeout=1)
    try:
        assert worker.call({"op":"reset"})["reset"] is True
        with pytest.raises(WorkerFailure):
            worker.call({"op":"reset"})
        assert worker.closed
    finally:
        worker.close()


def test_partial_initialization_interrupt_closes_child(tmp_path, monkeypatch):
    import lfm3_worker_client as client
    popen = client.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        return process

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(client.subprocess, "Popen", spawn)
    monkeypatch.setattr(client.os, "set_blocking", interrupt)
    with pytest.raises(KeyboardInterrupt):
        WorkerClient(executable(tmp_path, 'time.sleep(10)\n'), tmp_path, timeout=1)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed
