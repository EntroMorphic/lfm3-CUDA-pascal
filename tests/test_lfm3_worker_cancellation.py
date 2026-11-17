"""Control-pipe ownership/correlation tests with fake workers; no GPU."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lfm3_worker_client import WorkerClient, WorkerFailure


def executable(tmp_path, capable=True):
    path = tmp_path / "cancellable-worker"
    ready = dict(ready=True, protocol=2, max_sessions=2, aggregate_capacity=32768,
                 gpu_batching=False, cancellation=capable)
    path.write_text(f'''#!{sys.executable}
import json,os,sys,select,struct
fd=int(sys.argv[sys.argv.index("--cancel-fd")+1])
print({json.dumps(json.dumps(ready))},flush=True)
for line in sys.stdin:
 req=json.loads(line)
 if req["op"]=="generate":
  assert select.select([fd],[],[],3)[0]
  target=struct.unpack("<Q",os.read(fd,8))[0]
  assert target==req["rpc_id"]
  reply=dict(ok=True,rpc_id=target,session_id=req["session_id"],cancelled=True,generated_ids=[1])
 else:
  reply=dict(ok=True,rpc_id=req["rpc_id"])
 print(json.dumps(reply),flush=True)
''')
    path.chmod(0o700)
    return path


def test_pre_dispatch_cancellation_and_owned_pipe_cleanup(tmp_path):
    event = threading.Event()
    event.set()
    client = WorkerClient(executable(tmp_path), tmp_path, cancellation=True, timeout=2)
    fd = client.cancel_write_fd
    try:
        assert client.call_cancellable(dict(op="generate", session_id="A"), event)["cancelled"] is True
        assert client.call(dict(op="stats"))["ok"] is True
        assert client.cancel("A") is False
    finally:
        client.close()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_running_cancel_targets_only_active_generation(tmp_path):
    client = WorkerClient(executable(tmp_path), tmp_path, cancellation=True, timeout=2)
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(client.call, dict(op="generate", session_id="A"))
            deadline = time.monotonic() + 1
            while client.active_generation is None and time.monotonic() < deadline:
                time.sleep(0.001)
            assert client.active_generation is not None
            assert client.cancel("B") is False
            assert client.cancel("A") is True
            assert future.result(timeout=2)["cancelled"] is True
        assert client.call(dict(op="stats"))["ok"] is True
        assert client.cancel("A") is False
    finally:
        client.close()


def test_repeated_requests_write_exactly_one_targeted_frame():
    import struct
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    client = WorkerClient.__new__(WorkerClient)
    client.closed = False
    client.active_lock = threading.Lock()
    client.active_generation = dict(session_id="A", rpc_id=7, sent=False)
    client.cancel_write_fd = write_fd
    try:
        assert client.cancel("B") is False
        assert client.cancel("A") is True
        assert client.cancel("A") is True
        assert os.read(read_fd, 32) == struct.pack("<Q", 7)
        with pytest.raises(BlockingIOError):
            os.read(read_fd, 1)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_missing_cancel_capability_fails_handshake_without_fd_leak(tmp_path):
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(WorkerFailure, match="handshake"):
        WorkerClient(executable(tmp_path, capable=False), tmp_path, cancellation=True, timeout=1)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_cancel_pipe_backpressure_cannot_fake_pre_dispatch_success(tmp_path, monkeypatch):
    import lfm3_worker_client as module
    client = WorkerClient(executable(tmp_path), tmp_path, cancellation=True, timeout=1)
    original = module.os.write
    fd = client.cancel_write_fd
    def blocked(descriptor, data):
        if descriptor == fd:
            raise BlockingIOError()
        return original(descriptor, data)
    monkeypatch.setattr(module.os, "write", blocked)
    event = threading.Event()
    event.set()
    with pytest.raises(WorkerFailure):
        client.call_cancellable(dict(op="generate", session_id="A"), event)
    assert client.closed and client.process.poll() is not None
