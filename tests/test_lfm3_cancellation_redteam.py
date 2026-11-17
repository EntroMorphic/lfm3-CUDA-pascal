"""Regression reproductions for cancellation review, no GPU/listener."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_worker_client as module


def test_sigint_after_cancel_read_close_cannot_skip_child_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "worker"
    ready = dict(ready=True, protocol=2, max_sessions=2, aggregate_capacity=32768,
                 gpu_batching=False, cancellation=True)
    path.write_text(f"#!{sys.executable}\nimport time\nprint({json.dumps(json.dumps(ready))},flush=True)\ntime.sleep(30)\n")
    path.chmod(0o700)
    original_pipe, original_close, original_popen = os.pipe2, os.close, subprocess.Popen
    descriptors, children = [], []
    injected = False
    def pipe(flags):
        pair = original_pipe(flags)
        if not descriptors:
            descriptors.extend(pair)
        return pair
    def close(fd):
        nonlocal injected
        original_close(fd)
        if descriptors and fd == descriptors[0] and not injected:
            injected = True
            os.kill(os.getpid(), signal.SIGINT)
    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.os, "pipe2", pipe)
    monkeypatch.setattr(module.os, "close", close)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    try:
        with pytest.raises(KeyboardInterrupt):
            module.WorkerClient(path, tmp_path, cancellation=True, timeout=1)
        assert injected and len(children) == 1
        assert children[0].poll() is not None
        assert children[0].stdin.closed and children[0].stdout.closed
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        # Even the broken implementation must not leave the red-team fake child.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            child.stdin.close()
            child.stdout.close()
        for fd in descriptors:
            try:
                original_close(fd)
            except OSError:
                pass


def test_live_cancel_backpressure_is_failed_admission_not_inactive(monkeypatch):
    client = module.WorkerClient.__new__(module.WorkerClient)
    client.closed = False
    client.active_lock = threading.Lock()
    client.active_generation = dict(session_id="A", rpc_id=9, sent=False)
    client.cancel_write_fd = 999999
    def blocked(*args):
        raise BlockingIOError()
    monkeypatch.setattr(module.os, "write", blocked)
    with pytest.raises(module.WorkerFailure):
        client.cancel("A")
    assert client.active_generation["sent"] is False
