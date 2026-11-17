"""Host-only owned process-group cleanup; no CUDA, SSH, or external jobs."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import errno
import io
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from test_lfm3_worker_client import READY, WorkerClient, WorkerFailure, executable
import lfm3_worker_client as transport


def with_helper(tmp_path):
    code = '''
import signal
helper=os.fork()
if helper == 0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True: time.sleep(1)
request=json.loads(sys.stdin.readline())
print(json.dumps(dict(ok=True,rpc_id=request["rpc_id"],helper=helper)),flush=True)
os._exit(0)
'''
    client = WorkerClient(executable(tmp_path, code), tmp_path, timeout=2)
    try:
        helper = client.call(dict(op="helper"))["helper"]
        assert client.process.wait(timeout=1) == 0
        assert os.getpgid(helper) == client.process.pid
        assert helper in transport.live_group(client.process.pid)
        return client, helper
    except BaseException:
        client.close()
        raise


def test_helper_outlives_leader_but_not_successful_close(tmp_path, monkeypatch):
    client, helper = with_helper(tmp_path)
    pgid = client.process.pid
    signals = []
    original = transport.os.killpg
    def killpg(group, sig):
        assert group == pgid and group != os.getpgrp()
        signals.append(sig)
        return original(group, sig)
    monkeypatch.setattr(transport.os, "killpg", killpg)
    try:
        client.close()
        assert transport.live_group(pgid) == []
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert client.closed and not client.cleanup_failed
        assert client.process.stdin.closed and client.process.stdout.closed
        client.close()
    finally:
        if not client.closed:
            client.close()


def test_concurrent_close_waits_for_group_quiet_verification(tmp_path, monkeypatch):
    client, _ = with_helper(tmp_path)
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original = transport.live_group
    def group(pgid):
        assert pgid == client.process.pid
        entered.set()
        assert release.wait(2)
        return original(pgid)
    monkeypatch.setattr(transport, "live_group", group)
    def second_close():
        second_started.set()
        client.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.close)
        try:
            assert entered.wait(1)
            second = pool.submit(second_close)
            assert second_started.wait(1)
            with pytest.raises(FutureTimeout):
                second.result(timeout=0.05)
            assert not first.done()
        finally:
            release.set()
        first.result(timeout=1)
        second.result(timeout=1)
    assert original(client.process.pid) == [] and not client.cleanup_failed


def test_nonquiet_group_fails_bounded_without_descriptor_leaks(tmp_path, monkeypatch):
    ready = dict(READY, cancellation=True)
    client = WorkerClient(executable(tmp_path, "time.sleep(10)", ready), tmp_path,
                          timeout=1, cancellation=True)
    cancel_fd = client.cancel_write_fd
    group_queries = []
    def nonquiet(pgid):
        assert pgid == client.process.pid
        group_queries.append(pgid)
        return [pgid + 1]
    monkeypatch.setattr(transport, "live_group", nonquiet)
    start = time.monotonic()
    with pytest.raises(WorkerFailure, match="cleanup incomplete"):
        client.close()
    assert time.monotonic() - start < 3.5
    assert group_queries and client.closed and client.cleanup_failed
    assert isinstance(client.cleanup_error, WorkerFailure)
    assert client.process.poll() is not None
    assert client.process.stdin.closed and client.process.stdout.closed
    assert client.cancel_read_fd is None and client.cancel_write_fd is None
    with pytest.raises(OSError) as caught:
        os.fstat(cancel_fd)
    assert caught.value.errno == errno.EBADF
    with pytest.raises(WorkerFailure, match="previously failed"):
        client.close()


@pytest.mark.parametrize("pgid", ["parent", "different"])
def test_unowned_group_never_signalled_and_pipes_close(monkeypatch, pgid):
    client = WorkerClient.__new__(WorkerClient)
    client.closed = False
    client.cleanup_lock = threading.Lock()
    client.active_lock = threading.Lock()
    client.cancel_read_fd = client.cancel_write_fd = None
    pid = os.getpid() if pgid == "parent" else 987654321
    client.process = SimpleNamespace(pid=pid, stdin=io.BytesIO(), stdout=io.BytesIO())
    monkeypatch.setattr(transport.os, "getpgid", lambda pid: os.getpgrp() if pgid == "parent" else pid + 1)
    signals = []
    monkeypatch.setattr(transport.os, "killpg", lambda group, sig: signals.append((group, sig)))
    with pytest.raises(ValueError, match="own a new process session"):
        client.close()
    assert not signals and client.cleanup_failed
    assert client.process.stdin.closed and client.process.stdout.closed


def test_quiet_scan_failure_still_closes_pipes_and_records_failure(tmp_path, monkeypatch):
    client = WorkerClient(executable(tmp_path, "time.sleep(10)"), tmp_path, timeout=1)
    def failed_scan(pgid):
        raise RuntimeError("proc unavailable")
    monkeypatch.setattr(transport, "live_group", failed_scan)
    with pytest.raises(RuntimeError, match="proc unavailable"):
        client.close()
    assert client.closed and client.cleanup_failed
    assert client.process.stdin.closed and client.process.stdout.closed
    assert client.process.poll() is not None
