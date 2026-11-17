"""No listener, GPU, model, or tokenizer package is used by these tests."""
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_server as server


CONFIG = dict(model_type="lfm3", vocab_size=100, eos_token_id=9)
MESSAGES = [{"role": "user", "content": "hello"}]


class Frontend:
    def __init__(self):
        self.ids = [1, 2]
        self.calls = 0

    def encode_messages(self, messages, capacity, count):
        self.calls += 1
        return self.ids

    def decode(self, ids):
        return "text"


class Worker:
    timeout = 1

    def __init__(self):
        self.requests = []
        self.closed = False
        self.block = None
        self.entered = threading.Event()
        self.fail = False
        self.active = 0
        self.maximum_active = 0

    def call(self, request):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            self.requests.append(request)
            if request["op"] == "generate":
                self.entered.set()
                if self.block:
                    assert self.block.wait(3)
                if self.fail:
                    raise RuntimeError("GPU died")
                return dict(ok=True, session_id=request["session_id"], generated_ids=[3],
                            consumed_tokens=len(request["input_ids"]), reused_tokens=0,
                            do_sample=request["do_sample"], seed=request["seed"])
            if request["op"] == "create":
                return dict(ok=True, session_id=request["session_id"])
            return dict(ok=True, reset=True, closed=True)
        finally:
            self.active -= 1

    def close(self):
        self.closed = True


@pytest.fixture
def service():
    value = server.Service(Worker(), Frontend(), CONFIG)
    yield value
    value.close()


def create(service, capacity=100):
    return service.create({"capacity": capacity})["session_id"]


def test_cleanup_failure_keeps_service_fail_closed_and_owner_alive(service):
    sid = create(service)
    service.worker.fail = True
    def failed_close():
        raise TimeoutError("final reap deadline")
    service.worker.close = failed_close
    with pytest.raises(server.APIError) as caught:
        service.generate(generation(sid))
    assert caught.value.status == 503
    assert service.failed and service.cleanup_failed and not service.sessions
    assert service.owner.is_alive()
    with pytest.raises(server.APIError) as caught:
        service.create({"capacity": 32})
    assert caught.value.status == 503


def generation(sid, **kwargs):
    return dict(session_id=sid, messages=MESSAGES, max_new_tokens=1, **kwargs)


def test_lifecycle_and_sampling(service):
    a, b = create(service), create(service)
    assert a != b and len(a) == 32
    result = service.generate(generation(a, sampling={"do_sample": True, "seed": 42}))
    assert result["do_sample"] and result["seed"] == 42 and result["gpu_batching"] is False
    assert result["cancelled"] is False and result["cache_valid"] is True
    assert result["finish_reason"] == "length"
    native = service.worker.requests[-1]
    assert native["session_id"] == a and native["input_ids"] == [1, 2]
    assert not set(native) & {"capacity", "suppress_tokens", "experimental_long_context"}
    service.lifecycle(a, "reset", {})
    assert b in service.sessions
    service.lifecycle(a, "close", {})
    assert a not in service.sessions
    with pytest.raises(server.APIError, match="unknown_session"):
        service.generate(generation(a))


def test_quotas_and_32k(service):
    a = create(service, 32768)
    service.frontend.ids = [1] * 32767
    assert service.generate(generation(a))["consumed_tokens"] == 32767
    before = len(service.worker.requests)
    with pytest.raises(ValueError):
        service.generate({**generation(a), "max_new_tokens": 2})
    assert len(service.worker.requests) == before
    with pytest.raises(server.APIError, match="session_quota"):
        create(service, 2)
    service.lifecycle(a, "close", {})
    create(service)
    create(service)
    with pytest.raises(server.APIError, match="session_quota"):
        create(service)


@pytest.mark.parametrize("change", [
    {"messages": None}, {"messages": [{"role": "user", "content": "<|image|>"}]},
    {"messages": [{"role": "user", "content": "\ud800"}]},
    {"messages": [{"role": "user", "content": "x", "tool_calls": []}]},
    {"sampling": None}, {"sampling": {"temperature": 0}},
    {"sampling": {"seed": True}}, {"max_new_tokens": True}, {"model": "/other"},
])
def test_invalid_before_gpu(service, change):
    sid = create(service)
    before = len(service.worker.requests)
    with pytest.raises(ValueError):
        service.generate({**generation(sid), **change})
    assert len(service.worker.requests) == before
    assert not service.sessions[sid]["busy"]


def test_serial_owner_and_session_lease(service):
    a, b = create(service), create(service)
    worker = service.worker
    worker.block = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.generate, generation(a))
        assert worker.entered.wait(1)
        with pytest.raises(server.APIError, match="session_busy"):
            service.generate(generation(a))
        with pytest.raises(server.APIError, match="session_busy"):
            service.lifecycle(a, "reset", {})
        second = pool.submit(service.generate, generation(b))
        deadline = time.monotonic() + 1
        while service.jobs.qsize() != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert service.jobs.qsize() == 1
        worker.block.set()
        first.result(2)
        second.result(2)
    assert worker.maximum_active == 1


def test_bounded_queue_and_rollback():
    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG, server.Settings(queue_size=1))
    try:
        sid = create(service)
        worker.block = threading.Event()
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(service.generate, generation(sid))
            assert worker.entered.wait(1)
            queued = pool.submit(service.create, {"capacity": 100})
            deadline = time.monotonic() + 1
            while service.jobs.qsize() != 1 and time.monotonic() < deadline:
                time.sleep(0.001)
            # Submit directly to isolate inference-queue rejection from the
            # intentionally stricter two-session quota.
            cleanup = threading.Event()
            with pytest.raises(server.APIError, match="queue_full"):
                service._submit(lambda: None, cleanup.set)
            assert cleanup.is_set()
            worker.block.set()
            first.result(2)
            queued.result(2)
    finally:
        worker.block.set()
        service.close()


def test_worker_failure_invalidates_every_session(service):
    a, b = create(service), create(service)
    service.worker.fail = True
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate(generation(a))
    assert service.failed and not service.sessions and service.worker.closed
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate(generation(b))


@pytest.mark.parametrize("change", [
    {"generated_ids": []}, {"generated_ids": [100]}, {"generated_ids": [True]},
    {"consumed_tokens": 1}, {"reused_tokens": 3}, {"seed": True},
    {"do_sample": 0}, {"session_id": "another-session"}, {"ok": False},
    {"sampling_policy": {}}, {"sampled_policy": {}}, {"sampling": {}},
])
def test_bad_native_metadata_fails_closed(service, change):
    a, b = create(service), create(service)
    call = service.worker.call

    def bad(request):
        return {**call(request), **change}
    service.worker.call = bad
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate(generation(a))
    assert service.failed and not service.sessions and service.worker.closed
    assert invoke(service, server.encode_json(generation(b)), path="/generate")[0] == 503


def test_frontend_contention_releases_lease(service):
    sid = create(service)
    service.frontend_lock.acquire()
    try:
        with pytest.raises(server.APIError, match="frontend_busy"):
            service.generate(generation(sid))
        assert not service.sessions[sid]["busy"]
    finally:
        service.frontend_lock.release()


def test_legacy_worker_cancel_unavailable_does_not_set_generation_event(service):
    sid = create(service)
    event = threading.Event()
    service._lease(sid, "generate", event)
    try:
        status, body = invoke(service, b"{}", path=f"/sessions/{sid}/cancel")
        assert status == 501 and body == {"error": "cancellation_unavailable"}
        assert not event.is_set() and service.sessions[sid]["busy"]
    finally:
        service._release(sid, event)


def wait_until(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


@pytest.mark.parametrize("phase", ["encode", "decode"])
def test_slow_frontend_bounds_caller_and_owner_without_late_lease_release(phase):
    worker, frontend = Worker(), Frontend()
    entered, release = threading.Event(), threading.Event()
    original = frontend.encode_messages if phase == "encode" else frontend.decode

    def slow(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    setattr(frontend, "encode_messages" if phase == "encode" else "decode", slow)
    service = server.Service(worker, frontend, CONFIG, server.Settings(request_timeout=0.05))
    try:
        sid = create(service)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(invoke, service, server.encode_json(generation(sid)), path="/generate")
            try:
                assert entered.wait(1)
                status, body = future.result(timeout=0.5)
                assert status == 504 and "deadline" in body["error"]
                wait_until(lambda: not service.sessions[sid]["busy"])
                assert not service.failed and service.owner.is_alive()
                assert service.frontend_owner.daemon and service.frontend_owner.is_alive()
                assert service.frontend_jobs.maxsize == 1
                # Repeated admission cannot spawn CPU workers or native work.
                before = len(worker.requests)
                for _ in range(10):
                    with pytest.raises(server.APIError, match="frontend_busy"):
                        service.generate(generation(sid))
                assert len(worker.requests) == before
                expected = ["create"] + (["generate"] if phase == "decode" else [])
                assert [request["op"] for request in worker.requests] == expected
                # The native owner remains available even during slow decode.
                service.lifecycle(sid, "reset", {})
                service._lease(sid)
            finally:
                release.set()
            service.frontend_jobs.join()
            # A stale frontend completion must not release this newer lease.
            assert service.sessions[sid]["busy"]
            service._release(sid)
        setattr(frontend, "encode_messages" if phase == "encode" else "decode", original)
        assert service.generate(generation(sid))["text"] == "text"
        assert not service.failed
    finally:
        release.set()
        service.close()


def test_generation_propagates_one_deadline_to_encode_queue_and_decode(service, monkeypatch):
    sid = create(service)
    deadlines = []
    frontend_call, submit = service._frontend_call, service._submit

    def frontend(function, deadline, **kwargs):
        deadlines.append(deadline)
        return frontend_call(function, deadline, **kwargs)

    def queued(function, cleanup, **kwargs):
        deadlines.append(kwargs["deadline"])
        return submit(function, cleanup, **kwargs)

    monkeypatch.setattr(service, "_frontend_call", frontend)
    monkeypatch.setattr(service, "_submit", queued)
    service.generate(generation(sid))
    assert len(deadlines) == 3 and len(set(deadlines)) == 1


def test_expired_submission_releases_lease_without_native_execution(service):
    sid = create(service)
    service._lease(sid)
    before = list(service.worker.requests)
    with pytest.raises(server.APIError, match="queue_deadline"):
        service._submit(lambda: pytest.fail("expired native work ran"),
                        lambda: service._release(sid), deadline=time.monotonic() - 1)
    assert service.worker.requests == before
    assert not service.sessions[sid]["busy"] and not service.failed


def test_frontend_completion_racing_wait_timeout_does_not_fail_service(service, monkeypatch):
    sid = create(service)
    original = server.Future.result

    def raced(future, timeout=None):
        result = original(future, timeout=timeout)
        if timeout is not None:
            raise server.FutureTimeout
        return result

    monkeypatch.setattr(server.Future, "result", raced)
    assert service.generate(generation(sid))["text"] == "text"
    assert not service.failed and not service.sessions[sid]["busy"]


def test_late_owner_completion_cannot_publish_success_to_delayed_waiter(service, monkeypatch):
    original = server.Future.result
    cleaned = threading.Event()
    deadline = time.monotonic() + 0.03

    def delayed_waiter(future, timeout=None):
        # Model a caller descheduled across the timeout: owner publication must
        # independently enforce the deadline, not rely on waiter abandonment.
        return original(future, timeout=1 if timeout is not None else None)

    def late():
        time.sleep(0.08)
        return {"ok": True}

    monkeypatch.setattr(server.Future, "result", delayed_waiter)
    with pytest.raises(server.APIError, match="request_deadline_outcome_unknown"):
        service._submit(late, cleaned.set, deadline=deadline)
    assert cleaned.is_set() and not service.failed


def test_shutdown_does_not_join_nonpreemptible_frontend():
    entered, release = threading.Event(), threading.Event()
    frontend = Frontend()

    def stalled(*args):
        entered.set()
        assert release.wait(3)
        return [1, 2]

    frontend.encode_messages = stalled
    service = server.Service(Worker(), frontend, CONFIG, server.Settings(request_timeout=0.05))
    try:
        sid = create(service)
        with pytest.raises(server.APIError) as caught:
            service.generate(generation(sid))
        assert caught.value.status == 504 and entered.is_set()
        start = time.monotonic()
        service.close()
        assert time.monotonic() - start < 0.5
        assert service.frontend_owner.is_alive() and not release.is_set()
    finally:
        release.set()
        service.frontend_owner.join(timeout=1)
        service.close()


def test_expired_queue_job_never_calls_worker():
    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG, server.Settings(request_timeout=0.05))
    try:
        sid = create(service)
        worker.block = threading.Event()
        with pytest.raises(server.APIError, match="outcome_unknown"):
            service.generate(generation(sid))
        with pytest.raises(server.APIError, match="outcome_unknown"):
            create(service)
        worker.block.set()
        service.jobs.join()
        assert [request["op"] for request in worker.requests] == ["create", "generate"]
        assert list(service.sessions) == [sid]
    finally:
        if worker.block:
            worker.block.set()
        service.close()


def test_request_timeout_keeps_lease_until_owner_finishes():
    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG, server.Settings(request_timeout=0.05))
    try:
        sid = create(service)
        worker.block = threading.Event()
        with pytest.raises(server.APIError, match="outcome_unknown"):
            service.generate(generation(sid))
        assert service.sessions[sid]["busy"]
        worker.block.set()
        service.jobs.join()
        assert not service.sessions[sid]["busy"]
    finally:
        service.close()


def test_executing_create_deadline_reclaims_in_owner_thread():
    worker = Worker()
    entered, finish = threading.Event(), threading.Event()
    call = worker.call
    owners = []

    def slow(request):
        owners.append(threading.get_ident())
        if request["op"] == "create" and not entered.is_set():
            entered.set()
            assert finish.wait(3)
        return call(request)

    worker.call = slow
    service = server.Service(worker, Frontend(), CONFIG, server.Settings(request_timeout=0.05, max_sessions=1))
    try:
        with pytest.raises(server.APIError) as error:
            create(service)
        assert error.value.status == 504 and entered.is_set()
        assert len(service.sessions) == 1  # quota remains reserved until native completion
        finish.set()
        service.jobs.join()
        assert not service.sessions and not service.failed
        assert [request["op"] for request in worker.requests] == ["create", "close"]
        assert set(owners) == {service.owner.ident}
        assert create(service) in service.sessions
    finally:
        finish.set()
        service.close()


def test_lost_create_response_expires_and_reclaims_quota():
    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG,
                             server.Settings(max_sessions=1, session_idle_timeout=0.05))
    try:
        create(service)  # caller loses/ignores the otherwise successful response
        deadline = time.monotonic() + 2
        while service.sessions and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not service.sessions and not service.failed
        assert [request["op"] for request in worker.requests] == ["create", "close"]
        assert create(service) in service.sessions
    finally:
        service.close()


def test_idle_reaper_never_closes_busy_session():
    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG,
                             server.Settings(session_idle_timeout=0.05, request_timeout=0.05))
    try:
        sid = create(service)
        worker.block = threading.Event()
        with pytest.raises(server.APIError, match="outcome_unknown"):
            service.generate(generation(sid))
        time.sleep(0.1)
        assert service.sessions[sid]["busy"]
        assert not any(request["op"] == "close" for request in worker.requests)
        worker.block.set()
        service.jobs.join()
        assert sid in service.sessions
    finally:
        if worker.block:
            worker.block.set()
        service.close()


@pytest.mark.parametrize("idle", [False, 0, -1, float("nan"), float("inf"), 86401])
def test_invalid_idle_timeout(idle):
    with pytest.raises(ValueError):
        server.Settings(session_idle_timeout=idle)


def test_idle_close_failure_invalidates_all_sessions():
    worker = Worker()
    call = worker.call

    def bad_close(request):
        if request["op"] == "close":
            return {"ok": True, "closed": False}
        return call(request)

    worker.call = bad_close
    service = server.Service(worker, Frontend(), CONFIG, server.Settings(session_idle_timeout=0.05))
    try:
        create(service)
        create(service)
        deadline = time.monotonic() + 2
        while not service.failed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service.failed and not service.sessions and worker.closed
    finally:
        service.close()


def test_create_completion_racing_wait_timeout_does_not_abandon(service, monkeypatch):
    # Model Future.result's timeout racing an already-published result. The
    # completion/abandonment lock must return that result, not strand its quota.
    original = server.Future.result

    def raced(future, timeout=None):
        result = original(future, timeout=timeout)
        if timeout is not None:
            raise server.FutureTimeout
        return result

    monkeypatch.setattr(server.Future, "result", raced)
    sid = create(service)
    assert sid in service.sessions and not service.sessions[sid]["busy"]
    assert [request["op"] for request in service.worker.requests] == ["create"]


def test_sigterm_during_handshake_reaps_worker_descendant(tmp_path):
    pids = tmp_path / "pids"
    worker = tmp_path / "stalled-worker"
    worker.write_text(f'''#!{sys.executable}
import os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path({str(pids)!r}).write_text(str(os.getpid()) + " " + str(child.pid))
time.sleep(60)
''')
    worker.chmod(0o700)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    runner = "import lfm3_server as s; s.load_frontend=lambda checkpoint: None; s.main()"
    env = dict(os.environ, LFM3_SERVER_TOKEN="a" * 43,
               PYTHONPATH=str(Path(server.__file__).parent))
    process = subprocess.Popen([sys.executable, "-c", runner, "--worker", str(worker),
                                "--checkpoint", str(tmp_path), "--worker-timeout", "60"], env=env)
    owned = []

    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    try:
        deadline = time.monotonic() + 5
        while not pids.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pids.exists()
        owned = [int(pid) for pid in pids.read_text().split()]
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=6) == 0
        deadline = time.monotonic() + 2
        while any(alive(pid) for pid in owned) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(alive(pid) for pid in owned)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if owned:
            try:
                os.killpg(owned[0], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_token_permissions_and_binding(tmp_path, service):
    path = tmp_path / "token"
    path.write_text("a" * 43 + "\n")
    path.chmod(0o600)
    assert server.token_file(path) == "a" * 43
    path.chmod(0o644)
    with pytest.raises(ValueError):
        server.token_file(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        server.token_file(link)
    with pytest.raises(ValueError):
        server.LocalHTTPServer(("0.0.0.0", 0), service, "a" * 43)


def invoke(service, body=b'{"capacity":100}', headers=None, path="/sessions"):
    handler = server.Handler.__new__(server.Handler)
    handler.server = SimpleNamespace(service=service, token="a" * 43)
    handler.connection = SimpleNamespace(settimeout=lambda value: None)
    handler.path = path
    handler.rfile = io.BufferedReader(io.BytesIO(body))
    handler.headers = Message()
    entries = headers if headers is not None else [
        ("Authorization", "Bearer " + "a" * 43), ("Content-Type", "application/json"),
        ("Content-Length", str(len(body)))]
    for key, value in entries:
        handler.headers[key] = value
    replies = []
    handler._reply = lambda status, value: replies.append((status, value))
    handler.do_POST()
    return replies[0]


def test_http_validation_without_listener(service):
    assert invoke(service)[0] == 200
    assert invoke(service, headers=[])[0] == 401
    auth = [("Authorization", "Bearer " + "a" * 43)]
    assert invoke(service, headers=auth + auth)[0] == 401
    assert invoke(service, headers=auth)[0] == 400
    base = auth + [("Content-Type", "application/json")]
    assert invoke(service, headers=base + [("Content-Length", "2"), ("Content-Length", "2")])[0] == 400
    assert invoke(service, headers=base + [("Transfer-Encoding", "chunked"), ("Content-Length", "2")])[0] == 400
    assert invoke(service, headers=base + [("Content-Length", str(server.MAX_BODY + 1))])[0] == 413
    for body in (b'{"capacity":1,"capacity":2}', b'{"capacity":NaN}', b'{"capacity":1e999}', b'[]', b'\xff'):
        assert invoke(service, body)[0] == 400
    assert invoke(service, path="/batch")[0] == 404
