"""Service/HTTP cancellation contract: no listener, GPU, or native worker used."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from test_lfm3_server import (CONFIG, Frontend, Worker, create, generation, invoke,
                              wait_until)
import lfm3_server as server


class CancellableWorker(Worker):
    supports_cancellation = True

    def __init__(self):
        super().__init__()
        self.blocked_sid = None
        self.running = threading.Event()
        self.finish = threading.Event()
        self.partial = [3]
        self.completion_wins = False
        self.active_generation = None
        self.notifications = []
        self.rpc_threads = []
        self.caches = {}
        self.bad = {}

    def call(self, request):
        self.rpc_threads.append(threading.get_ident())
        result = super().call(request)
        if request["op"] == "create":
            self.caches[request["session_id"]] = "kept"
        elif request["op"] == "reset":
            self.caches[request["session_id"]] = None
            result.update(self.bad)
        return result

    def call_cancellable(self, request, cancel_event):
        self.rpc_threads.append(threading.get_ident())
        self.active_generation = (request["session_id"], cancel_event)
        try:
            if request["session_id"] == self.blocked_sid:
                self.running.set()
                assert self.finish.wait(3)
            if cancel_event.is_set() and not self.completion_wins:
                self.requests.append(request)
                self.caches[request["session_id"]] = None
                result = dict(ok=True, session_id=request["session_id"], cancelled=True,
                              generated_ids=self.partial, consumed_tokens=0, reused_tokens=0,
                              cache_valid=False, do_sample=request["do_sample"], seed=request["seed"])
                result.update(self.bad)
                return result
            result = super().call(request)
            result.update(self.bad)
            return result
        finally:
            self.active_generation = None

    def cancel(self, sid):
        self.notifications.append((sid, threading.get_ident()))
        return self.active_generation is not None and self.active_generation[0] == sid


@pytest.fixture
def service():
    value = server.Service(CancellableWorker(), Frontend(), CONFIG,
                           server.Settings(request_timeout=1))
    yield value
    value.worker.finish.set()
    value.close()


def cancel_http(service, sid, **kwargs):
    return invoke(service, b"{}", path=f"/sessions/{sid}/cancel", **kwargs)


def assert_terminal(result, ids):
    assert result["cancelled"] is True and result["finish_reason"] == "cancelled"
    assert result["generated_ids"] == ids and result["cache_valid"] is False
    assert result["consumed_tokens"] == result["reused_tokens"] == 0
    assert result["gpu_batching"] is False


def test_cancel_http_auth_validation_idle_and_unknown(service):
    sid = create(service)
    assert cancel_http(service, sid, headers=[])[0] == 401
    assert cancel_http(service, sid)[0] == 409
    assert cancel_http(service, "z" * 32)[0] == 404
    assert invoke(service, b'{"extra":true}', path=f"/sessions/{sid}/cancel")[0] == 400
    assert not service.worker.notifications


def test_unsupported_cancel_does_not_set_event_or_notify_worker(service):
    sid = create(service)
    service.worker.supports_cancellation = False
    service.worker.blocked_sid = sid
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert service.worker.running.wait(1)
            status, body = cancel_http(service, sid)
            assert (status, body) == (501, {"error": "cancellation_unavailable"})
            assert not service.sessions[sid]["cancel_event"].is_set()
            assert not service.worker.notifications
        finally:
            service.worker.finish.set()
        assert future.result(1)["cancelled"] is False


def test_cancel_before_client_active_registration_is_bridged_by_event(service):
    sid = create(service)
    entered, register = threading.Event(), threading.Event()
    original = service.worker.call_cancellable

    def delayed(request, event):
        entered.set()
        assert register.wait(3)
        return original(request, event)

    service.worker.call_cancellable = delayed
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert entered.wait(1) and service.worker.active_generation is None
            assert cancel_http(service, sid)[0] == 202
        finally:
            register.set()
        assert_terminal(future.result(1), [3])
    assert [r["op"] for r in service.worker.requests] == ["create", "generate"]


def test_failed_cancel_notification_is_not_a_success_or_handler_cache_mutation(service):
    sid = create(service)
    worker = service.worker
    worker.blocked_sid = sid

    def broken(sid):
        raise server.WorkerFailure("broken control pipe")

    worker.cancel = broken
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert worker.running.wait(1)
            assert cancel_http(service, sid) == (503, {"error": "cancellation_transport_failed"})
            assert service.sessions[sid]["busy"] and worker.caches[sid] == "kept"
            assert service.sessions[sid]["cancel_event"].is_set()
        finally:
            worker.finish.set()
        assert_terminal(future.result(1), [3])


@pytest.mark.parametrize("operation", ["create", "reset", "close", None])
def test_cancel_does_not_target_non_generation_lease(service, operation):
    sid = create(service)
    service._lease(sid, operation)
    try:
        assert cancel_http(service, sid)[0] == 409
        assert not service.worker.notifications
    finally:
        service._release(sid)


def test_queued_cancel_resets_only_target_on_owner_and_duplicate_is_requested(service):
    a, b = create(service), create(service)
    worker = service.worker
    worker.blocked_sid = b
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.generate, generation(b))
        assert worker.running.wait(1)
        queued = pool.submit(service.generate, generation(a, sampling={"do_sample": True, "seed": 17}))
        try:
            wait_until(lambda: service.jobs.qsize() == 1)
            for _ in range(2):
                status, response = cancel_http(service, a)
                assert status == 202 and response["cancel_requested"] is True
                assert "cancelled" not in response
            assert service.sessions[a]["busy"]
            with pytest.raises(server.APIError, match="session_busy"):
                service.lifecycle(a, "reset", {})
        finally:
            worker.finish.set()
        assert first.result(1)["cancelled"] is False
        result = queued.result(1)
    assert_terminal(result, [])
    assert result["text"] == "" and result["do_sample"] is True and result["seed"] == 17
    assert [r["op"] for r in worker.requests] == ["create", "create", "generate", "reset"]
    assert worker.caches[a] is None and worker.caches[b] == "kept"
    assert set(worker.rpc_threads) == {service.owner.ident}
    assert cancel_http(service, a)[0] == 409
    assert service.generate(generation(b))["text"] == "text"


@pytest.mark.parametrize("partial", [[], [3], [3, 9]])
def test_running_cancel_accepts_partial_and_preserves_peer(service, partial):
    a, b = create(service), create(service)
    worker = service.worker
    worker.blocked_sid, worker.partial = a, partial
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, {**generation(a), "max_new_tokens": 4})
        try:
            assert worker.running.wait(1)
            assert cancel_http(service, a)[0] == 202
            assert service.sessions[a]["busy"] and worker.caches[a] == "kept"
            assert worker.caches[b] == "kept"
        finally:
            worker.finish.set()
        result = future.result(1)
    assert_terminal(result, partial)
    assert result["text"] == ("text" if partial else "")
    assert worker.caches[a] is None and worker.caches[b] == "kept"
    assert not service.failed and not service.sessions[a]["busy"]
    assert set(worker.rpc_threads) == {service.owner.ident}
    assert worker.notifications[0][1] != service.owner.ident
    assert service.generate(generation(b))["cancelled"] is False
    assert service.generate(generation(a))["cancelled"] is False


def test_encoding_cancel_waits_for_cpu_then_resets_without_generate(service):
    sid = create(service)
    entered, release = threading.Event(), threading.Event()

    def encode(*args):
        entered.set()
        assert release.wait(3)
        return [1, 2]

    service.frontend.encode_messages = encode
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert entered.wait(1)
            assert cancel_http(service, sid)[0] == 202
            assert not future.done() and service.sessions[sid]["busy"]
            assert [r["op"] for r in service.worker.requests] == ["create"]
        finally:
            release.set()
        assert_terminal(future.result(1), [])
    assert [r["op"] for r in service.worker.requests] == ["create", "reset"]


def test_cancel_completion_may_win_and_old_event_cannot_release_new_generation(service):
    sid = create(service)
    worker = service.worker
    worker.blocked_sid, worker.completion_wins = sid, True
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert worker.running.wait(1)
            old_event = service.sessions[sid]["cancel_event"]
            assert cancel_http(service, sid)[0] == 202
        finally:
            worker.finish.set()
        assert future.result(1)["cancelled"] is False
    assert cancel_http(service, sid)[0] == 409
    new_event = threading.Event()
    service._lease(sid, "generate", new_event)
    try:
        old_event.set()
        service._release(sid, old_event)
        assert service.sessions[sid]["busy"]
        assert service.sessions[sid]["cancel_event"] is new_event and not new_event.is_set()
    finally:
        service._release(sid, new_event)


def test_cancel_during_decode_cannot_retroactively_cancel_native_completion(service):
    sid = create(service)
    entered, release = threading.Event(), threading.Event()

    def decode(ids):
        entered.set()
        assert release.wait(3)
        return "text"

    service.frontend.decode = decode
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert entered.wait(1) and service.worker.active_generation is None
            assert cancel_http(service, sid)[0] == 202
            assert service.sessions[sid]["busy"] and not future.done()
        finally:
            release.set()
        result = future.result(1)
    assert result["cancelled"] is False and result["cache_valid"] is True
    assert result["finish_reason"] == "length"
    assert [r["op"] for r in service.worker.requests] == ["create", "generate"]
    assert cancel_http(service, sid)[0] == 409


def test_cancel_request_does_not_release_running_lease_at_http_deadline(service):
    service.settings = server.Settings(request_timeout=0.05)
    a, b = create(service), create(service)
    worker = service.worker
    worker.blocked_sid = a
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(a))
        try:
            assert worker.running.wait(1)
            assert cancel_http(service, a)[0] == 202
            with pytest.raises(server.APIError) as caught:
                future.result(0.5)
            assert caught.value.status == 504
            assert service.sessions[a]["busy"] and worker.caches[a] == "kept"
            assert cancel_http(service, a)[0] == 202
        finally:
            worker.finish.set()
    wait_until(lambda: not service.sessions[a]["busy"])
    assert worker.caches[a] is None and worker.caches[b] == "kept"
    assert service.sessions[a]["cancel_event"] is None
    assert not service.failed


@pytest.mark.parametrize("bad", [
    {"cancelled": True, "cache_valid": False, "generated_ids": [], "consumed_tokens": 0},
    {"cancelled": 1}, {"cache_valid": False}, {"finish_reason": "cancelled"},
])
def test_unsolicited_cancel_metadata_invalidates_all_sessions(service, bad):
    a, b = create(service), create(service)
    service.worker.bad = bad
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate(generation(a))
    assert service.failed and not service.sessions and service.worker.closed
    assert cancel_http(service, b)[0] == 503


@pytest.mark.parametrize("partial", [[100], [True], [9, 3]])
def test_invalid_partial_cancel_tokens_fail_closed(service, partial):
    sid = create(service)
    service.worker.blocked_sid, service.worker.partial = sid, partial
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, {**generation(sid), "max_new_tokens": 4})
        try:
            assert service.worker.running.wait(1)
            assert cancel_http(service, sid)[0] == 202
        finally:
            service.worker.finish.set()
        with pytest.raises(server.APIError, match="worker_unavailable"):
            future.result(1)
    assert service.failed and not service.sessions


def test_suppressed_partial_cancel_tokens_fail_closed(service):
    service.suppress = [3]
    sid = create(service)
    service.worker.blocked_sid = sid
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert service.worker.running.wait(1)
            assert cancel_http(service, sid)[0] == 202
        finally:
            service.worker.finish.set()
        with pytest.raises(server.APIError, match="worker_unavailable"):
            future.result(1)
    assert service.failed


@pytest.mark.parametrize("bad", [
    {"cache_valid": True}, {"cache_valid": 0}, {"consumed_tokens": 1},
    {"reused_tokens": 1}, {"seed": True}, {"do_sample": 0},
    {"session_id": "z" * 32}, {"generated_ids": None},
])
def test_invalid_native_cancel_metadata_fails_closed(service, bad):
    sid = create(service)
    service.worker.blocked_sid = sid
    service.worker.bad = bad
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert service.worker.running.wait(1)
            assert cancel_http(service, sid)[0] == 202
        finally:
            service.worker.finish.set()
        with pytest.raises(server.APIError, match="worker_unavailable"):
            future.result(1)
    assert service.failed and not service.sessions and service.worker.closed


def test_pre_dispatch_cancel_reset_failure_fails_closed(service):
    sid = create(service)
    entered, release = threading.Event(), threading.Event()

    def encode(*args):
        entered.set()
        assert release.wait(3)
        return [1, 2]

    service.frontend.encode_messages = encode
    service.worker.bad = {"reset": False}
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(service.generate, generation(sid))
        try:
            assert entered.wait(1)
            assert cancel_http(service, sid)[0] == 202
        finally:
            release.set()
        with pytest.raises(server.APIError, match="worker_unavailable"):
            future.result(1)
    assert service.failed and not service.sessions and service.worker.closed
