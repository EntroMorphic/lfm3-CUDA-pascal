"""Host-only explicit group admission and validation; not GPU acceptance."""
from concurrent.futures import ThreadPoolExecutor
import io
from email.message import Message
import threading
import time

import pytest

from test_lfm3_server import CONFIG, Frontend, Worker, create, generation, server


class BatchWorker(Worker):
    supports_batching = True
    supports_cancellation = True

    def __init__(self):
        super().__init__()
        self.groups = []
        self.mutate = lambda result: None

    def cancel(self, sid):
        return any(sid == lane["session_id"] for lanes, _ in self.groups for lane in lanes)

    def call_batch_cancellable(self, lanes, events):
        self.groups.append((lanes, events))
        self.entered.set()
        if self.block:
            assert self.block.wait(3)
        results = []
        for lane, event in zip(lanes, events):
            cancelled = event.is_set()
            ids = ([] if cancelled else [3] * lane["max_new_tokens"] if lane["do_sample"]
                   else [3, 9] if lane["max_new_tokens"] > 1 else [3])
            results.append(dict(session_id=lane["session_id"], generated_ids=ids,
                                consumed_tokens=0 if cancelled else len(lane["input_ids"]) + len(ids) - 1,
                                reused_tokens=0, do_sample=lane["do_sample"], seed=lane["seed"],
                                cancelled=cancelled, cache_valid=not cancelled,
                                gpu_batching=False, paired_decode_calls=0))
        paired = max(0, min(len(lane["generated_ids"]) - 1 for lane in results))
        for lane in results:
            lane.update(gpu_batching=paired > 0, paired_decode_calls=paired)
        result = dict(ok=True, results=results, gpu_batching=paired > 0, paired_decode_calls=paired)
        self.mutate(result)
        return result


@pytest.fixture
def service():
    value = server.Service(BatchWorker(), Frontend(), CONFIG)
    yield value
    value.close()


def batch(a, b):
    return dict(requests=[{**generation(a, sampling=dict(do_sample=True, seed=17)), "max_new_tokens": 2},
                          {**generation(b, sampling=dict(seed=29)), "max_new_tokens": 3}])


def test_one_group_independent_policies_and_eos(service):
    a, b = create(service), create(service)
    result = service.generate_batch(batch(a, b))
    assert result["gpu_batching"] is True and result["paired_decode_calls"] == 1
    assert len(service.worker.groups) == 1
    assert not any(r["op"] == "generate" for r in service.worker.requests)
    lanes, events = service.worker.groups[0]
    assert [r["session_id"] for r in lanes] == [a, b]
    assert all("op" not in lane and "rpc_id" not in lane for lane in lanes)
    assert [r["seed"] for r in result["results"]] == [17, 29]
    assert [r["finish_reason"] for r in result["results"]] == ["length", "stop"]
    assert events[0] is not events[1]
    assert not service.sessions[a]["busy"] and not service.sessions[b]["busy"]
    service.lifecycle(a, "reset", {})
    service.lifecycle(b, "close", {})
    assert a in service.sessions and b not in service.sessions


@pytest.mark.parametrize("kind", ["duplicate", "sampling", "count", "messages", "unknown"])
def test_bad_second_lane_no_native_no_lease_leak(service, kind):
    a, b = create(service), create(service)
    body = batch(a, b)
    if kind == "duplicate":
        body["requests"][1]["session_id"] = a
    elif kind == "sampling":
        body["requests"][1]["sampling"] = {"seed": True}
    elif kind == "count":
        body["requests"][1]["max_new_tokens"] = True
    elif kind == "messages":
        body["requests"][1]["messages"] = [{"role": "invalid", "content": "bad"}]
    else:
        body["requests"][1]["session_id"] = "x" * 32
    with pytest.raises((ValueError, server.APIError)):
        service.generate_batch(body)
    assert not service.worker.groups and service.frontend.calls == 0
    assert not any(s["busy"] for s in service.sessions.values())


def test_busy_second_does_not_release_someone_elses_lease(service):
    a, b = create(service), create(service)
    event = threading.Event()
    service._lease(b, "generate", event)
    with pytest.raises(server.APIError, match="session_busy"):
        service.generate_batch(batch(a, b))
    assert not service.sessions[a]["busy"]
    assert service.sessions[b]["busy"] and service.sessions[b]["cancel_event"] is event
    service._release(b, event)


def test_cancel_one_running_preserves_peer_and_both_leases(service):
    a, b = create(service), create(service)
    service.worker.block = threading.Event()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.generate_batch, batch(a, b))
        assert service.worker.entered.wait(1)
        for sid in (a, b):
            with pytest.raises(server.APIError, match="session_busy"):
                service.lifecycle(sid, "reset", {})
        assert service.cancel(a, {})["cancel_requested"]
        service.worker.block.set()
        result = future.result(1)
    left, right = result["results"]
    assert left["cancelled"] and not left["cache_valid"] and left["generated_ids"] == []
    assert not right["cancelled"] and right["cache_valid"] and right["generated_ids"] == [3, 9]


@pytest.mark.parametrize("field,value", [
    ("session_id", "x" * 32), ("seed", 18), ("do_sample", False),
    ("consumed_tokens", 7), ("reused_tokens", -1), ("generated_ids", [101]),
    ("cancelled", True), ("cancelled", 0), ("cache_valid", False),
    ("paired_decode_calls", 2), ("gpu_batching", False), ("sampling", {}),
])
def test_bad_second_reply_fails_whole_service(service, field, value):
    a, b = create(service), create(service)
    # First lane remains valid; malformed second metadata must not be ignored.
    service.worker.mutate = lambda result: result["results"][1].update({field: value})
    if field in ("seed", "do_sample"):
        value = 18 if field == "seed" else True
        service.worker.mutate = lambda result: result["results"][1].update({field: value})
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed and service.worker.closed and not service.sessions


@pytest.mark.parametrize("paired,flag", [(0, True), (1, False), (True, True), (-1, False), (1, 1)])
def test_untruthful_outer_accounting_fatal(service, paired, flag):
    a, b = create(service), create(service)
    service.worker.mutate = lambda r: r.update(paired_decode_calls=paired, gpu_batching=flag)
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed


def test_zero_paired_calls_truthful(service):
    a, b = create(service), create(service)
    def unpaired(r):
        r.update(paired_decode_calls=0, gpu_batching=False)
        for lane in r["results"]:
            lane.update(paired_decode_calls=0, gpu_batching=False)
        # EOS sampled during prefill retires this lane before any paired work.
        r["results"][1].update(generated_ids=[9], consumed_tokens=2)
    service.worker.mutate = unpaired
    assert service.generate_batch(batch(a, b))["gpu_batching"] is False


def set_paired(result, count):
    result.update(paired_decode_calls=count, gpu_batching=count > 0)
    for lane in result["results"]:
        lane.update(paired_decode_calls=count, gpu_batching=count > 0)


@pytest.mark.parametrize("count", [0, 2, 999999])
def test_completed_group_requires_exact_actual_decode_count(service, count):
    a, b = create(service), create(service)
    service.worker.mutate = lambda result: set_paired(result, count)
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed and service.worker.closed and not service.sessions


@pytest.mark.parametrize("ids,count,valid", [([], 0, True), ([], 1, False),
                                            ([4], 1, True), ([4], 2, False),
                                            ([4, 5], 2, True)])
def test_cancelled_lane_pair_count_boundary(service, ids, count, valid):
    a, b = create(service), create(service)
    body = batch(a, b)
    for request in body["requests"]:
        request["max_new_tokens"] = 4
    def cancelled(result):
        # Native acknowledged cancellation after some paired forwards, possibly
        # before committing their just-sampled token. The peer still completes.
        service.worker.groups[-1][1][0].set()
        result["results"][0].update(cancelled=True, cache_valid=False,
                                    generated_ids=ids, consumed_tokens=0, reused_tokens=0)
        result["results"][1].update(generated_ids=[3, 3, 3, 3], consumed_tokens=5)
        set_paired(result, count)
    service.worker.mutate = cancelled
    if valid:
        result = service.generate_batch(body)
        assert result["paired_decode_calls"] == count
        assert result["results"][0]["cancelled"] and result["results"][1]["cache_valid"]
        assert not service.failed
    else:
        with pytest.raises(server.APIError, match="worker_unavailable"):
            service.generate_batch(body)
        assert service.failed


def test_pair_count_cannot_exceed_requested_budget_even_when_both_cancelled(service):
    a, b = create(service), create(service)
    body = batch(a, b)
    for request in body["requests"]:
        request["max_new_tokens"] = 2
    def cancelled(result):
        for event, lane in zip(service.worker.groups[-1][1], result["results"]):
            event.set()
            lane.update(cancelled=True, cache_valid=False, generated_ids=[3, 4],
                        consumed_tokens=0, reused_tokens=0)
        set_paired(result, 2)
    service.worker.mutate = cancelled
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(body)
    assert service.failed


def test_cancellation_intent_does_not_override_normal_completion(service):
    a, b = create(service), create(service)
    service.worker.mutate = lambda result: service.worker.groups[-1][1][0].set()
    result = service.generate_batch(batch(a, b))
    assert result["paired_decode_calls"] == 1
    assert all(not lane["cancelled"] and lane["cache_valid"] for lane in result["results"])
    assert not service.failed


@pytest.mark.parametrize("op", ["reset", "close"])
@pytest.mark.parametrize("identity", [None, 0, "other-session"])
def test_present_invalid_lifecycle_identity_fails_closed(service, op, identity):
    sid = create(service)
    original = service.worker.call
    def invalid(request):
        result = original(request)
        if request["op"] == op:
            result["session_id"] = identity
        return result
    service.worker.call = invalid
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.lifecycle(sid, op, {})
    assert service.failed and service.worker.closed and not service.sessions


@pytest.mark.parametrize("op", ["reset", "close"])
def test_matching_lifecycle_identity_accepted(service, op):
    sid = create(service)
    original = service.worker.call
    def matching(request):
        result = original(request)
        result["session_id"] = request["session_id"]
        return result
    service.worker.call = matching
    assert service.lifecycle(sid, op, {}) == {"ok": True}
    assert not service.failed


def test_partial_cancelled_lane_decodes_without_changing_peer(service):
    a, b = create(service), create(service)
    service.worker.block = threading.Event()
    def partial(result):
        result["results"][0]["generated_ids"] = [4]
    service.worker.mutate = partial
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.generate_batch, batch(a, b))
        assert service.worker.entered.wait(1)
        service.cancel(a, {})
        service.worker.block.set()
        left, right = future.result(1)["results"]
    assert left["cancelled"] and left["generated_ids"] == [4] and left["text"] == "text"
    assert left["consumed_tokens"] == 0 and left["reused_tokens"] == 0
    assert right["cache_valid"] and right["finish_reason"] == "stop"


@pytest.mark.parametrize("ids", [[3], [9, 4], [], [3, 4, 5, 6]])
def test_second_lane_stop_and_length_validation(service, ids):
    a, b = create(service), create(service)
    def corrupt(result):
        result["results"][1].update(generated_ids=ids, consumed_tokens=2 + len(ids) - 1)
    service.worker.mutate = corrupt
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed


def test_second_lane_suppression_validation(service):
    a, b = create(service), create(service)
    service.suppress = [4]
    service.worker.mutate = lambda r: r["results"][1].update(generated_ids=[4, 9], consumed_tokens=3)
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed


def test_encoding_timeout_no_late_gpu_or_second_encode(service):
    a, b = create(service), create(service)
    entered, release = threading.Event(), threading.Event()
    def slow(*args):
        entered.set()
        assert release.wait(2)
        return [1, 2]
    service.frontend.encode_messages = slow
    service.settings = server.Settings(request_timeout=0.06)
    try:
        with pytest.raises(server.APIError) as caught:
            service.generate_batch(batch(a, b))
        assert caught.value.status == 504 and entered.is_set()
        assert not service.worker.groups and not any(s["busy"] for s in service.sessions.values())
    finally:
        release.set()
    time.sleep(0.02)
    assert not service.worker.groups


def test_native_timeout_keeps_both_leases_until_owner_finishes(service):
    a, b = create(service), create(service)
    service.worker.block = threading.Event()
    service.settings = server.Settings(request_timeout=0.06)
    try:
        with pytest.raises(server.APIError) as caught:
            service.generate_batch(batch(a, b))
        assert caught.value.status == 504
        assert all(s["busy"] for s in service.sessions.values())
    finally:
        service.worker.block.set()
    service.jobs.join()
    assert not any(s["busy"] for s in service.sessions.values()) and not service.failed


def test_decode_timeout_releases_leases_and_never_fails_service(service):
    a, b = create(service), create(service)
    release = threading.Event()
    service.frontend.decode = lambda ids: release.wait(2) and "late"
    service.settings = server.Settings(request_timeout=0.06)
    try:
        with pytest.raises(server.APIError) as caught:
            service.generate_batch(batch(a, b))
        assert caught.value.status == 504
        service.jobs.join()
        assert not any(s["busy"] for s in service.sessions.values()) and not service.failed
    finally:
        release.set()


def test_encode_cancellation_bridged_to_group(service):
    a, b = create(service), create(service)
    entered, release = threading.Event(), threading.Event()
    def encode(*args):
        entered.set()
        assert release.wait(2)
        return [1, 2]
    service.frontend.encode_messages = encode
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.generate_batch, batch(a, b))
        assert entered.wait(1)
        service.cancel(a, {})
        release.set()
        result = future.result(1)
    assert result["results"][0]["cancelled"] and not result["results"][1]["cancelled"]


def test_endpoint_uses_same_auth_and_framing(service):
    a, b = create(service), create(service)
    payload = server.encode_json(batch(a, b))
    from types import SimpleNamespace
    def request(auth=True, duplicate=False):
        handler = server.Handler.__new__(server.Handler)
        handler.path = "/generate_batch"
        handler.headers = Message()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(payload))
        if duplicate:
            handler.headers["Content-Length"] = str(len(payload))
        if auth:
            handler.headers["Authorization"] = "Bearer " + "t" * 32
        handler.server = SimpleNamespace(service=service, token="t" * 32)
        handler.connection = SimpleNamespace(settimeout=lambda value: None)
        handler.rfile = io.BytesIO(payload)
        replies = []
        handler._reply = lambda status, body: replies.append((status, body))
        handler.do_POST()
        return replies[0]
    assert request(auth=False)[0] == 401
    assert request(duplicate=True)[0] == 400
    assert not service.worker.groups
    status, result = request()
    assert status == 200 and len(result["results"]) == 2 and len(service.worker.groups) == 1


def test_frontend_contention_releases_only_batch_leases(service):
    a, b = create(service), create(service)
    service.frontend_lock.acquire()
    try:
        with pytest.raises(server.APIError, match="frontend_busy"):
            service.generate_batch(batch(a, b))
        assert not service.worker.groups
        assert not any(s["busy"] for s in service.sessions.values())
    finally:
        service.frontend_lock.release()


def test_second_encode_bad_budget_no_native_and_no_leak(service):
    a, b = create(service, 10), create(service, 3)
    with pytest.raises(ValueError):
        service.generate_batch(batch(a, b))
    assert service.frontend.calls == 2 and not service.worker.groups
    assert not any(s["busy"] for s in service.sessions.values())


def test_queued_expiry_never_dispatches_group(service):
    a, b = create(service), create(service)
    entered, release = threading.Event(), threading.Event()
    def occupy():
        entered.set()
        assert release.wait(2)
    with ThreadPoolExecutor() as pool:
        owner = pool.submit(service._submit, occupy, lambda: None)
        assert entered.wait(1)
        service.settings = server.Settings(request_timeout=0.06)
        try:
            with pytest.raises(server.APIError) as caught:
                service.generate_batch(batch(a, b))
            assert caught.value.status == 504
            assert not service.worker.groups
        finally:
            release.set()
        owner.result(1)
        service.jobs.join()
    assert not service.worker.groups and not any(s["busy"] for s in service.sessions.values())


def test_queue_backpressure_preserves_bounded_single_group_admission(service):
    a, b = create(service), create(service)
    entered, release = threading.Event(), threading.Event()
    def occupy():
        entered.set()
        assert release.wait(3)
    with ThreadPoolExecutor(max_workers=6) as pool:
        owner = pool.submit(service._submit, occupy, lambda: None)
        assert entered.wait(1)
        queued = [pool.submit(service._submit, lambda: None, lambda: None) for _ in range(4)]
        end = time.monotonic() + 1
        while service.jobs.qsize() != 4 and time.monotonic() < end:
            time.sleep(0.001)
        try:
            assert service.jobs.qsize() == 4
            with pytest.raises(server.APIError, match="queue_full"):
                service.generate_batch(batch(a, b))
            assert not service.worker.groups and not any(s["busy"] for s in service.sessions.values())
        finally:
            release.set()
        owner.result(1)
        for future in queued:
            future.result(1)


def test_batching_unavailable_is_nonmutating(service):
    a, b = create(service), create(service)
    service.worker.supports_batching = False
    with pytest.raises(server.APIError, match="batching_unavailable"):
        service.generate_batch(batch(a, b))
    assert not service.frontend.calls and not service.worker.groups
    assert not any(s["busy"] for s in service.sessions.values())


@pytest.mark.parametrize("body", [{}, {"requests": []}, {"requests": [None, None]},
                                    {"requests": [1]}, {"requests": "two"},
                                    {"requests": [], "extra": True}])
def test_invalid_group_shape(service, body):
    with pytest.raises(ValueError):
        service.generate_batch(body)
    assert not service.worker.groups
