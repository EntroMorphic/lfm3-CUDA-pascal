"""Host-only integration counterexamples; no GPU, listeners or native execution.

These assertions specify fail-closed behavior and intentionally fail until the
main owner fixes the demonstrated reply-identity/accounting gaps.
"""
import pytest

from test_lfm3_server import CONFIG, Frontend, create, server
from test_lfm3_service_batching import BatchWorker, batch


@pytest.fixture
def service():
    value = server.Service(BatchWorker(), Frontend(), CONFIG)
    yield value
    value.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_lifecycle_reply_for_other_group_session_is_fatal(service, operation):
    a, b = create(service), create(service)
    service.generate_batch(batch(a, b))
    original = service.worker.call

    def wrong_identity(request):
        result = original(request)
        if request["op"] == operation and request["session_id"] == a:
            result["session_id"] = b
        return result

    service.worker.call = wrong_identity
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.lifecycle(a, operation, {})
    assert service.failed and service.worker.closed and not service.sessions


def test_abandoned_allocation_close_reply_identity_is_checked(service):
    """The discard/idle close helper must not trust another allocation's ACK."""
    a, b = create(service), create(service)
    original = service.worker.call

    def wrong_identity(request):
        result = original(request)
        if request["op"] == "close" and request["session_id"] == a:
            result["session_id"] = b
        return result

    service.worker.call = wrong_identity
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service._submit(lambda: service._close_allocation(a), lambda: None)
    assert service.failed and service.worker.closed and not service.sessions


def test_impossible_native_pair_count_is_not_published_as_http_fact(service):
    a, b = create(service), create(service)

    def impossible(result):
        result["paired_decode_calls"] = 999999
        for lane in result["results"]:
            lane["paired_decode_calls"] = 999999

    service.worker.mutate = impossible
    with pytest.raises(server.APIError, match="worker_unavailable"):
        service.generate_batch(batch(a, b))
    assert service.failed and service.worker.closed and not service.sessions
