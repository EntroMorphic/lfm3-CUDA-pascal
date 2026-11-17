"""Real local pipes/fake executables exercise group framing, not CUDA."""
import struct
import threading

import pytest

from test_lfm3_worker_client import READY, WorkerClient, WorkerFailure, executable


READY_BATCH = dict(READY, cancellation=True, decode_batching=True)
LANES = [dict(session_id="a", input_ids=[1], max_new_tokens=3, seed=17),
         dict(session_id="b", input_ids=[2], max_new_tokens=7, seed=29)]


@pytest.mark.parametrize("capability", [None, False, 1, "true"])
def test_production_requires_strict_decode_capability(tmp_path, capability):
    ready = dict(READY_BATCH)
    if capability is None:
        del ready["decode_batching"]
    else:
        ready["decode_batching"] = capability
    with pytest.raises(WorkerFailure):
        WorkerClient(executable(tmp_path, "", ready), tmp_path, timeout=1, batching=True)


def test_legacy_default_without_capability(tmp_path):
    client = WorkerClient(executable(tmp_path, "time.sleep(2)", READY), tmp_path, timeout=1)
    try:
        assert client.supports_batching is False
        with pytest.raises(WorkerFailure):
            client.call_batch_cancellable(LANES, [None, None])
    finally:
        client.close()


def test_one_group_write_and_both_pre_dispatch_tags(tmp_path):
    code = '''
import struct,select
fd=int(sys.argv[sys.argv.index("--cancel-fd")+1])
req=json.loads(sys.stdin.readline())
tags=[]
while len(tags)<2:
 assert select.select([fd],[],[],1)[0]
 tags += list(struct.unpack("<"+"Q"*(len(raw:=os.read(fd,16))//8),raw))
print(json.dumps(dict(ok=True,rpc_id=req["rpc_id"],echo=req,tags=tags)),flush=True)
'''
    client = WorkerClient(executable(tmp_path, code, READY_BATCH), tmp_path,
                          timeout=2, cancellation=True, batching=True)
    events = [threading.Event(), threading.Event()]
    for event in events:
        event.set()
    try:
        result = client.call_batch_cancellable(LANES, events)
        assert result["echo"] == dict(op="generate_batch", rpc_id=1, lanes=LANES)
        assert result["tags"] == [(1 << 32) | 1, (2 << 32) | 1]
        assert client.sequence == 1 and not client.active_generations
        assert client.cancel("a") is False
    finally:
        client.close()


def test_register_both_before_write_and_idempotent_cancel(tmp_path, monkeypatch):
    code = '''
req=json.loads(sys.stdin.readline())
print(json.dumps(dict(ok=True,rpc_id=req["rpc_id"])),flush=True)
'''
    client = WorkerClient(executable(tmp_path, code, READY_BATCH), tmp_path,
                          timeout=2, cancellation=True, batching=True)
    import lfm3_worker_client as module
    original = module.os.write
    tags = []
    def write(fd, data):
        if fd == client.process.stdin.fileno():
            assert set(client.active_generations) == {"a", "b"}
            for sid in ("a", "b", "a", "b"):
                assert client.cancel(sid)
        if fd == client.cancel_write_fd:
            tags.append(struct.unpack("<Q", data)[0])
        return original(fd, data)
    monkeypatch.setattr(module.os, "write", write)
    try:
        assert client.call_batch_cancellable(LANES, [None, None])["ok"]
        assert tags == [(1 << 32) | 1, (2 << 32) | 1]
    finally:
        client.close()


def test_batch_stale_ack_cannot_complete_next_rpc(tmp_path):
    code = '''
req=json.loads(sys.stdin.readline())
reply=dict(ok=True,rpc_id=req["rpc_id"])
print(json.dumps(reply),flush=True)
sys.stdin.readline()
print(json.dumps(reply),flush=True)
'''
    client = WorkerClient(executable(tmp_path, code, READY_BATCH), tmp_path, timeout=1, batching=True)
    try:
        client.call_batch_cancellable(LANES, [None, None])
        with pytest.raises(WorkerFailure):
            client.call_batch_cancellable(LANES, [None, None])
        assert client.closed
    finally:
        client.close()


def test_control_backpressure_never_acknowledges_cancel(tmp_path, monkeypatch):
    code = 'req=json.loads(sys.stdin.readline());print(json.dumps(dict(ok=True,rpc_id=req["rpc_id"])),flush=True)'
    client = WorkerClient(executable(tmp_path, code, READY_BATCH), tmp_path,
                          timeout=1, cancellation=True, batching=True)
    import lfm3_worker_client as module
    original = module.os.write
    def write(fd, data):
        if fd == client.cancel_write_fd:
            raise BlockingIOError()
        return original(fd, data)
    monkeypatch.setattr(module.os, "write", write)
    event = threading.Event()
    event.set()
    try:
        with pytest.raises(WorkerFailure):
            client.call_batch_cancellable(LANES, [event, None])
        assert client.closed
    finally:
        client.close()


@pytest.mark.parametrize("lanes", [[], [LANES[0]], [LANES[0], LANES[0]],
                                     [dict(LANES[0], op="generate"), LANES[1]],
                                     [dict(LANES[0], rpc_id=7), LANES[1]],
                                     [dict(LANES[0], session_id=None), LANES[1]]])
def test_invalid_lanes_not_written_or_sequence_consumed(tmp_path, lanes):
    client = WorkerClient(executable(tmp_path, "time.sleep(2)", READY_BATCH), tmp_path,
                          timeout=1, cancellation=True, batching=True)
    try:
        with pytest.raises(ValueError):
            client.call_batch_cancellable(lanes, [None, None])
        assert client.sequence == 0 and not client.active_generations and not client.closed
    finally:
        client.close()
