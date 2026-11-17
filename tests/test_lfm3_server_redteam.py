"""Independent regression repros: host only, no listener/model/GPU required.

These assert the desired safety properties and intentionally fail before fixes.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_server as server
from lfm3_worker_client import WorkerClient, WorkerFailure


CONFIG = dict(model_type="lfm3", vocab_size=100, eos_token_id=9)


class Worker:
    timeout = 1

    def __init__(self):
        self.requests = []

    def call(self, request):
        self.requests.append(request)
        if request["op"] == "create":
            return dict(ok=True, session_id=request["session_id"])
        return dict(ok=True, session_id=request["session_id"], generated_ids=[3],
                    consumed_tokens=2, reused_tokens=0,
                    do_sample=request["do_sample"], seed=request["seed"])

    def close(self):
        pass


def test_request_deadline_includes_tokenization_before_native_admission():
    entered, release = threading.Event(), threading.Event()

    class Frontend:
        def encode_messages(self, messages, capacity, count):
            entered.set()
            assert release.wait(2)
            return [1, 2]

        def decode(self, ids):
            return "text"

    worker = Worker()
    service = server.Service(worker, Frontend(), CONFIG,
                             server.Settings(request_timeout=0.05))
    try:
        sid = service.create({"capacity": 32})["session_id"]
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(service.generate, dict(
                session_id=sid, messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=1))
            try:
                assert entered.wait(1)
                # Caller must time out while encoding is STILL blocked, not
                # merely notice the deadline after the frontend returns.
                with pytest.raises(server.APIError) as caught:
                    future.result(timeout=0.5)
                assert caught.value.status == 504
                assert not release.is_set()
                assert [request["op"] for request in worker.requests] == ["create"]
                assert not service.sessions[sid]["busy"]
            finally:
                release.set()
            service.frontend_jobs.join()
        assert [request["op"] for request in worker.requests] == ["create"]
        assert not service.sessions[sid]["busy"]
    finally:
        release.set()
        service.close()


def test_duplicate_worker_reply_cannot_acknowledge_unexecuted_next_reset(tmp_path):
    ready = dict(ready=True, protocol=2, max_sessions=2,
                 aggregate_capacity=32768, gpu_batching=False)
    executable = tmp_path / "duplicate-reply-worker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys,time\n"
        f"print({json.dumps(json.dumps(ready))}, flush=True)\n"
        "request=json.loads(sys.stdin.readline())\n"
        "reply = (json.dumps(dict(ok=True,rpc_id=request['rpc_id'],reset=True,execution=1))+'\\n').encode()\n"
        # A single atomic pipe write makes both frames available together.
        "os.write(1, reply + reply)\n"
        "sys.stdin.readline()\n"
        # Second reset has NOT executed and will never produce its own reply.
        "time.sleep(10)\n"
    )
    executable.chmod(0o700)
    worker = WorkerClient(executable, tmp_path, timeout=0.2)
    try:
        # Reject either when the duplicate arrives or before accepting the next
        # reply. Returning success twice silently misattributes execution #1.
        with pytest.raises(WorkerFailure):
            assert worker.call({"op": "reset", "session_id": "a" * 32})["reset"] is True
            worker.call({"op": "reset", "session_id": "a" * 32})
        assert worker.closed
    finally:
        worker.close()
