"""Counterexamples for native cancellation acceptance, without GPU execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.mark.parametrize("damage", [None, "ignored", "peer", "transcript", "recovery", "stale"])
def test_cancellation_gate_rejects_invalid_behavior(tmp_path, monkeypatch, damage):
    tool = Path(__file__).resolve().parents[1] / "tools/check_lfm3_cancellation.py"
    spec = importlib.util.spec_from_file_location("cancel_gate", tool)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    reference = list(range(40))
    binary = tmp_path / "worker"
    binary.write_text("fixture")
    req, result = tmp_path / "request.json", tmp_path / "result.json"
    req.write_text(json.dumps(dict(input_ids=[1, 2, 3])))
    result.write_text(json.dumps(dict(generated_ids=reference)))
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", [str(tool), "--checkpoint", str(tmp_path), "--worker", str(binary),
        "--control-request", str(req), "--control-result", str(result), "--output", str(output)])
    instances = []
    class Worker:
        def __init__(self, *args, **kwargs):
            self.sequence = 0
            self.sessions = {}
            self.trace = kwargs["sampling_trace_root"]
            self.trace.mkdir()
            self.read_fd, self.cancel_write_fd = os.pipe2(os.O_NONBLOCK)
            self.running = threading.Event()
            self.active = None
            self.cancelled = False
            self.closed = False
            self.process = SimpleNamespace(wait=lambda timeout: 0)
            instances.append(self)
        def cancel(self, sid):
            if self.active != sid:
                return False
            self.running.set()
            return True
        def abort(self, sid, ids):
            s = self.sessions[sid]
            s.update(cache_valid=False, consumed_tokens=0, inflight=False)
            return dict(ok=True, cancelled=True, cache_valid=False, consumed_tokens=0,
                        generated_ids=ids, session_id=sid)
        def call_cancellable(self, request, event):
            self.sequence += 1
            assert event.is_set()
            return self.abort(request["session_id"], [])
        def call(self, request):
            self.sequence += 1
            op, sid = request["op"], request.get("session_id")
            if op == "create":
                self.sessions[sid] = dict(cache_valid=False, consumed_tokens=0, committed_consumed_tokens=0, transcript_tokens=0, inflight=False)
            elif op == "inspect":
                return dict(ok=True, **self.sessions[sid])
            elif op == "generate":
                if request.get("trace_sampling"):
                    self.active = sid
                    directory = self.trace / str(self.sequence)
                    directory.mkdir()
                    (directory / "2.logits.bf16").write_bytes(b"completed-step-proof")
                    assert self.running.wait(2)
                    self.active = None
                    self.cancelled = True
                    response = self.abort(sid, reference[:3])
                    if damage == "ignored":
                        response.update(cancelled=False, generated_ids=reference[:32])
                    if damage == "peer":
                        self.sessions["B"]["cache_valid"] = False
                    if damage == "transcript":
                        self.sessions[sid]["transcript_tokens"] = 0
                    return response
                try:
                    stale = os.read(self.read_fd, 8)
                except BlockingIOError:
                    stale = b""
                if stale and damage == "stale":
                    return self.abort(sid, [])
                count = request["max_new_tokens"]
                inputs = request["input_ids"]
                reused = self.sessions[sid]["consumed_tokens"] if len(inputs) > 3 else 0
                ids = reference[32:33] if count == 1 else reference[:32]
                if sid == "A" and self.cancelled and damage == "recovery":
                    ids = [99]
                self.sessions[sid].update(cache_valid=True, consumed_tokens=len(inputs)+len(ids)-1,
                                          committed_consumed_tokens=len(inputs)+len(ids)-1,
                                          transcript_tokens=len(inputs)+len(ids))
                return dict(ok=True, generated_ids=ids, reused_tokens=reused)
            elif op == "close":
                del self.sessions[sid]
            return dict(ok=True)
        def close(self):
            self.closed = True
            os.close(self.read_fd)
            os.close(self.cancel_write_fd)
    monkeypatch.setattr(gate, "WorkerClient", Worker)
    if damage:
        with pytest.raises(AssertionError):
            gate.main()
        assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    else:
        gate.main()
        assert json.loads((output / "summary.json").read_text())["status"] == "passed"
    assert instances[0].closed
