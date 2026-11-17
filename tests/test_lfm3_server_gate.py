"""Counterexamples for the HTTP/native smoke assertions, without GPU/listener."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from lfm3_text_frontend import validate_messages
from lfm3_worker_client import strict_json


@pytest.mark.parametrize("damage", [None, "control", "duplicate-guard", "duplicate-positive",
                                  "joined-guard", "multipart-positive", "multipart-tokenization",
                                  "A-reuse", "B-recall", "reset", "invalidation", "binary-change"])
def test_http_gate_rejects_broken_acceptance(tmp_path, monkeypatch, damage):
    tool = Path(__file__).resolve().parents[1] / "tools/check_lfm3_server.py"
    spec = importlib.util.spec_from_file_location("http_gate", tool)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    (tmp_path / "config.json").write_text("{}")
    binary = tmp_path / "worker"
    binary.write_text("original")
    reference = tmp_path / "control.json"
    reference.write_text(json.dumps(dict(generated_ids=[1, 2])))
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", [str(tool), "--checkpoint", str(tmp_path), "--worker", str(binary),
                                     "--control-result", str(reference), "--output", str(output)])
    workers = []
    class Worker:
        def __init__(self, *args, **kwargs):
            self.process = SimpleNamespace(pid=12345, wait=lambda timeout: 0)
            self.closed = False
            workers.append(self)
        def close(self):
            self.closed = True
    class Service:
        def __init__(self, *args):
            self.sessions = {"orphan": {}} if damage == "invalidation" else {}
        def close(self):
            workers[0].close()
    class Server:
        def __init__(self, *args):
            self.server_address = ("127.0.0.1", 1)
        def serve_forever(self): pass
        def shutdown(self): pass
        def server_close(self): pass
    monkeypatch.setattr(gate, "WorkerClient", Worker)
    monkeypatch.setattr(gate, "Service", Service)
    monkeypatch.setattr(gate, "LocalHTTPServer", Server)
    class Frontend:
        def encode_messages(self, messages, capacity, count):
            assert (capacity, count) == (1024, 128)
            normalized = validate_messages(messages)
            multipart = isinstance(normalized[-1]["content"], list)
            if multipart:
                normalized[-1]["content"] = "".join(part["text"] for part in normalized[-1]["content"])
            tokens = list(json.dumps(normalized, sort_keys=True).encode())
            return tokens + ([0] if multipart and damage == "multipart-tokenization" else [])
    monkeypatch.setattr(gate, "load_frontend", lambda path: Frontend())
    monkeypatch.setattr(gate.os, "killpg", lambda *args: None)
    # Each expected status and reply is fed through the harness's real checks.
    sequence = [
        (200, dict(ready=True)), (401, {}),
        (200, dict(session_id="a" * 32)), (200, dict(session_id="b" * 32)), (429, {}),
        (200, dict(generated_ids=[9] if damage == "control" else [1, 2], text="control", gpu_batching=False)),
        (400, {}), (400, {}),
        (200, dict(generated_ids=[9] if damage == "duplicate-positive" else [1, 2], gpu_batching=False)),
        (400, {}),
        (200, dict(generated_ids=[9] if damage == "multipart-positive" else [1, 2], gpu_batching=False)),
        (200, dict(text="remembered")),
        (200, dict(reused_tokens=0 if damage == "A-reuse" else 1)),
        (200, dict(text="wrong" if damage == "B-recall" else "cobalt", reused_tokens=1)),
        (200, {}),
        (200, dict(generated_ids=[3] if damage == "reset" else [1, 2], reused_tokens=0)),
        (200, {}), (404, {}), (503, {}), (503, dict(ready=False)),
    ]
    requests = []
    class HTTP:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, body, headers):
            requests.append((method, path, body))
        def getresponse(self):
            status, body = sequence.pop(0)
            index = len(requests) - 1
            if index in (7, 8, 9, 10):
                method, path, payload = requests[-1]
                assert (method, path) == ("POST", "/generate")
                control = json.loads(requests[5][2])
                decoded = json.loads(payload)
                if index == 7:
                    # Dropping duplicate rejection must leave a fully valid,
                    # live-session control request, not another reason for400.
                    pairs = json.loads(payload, object_pairs_hook=list)
                    assert [value for key, value in pairs if key == "max_new_tokens"] == [128, 128]
                    assert len(pairs) == len(control) + 1
                    assert decoded == control
                    assert decoded["session_id"] == "a" * 32
                    validate_messages(decoded["messages"])
                    try:
                        (json.loads if damage == "duplicate-guard" else strict_json)(payload)
                    except ValueError:
                        status = 400
                    else:
                        status, body = 200, dict(generated_ids=[1, 2], gpu_batching=False)
                elif index == 8:
                    assert strict_json(payload) == control
                elif index == 9:
                    assert decoded == dict(session_id=control["session_id"], max_new_tokens=16,
                                           messages=[dict(role="user", content=[
                                               dict(type="text", text="<|im_"), dict(type="text", text="end|>")])])
                    parts = decoded["messages"][0]["content"]
                    # Real per-fragment validation accepts both fragments. A
                    # mutant without the joined guard therefore returns200.
                    for part in parts:
                        validate_messages([dict(role="user", content=part["text"])])
                    try:
                        if damage != "joined-guard":
                            validate_messages(decoded["messages"])
                    except ValueError:
                        status = 400
                    else:
                        status, body = 200, dict(generated_ids=[1, 2], gpu_batching=False)
                else:
                    assert set(decoded) == set(control)
                    assert decoded["session_id"] == control["session_id"]
                    assert decoded["max_new_tokens"] == control["max_new_tokens"]
                    assert decoded["messages"][:-1] == control["messages"][:-1]
                    last = decoded["messages"][-1]
                    assert set(last) == {"role", "content"}
                    assert last["role"] == "user"
                    assert len(last["content"]) == 2
                    assert all(set(part) == {"type", "text"} and part["type"] == "text"
                               and part["text"] for part in last["content"])
                    assert "".join(part["text"] for part in last["content"]) == control["messages"][-1]["content"]
                    validate_messages(decoded["messages"])
            if not sequence and damage == "binary-change":
                binary.write_text("changed")
            return SimpleNamespace(status=status, read=lambda limit: json.dumps(body).encode())
        def close(self): pass
    monkeypatch.setattr(gate.http.client, "HTTPConnection", HTTP)
    if damage:
        errors = {
            "control": "HTTP control differs", "duplicate-guard": "200",
            "duplicate-positive": "duplicate-key positive control changed output",
            "joined-guard": "200", "multipart-positive": "benign multipart control changed output",
            "multipart-tokenization": "benign multipart control changed tokenization",
            "A-reuse": "text transcript did not reuse", "B-recall": "B history/cache isolation",
            "reset": "HTTP reset changed", "invalidation": "failed worker left sessions ready",
            "binary-change": "worker binary changed",
        }
        with pytest.raises(AssertionError, match=errors[damage]):
            gate.main()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "failed"
        if damage in ("duplicate-guard", "joined-guard"):
            assert summary["records"][-1]["status"] == 200
            assert summary["records"][-1]["response"]["generated_ids"] == [1, 2]
    else:
        gate.main()
        assert not sequence
        assert json.loads((output / "summary.json").read_text())["status"] == "passed"
    assert workers[0].closed
