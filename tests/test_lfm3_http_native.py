"""Host tests for the native HTTP server layer (native/include/lfm/lfm3_http.hpp), behind a fake engine.

The harness (native/tests/lfm3_http_harness.cpp) runs the real HTTP layer, Service, native tokenizer
and session ledger, and a fake engine that speaks resident protocol 2 like liquid-lfm3-worker. Each
build variant (plain, ASan+UBSan, TSan) runs every scenario; a sanitizer report fails the test.
Every JSON response is also checked byte for byte against Python's json.dumps of its own content.
"""
import contextlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from native_image_build import image_link_args, libjpeg_dir

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"
pytestmark = pytest.mark.skipif(not (CHECKPOINT / "tokenizer.json").exists() or sys.platform != "linux",
                                reason="Linux and the checkpoint tokenizer required")

TOKEN = secrets.token_urlsafe(32)
MESSAGES = [{"role": "system", "content": "Respond with text."}, {"role": "user", "content": "hello there"}]
GENERATION_KEYS = ["session_id", "text", "generated_ids", "reused_tokens", "consumed_tokens", "do_sample", "seed",
                   "gpu_batching", "cancelled", "cache_valid", "finish_reason"]
PRE_RUN_CANCEL_KEYS = ["session_id", "text", "generated_ids", "consumed_tokens", "reused_tokens", "cancelled",
                       "finish_reason", "cache_valid", "do_sample", "seed", "gpu_batching"]
# TSan is built with clang: GCC 11's TSan does not intercept pthread_cond_clockwait (used by libstdc++ for
# steady-clock waits) and reports a false "double lock" on every condition-variable mutex.
VARIANTS = {"plain": ("g++", []), "asan": ("g++", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]),
            "tsan": ("clang++", ["-fsanitize=thread"])}


@pytest.fixture(scope="module", params=list(VARIANTS))
def harness_binary(tmp_path_factory, request):
    # the modules that import this fixture do not inherit this module's pytestmark, so the fixture skips itself
    if not (CHECKPOINT / "tokenizer.json").exists() or not (CHECKPOINT / "config.json").exists() or sys.platform != "linux":
        pytest.skip("Linux and the checkpoint (config.json, tokenizer.json) required")
    compiler, flags = VARIANTS[request.param]
    if not shutil.which(compiler):
        pytest.skip(f"{compiler} required")
    work = tmp_path_factory.mktemp("http")
    out = work / f"harness-{request.param}"
    images = image_link_args(compiler, flags, work)  # the image front end, for image parts (#4)
    subprocess.run([compiler, "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-pedantic", *flags,
                    "-I", str(ROOT / "native/include"), "-I", str(libjpeg_dir() / "include"),
                    str(ROOT / "native/tests/lfm3_http_harness.cpp"), *images, "-pthread", "-o", str(out)], check=True)
    return out


class Harness:
    def __init__(self, binary, *options):
        # TSan cannot map its shadow under high-entropy ASLR on recent kernels; run it without ASLR.
        prefix = ["setarch", os.uname().machine, "-R"] if binary.name.endswith("-tsan") else []
        self.process = subprocess.Popen([*prefix, str(binary), str(CHECKPOINT), *map(str, options)], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=dict(os.environ, LFM3_SERVER_TOKEN=TOKEN,
                                                                         TSAN_OPTIONS="halt_on_error=1"))
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError(self.process.stderr.read().decode())
        self.port = json.loads(line)["port"]

    def raw(self, data, *, read=True, timeout=30):
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as s:
            s.sendall(data)
            if not read:
                return b""
            chunks = []
            while True:
                try:
                    block = s.recv(65536)
                except ConnectionResetError:  # closed with unread request bytes: no reply
                    block = b""
                if not block:
                    return b"".join(chunks)
                chunks.append(block)

    def call(self, method, path, body=None, *, raw=None, auth=True, headers=None, timeout=60):
        payload = raw if raw is not None else b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
        lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1"]
        if headers is None:
            headers = {}
            if method == "POST":
                headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
            if auth:
                headers["Authorization"] = "Bearer " + TOKEN
        for key, value in (headers.items() if isinstance(headers, dict) else headers):
            lines.append(f"{key}: {value}")
        response = self.raw(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + payload, timeout=timeout)
        return parse_response(response)

    def stop(self, timeout=30):
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(timeout=timeout)
        stderr = self.process.stderr.read().decode()
        assert code == 0 and not stderr, (code, stderr)


def parse_response(response):
    head, _, body = response.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    version, status, _ = lines[0].split(" ", 2)
    assert version == "HTTP/1.0"
    headers = dict(line.split(": ", 1) for line in lines[1:])
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Content-Length"] == str(len(body)) and headers["Cache-Control"] == "no-store"
    assert headers["Connection"] == "close"
    value = json.loads(body)
    # The exact bytes Python's lfm3_server.encode_json produces for this content.
    assert body == json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    return int(status), value


@contextlib.contextmanager
def running(binary, *options):
    h = Harness(binary, *options)
    try:
        yield h
    except BaseException:
        h.process.kill()
        raise
    h.stop()


def session(h, capacity=1024):
    status, body = h.call("POST", "/sessions", {"capacity": capacity})
    assert status == 200 and list(body) == ["session_id", "capacity", "gpu_batching"], body
    assert len(body["session_id"]) == 32 and body["capacity"] == capacity and body["gpu_batching"] is False
    return body["session_id"]


def generate(h, sid, count=64, messages=MESSAGES, **extra):
    return h.call("POST", "/generate", dict(session_id=sid, messages=messages, max_new_tokens=count, **extra))


def in_thread(function, *args):
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("value", function(*args)))
    thread.start()
    return thread, result


def test_health_routing_and_methods(harness_binary):
    with running(harness_binary) as h:
        assert h.call("GET", "/health", auth=False) == (200, {"ready": True, "gpu_batching": False})
        assert h.call("GET", "//health", auth=False)[0] == 200          # http.server path normalization
        assert h.call("GET", "/health?x=1", auth=False) == (404, {"error": "not_found"})
        assert h.call("GET", "/sessions", auth=False) == (404, {"error": "not_found"})
        assert h.call("PUT", "/sessions", {}) == (501, {"error": "unsupported_method"})
        assert h.call("POST", "/nope", {}, auth=False) == (401, {"error": "unauthorized"})
        assert h.call("POST", "/nope", {}) == (404, {"error": "not_found"})
        assert h.call("POST", "/sessions/" + "a" * 31 + "/reset", {}) == (404, {"error": "not_found"})
        assert h.call("POST", "/sessions/" + "a" * 32 + "/rewind", {}) == (404, {"error": "not_found"})


def test_authorization(harness_binary):
    body = json.dumps({"capacity": 64})
    base = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    with running(harness_binary) as h:
        for auth in ([], [("Authorization", "Bearer " + TOKEN[:-1])], [("Authorization", "bearer " + TOKEN)],
                     [("Authorization", "Bearer " + TOKEN)] * 2, [("Authorization", "Bearer  " + TOKEN)],
                     [("Authorization", "Bearer " + TOKEN + " ")], [("Authorization", "Bearer " + TOKEN + "\xe9")],
                     [("Authorization", "Bearer " + "a" * 300)]):
            assert h.call("POST", "/sessions", raw=body.encode(), headers=base + auth) == (401, {"error": "unauthorized"})
        assert h.call("POST", "/sessions", raw=body.encode(), headers=base + [("authorization", "Bearer " + TOKEN)])[0] == 200


def test_body_framing(harness_binary):
    auth = [("Authorization", "Bearer " + TOKEN)]
    body = b'{"capacity":64}'
    with running(harness_binary) as h:
        def post(headers, payload=body):
            return h.call("POST", "/sessions", raw=payload, headers=auth + headers)
        typed = [("Content-Type", "application/json")]
        assert post(typed) == (400, {"error": "content_length_required"})
        assert post(typed + [("Content-Length", "15"), ("Transfer-Encoding", "chunked")]) == (400, {"error": "content_length_required"})
        assert post(typed + [("Content-Length", "15")] * 2) == (400, {"error": "content_length_required"})
        for bad in ("+15", "15 ", "1e1", "abc", "123456789", ""):
            assert post(typed + [("Content-Length", bad)]) == (400, {"error": "invalid_content_length"}), bad
        assert post(typed + [("Content-Length", "0")]) == (413, {"error": "body_limit"})
        assert post(typed + [("Content-Length", str(4 * 1024 * 1024 + 1))], b"") == (413, {"error": "body_limit"})
        length = [("Content-Length", "15")]
        for content_type in ([], [("Content-Type", "text/plain")], [("Content-Type", "application/json; charset=utf-8")],
                             [("Content-Type", "application/json")] * 2):
            assert post(length + content_type) == (415, {"error": "application_json_required"})
        status, created = post(length + [("Content-Type", "APPLICATION/JSON")])
        assert status == 200
        assert h.call("POST", f"/sessions/{created['session_id']}/close", {}) == (200, {"ok": True})


def test_strict_json(harness_binary):
    with running(harness_binary) as h:
        for payload in (b'{"capacity":64', b'{"capacity":64,"capacity":64}', b'{"capacity":NaN}', b'{"capacity":Infinity}',
                        b'{"capacity":1e400}', b'[{"capacity":64}]', b'"x"', b'{"capacity":64} x', b'\xef\xbb\xbf{"capacity":64}',
                        b'{"capacity":64,"x":"\xff"}', b'{"capacity":64,"x":"\xc0\xaf"}', b'{"capacity":64,"x":"\xed\xa0\x80"}',
                        b'{"capacity":64,"x":"\\ud800"}', b"[" * 100000 + b"]" * 100000, b'{"a":' * 65 + b'1' + b'}' * 65):
            assert h.call("POST", "/sessions", raw=payload) == (400, {"error": "invalid_request"}), payload[:40]
        assert h.call("POST", "/sessions", raw=b' \t\r\n{"capacity" : 64}\n ')[0] == 200


def test_slow_and_incomplete_bodies(harness_binary):
    with running(harness_binary, "--socket-timeout", "1") as h:
        head = ("POST /sessions HTTP/1.0\r\nContent-Type: application/json\r\nAuthorization: Bearer " + TOKEN +
                "\r\nContent-Length: 100\r\n\r\n").encode()
        with socket.create_connection(("127.0.0.1", h.port), timeout=10) as s:
            s.sendall(head + b'{"capacity":')
            s.shutdown(socket.SHUT_WR)
            assert parse_response(s.recv(65536)) == (400, {"error": "incomplete_body"})
        with socket.create_connection(("127.0.0.1", h.port), timeout=10) as s:
            s.sendall(head + b'{"capacity":')
            started = time.monotonic()
            assert parse_response(s.recv(65536)) == (408, {"error": "socket_deadline"})
            assert 0.8 < time.monotonic() - started < 3
        # Headers trickled one byte at a time: closed with no reply after the header deadline.
        with socket.create_connection(("127.0.0.1", h.port), timeout=10) as s:
            started = time.monotonic()
            closed = False
            for byte in b"GET /health HTTP/1.0\r\nX: " + b"y" * 100:
                try:
                    s.sendall(bytes([byte]))
                except OSError:
                    closed = True
                    break
                time.sleep(0.05)
                if time.monotonic() - started > 2.5:
                    break
            s.settimeout(3)
            assert closed or s.recv(65536) == b""
            assert time.monotonic() - started < 4


def test_request_line_and_header_limits(harness_binary):
    with running(harness_binary) as h:
        def status(data):
            raw = h.raw(data)
            return parse_response(raw)[0] if raw else None
        assert status(b"GET /health HTTP/2.0\r\n\r\n") == 505
        for line in (b"GET /health HTTP/1\r\n\r\n", b"GET /health FTP/1.0\r\n\r\n", b"GET /health HTTP/1.x\r\n\r\n",
                     b"GET /health\r\n\r\n", b"GET / health HTTP/1.0\r\n\r\n"):
            assert status(line) == 400, line
        # email.feedparser: a line without a colon ends the headers; a leading continuation is dropped.
        assert status(b"GET /health HTTP/1.0\r\nNoColon\r\n\r\n") == 200
        assert status(b"GET /health HTTP/1.0\r\n folded: x\r\n\r\n") == 200
        assert status(b"GET /" + b"a" * 70000 + b" HTTP/1.0\r\n\r\n") == 414
        assert status(b"GET /health HTTP/1.0\r\nX: " + b"a" * 70000 + b"\r\n\r\n") == 431
        # http.client counts the terminating blank line toward its 100-line limit.
        assert status(b"GET /health HTTP/1.0\r\n" + b"X: y\r\n" * 100 + b"\r\n") == 431
        assert status(b"GET /health HTTP/1.0\r\n" + b"X: y\r\n" * 99 + b"\r\n") == 200
        assert status(b"GET /health HTTP/1.0\n\n") == 200
        assert status(b"\r\n") is None


def test_session_creation_rules(harness_binary):
    with running(harness_binary, "--total-capacity", "8192") as h:
        for body in ({}, {"capacity": 64, "x": 1}, {"capacity": 1}, {"capacity": 64.0}, {"capacity": True}, {"capacity": "64"},
                     {"capacity": 8193}, {"capacity": None}, {"Capacity": 64}):
            assert h.call("POST", "/sessions", body) == (400, {"error": "invalid_request"}), body
        a = session(h, 6000)
        assert h.call("POST", "/sessions", {"capacity": 4000}) == (429, {"error": "session_quota"})
        b = session(h, 2192)
        assert h.call("POST", "/sessions", {"capacity": 2}) == (429, {"error": "session_quota"})
        for sid in (a, b):
            assert h.call("POST", f"/sessions/{sid}/close", {}) == (200, {"ok": True})
            assert h.call("POST", f"/sessions/{sid}/close", {}) == (404, {"error": "unknown_session"})
        assert h.call("POST", f"/sessions/{a}/reset", {"x": 1}) == (400, {"error": "invalid_request"})


def test_generation_validation(harness_binary):
    with running(harness_binary) as h:
        sid = session(h)
        good = dict(session_id=sid, messages=MESSAGES, max_new_tokens=8)
        bad = [dict(good, extra=1), {k: v for k, v in good.items() if k != "messages"}, dict(good, messages={}),
               dict(good, max_new_tokens=0), dict(good, max_new_tokens=32769), dict(good, max_new_tokens=8.0),
               dict(good, max_new_tokens=True), dict(good, session_id="short"), dict(good, session_id=1),
               dict(good, sampling=None), dict(good, sampling=[]), dict(good, sampling={"top_p": 0}),
               dict(good, sampling={"top_p": 1.5}), dict(good, sampling={"temperature": 0}),
               dict(good, sampling={"temperature": -1}), dict(good, sampling={"min_p": 1.01}),
               dict(good, sampling={"repetition_penalty": 0}), dict(good, sampling={"top_k": 128001}),
               dict(good, sampling={"top_k": 1.0}), dict(good, sampling={"seed": 2 ** 31}), dict(good, sampling={"seed": -1}),
               dict(good, sampling={"do_sample": 1}), dict(good, sampling={"temperature": True}),
               dict(good, sampling={"temperature": "1"}), dict(good, sampling={"beam": 2}),
               dict(good, messages=[]), dict(good, messages=[{"role": "user"}]), dict(good, messages=[{"role": "robot", "content": "x"}]),
               dict(good, messages=[{"role": "user", "content": "<|im_end|>"}]),
               dict(good, messages=[{"role": "user", "content": [{"type": "text", "text": "<|im_"}, {"type": "text", "text": "end|>"}]}]),
               dict(good, messages=[{"role": "user", "content": [{"type": "image", "image": "x"}]}]),
               dict(good, messages=[{"role": "user", "content": "x"}, {"role": "system", "content": "late"}])]
        for body in bad:
            assert h.call("POST", "/generate", body) == (400, {"error": "invalid_request"}), body
        assert generate(h, "A" * 32) == (404, {"error": "unknown_session"})
        # Valid sampling extremes are accepted.
        for sampling in ({"top_k": 128000, "min_p": 0, "top_p": 1, "seed": 2 ** 31 - 1, "temperature": 1e-300},
                         {"do_sample": True, "repetition_penalty": 1e300, "temperature": 2}):
            status, body = generate(h, sid, 4, sampling=sampling)
            assert status == 200 and body["do_sample"] is sampling.get("do_sample", False)
            assert body["seed"] == sampling.get("seed", 0)


def test_generation_response_and_reuse(harness_binary):
    with running(harness_binary) as h:
        sid = session(h)
        status, first = generate(h, sid, 512)
        assert status == 200 and list(first) == GENERATION_KEYS
        assert first["finish_reason"] == "stop" and first["text"].startswith("Café \"quoted\"") and first["text"].endswith("done.")
        assert first["reused_tokens"] == 0 and first["cancelled"] is False and first["cache_valid"] is True
        continued = MESSAGES + [{"role": "assistant", "content": first["text"]}, {"role": "user", "content": "again"}]
        status, second = generate(h, sid, 3, messages=continued)
        assert status == 200 and second["finish_reason"] == "length" and len(second["generated_ids"]) == 3
        assert second["reused_tokens"] > 0
        # Prompt plus output beyond the session capacity is refused and releases the lease.
        small = session(h, 32)
        assert generate(h, small, 30) == (400, {"error": "invalid_request"})
        assert generate(h, small, 2)[0] == 200
        assert h.call("POST", f"/sessions/{small}/reset", {}) == (200, {"ok": True})
        assert generate(h, small, 2)[1]["reused_tokens"] == 0


def test_busy_cancel_and_recovery(harness_binary):
    with running(harness_binary, "--token-delay-ms", "40") as h:
        a = session(h)
        assert h.call("POST", f"/sessions/{a}/cancel", {}) == (409, {"error": "generation_not_active"})
        assert h.call("POST", f"/sessions/{'B' * 32}/cancel", {}) == (404, {"error": "unknown_session"})
        thread, result = in_thread(generate, h, a, 64)
        time.sleep(0.5)
        assert generate(h, a, 4) == (409, {"error": "session_busy"})
        assert h.call("POST", f"/sessions/{a}/reset", {}) == (409, {"error": "session_busy"})
        assert h.call("POST", f"/sessions/{a}/cancel", {"x": 1}) == (400, {"error": "invalid_request"})
        status, ack = h.call("POST", f"/sessions/{a}/cancel", {})
        assert status == 202 and ack == {"session_id": a, "cancel_requested": True, "gpu_batching": False}
        thread.join(30)
        status, cancelled = result["value"]
        assert status == 200 and list(cancelled) == GENERATION_KEYS
        assert cancelled["cancelled"] is True and cancelled["finish_reason"] == "cancelled" and cancelled["cache_valid"] is False
        assert cancelled["consumed_tokens"] == 0 and cancelled["reused_tokens"] == 0
        assert 2 <= len(cancelled["generated_ids"]) < 64
        assert h.call("POST", f"/sessions/{a}/cancel", {}) == (409, {"error": "generation_not_active"})
        status, full = generate(h, a, 64)
        assert status == 200 and full["generated_ids"][:len(cancelled["generated_ids"])] == cancelled["generated_ids"]
        assert full["text"].startswith(cancelled["text"])


def test_cancel_while_queued_resets(harness_binary):
    with running(harness_binary, "--token-delay-ms", "40") as h:
        a, b = session(h), session(h)
        running_a, result_a = in_thread(generate, h, a, 20)
        time.sleep(0.3)
        queued_b, result_b = in_thread(generate, h, b, 20)
        time.sleep(0.3)
        assert h.call("POST", f"/sessions/{b}/cancel", {})[0] == 202
        queued_b.join(30)
        running_a.join(30)
        status, body = result_b["value"]
        assert status == 200 and list(body) == PRE_RUN_CANCEL_KEYS
        assert body == dict(session_id=b, text="", generated_ids=[], consumed_tokens=0, reused_tokens=0, cancelled=True,
                            finish_reason="cancelled", cache_valid=False, do_sample=False, seed=0, gpu_batching=False)
        assert result_a["value"][1]["cancelled"] is False


def test_request_deadline_keeps_lease_until_done(harness_binary):
    with running(harness_binary, "--token-delay-ms", "100", "--request-timeout", "1") as h:
        a = session(h)
        started = time.monotonic()
        assert generate(h, a, 30) == (504, {"error": "request_deadline_outcome_unknown"})
        assert time.monotonic() - started < 2
        assert generate(h, a, 2) == (409, {"error": "session_busy"})
        time.sleep(2.5)
        assert generate(h, a, 2)[0] == 200


def test_abandoned_create_is_closed(harness_binary):
    with running(harness_binary, "--create-delay-ms", "1500", "--request-timeout", "1") as h:
        assert h.call("POST", "/sessions", {"capacity": 64}) == (504, {"error": "request_deadline_outcome_unknown"})
        time.sleep(1)
    with running(harness_binary, "--create-delay-ms", "1500", "--request-timeout", "3") as h:
        a, b = session(h, 64), session(h, 64)
        assert h.call("POST", "/sessions", {"capacity": 64}) == (429, {"error": "session_quota"})
    with running(harness_binary, "--create-delay-ms", "1200", "--request-timeout", "1") as h:
        assert h.call("POST", "/sessions", {"capacity": 64})[0] == 504
        time.sleep(0.6)
        assert h.call("POST", "/sessions", {"capacity": 64}, timeout=10)[0] == 504
        time.sleep(1.5)
        # Both abandoned allocations were released by the owner; the quota is free again.
        assert h.call("POST", "/sessions", {"capacity": 64}, timeout=10)[0] == 504


def test_idle_sessions_are_reaped(harness_binary):
    with running(harness_binary, "--session-idle-timeout", "1") as h:
        a = session(h)
        assert generate(h, a, 2)[0] == 200
        time.sleep(1.6)
        assert generate(h, a, 2) == (404, {"error": "unknown_session"})
        session(h)
        session(h)


# text generation is ops text_begin and text_step since validation/text-wait-v1 (op generate is no longer sent)
@pytest.mark.parametrize("operation", ["text_begin", "text_step", "create", "reset", "generate_batch"])
def test_engine_failure_is_fatal(harness_binary, operation):
    fail_after = "2" if operation == "create" else "1"
    with running(harness_binary, "--fail-op", operation, "--fail-after", fail_after) as h:
        a = session(h)
        if operation == "create":
            assert h.call("POST", "/sessions", {"capacity": 64}) == (503, {"error": "worker_unavailable"})
        elif operation == "reset":
            assert h.call("POST", f"/sessions/{a}/reset", {}) == (503, {"error": "worker_unavailable"})
        elif operation in ("text_begin", "text_step"):
            assert generate(h, a, 4) == (503, {"error": "worker_unavailable"})
        else:
            b = session(h)
            batch = {"requests": [dict(session_id=s, messages=MESSAGES, max_new_tokens=4) for s in (a, b)]}
            assert h.call("POST", "/generate_batch", batch) == (503, {"error": "worker_unavailable"})
        assert h.call("GET", "/health", auth=False) == (503, {"ready": False, "gpu_batching": False})
        assert generate(h, a, 4) == (503, {"error": "worker_unavailable"})
        assert h.call("POST", "/sessions", {"capacity": 64}) == (503, {"error": "worker_unavailable"})
        assert h.call("POST", "/sessions", {"capacity": 1}) == (400, {"error": "invalid_request"})


def test_handler_saturation_closes_without_work(harness_binary):
    with running(harness_binary, "--handlers", "1", "--socket-timeout", "2") as h:
        with socket.create_connection(("127.0.0.1", h.port), timeout=10) as hold:
            hold.sendall(b"GET /health HTTP/1.0\r\n")
            time.sleep(0.3)
            assert h.raw(b"GET /health HTTP/1.0\r\n\r\n", timeout=5) == b""
            hold.sendall(b"\r\n")
            assert parse_response(hold.recv(65536))[0] == 200
        time.sleep(0.2)
        assert h.call("GET", "/health", auth=False)[0] == 200


def test_generate_batch(harness_binary):
    with running(harness_binary, "--token-delay-ms", "20") as h:
        a, b = session(h), session(h)
        lanes = [dict(session_id=a, messages=MESSAGES, max_new_tokens=12),
                 dict(session_id=b, messages=MESSAGES, max_new_tokens=7, sampling={"do_sample": True, "seed": 9})]
        status, body = h.call("POST", "/generate_batch", {"requests": lanes})
        assert status == 200 and list(body) == ["results", "gpu_batching", "paired_decode_calls"]
        assert body["gpu_batching"] is True and body["paired_decode_calls"] == 6
        for lane, result in zip(lanes, body["results"]):
            assert list(result) == GENERATION_KEYS + ["paired_decode_calls"]
            assert result["session_id"] == lane["session_id"] and len(result["generated_ids"]) == lane["max_new_tokens"]
            assert result["gpu_batching"] is True and result["paired_decode_calls"] == 6
        assert body["results"][1]["do_sample"] is True and body["results"][1]["seed"] == 9
        for requests in ([lanes[0]], [lanes[0], lanes[0]], lanes + lanes[:1], [lanes[0], dict(lanes[1], max_new_tokens=0)]):
            assert h.call("POST", "/generate_batch", {"requests": requests}) == (400, {"error": "invalid_request"})
        assert h.call("POST", "/generate_batch", {"requests": lanes, "x": 1}) == (400, {"error": "invalid_request"})
        assert h.call("POST", "/generate_batch", {"requests": [lanes[0], dict(lanes[1], session_id="Z" * 32)]}) == (
            404, {"error": "unknown_session"})
        # A busy peer refuses the whole group without touching either lease.
        slow, _ = in_thread(generate, h, b, 100)  # the fake's phrase stops after ~23 tokens (~0.46 s)
        time.sleep(0.15)
        assert h.call("POST", "/generate_batch", {"requests": lanes}) == (409, {"error": "session_busy"})
        assert generate(h, a, 2)[0] == 200
        slow.join(30)


def test_batch_lane_cancellation(harness_binary):
    with running(harness_binary, "--token-delay-ms", "40") as h:
        a, b = session(h), session(h)
        lanes = [dict(session_id=s, messages=MESSAGES, max_new_tokens=16) for s in (a, b)]
        thread, result = in_thread(h.call, "POST", "/generate_batch", {"requests": lanes})
        time.sleep(0.5)
        assert h.call("POST", f"/sessions/{b}/cancel", {})[0] == 202
        thread.join(30)
        status, body = result["value"]
        x, y = body["results"]
        assert status == 200 and x["cancelled"] is False and len(x["generated_ids"]) == 16
        assert y["cancelled"] is True and y["cache_valid"] is False and y["consumed_tokens"] == 0
        assert 2 <= len(y["generated_ids"]) < 16 and y["generated_ids"] == x["generated_ids"][:len(y["generated_ids"])]
        assert body["paired_decode_calls"] == y["paired_decode_calls"] <= len(y["generated_ids"])
        assert h.call("POST", f"/sessions/{b}/cancel", {}) == (409, {"error": "generation_not_active"})


def test_frontend_is_serialized(harness_binary):
    with running(harness_binary) as h:
        a, b = session(h), session(h)
        heavy = [{"role": "user", "content": "éa" * 1_200_000}]  # 3.6 MB, under the 4 MiB body limit
        def heavy_until_served():  # the probes below can hold the front end first; the heavy request retries then
            while (r := generate(h, a, 8, heavy))[0] == 429:
                time.sleep(0.01)
            return r

        thread, result = in_thread(heavy_until_served)
        statuses = []
        deadline = time.monotonic() + 20
        while thread.is_alive() and time.monotonic() < deadline:
            statuses.append(generate(h, b, 2)[0])
            time.sleep(0.01)
        thread.join(60)
        assert result["value"] == (400, {"error": "invalid_request"})
        assert 429 in statuses, statuses
        assert generate(h, a, 2)[0] == 200


def test_shutdown_cancels_running_generation(harness_binary):
    h = Harness(harness_binary, "--token-delay-ms", "50")
    try:
        a = session(h)
        thread, result = in_thread(generate, h, a, 200)
        time.sleep(0.5)
        started = time.monotonic()
        h.process.send_signal(signal.SIGTERM)
        thread.join(10)
        assert h.process.wait(timeout=10) == 0 and time.monotonic() - started < 3
        assert h.process.stderr.read() == b""
        status, body = result["value"]
        assert status == 200 and body["cancelled"] is True
    finally:
        h.process.kill()


def peak_rss(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("no VmHWM")


def test_large_bodies_are_parsed_one_at_a_time(harness_binary):
    if not harness_binary.name.endswith("-plain"):
        pytest.skip("memory is measured on the uninstrumented build")
    with running(harness_binary) as h:
        before = peak_rss(h.process.pid)
        body = b'{"session_id":"' + b"A" * 32 + b'","max_new_tokens":8,"messages":[' + b",".join([b"0"] * 2_000_000) + b"]}"
        results = []

        def send():
            results.append(h.call("POST", "/generate", raw=body, timeout=120))
        threads = [threading.Thread(target=send) for _ in range(8)]
        [t.start() for t in threads]
        [t.join(120) for t in threads]
        assert results == [(400, {"error": "invalid_request"})] * 8
        # One parse tree is ~275 MB here; eight unserialized parses plus a deep copy reached 2.2 GB.
        assert peak_rss(h.process.pid) - before < 1_000_000_000


def test_engine_timeout_fails_closed(harness_binary):
    with running(harness_binary, "--token-delay-ms", "400", "--engine-timeout", "1") as h:
        a = session(h)
        started = time.monotonic()
        assert generate(h, a, 20) == (503, {"error": "worker_unavailable"})
        assert time.monotonic() - started < 4
        assert h.call("GET", "/health", auth=False) == (503, {"ready": False, "gpu_batching": False})
        assert h.call("POST", "/sessions", {"capacity": 64}) == (503, {"error": "worker_unavailable"})


def test_listener_closes_and_second_signal_exits(harness_binary):
    h = Harness(harness_binary, "--create-delay-ms", "8000")
    try:
        thread, _ = in_thread(h.call, "POST", "/sessions", {"capacity": 64})
        time.sleep(0.5)
        h.process.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", h.port), timeout=2)
        assert h.process.poll() is None  # still waiting for the engine call
        started = time.monotonic()
        h.process.send_signal(signal.SIGTERM)
        assert h.process.wait(timeout=5) == 1 and time.monotonic() - started < 2
        thread.join(10)
    finally:
        h.process.kill()
