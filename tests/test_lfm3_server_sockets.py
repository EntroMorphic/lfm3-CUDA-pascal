"""Actual HTTP parser/framing/deadline tests over socket pairs; no listener/GPU."""
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_server as server

TOKEN = "a" * 32


class Service:
    failed = False
    stopping = threading.Event()

    def __init__(self):
        self.settings = server.Settings(socket_timeout=0.2)
        self.calls = []

    def create(self, body):
        self.calls.append(body)
        return {"session_id": "b" * 32, "capacity": body["capacity"]}


class Connection:
    def __init__(self):
        self.client, self.peer = socket.socketpair()
        self.client.settimeout(2)
        self.service = Service()
        self.errors = []
        self.owner = SimpleNamespace(service=self.service, token=TOKEN)
        def handle():
            try:
                server.Handler(self.peer, ("127.0.0.1", 1), self.owner)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                self.peer.close()
        self.thread = threading.Thread(target=handle, daemon=True)
        self.thread.start()

    def read(self):
        chunks = []
        while True:
            try:
                block = self.client.recv(65536)
            except ConnectionResetError:
                # Closing a socket with unread trickle bytes can send RST after
                # its response. Keep checking the complete response below.
                break
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)

    def close(self):
        self.client.close()
        self.thread.join(2)
        assert not self.thread.is_alive(), "HTTP handler escaped its deadline"
        assert not self.errors, self.errors


@pytest.fixture
def connection():
    value = Connection()
    yield value
    value.close()


def request(headers=(), body=b'{"capacity":32}', auth=True):
    fields = [b"POST /sessions HTTP/1.1", b"Host: localhost"]
    if auth:
        fields.append(b"Authorization: Bearer " + TOKEN.encode())
    fields.extend(headers or [b"Content-Type: application/json", b"Content-Length: " + str(len(body)).encode()])
    return b"\r\n".join(fields) + b"\r\n\r\n" + body


def status(response):
    return int(response.split(b" ", 2)[1])


def test_real_response_framing_and_single_request(connection):
    connection.client.sendall(request() + b"GET /health HTTP/1.1\r\n\r\n")
    response = connection.read()
    head, body = response.split(b"\r\n\r\n", 1)
    headers = dict(line.split(b": ", 1) for line in head.split(b"\r\n")[1:])
    assert status(response) == 200
    assert int(headers[b"Content-Length"]) == len(body)
    assert headers[b"Connection"] == b"close"
    assert headers[b"Cache-Control"] == b"no-store"
    assert len(connection.service.calls) == 1
    assert b"HTTP/" not in body


@pytest.mark.parametrize("headers,body,expected", [
    ([b"Content-Length: 2", b"Content-Length: 2", b"Content-Type: application/json"], b"{}", 400),
    ([b"Content-Length: 2", b"Transfer-Encoding: chunked", b"Content-Type: application/json"], b"{}", 400),
    ([b"Content-Length: -1", b"Content-Type: application/json"], b"", 400),
    ([b"Content-Length: 4194305", b"Content-Type: application/json"], b"", 413),
    ([b"Content-Length: 2", b"Content-Type: text/plain"], b"{}", 415),
    ([b"Content-Length: 3", b"Content-Type: application/json"], b"{\xff}", 400),
    ([b"Content-Length: 13", b"Content-Type: application/json"], b'{"a":1,"a":2}', 400),
])
def test_real_parser_rejects_framing_and_json(connection, headers, body, expected):
    connection.client.sendall(request(headers, body))
    assert status(connection.read()) == expected
    assert not connection.service.calls


def test_authentication_rejects_before_reading_body(connection):
    connection.client.sendall(request([b"Content-Length: 4000000"], b"", auth=False))
    assert status(connection.read()) == 401
    assert not connection.service.calls


def test_duplicate_authorization_rejected(connection):
    connection.client.sendall(request([b"Authorization: Bearer " + TOKEN.encode(), b"Content-Length: 2"], b"{}"))
    assert status(connection.read()) == 401


def test_incomplete_body_rejected(connection):
    connection.client.sendall(request([b"Content-Length: 30", b"Content-Type: application/json"], b"{"))
    connection.client.shutdown(socket.SHUT_WR)
    assert status(connection.read()) == 400


@pytest.mark.parametrize("part", ["headers", "body"])
def test_absolute_deadlines_bound_trickling(connection, part):
    if part == "body":
        connection.client.sendall(request([b"Content-Length: 100", b"Content-Type: application/json"], b""))
    else:
        connection.client.sendall(b"POST /sessions HTTP/1.1\r\nX-Trickle: ")
    start = time.monotonic()
    for _ in range(10):
        try:
            connection.client.sendall(b"x")
        except (BrokenPipeError, ConnectionResetError):
            break
        if not connection.thread.is_alive():
            break
        connection.thread.join(0.05)
    connection.thread.join(0.5)
    assert not connection.thread.is_alive()
    assert time.monotonic() - start < 1
    assert not connection.service.calls
    reply = connection.read()
    if part == "body":
        assert status(reply) == 408


def test_handler_saturation_prevents_thread_creation(monkeypatch):
    # Exercise real admission method without binding a listening socket.
    owner = object.__new__(server.LocalHTTPServer)
    owner.slots = threading.BoundedSemaphore(1)
    dropped, started = [], []
    owner.shutdown_request = lambda request: dropped.append(request)
    monkeypatch.setattr(server.ThreadingHTTPServer, "process_request", lambda self, req, addr: started.append(req))
    owner.process_request("accepted", ("127.0.0.1", 1))
    owner.process_request("saturated", ("127.0.0.1", 2))
    assert started == ["accepted"] and dropped == ["saturated"]
    monkeypatch.setattr(server.ThreadingHTTPServer, "process_request_thread", lambda *args: None)
    owner.process_request_thread("accepted", ("127.0.0.1", 1))
    owner.process_request("after-release", ("127.0.0.1", 3))
    assert started == ["accepted", "after-release"]


def test_nonreading_peer_cannot_hold_response_writer(connection):
    connection.peer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    connection.service.create = lambda body: {"text": "x" * (2 * 1024 * 1024)}
    connection.client.sendall(request())
    # Intentionally don't read until the write deadline has released the handler.
    connection.thread.join(1)
    assert not connection.thread.is_alive()
    assert not connection.errors
    response = connection.read()
    head, partial = response.split(b"\r\n\r\n", 1)
    headers = dict(line.split(b": ", 1) for line in head.split(b"\r\n")[1:])
    assert len(partial) < int(headers[b"Content-Length"])
    assert response.count(b"HTTP/1.0") == 1  # No erroneous second reply on timeout.
