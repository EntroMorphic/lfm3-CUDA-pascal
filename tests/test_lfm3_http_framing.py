"""HTTP framing parity: the native layer (lfm3_http.hpp) against tools/lfm3_server.py's real
LocalHTTPServer/Handler (http.server, http.client.parse_headers, email.feedparser) on raw requests.

The Python side runs with a stub service whose create() refuses every body, and the native harness
gets bodies its create() refuses too, so any request that reaches routing answers 400 invalid_request
on both sides. Every remaining difference is framing: request line, header parsing, auth, body
framing. Statuses must match; JSON error codes must match; http.server's HTML error pages (malformed
requests) are a documented difference and are compared by status only.
"""
import json
import os
from pathlib import Path
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
TOKEN = "A" * 40


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("g++ required")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT / "tools"))
    import lfm3_server

    class Stub:
        failed = False
        stopping = threading.Event()
        settings = lfm3_server.Settings(socket_timeout=2)

        def create(self, body):
            raise ValueError("refused")

    python = lfm3_server.LocalHTTPServer(("127.0.0.1", 0), Stub(), TOKEN)
    thread = threading.Thread(target=python.serve_forever, daemon=True)
    thread.start()
    work = tmp_path_factory.mktemp("framing")
    binary = work / "harness"
    subprocess.run(["g++", "-std=c++20", "-O1", "-I", str(ROOT / "native/include"), "-I", str(libjpeg_dir() / "include"),
                    str(ROOT / "native/tests/lfm3_http_harness.cpp"), *image_link_args("g++", [], work),
                    "-pthread", "-o", str(binary)], check=True)
    native = subprocess.Popen([str(binary), str(CHECKPOINT), "--socket-timeout", "2", "--total-capacity", "32"],
                              stdout=subprocess.PIPE, env=dict(os.environ, LFM3_SERVER_TOKEN=TOKEN))
    port = json.loads(native.stdout.readline())["port"]
    yield python.server_address[1], port
    native.send_signal(signal.SIGTERM)
    native.wait(timeout=30)
    python.shutdown()
    python.server_close()


def exchange(port, data):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        try:
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        chunks = []
        while True:
            try:
                block = s.recv(65536)
            except (ConnectionResetError, socket.timeout):
                break
            if not block:
                break
            chunks.append(block)
    raw = b"".join(chunks)
    if not raw:
        return None, None
    if not raw.startswith(b"HTTP/"):
        return "no-status-line", "html"  # http.server answering an HTTP/0.9-style request: body only
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    try:
        return status, json.loads(body)
    except ValueError:
        return status, "html"


def post(headers, body=b'{"capacity":1}', path="/sessions", line=None):
    start = line if line is not None else f"POST {path} HTTP/1.0".encode()
    return start + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"\r\n" + body


AUTH = b"Authorization: Bearer " + TOKEN.encode()
TYPE = b"Content-Type: application/json"
LEN = b"Content-Length: 14"
GOOD = [AUTH, TYPE, LEN]

CASES = {
    "good": post(GOOD),
    "health": b"GET /health HTTP/1.0\r\n\r\n",
    "health-lf-only": b"GET /health HTTP/1.0\n\n",
    "health-1.1": b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n",
    "health-1.10": b"GET /health HTTP/1.10\r\n\r\n",
    "health-01.0": b"GET /health HTTP/01.0\r\n\r\n",
    "double-slash": b"GET //health HTTP/1.0\r\n\r\n",
    "query": b"GET /health?x HTTP/1.0\r\n\r\n",
    "unknown-get": b"GET /nope HTTP/1.0\r\n\r\n",
    "unknown-post-unauth": post([TYPE, LEN], path="/nope"),
    "unknown-post": post(GOOD, path="/nope"),
    "version-2": b"GET /health HTTP/2.0\r\n\r\n",
    "version-bad": b"GET /health HTTP/1\r\n\r\n",
    "version-word": b"GET /health FTP/1.0\r\n\r\n",
    "two-words-post": b"POST /sessions\r\n\r\n",
    "four-words": b"GET / health HTTP/1.0\r\n\r\n",
    "nbsp-separator": b"GET\xa0/health HTTP/1.0\r\n\r\n",
    "nel-separator": b"GET /health\x85HTTP/1.0\r\n\r\n",
    "tab-separator": b"GET\t/health\tHTTP/1.0\r\n\r\n",
    "head": b"HEAD /health HTTP/1.0\r\n\r\n",
    "put": post(GOOD, line=b"PUT /sessions HTTP/1.0"),
    "no-auth": post([TYPE, LEN]),
    "bad-token": post([b"Authorization: Bearer " + b"B" * 40, TYPE, LEN]),
    "lower-bearer": post([b"Authorization: bearer " + TOKEN.encode(), TYPE, LEN]),
    "auth-twice": post([AUTH, AUTH, TYPE, LEN]),
    "auth-trailing-space": post([AUTH + b" ", TYPE, LEN]),
    "auth-leading-spaces": post([b"Authorization:    Bearer " + TOKEN.encode(), TYPE, LEN]),
    "auth-lower-name": post([b"authorization: Bearer " + TOKEN.encode(), TYPE, LEN]),
    "auth-folded": post([b"Authorization: Bearer", b" " + TOKEN.encode(), TYPE, LEN]),
    "auth-non-ascii": post([AUTH + b"\xe9", TYPE, LEN]),
    "no-length": post([AUTH, TYPE]),
    "two-lengths": post([AUTH, TYPE, LEN, LEN]),
    "chunked": post([AUTH, TYPE, LEN, b"Transfer-Encoding: chunked"]),
    "empty-te": post([AUTH, TYPE, LEN, b"Transfer-Encoding:"]),
    "length-plus": post([AUTH, TYPE, b"Content-Length: +14"]),
    "length-space": post([AUTH, TYPE, b"Content-Length: 14 "]),
    "length-zero": post([AUTH, TYPE, b"Content-Length: 0"], body=b""),
    "length-leading-zeros": post([AUTH, TYPE, b"Content-Length: 0014"]),
    "length-9-digits": post([AUTH, TYPE, b"Content-Length: 123456789"]),
    "length-too-big": post([AUTH, TYPE, b"Content-Length: 4194305"], body=b""),
    "length-short-body": post([AUTH, TYPE, b"Content-Length: 40"]),
    "no-type": post([AUTH, LEN]),
    "type-charset": post([AUTH, b"Content-Type: application/json; charset=utf-8", LEN]),
    "type-upper": post([AUTH, b"Content-Type: APPLICATION/JSON", LEN]),
    "type-twice": post([AUTH, TYPE, TYPE, LEN]),
    "bad-json": post([AUTH, TYPE, b"Content-Length: 5"], body=b"{nope"),
    # http.client / email.feedparser header parsing
    "bare-cr-splits": post([b"X: a\r" + AUTH, TYPE, LEN]),
    "bare-cr-blank-ends": post([b"X: a\r", AUTH, TYPE, LEN]),
    "type-cr-cr": post([AUTH, TYPE + b"\r", LEN]),
    "space-in-name-ends": post([b"X Y: 1", AUTH, TYPE, LEN]),
    "space-in-name-after": post([AUTH, TYPE, LEN, b"Bad Name: 1", LEN]),
    "empty-name": b"GET /health HTTP/1.0\r\n: 1\r\n\r\n",
    "empty-name-post": post([b": 1", AUTH, TYPE, LEN]),
    "no-colon-ends": post([b"Garbage", AUTH, TYPE, LEN]),
    "leading-continuation": post([b" folded", AUTH, TYPE, LEN]),
    "from-line": post([b"From someone", AUTH, TYPE, LEN]),
    "name-with-del": post([b"X\x7fY: 1", AUTH, TYPE, LEN]),
    "100-headers": b"GET /health HTTP/1.0\r\n" + b"X: y\r\n" * 100 + b"\r\n",
    "101-headers": b"GET /health HTTP/1.0\r\n" + b"X: y\r\n" * 101 + b"\r\n",
}
for size in (65533, 65534, 65535, 65536, 65537):
    path = b"/" + b"a" * (size - len(b"GET  HTTP/1.0\r\n") - 1)
    CASES[f"request-line-{size}-crlf"] = b"GET " + path + b" HTTP/1.0\r\n\r\n"
    CASES[f"request-line-{size}-lf"] = b"GET " + path + b"x HTTP/1.0\n\n"
    value = b"a" * (size - len(b"X: \r\n"))
    CASES[f"header-line-{size}-crlf"] = b"GET /health HTTP/1.0\r\nX: " + value + b"\r\n\r\n"
    CASES[f"header-line-{size}-lf"] = b"GET /health HTTP/1.0\nX: " + value + b"a\n\n"


@pytest.mark.parametrize("name", list(CASES))
def test_framing_matches_python(servers, name):
    python_port, native_port = servers
    expected = exchange(python_port, CASES[name])
    got = exchange(native_port, CASES[name])
    if expected[0] == "no-status-line":
        # Documented difference: where http.server answers body-only (no status line), native sends a
        # real status line with a JSON error.
        assert got in ((400, {"error": "bad_request"}), (505, {"error": "http_version"})), (name, expected, got)
        return
    assert got[0] == expected[0], (name, expected, got)
    if expected[1] != "html":
        assert got[1] == expected[1], (name, expected, got)
