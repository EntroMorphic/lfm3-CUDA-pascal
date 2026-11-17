"""Host tests for concurrent speech (validation/speech-batch-v1): the Service's speech batch (native/include/lfm/lfm3_http.hpp,
ops speech_admit/speech_step) behind the harness's fake engine, on every build variant (plain, ASan+UBSan, TSan).

The fake runs one chunk of every busy row per step, after --token-delay-ms (native/tests/lfm3_http_harness.cpp), so
times show what ran together. Checked: requests in a batch get the replies they get alone; rows fill and refill from
the queue; the queue's limit; a text job runs between speech steps and holds back no later speech (validation/text-wait-v1;
before, the batch drained for it); one request cancelled or
abandoned leaves the others; shutdown cancels every row and fails the queued; an engine failure fails them all; the
single-request path (--speech-rows 0) still serves.
"""
import threading
import time

import pytest

from test_lfm3_http_native import harness_binary, running  # noqa: F401  (fixture)
from test_lfm3_http_speech import LONG, tts
from test_lfm3_http_speech_stream import pcm, stream

SHORT = "Short sentence here."


def together(calls):
    """Runs the calls at once (each a function of no arguments); returns their results in order and the wall time."""
    out = [None] * len(calls)

    def run(i):
        out[i] = calls[i]()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    t0 = time.monotonic()
    for t in threads:
        t.start()
        if len(calls) <= 16: time.sleep(0.02)  # one at a time through the frontend (else 429 frontend_busy)
    for t in threads:
        t.join(60)
    return out, time.monotonic() - t0


def test_requests_in_a_batch_get_their_replies_alone(harness_binary):
    with running(harness_binary, "--speech-rows", "0") as h:
        alone = [h.call("POST", "/speech", tts(LONG, seed=i)) for i in range(4)]
    with running(harness_binary, "--speech-rows", "4", "--token-delay-ms", "300") as h:
        got, wall = together([lambda i=i: h.call("POST", "/speech", tts(LONG, seed=i)) for i in range(4)])
    assert got == alone and all(s == 200 for s, _ in got)
    assert wall < 2.0, wall  # 3 steps of 0.3 s together, not 12 one after another


def test_rows_fill_then_refill_from_the_queue(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "300", "--queue-size", "4") as h:
        got, wall = together([lambda i=i: h.call("POST", "/speech", tts(LONG, seed=i)) for i in range(4)])
    assert all(s == 200 and b["cancelled"] is False for s, b in got)
    assert 1.5 < wall < 3.5, wall  # two rounds of 3 steps


def test_the_queue_has_a_limit(harness_binary):
    with running(harness_binary, "--speech-rows", "1", "--token-delay-ms", "300", "--queue-size", "1") as h:
        got, _ = together([lambda i=i: h.call("POST", "/speech", tts(LONG, seed=i)) for i in range(3)])
    statuses = sorted(s for s, _ in got)
    assert statuses == [200, 200, 429] and any(b == {"error": "queue_full"} for _, b in got)


def test_a_text_job_runs_between_speech_steps_and_holds_back_no_later_speech(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "200") as h:
        status, created = h.call("POST", "/sessions", {"capacity": 512})
        assert status == 200
        done = {}

        def mark(name, f):
            r = f()
            done[name] = time.monotonic()
            return r
        long_speech = lambda: mark("speech", lambda: h.call("POST", "/speech", tts(" ".join([LONG] * 3))))  # 9 steps
        text = lambda: mark("text", lambda: h.call("POST", "/generate", dict(  # noqa: E731
            session_id=created["session_id"], max_new_tokens=4, messages=[{"role": "user", "content": "hello there"}])))
        later = lambda: mark("later", lambda: h.call("POST", "/speech", tts(SHORT)))  # noqa: E731
        t = threading.Thread(target=long_speech)
        t.start()
        time.sleep(0.5)
        t2 = threading.Thread(target=text)
        t2.start()
        time.sleep(0.2)
        t3 = threading.Thread(target=later)
        t3.start()
        for x in (t, t2, t3):
            x.join(60)
        # validation/text-wait-v1: the text job runs between speech steps (one token each), so it ends before the
        # 9-step speech; the later speech takes the free row meanwhile instead of waiting for a drain
        assert done["text"] < done["speech"] and done["later"] < done["speech"], done


def test_cancelling_one_stream_leaves_the_other(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        long = " ".join([LONG] * 4)  # 12 steps
        got, _ = together([lambda: stream(h, tts(long, seed=3, stream=True), close_after=1),
                           lambda: stream(h, tts(long, seed=5, stream=True))])
        status, events, _ = got[1]
        assert status == 200 and events[-1]["event"] == "done" and events[-1]["cancelled"] is False
        assert pcm(events) == b"".join(bytes([k, 5, 255, 127]) for k in range(12))


def test_an_admitted_request_runs_past_its_deadline_to_its_reply(harness_binary):
    # in a batch every row runs slower than alone, so the request deadline bounds only the wait in the queue: once
    # admitted, a request runs to its end (bounded by the step budget), and its reply equals its reply alone
    long = " ".join([LONG] * 3)  # 9 steps of 0.4 s, past the 1 s deadline
    with running(harness_binary, "--speech-rows", "0") as h:
        alone = h.call("POST", "/speech", tts(long, seed=2))
    with running(harness_binary, "--speech-rows", "1", "--token-delay-ms", "400", "--request-timeout", "1") as h:
        assert h.call("POST", "/speech", tts(long, seed=2)) == alone
        status, events, _ = stream(h, tts(long, seed=2, stream=True))
        assert status == 200 and events[-1]["event"] == "done" and events[-1]["cancelled"] is False
        assert {k: v for k, v in events[-1].items() if k != "event"} == {k: v for k, v in alone[1].items()
                                                                        if k != "audio_wav_base64"}


def test_a_request_still_queued_at_its_deadline_is_a_504_and_never_runs(harness_binary):
    import socket
    from test_lfm3_http_speech_stream import request_bytes
    with running(harness_binary, "--speech-rows", "1", "--token-delay-ms", "400", "--request-timeout", "1") as h:
        long = " ".join([LONG] * 3)
        first = threading.Thread(target=lambda: results.update(first=h.call("POST", "/speech", tts(long))))
        results = {}
        first.start()
        time.sleep(0.2)
        assert h.call("POST", "/speech", tts(SHORT)) == (504, {"error": "queue_deadline"})  # queued all along
        # a queued stream whose reply was committed (its client half-closed) ends with an error event instead
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(SHORT, stream=True)))
        s.shutdown(socket.SHUT_WR)
        data = b""
        while block := s.recv(65536):
            data += block
        s.close()
        assert data.startswith(b"HTTP/1.0 200 OK") and data.rstrip().endswith(b'{"event":"error","error":"queue_deadline"}')
        first.join(30)
        assert results["first"][0] == 200 and results["first"][1]["cancelled"] is False
        t0 = time.monotonic()
        assert h.call("POST", "/speech", tts(SHORT))[0] == 200 and time.monotonic() - t0 < 1.0  # the abandoned never ran


def test_the_accept_queue_holds_a_crowd(harness_binary):
    # the listening socket's backlog (ss shows it as Send-Q) is max(16, handlers), not http.server's 5
    import shutil
    import subprocess
    if not shutil.which("ss"):
        pytest.skip("ss required")
    for handlers, want in (("8", 16), ("40", 40)):
        with running(harness_binary, "--handlers", handlers) as h:
            out = subprocess.run(["ss", "-ltnH", f"sport = :{h.port}"], capture_output=True, text=True, check=True).stdout
            assert int(out.split()[2]) == want, out


def test_shutdown_cancels_every_row_and_fails_the_queued(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "300") as h:
        long = " ".join([LONG] * 4)
        results = {}

        def call(i):
            results[i] = h.call("POST", "/speech", tts(long, seed=i))
        threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        time.sleep(0.6)
        h.stop()
        for t in threads:
            t.join(30)
    cancelled = [r for r in results.values() if r[0] == 200]
    assert len(cancelled) == 2 and all(b["cancelled"] is True for _, b in cancelled), results
    assert [r for r in results.values() if r[0] != 200] == [(503, {"error": "worker_unavailable"})], results


def test_an_engine_failure_fails_every_row(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--fail-op", "speech_step",
                 "--fail-after", "3") as h:
        got, _ = together([lambda: h.call("POST", "/speech", tts(LONG, seed=1)),
                           lambda: stream(h, tts(LONG, seed=2, stream=True))])
        assert got[0] == (503, {"error": "worker_unavailable"}), got[0]
        status, events, _ = got[1]
        assert (status, events[-1]) in ((200, {"event": "error", "error": "worker_unavailable"}),
                                        (503, {"error": "worker_unavailable"})), got[1]
        assert h.call("POST", "/speech", tts(SHORT))[0] == 503


@pytest.mark.parametrize("rows", ["0", "1", "3"])
def test_streamed_and_unstreamed_replies_are_the_same_at_any_row_count(harness_binary, rows):
    with running(harness_binary, "--speech-rows", rows) as h:
        status, plain = h.call("POST", "/speech", tts(LONG, seed=6))
        status2, events, _ = stream(h, tts(LONG, seed=6, stream=True))
        assert status == status2 == 200 and events[-1]["event"] == "done"
        assert {k: v for k, v in events[-1].items() if k != "event"} == {k: v for k, v in plain.items()
                                                                       if k != "audio_wav_base64"}


def test_steady_text_traffic_does_not_starve_speech(harness_binary):
    # the second red-team's reproduction: two clients keep /generate busy; a /speech arriving among them is served in
    # its arrival order, not after every later text job
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "20", "--request-timeout", "4") as h:
        sessions = [h.call("POST", "/sessions", {"capacity": 512})[1]["session_id"] for _ in range(2)]
        stop = threading.Event()

        def loop(sid):
            while not stop.is_set():
                h.call("POST", "/generate", dict(session_id=sid, max_new_tokens=16,
                                                 messages=[{"role": "user", "content": "hello there"}]))
        threads = [threading.Thread(target=loop, args=(sid,)) for sid in sessions]
        for t in threads:
            t.start()
        time.sleep(0.5)
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts(SHORT))
        wait = time.monotonic() - t0
        stop.set()
        for t in threads:
            t.join(30)
        assert status == 200 and body["cancelled"] is False and wait < 2.0, (status, body, wait)


def test_a_client_that_leaves_while_queued_frees_its_place(harness_binary):
    import socket
    from test_lfm3_http_speech_stream import request_bytes
    with running(harness_binary, "--speech-rows", "1", "--token-delay-ms", "300", "--queue-size", "1") as h:
        busy = threading.Thread(target=lambda: h.call("POST", "/speech", tts(" ".join([LONG] * 2))))  # 6 steps
        busy.start()
        time.sleep(0.2)
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(LONG, stream=True)))  # queued: the one place
        time.sleep(0.3)
        s.close()  # gone while queued
        time.sleep(0.5)  # the next cycle drops it
        status, body = h.call("POST", "/speech", tts(SHORT))  # the place is free again: queued, not a 429
        busy.join(30)
        assert status == 200 and body["cancelled"] is False, (status, body)


def test_a_crowd_arriving_at_once_gets_replies_not_resets(harness_binary):
    # 120 clients at once: each gets an HTTP reply (200, or a 429 to retry); none is reset. The accept queue was
    # http.server's 5, which overflowed under such a crowd and reset about one connection in 360.
    import json
    import socket
    from test_lfm3_http_native import parse_response
    from test_lfm3_http_speech_stream import request_bytes

    def one(i):
        s = socket.create_connection(("127.0.0.1", h.port), timeout=60)
        try:
            s.sendall(request_bytes(tts(LONG, seed=i)))
            data = b""
            while block := s.recv(65536):
                data += block
            return parse_response(data)[0]
        except ConnectionResetError:
            return "reset"
        finally:
            s.close()
    with running(harness_binary, "--handlers", "128", "--queue-size", "64", "--speech-rows", "8", "--token-delay-ms", "50") as h:
        for _ in range(3):
            got, _ = together([lambda i=i: one(i) for i in range(120)])
            assert "reset" not in got and set(got) <= {200, 429}, got


def test_an_unstreamed_client_that_resets_frees_its_row(harness_binary):
    # audio-assurance-v1 red-team: an unstreamed request whose connection is broken is cancelled at the next step;
    # before, it ran its 12 steps (3.6 s) for no one while the next request waited
    import socket
    import struct
    from test_lfm3_http_speech_stream import request_bytes
    with running(harness_binary, "--speech-rows", "1", "--token-delay-ms", "300") as h:
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(" ".join([LONG] * 4))))  # 12 chunks, one a step
        time.sleep(0.5)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()  # a reset
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts(SHORT))
        assert status == 200 and body["cancelled"] is False, (status, body)
        assert time.monotonic() - t0 < 2.0, time.monotonic() - t0
