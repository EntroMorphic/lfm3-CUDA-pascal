"""Host tests for streamed POST /speech ("stream": true; native/include/lfm/lfm3_http.hpp, validation/speech-stream-v1)
behind the harness's fake engine, on every build variant (plain, ASan+UBSan, TSan).

The fake streams as WorkerCore does (native/tests/lfm3_http_harness.cpp): per chunk, the phrase's text slots (chunk 0,
not in audio mode), one audio piece of 2 samples (bytes k, seed % 256, 255, 127 for chunk k) and the chunk's end. Checked: the
reply's framing; every event line byte for byte against Python's json.dumps of its content; the events' order and
content against the unstreamed reply for the same request; refusals and failures before the first event as ordinary JSON
errors; failures, deadlines and shutdown after it as a last event; a client that goes away cancels its generation.
The harness runs the concurrent speech batch (2 rows by default); tests that show a cancellation by a follow-up request
waiting behind the cancelled one run with one row, where it must wait.
"""
import base64
import json
import signal
import socket
import threading
import time

import pytest

from test_lfm3_http_native import TOKEN, harness_binary, parse_response, running  # noqa: F401  (fixture)
from test_lfm3_http_speech import LONG, SYSTEM, tts

INTERLEAVED = [{"role": "system", "content": "Respond with interleaved text and audio."},
               {"role": "user", "content": "What are three tips for staying focused while studying?"}]


def request_bytes(body):
    payload = json.dumps(body, ensure_ascii=False).encode()
    head = ["POST /speech HTTP/1.1", "Host: 127.0.0.1", "Content-Type: application/json",
            f"Content-Length: {len(payload)}", "Authorization: Bearer " + TOKEN]
    return ("\r\n".join(head) + "\r\n\r\n").encode() + payload


def stream(h, body, *, close_after=None, timeout=60):
    """Sends a /speech request and reads the reply as it arrives. A streamed reply (200, NDJSON) returns
    (200, events, arrival times); anything else returns parse_response's (status, body) and None. close_after: close
    the connection once that many events have arrived (returns what arrived)."""
    s = socket.create_connection(("127.0.0.1", h.port), timeout=timeout)
    try:
        s.sendall(request_bytes(body))
        data, head, events, times = b"", None, [], []
        while True:
            try:
                block = s.recv(65536)
            except ConnectionResetError:
                block = b""
            if head is None:
                data += block
                if b"\r\n\r\n" not in data and block:
                    continue
                if b"x-ndjson" not in data.partition(b"\r\n\r\n")[0]:
                    while block:  # an ordinary reply: read it all
                        block = s.recv(65536)
                        data += block
                    status, value = parse_response(data)
                    return status, value, None
                head, _, data = data.partition(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                assert lines[0] == "HTTP/1.0 200 OK"
                headers = dict(line.split(": ", 1) for line in lines[1:])
                assert headers["Content-Type"] == "application/x-ndjson; charset=utf-8"
                assert "Content-Length" not in headers and headers["Cache-Control"] == "no-store"
                assert headers["Connection"] == "close"
            else:
                data += block
            while b"\n" in data:
                line, _, data = data.partition(b"\n")
                event = json.loads(line)
                assert line == json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                events.append(event)
                times.append(time.monotonic())
                if close_after is not None and len(events) >= close_after:
                    return 200, events, times
            if not block:
                assert data == b"", data  # the body ends with a whole line
                return 200, events, times
    finally:
        s.close()


def pcm(events):
    return b"".join(base64.b64decode(e["pcm_base64"]) for e in events if e["event"] == "audio")


def unstreamed_fields(body):
    return {k: v for k, v in body.items() if k != "audio_wav_base64"}


def test_split_audio_streams_each_chunk_then_the_reply(harness_binary):
    with running(harness_binary) as h:
        status, plain = h.call("POST", "/speech", tts(LONG, seed=7))
        assert status == 200
        status, events, _ = stream(h, tts(LONG, seed=7, stream=True))
        assert status == 200
        kinds = [e["event"] for e in events]
        assert kinds == ["audio", "chunk"] * 3 + ["done"]  # no text in audio mode
        assert [e["chunk"] for e in events[:-1]] == [0, 0, 1, 1, 2, 2]
        assert pcm(events) == b"".join(bytes([k, 7, 255, 127]) for k in range(3))
        chunks = [e for e in events if e["event"] == "chunk"]
        assert [{k: v for k, v in c.items() if k not in ("event", "chunk")} for c in chunks] == plain["chunks"]
        done = events[-1]
        assert list(done)[0] == "event" and {k: v for k, v in done.items() if k != "event"} == unstreamed_fields(plain)
        assert list(done)[1:] == [k for k in plain if k != "audio_wav_base64"]  # same fields, same order


@pytest.mark.parametrize("mode", ["interleaved", "text"])
def test_text_streams_in_pieces_that_join_to_the_reply_text(harness_binary, mode):
    with running(harness_binary) as h:
        status, plain = h.call("POST", "/speech", dict(messages=INTERLEAVED, mode=mode, max_new_tokens=256))
        assert status == 200 and "\U0001F642" in plain["text"] and "\u2028" in plain["text"]
        status, events, _ = stream(h, dict(messages=INTERLEAVED, mode=mode, max_new_tokens=256, stream=True))
        assert status == 200
        texts = [e for e in events if e["event"] == "text"]
        assert len(texts) > 3 and all(e["chunk"] == 0 and e["text"] for e in texts)
        assert "".join(e["text"] for e in texts) == plain["text"] == events[-1]["text"]
        assert any("\U0001F642" in e["text"] for e in texts)  # a 4-byte character is never split across pieces
        first_audio = [e["event"] for e in events].index("audio")
        assert all(e["event"] == "text" for e in events[:first_audio])  # the fake's text comes first
        assert {k: v for k, v in events[-1].items() if k != "event"} == unstreamed_fields(plain)


def test_stream_false_is_the_unstreamed_reply(harness_binary):
    with running(harness_binary) as h:
        assert h.call("POST", "/speech", tts(LONG, seed=3, stream=False)) == h.call("POST", "/speech", tts(LONG, seed=3))


@pytest.mark.parametrize("extra", [dict(stream="yes"), dict(stream=1), dict(stream=None),
                                   dict(stream=True, mode="interleaved", split=True),  # split is audio-mode only
                                   dict(stream=True, max_new_tokens=0), dict(stream=True, extra=1)])
def test_refusals_are_ordinary_errors(harness_binary, extra):
    with running(harness_binary) as h:
        status, body, events = stream(h, dict(tts(LONG), **extra))
        assert events is None and status == 400 and body == {"error": "invalid_request"}
        assert stream(h, tts(LONG, stream=True))[0] == 200


def test_unsupported_speech_is_an_ordinary_501(harness_binary):
    with running(harness_binary, "--speech", "off") as h:
        status, body, events = stream(h, tts(LONG, stream=True))
        assert events is None and status == 501 and body == {"error": "speech_unsupported"}


def test_an_engine_fault_before_any_event_is_an_ordinary_503(harness_binary):
    # a malformed first audio piece: nothing was sent, so the failure is a plain JSON 503, and the service fails closed
    with running(harness_binary, "--speech-stream-fault", "malformed") as h:
        status, body, events = stream(h, tts(LONG, stream=True))
        assert events is None and status == 503 and body == {"error": "worker_unavailable"}
        assert h.call("POST", "/speech", tts(LONG))[0] == 503


@pytest.mark.parametrize("fault", ["samples", "slots", "chunk"])
def test_a_reply_that_disagrees_with_the_stream_ends_it_with_an_error(harness_binary, fault):
    with running(harness_binary, "--speech-stream-fault", fault) as h:
        status, events, _ = stream(h, dict(messages=INTERLEAVED, mode="interleaved", max_new_tokens=256, stream=True))
        assert status == 200 and events[-1] == {"event": "error", "error": "worker_unavailable"}
        assert all(e["event"] != "done" for e in events) and len(events) > 1
        assert h.call("POST", "/speech", tts(LONG))[0] == 503  # an engine fault fails the service closed


def test_events_arrive_as_the_engine_makes_them(harness_binary):
    with running(harness_binary, "--token-delay-ms", "300") as h:
        t0 = time.monotonic()
        status, events, times = stream(h, tts(LONG, stream=True))
        assert status == 200
        audio = [t - t0 for e, t in zip(events, times) if e["event"] == "audio"]
        # one chunk every 300 ms: the first arrives long before the last, not all at the end
        assert len(audio) == 3 and audio[0] < 0.6 and audio[2] - audio[0] > 0.45, audio


def test_a_client_that_goes_away_is_cancelled(harness_binary):
    with running(harness_binary, "--token-delay-ms", "300", "--speech-rows", "1") as h:
        long = " ".join([LONG] * 6)  # 18 chunks, ~5.4 s if run to the end
        status, events, _ = stream(h, tts(long, stream=True), close_after=1)
        assert status == 200 and events[0]["event"] == "audio"
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts(LONG))  # queued behind the abandoned one until it is cancelled
        assert status == 200 and body["cancelled"] is False
        assert time.monotonic() - t0 < 2.5, time.monotonic() - t0


def test_a_client_that_goes_away_before_the_first_event_is_cancelled(harness_binary):
    with running(harness_binary, "--token-delay-ms", "1000", "--speech-rows", "1") as h:
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(" ".join([LONG] * 3), stream=True)))  # 9 chunks, 9 s if run to the end
        time.sleep(0.2)
        s.close()  # before its first event (1 s)
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts(LONG))  # queued behind it until it is cancelled
        assert status == 200 and body["cancelled"] is False
        # the leaver is cancelled at its first chunk boundary (1 s), then this request's 3 chunks run (3 s)
        assert time.monotonic() - t0 < 5.5, time.monotonic() - t0


def test_shutdown_mid_stream_ends_with_a_cancelled_done(harness_binary):
    with running(harness_binary, "--token-delay-ms", "400") as h:
        result = {}
        t = threading.Thread(target=lambda: result.update(r=stream(h, tts(" ".join([LONG] * 3), stream=True))))
        t.start()
        time.sleep(0.6)
        h.stop()
        t.join(30)
        status, events, _ = result["r"]
        assert status == 200 and events[0]["event"] == "audio"
        assert events[-1] == {"event": "done", "mode": "audio", "cancelled": True, "seed": 0, "split": True,
                              "normalize": True}


def test_deadline_after_the_start_is_an_error_event(harness_binary):
    # the single-request path (--speech-rows 0); in the speech batch an admitted request runs on past its deadline
    # (test_lfm3_http_speech_batch.py)
    with running(harness_binary, "--token-delay-ms", "300", "--request-timeout", "1", "--speech-rows", "0") as h:
        status, events, _ = stream(h, tts(" ".join([LONG] * 3), stream=True))
        assert status == 200 and events[0]["event"] == "audio"
        assert events[-1] == {"event": "error", "error": "request_deadline_outcome_unknown"}
        status, body = h.call("POST", "/speech", tts("Short sentence here."))  # the abandoned one was cancelled
        assert status == 200 and body["cancelled"] is False


def test_deadline_before_the_start_is_an_ordinary_504(harness_binary):
    with running(harness_binary, "--token-delay-ms", "1500", "--request-timeout", "1", "--speech-rows", "0") as h:
        status, body, events = stream(h, tts(LONG, stream=True))
        assert events is None and status == 504 and body == {"error": "request_deadline_outcome_unknown"}


def test_concurrent_streams_each_get_their_own_events(harness_binary):
    with running(harness_binary, "--token-delay-ms", "50") as h:
        results = [None] * 3
        def run(i):
            for _ in range(200):  # one request is prepared at a time; a second meanwhile gets 429 frontend_busy
                results[i] = stream(h, tts(LONG, seed=i, stream=True))
                if results[i][:2] != (429, {"error": "frontend_busy"}):
                    return
                time.sleep(0.01)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        for i, (status, events, _) in enumerate(results):
            assert status == 200 and events[-1]["event"] == "done" and events[-1]["seed"] == i
            assert pcm(events) == b"".join(bytes([k, i, 255, 127]) for k in range(3))  # its own audio


def test_a_client_that_half_closes_after_its_request_gets_the_whole_stream(harness_binary):
    # shutdown(SHUT_WR) after the request (nc -N, socat) is not leaving: the reply is committed and streamed in full
    with running(harness_binary, "--token-delay-ms", "100") as h:
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(LONG, seed=4, stream=True)))
        s.shutdown(socket.SHUT_WR)
        data = b""
        while block := s.recv(65536):
            data += block
        s.close()
        head, _, body = data.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.0 200 OK") and b"application/x-ndjson" in head
        events = [json.loads(line) for line in body.split(b"\n") if line]
        assert events[-1]["event"] == "done" and events[-1]["cancelled"] is False
        assert pcm(events) == b"".join(bytes([k, 4, 255, 127]) for k in range(3))


def test_a_stalled_reader_is_cut_off_and_cancelled(harness_binary):
    # 1 MB audio pieces into a client that never reads: once the socket buffers fill, a send makes no progress for the
    # socket timeout (1 s), the server closes the connection and cancels the generation
    with running(harness_binary, "--token-delay-ms", "100", "--socket-timeout", "1",
                 "--speech-stream-piece-bytes", str(1 << 20), "--speech-rows", "1") as h:
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(" ".join([LONG] * 20), stream=True)))  # 60 chunks, 6 s and 60 MB in full
        time.sleep(0.5)
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts("Short sentence here."))  # queued behind the stalled one
        assert status == 200 and body["cancelled"] is False
        assert time.monotonic() - t0 < 4.5, time.monotonic() - t0
        s.settimeout(5)
        data = b""
        try:
            while block := s.recv(1 << 20):
                data += block
        except (ConnectionResetError, socket.timeout):
            pass
        s.close()
        assert b'"event":"done"' not in data  # cut short: no done line, which a client must treat as failure


def test_streams_beyond_half_the_handlers_are_refused(harness_binary):
    with running(harness_binary, "--token-delay-ms", "400", "--handlers", "2") as h:  # one stream at a time
        result = {}
        t = threading.Thread(target=lambda: result.update(r=stream(h, tts(LONG, stream=True))))
        t.start()
        time.sleep(0.2)
        status, body, events = stream(h, tts(LONG, stream=True))
        assert events is None and status == 429 and body == {"error": "stream_limit"}
        t.join(30)
        assert result["r"][0] == 200 and result["r"][1][-1]["event"] == "done"
        assert stream(h, tts(LONG, stream=True))[1][-1]["event"] == "done"  # the slot is free again


def test_one_client_leaving_leaves_another_stream_alone(harness_binary):
    with running(harness_binary, "--token-delay-ms", "100", "--speech-rows", "1") as h:
        result = {}
        def other():
            time.sleep(0.05)  # queued behind the leaver
            for _ in range(100):
                result["r"] = stream(h, tts(LONG, seed=9, stream=True))
                if result["r"][:2] != (429, {"error": "frontend_busy"}):
                    return
                time.sleep(0.01)
        t = threading.Thread(target=other)
        t.start()
        status, events, _ = stream(h, tts(" ".join([LONG] * 6), stream=True), close_after=1)
        assert status == 200
        t.join(30)
        status, events, _ = result["r"]
        assert status == 200 and events[-1]["event"] == "done" and events[-1]["cancelled"] is False
        assert pcm(events) == b"".join(bytes([k, 9, 255, 127]) for k in range(3))


def test_a_reset_after_the_reply_began_cancels_without_waiting_for_a_send(harness_binary):
    # a client half-closes (the reply is committed), then resets the connection during a long gap with no events: the
    # socket's reset state cancels it within the poll, not at the next event's send (2 s later here)
    import struct
    with running(harness_binary, "--token-delay-ms", "2000", "--speech-rows", "1") as h:
        s = socket.create_connection(("127.0.0.1", h.port), timeout=30)
        s.sendall(request_bytes(tts(LONG, stream=True)))  # 3 chunks, 6 s in full
        s.shutdown(socket.SHUT_WR)
        assert s.recv(65536).startswith(b"HTTP/1.0 200 OK")  # committed on the half-close
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()  # a reset
        t0 = time.monotonic()
        status, body = h.call("POST", "/speech", tts("Short sentence here."))  # 1 chunk, 2 s
        assert status == 200 and body["cancelled"] is False
        # cancelled at once: the fake notices at its first chunk boundary (~1.7 s from here), then this one runs 2 s;
        # found only at the next send, it would be cancelled at the second boundary (~3.7 s) and finish at ~5.7 s
        assert time.monotonic() - t0 < 4.7, time.monotonic() - t0


def test_a_malformed_request_at_the_stream_cap_is_a_400(harness_binary):
    with running(harness_binary, "--token-delay-ms", "400", "--handlers", "2") as h:
        result = {}
        t = threading.Thread(target=lambda: result.update(r=stream(h, tts(LONG, stream=True))))
        t.start()
        time.sleep(0.2)
        status, body, events = stream(h, dict(tts(LONG, stream=True), mode="bogus"))
        assert events is None and status == 400 and body == {"error": "invalid_request"}
        t.join(30)
