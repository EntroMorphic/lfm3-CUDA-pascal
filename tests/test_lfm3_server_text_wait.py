"""Host tests for text sliced among speech steps (validation/text-wait-v1; native/include/lfm/lfm3_http.hpp run,
begin_text, text_cycle; ops text_begin/text_step), behind the harness's fake engine, on every build variant.

The fake sleeps --token-delay-ms per speech step and per text token, so times show what ran together. Written before
the change; on the code before it (one engine owner that runs only speech while rows run), the timing tests fail:
text waited for every running row to finish, a long text generation held back new speech, and a text request behind
the rows passed its deadline.
"""
import signal
import threading
import time


from test_lfm3_http_native import harness_binary, running  # noqa: F401  (fixture)
from test_lfm3_http_speech import LONG, tts
from test_lfm3_http_speech_stream import stream

PROMPT = [{"role": "user", "content": "hello there"}]  # the fake's phrase: 23 tokens, then a stop


def session(h, capacity=512):
    status, created = h.call("POST", "/sessions", {"capacity": capacity})
    assert status == 200, created
    return created["session_id"]


def generate(h, sid, tokens=64, timeout=60):
    return h.call("POST", "/generate", dict(session_id=sid, max_new_tokens=tokens, messages=PROMPT), timeout=timeout)


def in_thread(f):
    box = {}

    def run():
        box["result"] = f()
        box["done"] = time.monotonic()
    t = threading.Thread(target=run)
    t.start()
    return t, box


def long_speech(steps):
    return tts(" ".join([LONG] * (steps // 3)))  # LONG is 3 sentences: one chunk, one step, each


def test_text_runs_between_speech_steps_instead_of_waiting_for_the_rows(harness_binary):
    # rule 2: 90 speech steps of 0.1 s run; a client then creates a session and generates. Before, both calls waited
    # for the rows to finish (3 s and more); now each runs between speech steps
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        t, speech = in_thread(lambda: h.call("POST", "/speech", long_speech(90)))
        time.sleep(0.5)
        t0 = time.monotonic()
        sid = session(h)
        status, reply = generate(h, sid)
        took = time.monotonic() - t0
        t.join(60)
        assert status == 200 and len(reply["generated_ids"]) == 23 and reply["cancelled"] is False
        assert speech["result"][0] == 200
        # 23 tokens, each beside one speech step: about 4.8 s. Before, it waited for the rows' 9 s first
        assert took < 6.5, took
        assert t0 + took < speech["done"], (t0 + took, speech["done"])  # text finished first; the rows ran on


def test_the_engine_calls_alternate_speech_and_text_steps(harness_binary, tmp_path):
    trace = tmp_path / "ops.txt"
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "50", "--trace-ops", trace) as h:
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.3)
        sid = session(h)
        assert generate(h, sid)[0] == 200
        t.join(60)
    ops = [o for o in trace.read_text().split() if o in ("speech_step", "text_step", "text_begin")]
    first, last = ops.index("text_begin"), len(ops) - 1 - ops[::-1].index("text_step")
    window = ops[first + 1:last + 1]
    assert window.count("text_step") == 23  # one token per slice while rows run
    # between two text slices there is exactly one speech step: one each, in turn
    assert all(a != b for a, b in zip(window, window[1:])), window


def test_a_long_text_generation_does_not_hold_back_new_speech(harness_binary):
    # rule 4: no rows run, so text runs in slices of --text-slice tokens; speech arriving meanwhile is admitted at
    # the next slice. Before, it waited for the whole generation (2.3 s)
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--text-slice", "2") as h:
        sid = session(h)
        t, text = in_thread(lambda: generate(h, sid))
        time.sleep(0.4)
        sent = time.monotonic()
        status, events, times = stream(h, tts(LONG, stream=True))  # times: the clock at each event's arrival
        t.join(60)
        first = next(tm for e, tm in zip(events, times) if e["event"] == "audio") - sent
        assert status == 200 and events[-1]["event"] == "done"
        assert first < 0.8, first  # a slice (0.2 s) and a step (0.1 s), with margin
        assert text["result"][0] == 200 and len(text["result"][1]["generated_ids"]) == 23


def test_a_text_request_behind_long_rows_meets_its_deadline(harness_binary):
    # rule 6, first case: a 4 s request timeout, rows with about 6 s of work. Before: 504 queue_deadline
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--request-timeout", "4") as h:
        sid = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60), timeout=60))
        time.sleep(0.3)
        status, reply = h.call("POST", "/generate", dict(session_id=sid, max_new_tokens=8, messages=PROMPT))
        t.join(60)
        assert status == 200 and len(reply["generated_ids"]) == 8, reply


def test_a_text_reply_under_speech_is_its_reply_alone(harness_binary):
    with running(harness_binary, "--speech-rows", "2") as h:
        alone = generate(h, session(h))
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "30") as h:
        sid = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        loaded = generate(h, sid)
        t.join(60)
    strip = lambda r: (r[0], {k: v for k, v in r[1].items() if k != "session_id"})  # noqa: E731
    assert strip(loaded) == strip(alone)


def test_cancel_reaches_a_sliced_generation_and_resets_its_cache(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        fresh = generate(h, session(h), tokens=4)
        sid = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        g, box = in_thread(lambda: generate(h, sid))
        time.sleep(0.8)
        assert h.call("POST", f"/sessions/{sid}/cancel", {})[0] == 202
        g.join(60)
        status, reply = box["result"]
        assert status == 200 and reply["cancelled"] is True and reply["cache_valid"] is False
        assert 0 < len(reply["generated_ids"]) < 23
        assert generate(h, sid, tokens=4) == (fresh[0], dict(fresh[1], session_id=sid))  # as a fresh session's
        t.join(60)


def test_a_generation_past_its_deadline_is_cancelled_at_its_next_slice(harness_binary, tmp_path):
    # the behavior change: before, it ran to its end (cache kept); now it stops at the next slice (cache reset)
    trace = tmp_path / "ops.txt"
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--request-timeout", "1",
                 "--text-slice", "1", "--trace-ops", trace) as h:
        sid = session(h)
        t0 = time.monotonic()
        status, reply = generate(h, sid)  # 23 tokens of 0.1 s against a 1 s deadline
        assert status == 504 and reply == {"error": "request_deadline_outcome_unknown"}
        time.sleep(0.5)
        steps = trace.read_text().split().count("text_step")
        assert steps < 15, steps  # one token per slice: stopped about 10 tokens in, not run to its 23
        status, again = generate(h, sid, tokens=4)  # the session is free again, its cache reset
        assert status == 200 and again["reused_tokens"] == 0, again
        assert time.monotonic() - t0 < 2.0 + 1.5


def test_shutdown_with_text_and_rows_running_exits_0(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        sid = session(h)
        t1, speech = in_thread(lambda: h.call("POST", "/speech", long_speech(60), timeout=60))
        time.sleep(0.2)  # one at a time through the frontend (else 429 frontend_busy)
        t2, text = in_thread(lambda: generate(h, sid))
        time.sleep(0.6)
        h.process.send_signal(signal.SIGTERM)
        assert h.process.wait(30) == 0
        t1.join(60)
        t2.join(60)
    # both are cancelled at their next step, with their cancelled replies (red-team finding 1: the text got 503)
    for box in (speech, text):
        status, body = box["result"]
        assert status == 200 and body.get("cancelled") is True, box
    assert text["result"][1]["cache_valid"] is False and 0 < len(text["result"][1]["generated_ids"]) < 23


def test_an_engine_failure_in_a_text_slice_fails_the_service_closed(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "50", "--fail-op", "text_step",
                 "--fail-after", "3") as h:
        sid = session(h)
        t, speech = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        assert generate(h, sid)[0] == 503
        t.join(60)
        assert speech["result"][0] == 503
        assert h.call("POST", "/sessions", {"capacity": 512})[0] == 503


def test_generate_batch_still_waits_for_the_rows_and_holds_back_later_speech(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        a, b = session(h), session(h)
        done = {}

        def mark(name, f):
            r = f()
            done[name] = time.monotonic()
            return r
        t1 = threading.Thread(target=lambda: mark("speech", lambda: h.call("POST", "/speech", long_speech(15))))
        t1.start()
        time.sleep(0.3)
        t2 = threading.Thread(target=lambda: mark("batch", lambda: h.call("POST", "/generate_batch", {"requests": [
            dict(session_id=a, max_new_tokens=4, messages=PROMPT), dict(session_id=b, max_new_tokens=4, messages=PROMPT)]})))
        t2.start()
        time.sleep(0.2)
        t3 = threading.Thread(target=lambda: mark("later", lambda: h.call("POST", "/speech", tts(LONG))))
        t3.start()
        for x in (t1, t2, t3):
            x.join(60)
        assert done["speech"] <= done["batch"] <= done["later"], done


def test_a_speech_failure_with_text_active_fails_both_closed(harness_binary):
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "50", "--fail-op", "speech_step",
                 "--fail-after", "10") as h:
        sid = session(h)
        t, speech = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        assert generate(h, sid)[0] == 503
        t.join(60)
        assert speech["result"][0] == 503
        assert h.call("POST", "/sessions", {"capacity": 512})[0] == 503




def test_an_idle_session_expires_while_speech_runs(harness_binary):
    # rule 9: the idle timer counts time a session sits unused, whatever else runs
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--session-idle-timeout", "1") as h:
        idle = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(30)))
        time.sleep(2.0)
        assert generate(h, idle)[0] == 404  # unused for 2 s > 1 s: reaped while the rows ran
        t.join(60)


def test_a_generation_queued_longer_than_the_idle_timeout_gets_its_reply(harness_binary):
    # rule 9: a session whose generation waits in the queue (behind another text generation, beside speech) is
    # leased the whole time, so the idle timer never closes it
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100", "--session-idle-timeout", "1") as h:
        a, b = session(h), session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(90)))
        time.sleep(0.2)
        first, box_a = in_thread(lambda: generate(h, a))  # 23 slices beside speech: about 4.6 s
        time.sleep(0.3)
        t0 = time.monotonic()
        status, reply = generate(h, b)  # queued behind a's generation for about 4 s, then its own 4.6 s
        first.join(60)
        assert box_a["result"][0] == 200
        assert status == 200 and len(reply["generated_ids"]) == 23, reply
        assert time.monotonic() - t0 > 2.0  # it really waited longer than the idle timeout
        t.join(60)


def test_a_client_that_resets_mid_generation_leaves_its_session_as_a_fresh_one(harness_binary):
    # rule 6, second case: a text client resets (RST) while its generation runs under speech; the session's cache is
    # reset, so a continuation on it reuses nothing, as on a fresh session
    import json
    import socket
    import struct
    from test_lfm3_http_native import TOKEN
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        kept = session(h)
        status, full = generate(h, kept)
        assert status == 200 and full["finish_reason"] == "stop", full
        follow = PROMPT + [dict(role="assistant", content=full["text"]), dict(role="user", content="hello there")]

        def go_on(sid):
            return h.call("POST", "/generate", dict(session_id=sid, max_new_tokens=4, messages=follow))
        # the control: a client that stays for its reply keeps its cache, and the continuation reuses it
        status, reused = go_on(kept)
        assert status == 200 and reused["reused_tokens"] > 0, reused

        sid = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        body = json.dumps(dict(session_id=sid, max_new_tokens=64, messages=PROMPT)).encode()
        s = socket.create_connection(("127.0.0.1", h.port))
        s.sendall(b"POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Authorization: Bearer " + TOKEN.encode() + b"\r\nContent-Length: " + str(len(body)).encode()
                  + b"\r\n\r\n" + body)
        time.sleep(0.8)  # a few tokens in (23 of 0.1 s, among speech steps)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()  # RST
        deadline = time.monotonic() + 30
        while True:  # the session is busy until the server lets the generation go
            status, again = go_on(sid)
            if status != 409 or time.monotonic() > deadline:
                break
            time.sleep(0.1)
        assert status == 200 and again["reused_tokens"] == 0, again
        t.join(60)


def test_a_client_that_half_closes_still_gets_its_reply(harness_binary):
    # a half-close (FIN after the request) is not a reset: the generation runs to its end and the reply is read
    import json
    import socket
    from test_lfm3_http_native import TOKEN, parse_response
    with running(harness_binary, "--speech-rows", "2", "--token-delay-ms", "100") as h:
        alone = generate(h, session(h))
        sid = session(h)
        t, _ = in_thread(lambda: h.call("POST", "/speech", long_speech(60)))
        time.sleep(0.2)
        body = json.dumps(dict(session_id=sid, max_new_tokens=64, messages=PROMPT)).encode()
        with socket.create_connection(("127.0.0.1", h.port), timeout=60) as s:
            s.sendall(b"POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Authorization: Bearer " + TOKEN.encode() + b"\r\nContent-Length: " + str(len(body)).encode()
                      + b"\r\n\r\n" + body)
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while block := s.recv(65536):
                chunks.append(block)
        status, reply = parse_response(b"".join(chunks))
        assert (status, reply) == (alone[0], dict(alone[1], session_id=sid)), reply
        assert reply["cancelled"] is False and reply["cache_valid"] is True
        t.join(60)
