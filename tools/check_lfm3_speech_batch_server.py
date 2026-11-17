#!/usr/bin/env python3
"""GPU check for concurrent speech in liquid-lfm3-server (validation/speech-batch-v1; Thor, lfm3-profile-v1). Black box:
the server is started like a deployment and driven only over HTTP.

Rules (set in the README before any batched run):
  1. exact: every case's reply in a crowd equals its reply alone on the single-request server (OLD_SERVER: all fields
     but prefill_seconds, generate_seconds; streamed, every event and the done event), for the cases of
     check_lfm3_speech_stream.py (41 and the budget-truncated one), sent all at once and in a random order with random
     gaps (0-2 s), each unstreamed and streamed (the crowd server takes them all: 96 handlers, a queue of 64);
  3. real time: 8 (then 12, 16) concurrent streamed TTS requests: no player runs dry (largest N reported); also
     reported: 7 TTS streams while the five spoken questions (audio input: encoding and prefill inside a step) join;
  4. first audio under load: a TTS request arriving while 7 others stream: median of 10 first-audio times <= 1.0 s;
  5. text: a /generate arriving under continuous speech load: its ids equal its reference; its latency is reported
     beside the time the rows running at its arrival took to finish (it waits for them, then runs).
Writes OUT_JSON and exits 1 on any failure.
Usage: check_lfm3_speech_batch_server.py SERVER_BINARY OLD_SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("check", Path(__file__).with_name("check_lfm3_speech_stream.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
TIMING = ("prefill_seconds", "generate_seconds")


def strip(reply):
    return {k: v for k, v in reply.items() if k not in TIMING}


def start(binary, model, scratch, *options):
    s = check.Server.__new__(check.Server)
    import os
    import secrets
    import subprocess
    s.token = secrets.token_urlsafe(32)
    token_path = Path(scratch) / "batch-check.token"
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(s.token + "\n")
    s.proc = subprocess.Popen([binary, "--experimental", model, "--token-file", str(token_path), "--port", "0", *options],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    s.port = json.loads(s.proc.stdout.readline())["port"]
    token_path.unlink()
    return s


def generate(server, body):
    import urllib.request
    def post(path, b):
        req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(b).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    sid = post("/sessions", {"capacity": 1024})["session_id"]
    try:
        return post("/generate", dict(body, session_id=sid))
    finally:
        post(f"/sessions/{sid}/close", {})


def retried(call):
    """A 429 (the frontend prepares one request at a time; a full queue) is retried, as a client does."""
    for _ in range(3000):
        r = call()
        if r[0] != 429:
            return r
        time.sleep(0.01)
    return r


def post(server, body):
    return retried(lambda: server.post(body))


def stream(server, body):
    """As Server.stream, its times measured from the first attempt (time spent retrying counts)."""
    first = time.monotonic()
    status, events, times, t0 = retried(lambda: server.stream(body))
    return status, events, [t + (t0 - first) for t in times], first


def first_audio(events, times):
    return next((t for e, t in zip(events, times) if e.get("event") == "audio"), None)


def main(binary, old, model, build, libri, out_json):
    out = Path(out_json)
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  reference_server_sha256=hashlib.sha256(Path(old).read_bytes()).hexdigest())
    cases = check.cases(build, libri)
    ok = True
    # alone: each case unstreamed and streamed, one at a time, on the single-request server
    reference = start(old, model, out.parent)
    alone, alone_stream = {}, {}
    try:
        reference.post(cases[0][2])  # warm
        for name, _, body in cases:
            status, reply = reference.post(body)
            assert status == 200, (name, status, reply)
            alone[name] = strip(reply)
            _, events, _, _ = reference.stream(body)
            alone_stream[name] = events
    finally:
        code, _ = reference.stop()
        ok &= code == 0
    print("alone done", flush=True)

    def same(name, streamed, result):
        if streamed:
            status, events, _, _ = result
            want = alone_stream[name]
            return status == 200 and events[-1].get("event") == "done" and events[:-1] == want[:-1] and \
                strip(events[-1]) == strip(want[-1])
        status, reply = result
        return status == 200 and strip(reply) == alone[name]

    server = start(binary, model, out.parent, "--handlers", "96", "--queue-size", "64")
    try:
        server.post(cases[0][2])  # warm
        rng = random.Random(7)
        report["crowds"] = {}
        for streamed in (False, True):
            for arrival in ("at once", "random"):
                order = list(cases)
                delays = {n: 0.0 for n, _, _ in cases}
                if arrival == "random":
                    rng.shuffle(order)
                    t = 0.0
                    for n, _, _ in order:
                        t += rng.uniform(0, 2)
                        delays[n] = t

                def send(c, streamed=streamed):
                    time.sleep(delays[c[0]])
                    return c[0], (stream(server, c[2]) if streamed else post(server, c[2]))
                t0 = time.time()
                with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
                    got = dict(pool.map(send, order))
                differ = [n for n, _, _ in cases if not same(n, streamed, got[n])]
                tag = ("streamed" if streamed else "unstreamed") + ", " + arrival
                report["crowds"][tag] = dict(identical=len(cases) - len(differ), differ=differ, wall=round(time.time() - t0, 1))
                print("crowd", tag, json.dumps(report["crowds"][tag]), flush=True)
        report["rule1_pass"] = all(not c["differ"] for c in report["crowds"].values())
        ok &= report["rule1_pass"]

        tts = [b for n, k, b in cases if k == "tts" and n.startswith("tts-")]
        light = next(b for n, _, b in cases if n == "lighthouse-split-s1")
        # 4. first audio under load: 7 lighthouse streams running, then a TTS arrives (10 times, rows refilled)
        firsts = []
        stop = threading.Event()

        def load():
            while not stop.is_set():
                stream(server, light)
        loaders = [threading.Thread(target=load) for _ in range(7)]
        for t in loaders:
            t.start()
        time.sleep(3)
        for i in range(10):
            _, events, times, _ = stream(server, tts[i % len(tts)])
            firsts.append(first_audio(events, times))
            time.sleep(0.5)
        # 5. text under the same load: its ids against its reference (taken before the load)
        stop.set()
        report["first_audio_under_load"] = dict(times=firsts, median=statistics.median(firsts), pass_=statistics.median(firsts) <= 1.0)
        ok &= report["first_audio_under_load"]["pass_"]
        print("first audio under load", json.dumps(report["first_audio_under_load"]), flush=True)
        for t in loaders:
            t.join()
        text_body = dict(max_new_tokens=128, messages=[{"role": "user", "content": "Explain in one paragraph how a lighthouse warns ships at night."}])
        text_reference = generate(server, text_body)
        stop.clear()
        spans = []  # each load stream's (start, end)
        lock = threading.Lock()

        def timed_load():
            while not stop.is_set():
                t_start = time.time()
                stream(server, light)
                with lock:
                    spans.append((t_start, time.time()))
        loaders = [threading.Thread(target=timed_load) for _ in range(8)]
        for t in loaders:
            t.start()
        time.sleep(3)
        t0 = time.time()
        under = generate(server, text_body)
        latency = time.time() - t0
        stop.set()
        for t in loaders:
            t.join()
        drain = max((end for begin, end in spans if begin <= t0 < end), default=t0) - t0
        report["text_under_load"] = dict(latency=round(latency, 2), running_rows_finished_after=round(drain, 2),
                                         identical=under["generated_ids"] == text_reference["generated_ids"])
        ok &= report["text_under_load"]["identical"]
        print("text under load", json.dumps(report["text_under_load"]), flush=True)
    finally:
        code, stderr = server.stop()
        report["server_exit"], report["server_stderr"] = code, stderr[-1500:]
        ok &= code == 0
    # 3. real time at N concurrent streams, a server per N (rows = N)
    report["real_time"] = {}
    for n in (8, 12, 16):
        server = start(binary, model, out.parent, "--speech-rows", str(n))
        try:
            server.post(tts[0])
            with concurrent.futures.ThreadPoolExecutor(n) as pool:
                got = list(pool.map(lambda i: stream(server, tts[i % len(tts)]), range(n)))
            stalls = [check.gaps(events, times)[0] for _, events, times, _ in got]
            exact = all(events[-1].get("event") == "done" for _, events, _, _ in got)
            report["real_time"][n] = dict(stalls=stalls, firsts=[first_audio(e, t) for _, e, t, _ in got], all_done=exact)
            print("real time", n, json.dumps(report["real_time"][n]), flush=True)
        finally:
            code, _ = server.stop()
            ok &= code == 0
    # spoken questions joining 7 running TTS streams: their admissions (audio encoding, prefill) run inside steps
    server = start(binary, model, out.parent)
    try:
        server.post(tts[0])
        spoken = [b for n, _, b in cases if n.startswith("spoken-q")]

        def later(i):
            time.sleep(1.0 + 0.7 * i)
            return stream(server, spoken[i])
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            running = [pool.submit(stream, server, tts[i % len(tts)]) for i in range(7)]
            joining = [pool.submit(later, i) for i in range(len(spoken))]
            got = [f.result() for f in running + joining]
        stalls = [check.gaps(events, times)[0] for _, events, times, _ in got]
        report["real_time"]["7 tts + 5 spoken joining"] = dict(stalls=stalls, all_done=all(e[-1].get("event") == "done" for _, e, _, _ in got))
        print("real time 7 tts + 5 spoken joining", json.dumps(report["real_time"]["7 tts + 5 spoken joining"]), flush=True)
    finally:
        code, _ = server.stop()
        ok &= code == 0
    report["rule3_pass"] = sum(report["real_time"][8]["stalls"]) == 0 and report["real_time"][8]["all_done"]
    report["largest_n_without_gaps"] = max([n for n, r in report["real_time"].items() if isinstance(n, int) and sum(r["stalls"]) == 0],
                                           default=0)
    ok &= report["rule3_pass"]
    report["pass"] = bool(ok)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("BATCH SERVER CHECK", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
