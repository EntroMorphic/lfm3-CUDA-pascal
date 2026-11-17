#!/usr/bin/env python3
"""Text under speech load, black box (validation/text-wait-v1 rules 2, 3, 4 and 6's first case; Thor, quiet).

Phases (any subset, in this order):
  alone     10 text arrivals on an idle server: create a session, generate (rule 2's prompt, 128 tokens), close.
  rule2     10 arrivals, each behind 8 fresh streams of budget-truncated-s1 that have each played 5 s of audio.
            Pass: median <= 3.5 x alone's median and <= 10 s; none over 15 s.
  rule3     7 streams (4 TTS long-form, 3 interleaved) and a text client generating back to back; 10 TTS arrivals,
            streamed, one at a time as the 8th row. Pass: no stream (the 7 and the 10) runs dry; first audio of the
            arrivals: median <= 0.6 s.
  rule4     a 2,000-token text generation running; 2 s in, a TTS request (streamed) arrives. 5 trials.
            Pass: every first audio within 1.0 s.
  deadline  a server with --request-timeout 20; one text arrival behind 8 streams as in rule2. Pass: 200.
Each text reply's token count and ids hash are recorded; alone and loaded, the ids must agree (rule 1 checks this
fully). Every stream's events are checked for gaps by check_lfm3_speech_stream.gaps.
Usage: check_lfm3_text_wait.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [PHASES]
"""
import hashlib
import importlib.util
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check", HERE / "check_lfm3_speech_stream.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
spec = importlib.util.spec_from_file_location("batch", HERE / "check_lfm3_speech_batch_server.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)

RULE2 = dict(max_new_tokens=128, messages=[{"role": "user", "content":
    "Hypothetically, might reflective recursion be a function of cognition? Answer in at least 300 words."}])
RULE4 = dict(max_new_tokens=2000, messages=[{"role": "user", "content":
    "Write a long, detailed story about a lighthouse keeper, at least 3,000 words."}])


def call(server, path, body, timeout=1200):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def text_arrival(server, body, capacity=4096):
    """A client that creates a session, generates and closes: (seconds to the reply, status, reply summary)."""
    t0 = time.monotonic()
    status, created = call(server, "/sessions", {"capacity": capacity})
    while status == 429:
        time.sleep(0.01)
        status, created = call(server, "/sessions", {"capacity": capacity})
    if status != 200:
        return time.monotonic() - t0, status, dict(stage="create", error=created)
    try:
        status, reply = call(server, "/generate", dict(body, session_id=created["session_id"]))
        while status == 429:
            time.sleep(0.01)
            status, reply = call(server, "/generate", dict(body, session_id=created["session_id"]))
        seconds = time.monotonic() - t0
    finally:
        call(server, f"/sessions/{created['session_id']}/close", {})
    ids = reply.get("generated_ids") if status == 200 else None
    return seconds, status, dict(tokens=len(ids) if ids else None, ids_sha=hashlib.sha256(json.dumps(ids).encode())
                                 .hexdigest()[:16] if ids else None, error=None if status == 200 else reply)


def summarize(results):
    out = []
    for status, events, times, _ in results:
        stalls, stalled, _ = check.gaps(events, times)
        first = next((t for e, t in zip(events, times) if e.get("event") == "audio"), None)
        out.append(dict(status=status, done=bool(events) and events[-1].get("event") == "done", stalls=stalls,
                        stalled_seconds=round(stalled, 3), first_audio=first))
    return out


class LiveStream(threading.Thread):
    """One streamed /speech read on its own socket, exposing the audio received so far (seconds)."""
    def __init__(self, server, body):
        super().__init__()
        self.server, self.body = server, body
        self.audio = 0.0
        self.result = None

    def run(self):
        for _ in range(3000):  # a 429 (the frontend prepares one request at a time) is retried, as a client does
            self.once()
            if self.result[0] != 429:
                return
            time.sleep(0.01)

    def once(self):
        import base64
        import socket
        payload = json.dumps(dict(self.body, stream=True)).encode()
        head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {self.server.token}\r\n\r\n").encode()
        s = socket.create_connection(("127.0.0.1", self.server.port), timeout=1800)
        t0 = time.monotonic()
        s.sendall(head + payload)
        data, status, events, times = b"", None, [], []
        try:
            while True:
                block = s.recv(1 << 16)
                now = time.monotonic()
                data += block
                if status is None:
                    if b"\r\n\r\n" not in data and block:
                        continue
                    head_bytes, _, data = data.partition(b"\r\n\r\n")
                    status = int(head_bytes.split(b" ", 2)[1])
                while b"\n" in data:
                    line, _, data = data.partition(b"\n")
                    e = json.loads(line)
                    events.append(e)
                    times.append(now - t0)
                    if e.get("event") == "audio":
                        self.audio += len(base64.b64decode(e["pcm_base64"])) / 2 / 24000
                if not block:
                    break
        finally:
            s.close()
        self.result = (status, events, times, t0)


def start_streams(server, bodies, audio_seconds):
    streams = [LiveStream(server, b) for b in bodies]
    for s in streams:
        s.start()
    deadline = time.monotonic() + 600
    while min(s.audio for s in streams) < audio_seconds:
        if time.monotonic() > deadline or any(not s.is_alive() for s in streams):
            raise RuntimeError("streams did not reach their audio before the text arrival")
        time.sleep(0.05)
    return streams


def main(binary, model, build, libri, out_json, phases="alone rule2 rule3 rule4 deadline"):
    out = Path(out_json)
    cases = {name: body for name, _, body in check.cases(build, libri)}
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), phases={})
    long8 = [cases["budget-truncated-s1"]] * 8

    def save():
        out.write_text(json.dumps(report, indent=1))

    phases = phases.split()
    server = batch.start(binary, model, out.parent)
    try:
        server.post(cases["tts-fox-female-s1"])  # warm the speech path
        text_arrival(server, RULE2)  # warm the text path
        alone = None
        if "alone" in phases or "rule2" in phases:
            runs = [text_arrival(server, RULE2) for _ in range(10)]
            alone = statistics.median(r[0] for r in runs)
            report["phases"]["alone"] = dict(seconds=[round(r[0], 3) for r in runs], median=round(alone, 3),
                                             statuses=[r[1] for r in runs], replies=[r[2] for r in runs])
            save()
            print("alone", report["phases"]["alone"]["median"], flush=True)
        if "rule2" in phases:
            arrivals = []
            for i in range(10):
                streams = start_streams(server, long8, 5.0)
                seconds, status, summary = text_arrival(server, RULE2)
                for s in streams:
                    s.join()
                arrivals.append(dict(seconds=round(seconds, 3), status=status, reply=summary,
                                     streams=summarize([s.result for s in streams])))
                print(f"rule2 arrival {i}: {seconds:.2f} s status {status}", flush=True)
                report["phases"]["rule2"] = dict(arrivals=arrivals)
                save()
            secs = [a["seconds"] for a in arrivals]
            med = statistics.median(secs)
            report["phases"]["rule2"].update(
                median=round(med, 3), ratio=round(med / alone, 3), max=max(secs),
                same_ids_as_alone=all(a["reply"].get("ids_sha") == report["phases"]["alone"]["replies"][0]["ids_sha"]
                                      for a in arrivals),
                pass_=all(a["status"] == 200 for a in arrivals) and med <= 3.5 * alone and med <= 10 and max(secs) <= 15)
            save()
        if "rule3" in phases:
            # 7 load rows kept busy until the 10 arrivals are done: each row's stream starts again when it ends
            # (attempt 1, twa-1: the 7 streams all ended after 9 arrivals, short of the rule's 10)
            load = [cases["tts-long-male-s1"], cases["tts-long-male-s2"], cases["lighthouse-split-s1"],
                    cases["bread-split-s1"], cases["il-reflective-s1"], cases["il-fridge-s1"], cases["il-study-s1"]]
            stop = threading.Event()
            texts, finished = [], []
            lock = threading.Lock()

            def texter():
                while not stop.is_set():
                    texts.append(text_arrival(server, RULE2))

            def row(body):
                while not stop.is_set():
                    s = LiveStream(server, body)
                    s.run()
                    with lock:
                        finished.append(s.result)
            rows = [threading.Thread(target=row, args=(b,)) for b in load]
            for r in rows:
                r.start()
            time.sleep(3.0)  # every row has started streaming
            tt = threading.Thread(target=texter)
            tt.start()
            arrivals = []
            for i in range(10):
                r = batch.retried(lambda: server.stream(cases["tts-fox-female-s1"] if i % 2 == 0 else cases["tts-fox-male-s2"], timeout=600))
                arrivals.append(r)
            stop.set()
            for r in rows:
                r.join()
            tt.join()
            arr = summarize(arrivals)
            loads = summarize(finished)
            firsts = [a["first_audio"] for a in arr if a["first_audio"] is not None]
            report["phases"]["rule3"] = dict(
                load=loads, arrivals=arr, texts=[dict(seconds=round(t[0], 3), status=t[1]) for t in texts],
                first_audio_median=round(statistics.median(firsts), 3) if firsts else None,
                pass_=len(arr) == 10 and all(a["status"] == 200 and a["done"] and a["stalls"] == 0 for a in arr + loads)
                and bool(firsts) and statistics.median(firsts) <= 0.6 and all(t[1] == 200 for t in texts))
            save()
            print("rule3", report["phases"]["rule3"]["first_audio_median"], report["phases"]["rule3"]["pass_"], flush=True)
        if "rule4" in phases:
            trials = []
            for i in range(5):
                holder = {}
                tt = threading.Thread(target=lambda: holder.update(r=text_arrival(server, RULE4)))
                tt.start()
                time.sleep(2.0)
                r = batch.retried(lambda: server.stream(cases["tts-fox-female-s1"], timeout=1800))
                tt.join()
                s = summarize([r])[0]
                trials.append(dict(first_audio=s["first_audio"], speech_status=s["status"], speech_done=s["done"],
                                   text_seconds=round(holder["r"][0], 3), text_status=holder["r"][1],
                                   text_tokens=holder["r"][2].get("tokens")))
                print(f"rule4 trial {i}: first audio {s['first_audio']}", flush=True)
            report["phases"]["rule4"] = dict(
                trials=trials, pass_=all(t["first_audio"] is not None and t["first_audio"] <= 1.0 and t["speech_done"]
                                         and t["text_status"] == 200 and t["text_seconds"] > t["first_audio"] + 2.0
                                         for t in trials))
            save()
    finally:
        code, _ = server.stop()
        report["server_exit"] = code
        save()
    if "deadline" in phases:
        server = batch.start(binary, model, out.parent, "--request-timeout", "20")
        try:
            server.post(cases["tts-fox-female-s1"])
            text_arrival(server, RULE2)
            streams = start_streams(server, long8, 5.0)
            seconds, status, summary = text_arrival(server, RULE2)
            for s in streams:
                s.join()
            report["phases"]["deadline"] = dict(seconds=round(seconds, 3), status=status, reply=summary,
                                                pass_=status == 200)
        finally:
            code, _ = server.stop()
            report["deadline_server_exit"] = code
            save()
        print("deadline", report["phases"]["deadline"]["status"], flush=True)
    verdicts = {k: v.get("pass_") for k, v in report["phases"].items() if "pass_" in v}
    report["verdicts"] = verdicts
    save()
    print(json.dumps(verdicts))


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        sys.exit(__doc__)
    main(*sys.argv[1:])
