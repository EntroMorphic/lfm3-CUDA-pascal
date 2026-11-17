#!/usr/bin/env python3
"""GPU check for streamed POST /speech (validation/speech-stream-v1; Thor, lfm3-profile-v1). Black box: the server is
started like a deployment (token file, --port 0) and driven only over HTTP, each case sent unstreamed and streamed.

Rules (set in the README before any streamed run):
  1. exact: streamed PCM == the unstreamed WAV's data; the done event == the unstreamed reply but audio_wav_base64,
     prefill_seconds and generate_seconds; the text events joined == the done event's text;
  2. first audio (client, loopback, request's first byte to the first audio event): TTS median <= 1.0 s and max <= 1.5 s;
     interleaved median <= 1.5 s;
  3. no gaps: a player starting at the first audio event, at 24 kHz, never runs dry before the done event;
  5. cancellation: a client closing after its first audio event; the next request starts within 1 s (its first audio
     arrives within 1 s of its own unhindered first-audio time after the close) and equals its reference;
  6. cost: lighthouse streamed generate_seconds within 3% of unstreamed (median of 5 each).
(Rule 4, nothing else moves, is the existing checks on the same build.)
Writes OUT_JSON and exits 1 on any failure.
Usage: check_lfm3_speech_stream.py SERVER_BINARY MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON
  (BUILD_DIR: build/ on Thor, holding speech-e2e-v1, speech-chunk-native-v1 and audio-in-listen)
"""
import base64
import hashlib
import json
import os
import secrets
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ASR = "Perform ASR in English."
NUMBERS = ("The meeting starts at 7:45 and ends by 9:00. Order number 389 cost $3.50, about 2.75% more than in 1984. "
           "Dr. Patel met Mr. Lee at St. Mary's on Nov. 9, 2026.")
RATE = 24_000
TIMING = ("prefill_seconds", "generate_seconds")


def cases(build, libri):
    b = Path(build)
    out = []  # name, kind (tts | interleaved | asr), body
    e2e = sorted((b / "speech-e2e-v1/requests").glob("*.req.json"))
    assert len(e2e) == 30, f"expected the 30 speech-e2e-v1 requests, found {len(e2e)}"
    for p in e2e:
        r = json.loads(p.read_text())
        body = dict(messages=r["conversation"], mode=r["mode"], max_new_tokens=r["max_new_tokens"], seed=r["seed"])
        if r["mode"] == "audio":
            body.update(normalize=False)
        out.append((p.name[:-9], "tts" if r["mode"] == "audio" else "interleaved", body))
    for name, path in (("lighthouse-split-s1", "speech-chunk-native-v1/requests/tts-long-male-s1.req.json"),
                       ("bread-split-s1", "speech-chunk-native-v1/requests/probe-other-long-s1.req.json")):
        r = json.loads((b / path).read_text())
        out.append((name, "tts", dict(messages=r["conversation"], mode="audio", normalize=False, max_new_tokens=1536, seed=1)))
    # long enough that the step budget (6144) cuts it: truncated, chunks skipped (added after the second red-team)
    r = json.loads((b / "speech-chunk-native-v1/requests/tts-long-male-s1.req.json").read_text())
    conv = [dict(m, content=" ".join([m["content"]] * 14)) if m["role"] == "user" else m for m in r["conversation"]]
    out.append(("budget-truncated-s1", "tts", dict(messages=conv, mode="audio", normalize=False, max_new_tokens=1536, seed=1)))
    out.append(("numbers-normalized-s1", "tts", dict(
        messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                  {"role": "user", "content": NUMBERS}], mode="audio", max_new_tokens=512, seed=1)))

    def clip(path):
        return [{"type": "audio", "audio": base64.b64encode(Path(path).read_bytes()).decode()}]
    for i in range(5):
        out.append((f"spoken-q{i}", "interleaved", dict(
            messages=[{"role": "system", "content": "Respond with interleaved text and audio."},
                      {"role": "user", "content": clip(b / f"audio-in-listen/q{i}-16k.wav")}],
            mode="interleaved", max_new_tokens=1024, seed=1)))
    for u in ("1089-134686-0000", "4446-2275-0004", "121-123859-0002"):
        out.append((f"asr-{u}", "asr", dict(messages=[{"role": "system", "content": ASR},
                                                       {"role": "user", "content": clip(Path(libri) / f"{u}.wav")}],
                                             mode="text", max_new_tokens=512, seed=0)))
    return out


class Server:
    def __init__(self, binary, model, scratch, extra=()):
        self.token = secrets.token_urlsafe(32)
        token_path = Path(scratch) / "stream-check.token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.token + "\n")
        self.proc = subprocess.Popen([binary, "--experimental", model, "--token-file", str(token_path), "--port", "0",
                                      *extra],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.port = json.loads(self.proc.stdout.readline())["port"]
        token_path.unlink()

    def post(self, body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/speech", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def stream(self, body, close_after_audio=False, timeout=600):
        """(status, events, arrival seconds from the request's first byte, the first byte's monotonic time)."""
        payload = json.dumps(dict(body, stream=True)).encode()
        head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {self.token}\r\n\r\n").encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
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
                    if b"application/x-ndjson" not in head_bytes:
                        while block:
                            block = s.recv(1 << 16)
                            data += block
                        return status, [json.loads(data)], [now - t0], t0
                while b"\n" in data:
                    line, _, data = data.partition(b"\n")
                    events.append(json.loads(line))
                    times.append(now - t0)
                    if close_after_audio and events[-1]["event"] == "audio":
                        return status, events, times, t0
                if not block:
                    return status, events, times, t0
        finally:
            s.close()

    def stop(self):
        self.proc.terminate()  # SIGTERM only
        self.proc.wait(timeout=600)
        return self.proc.returncode, self.proc.stderr.read()


def wav_data(wav):
    assert wav[:4] == b"RIFF" and wav[36:40] == b"data" and len(wav) == 44 + int.from_bytes(wav[40:44], "little")
    return wav[44:]


def gaps(events, times):
    """A player that starts at the first audio event and plays at 24 kHz: (stalls, total stall seconds, slack min)."""
    end, stalls, stalled, slack = None, 0, 0.0, None
    for e, t in zip(events, times):
        if e["event"] != "audio":
            continue
        n = len(base64.b64decode(e["pcm_base64"])) // 2
        if end is None:
            end = t
        elif t > end:
            stalls, stalled = stalls + 1, stalled + (t - end)
            end = t
        else:
            slack = end - t if slack is None else min(slack, end - t)
        end += n / RATE
    return stalls, stalled, slack


def compare(plain, events):
    """Rule 1 for one case: (pass, details)."""
    if not events or events[-1].get("event") != "done":
        return False, "no done event"
    done = {k: v for k, v in events[-1].items() if k != "event"}
    want = {k: v for k, v in plain.items() if k not in ("audio_wav_base64",) + TIMING}
    got = {k: v for k, v in done.items() if k not in TIMING}
    if got != want or list(done) != [k for k in plain if k != "audio_wav_base64"]:
        return False, {"done": got, "want": want}
    pcm = b"".join(base64.b64decode(e["pcm_base64"]) for e in events if e["event"] == "audio")
    if pcm != wav_data(base64.b64decode(plain["audio_wav_base64"])):
        return False, f"pcm differs ({len(pcm)} bytes streamed)"
    text = "".join(e["text"] for e in events if e["event"] == "text")
    if text != done["text"]:
        return False, {"text_events": text, "done_text": done["text"]}
    chunks = [{k: v for k, v in e.items() if k not in ("event", "chunk")} for e in events if e["event"] == "chunk"]
    if chunks != plain["chunks"]:
        return False, "chunk events differ"
    ended = 0  # every audio and text event belongs to the chunk running, and chunk events count up from 0
    for e in events[:-1]:
        if e["event"] in ("audio", "text") and e["chunk"] != ended:
            return False, f"{e['event']} event in chunk {e['chunk']} while chunk {ended} runs"
        if e["event"] == "chunk":
            if e["chunk"] != ended:
                return False, "chunk events out of order"
            ended += 1
    return True, ""


def main(binary, model, build, libri, out_json):
    out = Path(out_json)
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), cases={})
    server = Server(binary, model, out.parent)
    ok = True
    try:
        server.post(cases(build, libri)[0][2])  # warm: first-use allocation of the speech runner
        all_cases = cases(build, libri)
        for name, kind, body in all_cases:
            status, plain = server.post(body)
            s_status, events, times, _ = server.stream(body)
            exact, why = compare(plain, events) if status == 200 and s_status == 200 else (False, (status, s_status))
            audio_times = [t for e, t in zip(events, times) if e.get("event") == "audio"]
            stalls, stalled, slack = gaps(events, times)
            res = dict(kind=kind, exact=exact, why=why, first_audio=audio_times[0] if audio_times else None,
                       first_event=times[0] if times else None, done_at=times[-1] if times else None,
                       audio_events=len(audio_times), text_events=sum(e.get("event") == "text" for e in events),
                       audio_seconds=plain.get("audio_seconds"), stalls=stalls, stalled_seconds=round(stalled, 4),
                       min_slack=None if slack is None else round(slack, 4),
                       generate_seconds=(plain.get("generate_seconds"), events[-1].get("generate_seconds")))
            res["truncated"] = plain.get("truncated")
            res["pass"] = exact and (kind == "asr" or (audio_times and stalls == 0)) and \
                (name != "budget-truncated-s1" or plain.get("truncated") is True)
            ok &= bool(res["pass"])
            report["cases"][name] = res
            print(name, json.dumps(res), flush=True)
        firsts = {k: [c["first_audio"] for c in report["cases"].values() if c["kind"] == k and c["first_audio"] is not None]
                  for k in ("tts", "interleaved")}
        report["first_audio"] = {k: dict(median=statistics.median(v), max=max(v), n=len(v)) for k, v in firsts.items()}
        report["rule2_pass"] = (report["first_audio"]["tts"]["median"] <= 1.0 and report["first_audio"]["tts"]["max"] <= 1.5
                                and report["first_audio"]["interleaved"]["median"] <= 1.5)
        report["rule1_pass"] = all(c["exact"] for c in report["cases"].values())
        report["rule3_pass"] = all(c["stalls"] == 0 for c in report["cases"].values() if c["kind"] != "asr")
        ok &= report["rule2_pass"]

        # Rule 5: walk away after the first audio of the lighthouse; the next request (fox, unsplit) must start at once.
        light = next(b for n, _, b in all_cases if n == "lighthouse-split-s1")
        fox = next(b for n, _, b in all_cases if n == "tts-fox-female-s1")
        _, fox_plain = server.post(fox)
        _, alone, alone_t, _ = server.stream(fox)
        fox_first = next(t for e, t in zip(alone, alone_t) if e["event"] == "audio")
        _, ev, _, _ = server.stream(light, close_after_audio=True)
        closed = time.monotonic()
        _, after, after_t, t_sent = server.stream(fox)
        after_first = t_sent - closed + next(t for e, t in zip(after, after_t) if e["event"] == "audio")
        exact, why = compare(fox_plain, after)
        report["cancel"] = dict(first_audio_alone=fox_first, first_audio_after_close=after_first,
                                delay=after_first - fox_first, next_exact=exact, why=why,
                                pass_=exact and after_first - fox_first <= 1.0)
        ok &= report["cancel"]["pass_"]
        print("cancel", json.dumps(report["cancel"]), flush=True)

        # Rule 6: the cost of streaming on the lighthouse.
        plain_g, stream_g = [], []
        for _ in range(5):
            plain_g.append(server.post(light)[1].get("generate_seconds", float("inf")))
            stream_g.append(server.stream(light)[1][-1].get("generate_seconds", float("inf")))  # an error event: inf
        mp, ms = statistics.median(plain_g), statistics.median(stream_g)
        report["cost"] = dict(plain=plain_g, stream=stream_g, median_plain=mp, median_stream=ms, ratio=ms / mp,
                              pass_=ms <= mp * 1.03)
        ok &= report["cost"]["pass_"]
        print("cost", json.dumps(report["cost"]), flush=True)
    finally:
        code, stderr = server.stop()
        report["server_exit"], report["server_stderr"] = code, stderr[-2000:]
        ok &= code == 0
    report["pass"] = bool(ok)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("STREAM CHECK", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
