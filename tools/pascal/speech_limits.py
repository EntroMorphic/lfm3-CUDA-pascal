"""speech-v1 rule 13, through liquid-lfm3-server over HTTP (its defaults): a speech request past each of its limits,
and one more than the queue holds, are refused with the reference's status (the HTTP layer as imported,
lfm/lfm3_http.hpp), and the controls then give the same bytes as before.

Controls, before and after: a pinned greedy /generate (a session of 1,024; 128 new tokens; ids) and a short /speech
(one sentence, audio, seed 1; the WAV's bytes). Refusals, each sent once (a 429 frontend_busy is sent again, as a
client does):
  text      the messages' text just over 64 KiB (kSpeechMaxTextBytes) in under 512 long sentences, split: within
            every other limit (chunks, prompt tokens, each chunk's capacity), so only the
            byte limit can refuse it (a broken one would answer 200)                    400 invalid_request
  chunks    513 sentences, split (kSpeechMaxChunks 512)                                  400 invalid_request
  prompt    512 sentences, each chunk's prompt with a ~300-token system message: over
            131,072 prompt tokens in all (kSpeechMaxPromptTokens), every chunk within
            its capacity and the text within 64 KiB                                    400 invalid_request
  capacity  one unsplit prompt of ~4,000 tokens with max_new_tokens 1,536: over the
            chunk's 4,096 (kSpeechCapacity)                                            400 invalid_request
  clips     65 audio clips of 0.1 s (kSpeechMaxClips 64)                               400 invalid_request
  body      a body of 4 MiB + 1 with no image part (kMaxBody)                            413 body_limit
  body-16   a declared Content-Length of 16 MiB + 1 (kMaxSpeechBody), headers only       413 body_limit
  queue     with the speech rows busy and the queue full (queue_size requests waiting),
            one more /speech                                                            429 queue_full
The queue's requests (a paragraph twice, audio, max_new_tokens 300: cut at 24 s of audio; seeds 1 to rows +
queue_size, sent 1 s apart) must all succeed, every one still running or waiting when the extra one is refused.
    python3 tools/pascal/speech_limits.py SERVER CHECKPOINT OUTPUT [ROWS=1] [QUEUE=4]
"""
import base64
import io
import json
import socket
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402

server_path, checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 1
queue = int(sys.argv[5]) if len(sys.argv) > 5 else 4
out.mkdir(parents=True, exist_ok=False)
records, report, ok = [], dict(rows=rows, queue_size=queue, refusals={}), True
TTS = "Perform TTS in American English using a female voice."
SENTENCE = "The small boat drifted slowly past the quiet harbor wall."
LONG_SENTENCE = ("The small boat drifted slowly past the quiet harbor wall while the gulls circled overhead and the "
                 "fishermen mended their long nets.")
LONG_COUNT = 65536 // (len(LONG_SENTENCE) + 1) + 1  # just over 64 KiB with the spaces
assert LONG_COUNT < 512 and len(" ".join([LONG_SENTENCE] * LONG_COUNT)) > 65536
PARAGRAPH = ("The river leaves the mountains in early spring, carrying silt and snowmelt past the old mill, the orchards "
             "and the long fields of the valley floor, until it widens into the slow brown estuary where the herons wait.")


def clip(seconds):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b"\0\0" * int(16000 * seconds))
    return base64.b64encode(buf.getvalue()).decode()


def speech(body, expected=(200, 400, 413, 429, 503), raw=None):
    for _ in range(400):  # 429 frontend_busy: the front end prepares one request at a time
        status, reply = client.call("POST", "/speech", body, expected=expected, raw=raw)
        if (status, reply.get("error")) != (429, "frontend_busy"): return status, reply
        time.sleep(0.25)
    return status, reply


def declared_only(length):
    """Headers of a /speech request declaring `length` body bytes, and no body: the server judges the length first."""
    s = socket.create_connection(("127.0.0.1", server.port), timeout=60)
    try:
        s.sendall((f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                   f"Authorization: Bearer {server.token}\r\nContent-Length: {length}\r\n\r\n").encode())
        data = b""
        while True:
            if b"\r\n\r\n" in data:
                head, rest = data.split(b"\r\n\r\n", 1)
                lengths = [l.split(b":", 1)[1] for l in head.split(b"\r\n") if l.lower().startswith(b"content-length:")]
                if lengths and len(rest) >= int(lengths[0]): break
            chunk = s.recv(65536)
            if not chunk: break
            data += chunk
    finally:
        s.close()
    head, body = data.split(b"\r\n\r\n", 1)
    return int(head.split()[1]), json.loads(body)


def controls():
    sid = client.call("POST", "/sessions", dict(capacity=1024))["session_id"]
    text = client.call("POST", "/generate", dict(session_id=sid, max_new_tokens=128, messages=[
        {"role": "user", "content": "Explain in one paragraph how a lighthouse warns ships at night."}]))["generated_ids"]
    client.call("POST", f"/sessions/{sid}/close", {})
    status, reply = speech(dict(messages=[{"role": "system", "content": TTS},
                                          {"role": "user", "content": "The quick brown fox jumps over the lazy dog."}],
                                mode="audio", max_new_tokens=512, seed=1))
    assert status == 200, (status, reply)
    return text, base64.b64decode(reply["audio_wav_base64"])


def refused(name, want, got):
    global ok
    status, reply = got
    entry = dict(status=status, error=reply.get("error"), expected=list(want))
    entry["pass"] = (status, reply.get("error")) == want
    ok &= entry["pass"]; report["refusals"][name] = entry
    print(f"rule 13 {name}: {status} {reply.get('error')} (expected {want[0]} {want[1]}):", "ok" if entry["pass"] else "FAIL", flush=True)


server = NativeServer(server_path, checkpoint, out, "server", None)
client = Client(server.port, server.token, records)
try:
    text0, wav0 = controls()
    report["control_tokens"], report["control_wav_bytes"] = len(text0), len(wav0)
    refused("text", (400, "invalid_request"), speech(dict(messages=[{"role": "system", "content": TTS},
            {"role": "user", "content": " ".join([LONG_SENTENCE] * LONG_COUNT)}], mode="audio", max_new_tokens=4, seed=1,
            split=True)))
    refused("chunks", (400, "invalid_request"), speech(dict(messages=[{"role": "system", "content": TTS},
            {"role": "user", "content": " ".join([SENTENCE] * 513)}], mode="audio", max_new_tokens=12, seed=1, split=True)))
    long_system = TTS + " " + " ".join(["lighthouse keeper"] * 100)
    refused("prompt", (400, "invalid_request"), speech(dict(messages=[{"role": "system", "content": long_system},
            {"role": "user", "content": " ".join([SENTENCE] * 512)}], mode="audio", max_new_tokens=12, seed=1, split=True)))
    refused("capacity", (400, "invalid_request"), speech(dict(messages=[{"role": "system", "content": TTS},
            {"role": "user", "content": " ".join(["lighthouse keeper"] * 1400)}], mode="audio", max_new_tokens=1536, seed=1,
            split=False)))
    refused("clips", (400, "invalid_request"), speech(dict(messages=[{"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": clip(0.1)} for _ in range(65)]}], mode="text",
            max_new_tokens=16, seed=0)))
    body = json.dumps(dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                                     {"role": "user", "content": [{"type": "audio", "audio": ""}]}], mode="text",
                           max_new_tokens=16, seed=0))
    pad = 4 * 1024 * 1024 + 1 - len(body)
    raw = body.replace('"audio": ""', '"audio": "' + "A" * pad + '"').encode()
    assert len(raw) == 4 * 1024 * 1024 + 1, len(raw)
    refused("body", (413, "body_limit"), speech(None, raw=raw))
    refused("body-16", (413, "body_limit"), declared_only(16 * 1024 * 1024 + 1))

    # queue: rows running, `queue` waiting, then one more
    held = [dict() for _ in range(rows + queue)]

    def hold(k):
        held[k]["start"] = time.monotonic()
        held[k]["status"], reply = speech(dict(messages=[{"role": "system", "content": TTS}, {"role": "user", "content": PARAGRAPH + " " + PARAGRAPH}],
                                               mode="audio", max_new_tokens=300, seed=k + 1))
        held[k]["error"], held[k]["end"] = reply.get("error"), time.monotonic()
    threads = []
    for k in range(rows + queue):
        threads.append(threading.Thread(target=hold, args=(k,))); threads[-1].start(); time.sleep(1.0)
    time.sleep(3)
    t_extra = time.monotonic()
    refused("queue", (429, "queue_full"), speech(dict(messages=[{"role": "system", "content": TTS},
            {"role": "user", "content": "One more."}], mode="audio", max_new_tokens=64, seed=1)))
    [t.join() for t in threads]
    report["queue_requests"] = [dict(status=h["status"], error=h["error"], start=round(h["start"] - t_extra, 1),
                                     end=round(h["end"] - t_extra, 1)) for h in held]
    report["queue_requests_pass"] = all(h["status"] == 200 for h in held) and \
        sum(h["end"] > t_extra for h in held) == rows + queue  # all still running or waiting when the extra one came
    ok &= report["queue_requests_pass"]
    print("rule 13 queue's requests:", [h["status"] for h in held], "all held at the extra one:", report["queue_requests_pass"], flush=True)

    text1, wav1 = controls()
    report["text_control_identical"], report["speech_control_identical"] = text1 == text0, wav1 == wav0
    ok &= text1 == text0 and wav1 == wav0
    report["health"] = client.call("GET", "/health", auth=False)
    ok &= report["health"].get("ready") is True
    print("rule 13 controls: text identical", text1 == text0, "| speech identical", wav1 == wav0, flush=True)
finally:
    report["server_exit"] = server.stop()
    ok &= report["server_exit"] == 0
    report["pass"] = bool(ok)
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    for r in records:
        if isinstance(r.get("response"), dict) and "audio_wav_base64" in r["response"]:
            r["response"] = dict(r["response"], audio_wav_base64=f"<{len(r['response']['audio_wav_base64'])} base64 chars>")
    (out / "records.json").write_text(json.dumps(records, indent=1) + "\n")
print("SPEECH LIMITS (rule 13)", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
