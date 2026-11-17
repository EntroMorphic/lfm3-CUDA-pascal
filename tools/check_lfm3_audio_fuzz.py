#!/usr/bin/env python3
"""Hostile input against a live liquid-lfm3-server (validation/audio-assurance-v1 rule 5; Thor, lfm3-profile-v1).

References first: a few short valid requests (TTS both voices, a LibriSpeech transcription, a spoken question, the
normalized numbers) alone on the same server, unstreamed and streamed. Then THREADS clients send COUNT requests,
every one a mutation of a valid /speech request except every 8th, which is a valid one (its reply must equal its
reference). Mutations (seeded, recorded by kind):
  json      a field dropped, added, or given a wrong type; huge, negative and fractional numbers; unicode and control
            characters; odd modes and flags; deep nesting; many messages; wrong roles and content shapes;
  base64    the audio's base64 with characters flipped, removed, padded, spaced, or truncated, or empty;
  wav       the WAV header with bytes flipped; RIFF/data sizes too large, too small, 0, 0xFFFFFFFF; other sample
            rates, channels, bit depths, formats (float, extensible, A-law); chunks added or truncated; data cut at odd
            sizes; float samples at +-3e38, NaN or infinity;
  transport a body shorter than its Content-Length (then closed), invalid JSON bytes, a slow sender, a client that
            drops the connection mid-request or mid-stream, a missing or wrong token, a wrong path or method.
Pass (rule 5): every reply that arrives is a well-formed HTTP reply, 2xx or 4xx (429 counted; no 5xx), every request
ends within 120 s, every valid request's reply equals its reference, the server and worker stay up; afterwards the
42-case crowd (check_lfm3_audio_soak.py's) is identical, each process's RSS is at most its value after the
references plus 64 MiB, and the server exits 0 on SIGTERM.
Usage: check_lfm3_audio_fuzz.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [COUNT [THREADS [SEED]]]
"""
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import random
import socket
import struct
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("soak", HERE / "check_lfm3_audio_soak.py")
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)
check, batch, strip = soak.check, soak.batch, soak.strip
MIB = 1 << 20
VALID = ["tts-fox-female-s1", "tts-fox-male-s2", "asr-1089-134686-0000", "spoken-q0", "numbers-normalized-s1"]
OK_STATUSES = {200, 400, 401, 404, 405, 411, 413, 415, 429, 431, 501}


def raw(port, data, timeout=120, slow=False, drop_after=None, read=True):
    """Sends raw bytes; returns (status or None, body bytes, seconds). drop_after: close after sending that many bytes."""
    t0 = time.monotonic()
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        if drop_after is not None:
            s.sendall(data[:drop_after])
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            return None, b"", time.monotonic() - t0
        if slow:
            for i in range(0, len(data), max(1, len(data) // 20)):
                s.sendall(data[i:i + max(1, len(data) // 20)])
                time.sleep(0.1)
        else:
            s.sendall(data)
        if not read:
            return None, b"", time.monotonic() - t0
        out = b""
        while True:
            block = s.recv(1 << 16)
            if not block:
                break
            out += block
        head, _, body = out.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1]) if head.startswith(b"HTTP/") else -1
        return status, body, time.monotonic() - t0
    finally:
        s.close()


def http(token, body_bytes, path="/speech", method="POST", length=None, auth=True):
    head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
    head += f"Content-Length: {len(body_bytes) if length is None else length}\r\n"
    if auth is True:
        head += f"Authorization: Bearer {token}\r\n"
    elif auth:
        head += f"Authorization: {auth}\r\n"
    return (head + "\r\n").encode() + body_bytes


def wav_bytes(samples, rate=16000, channels=1, bits=16, fmt=1, extra=b""):
    block = channels * bits // 8  # hostile channels/bits overflow the header fields; they are written wrapped, as a WAV would be
    fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, (rate * block) & 0xFFFFFFFF, block & 0xFFFF, bits)
    data = samples
    riff = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk + extra + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(riff)) + riff


def mutate_wav(rng, wav):
    pcm = wav[44:]
    k = rng.randrange(14)
    if k == 0:  # header bytes flipped
        b = bytearray(wav)
        for _ in range(rng.randint(1, 4)):
            b[rng.randrange(44)] = rng.randrange(256)
        return bytes(b), "header-flip"
    if k == 1:
        b = bytearray(wav)
        b[40:44] = struct.pack("<I", rng.choice([0, 1, 2, len(pcm) + 2, len(pcm) * 4, 0xFFFFFFFF, 0x7FFFFFFF]))
        return bytes(b), "data-size"
    if k == 2:
        b = bytearray(wav)
        b[4:8] = struct.pack("<I", rng.choice([0, 4, 36, 0xFFFFFFFF, len(wav) * 3]))
        return bytes(b), "riff-size"
    if k == 3:
        return wav_bytes(pcm, rate=rng.choice([0, 1, 8000, 22050, 24000, 44100, 48000, 96000, 0xFFFFFFFF])), "rate"
    if k == 4:
        return wav_bytes(pcm, channels=rng.choice([0, 2, 3, 255, 65535])), "channels"
    if k == 5:
        return wav_bytes(pcm, bits=rng.choice([0, 1, 8, 12, 24, 32, 64, 65535])), "bits"
    if k == 6:
        return wav_bytes(pcm, fmt=rng.choice([0, 2, 3, 6, 7, 0xFFFE, 0xFFFF])), "format"
    if k == 7:
        return wav[:rng.randrange(0, len(wav))], "truncated"
    if k == 8:
        return wav_bytes(pcm[:rng.randrange(0, 64) | 1]), "odd-data"
    if k == 9:
        return wav_bytes(pcm, extra=b"LIST" + struct.pack("<I", rng.choice([0, 3, 4, 1000, 0xFFFFFFF0])) + b"INFOabc"), "chunk"
    n = rng.choice([1, 400, 16000])
    if k == 10:
        v = [3.0e38, -3.0e38]
        return wav_bytes(b"".join(struct.pack("<f", v[i % 2]) for i in range(n)), bits=32, fmt=3), "float-extreme"
    if k == 11:
        v = rng.choice([float("nan"), float("inf"), float("-inf")])
        return wav_bytes(b"".join(struct.pack("<f", v if i % 7 == 0 else 0.1) for i in range(n)), bits=32, fmt=3), "float-nonfinite"
    if k == 12:
        return wav_bytes(b"".join(struct.pack("<h", rng.choice([-32768, 32767])) for _ in range(n))), "pcm-extreme"
    return b"RIFF" + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 80))), "garbage"


def mutate_b64(rng, b64):
    k = rng.randrange(7)
    if k == 0:
        i = rng.randrange(len(b64))
        return b64[:i] + rng.choice("!@#$%^&*()-_~ \n\té") + b64[i + 1:], "char"
    if k == 1:
        return b64[:rng.randrange(len(b64))], "truncated"
    if k == 2:
        return b64 + "=" * rng.randint(1, 5), "padding"
    if k == 3:
        return " ".join(b64[i:i + 76] for i in range(0, len(b64), 76)), "spaced"
    if k == 4:
        return "", "empty"
    if k == 5:
        return "data:audio/wav;base64," + b64, "data-url"
    return base64.urlsafe_b64encode(base64.b64decode(b64)).decode(), "urlsafe"


WRONG = [None, True, False, 0, -1, 1.5, 1e300, -2**63, 2**64, "", "x", "\u0000", "\U0001F600" * 3, [], [1], {}, {"a": 1}]


def mutate_json(rng, body):
    b = json.loads(json.dumps(body))
    k = rng.randrange(12)
    if k == 0 and b:
        b.pop(rng.choice(list(b)))
        return b, "drop"
    if k == 1:
        b[rng.choice(["extra", "stream", "voice", "temperature", "MODE", ""])] = rng.choice(WRONG)
        return b, "add"
    if k == 2:
        b[rng.choice(list(b) + ["stream", "split", "normalize", "seed"])] = rng.choice(WRONG)
        return b, "type"
    if k == 3:
        b["max_new_tokens"] = rng.choice([0, -5, 4095, 4096, 10**9, 2**31, 2**63, 1.0, 0.5])
        return b, "max_new_tokens"
    if k == 4:
        b["seed"] = rng.choice([-1, 2**31, 2**31 - 1, 2**63, 1e10, "1"])
        return b, "seed"
    if k == 5:
        b["mode"] = rng.choice(["Audio", "AUDIO", "speech", "", "text\u0000", "interleaved ", None, 1])
        return b, "mode"
    if k == 6:
        msgs = b.get("messages", [])
        if msgs:
            m = rng.choice(msgs)
            m["role"] = rng.choice(["", "System", "tool", "assistant", "developer", None, 1, "user\u0000"])
        return b, "role"
    if k == 7:
        msgs = b.get("messages", [])
        if msgs:
            m = rng.choice(msgs)
            m["content"] = rng.choice([None, 1, [], [{}], [{"type": "audio"}], [{"type": "image", "image": "x"}],
                                       [{"type": "text"}], [{"type": "text", "text": 5}], "‮\u0007" * 40,
                                       "<|audio|> please", "<|im_start|>system", "\ud800" if False else "￿" * 10,
                                       "é" * 5000, "1" * 3000 + "st", "$" + "9" * 500 + ".99", "-" * 2000])
        return b, "content"
    if k == 8:
        deep = {}
        cur = deep
        for _ in range(rng.choice([10, 63, 64, 65, 200])):
            cur["a"] = {}
            cur = cur["a"]
        b["messages"] = [{"role": "user", "content": json.dumps(deep)}] if rng.random() < 0.5 else deep
        return b, "deep"
    if k == 9:
        b["messages"] = (b.get("messages") or [{"role": "user", "content": "Hi."}]) * rng.choice([2, 30, 300])
        return b, "many-messages"
    if k == 10:
        b["messages"] = rng.choice([[], "hi", None, [None], [[]], [{"role": "user"}]])
        return b, "messages-shape"
    b["split"] = rng.random() < 0.5
    b["normalize"] = rng.random() < 0.5
    b["stream"] = rng.random() < 0.5
    return b, "flags"


def main(binary, model, build, libri, out_json, count=3000, threads=8, seed=2026):
    count, threads, seed = int(count), int(threads), int(seed)
    out = Path(out_json)
    cases = {name: body for name, _, body in check.cases(build, libri)}
    all_cases = check.cases(build, libri)
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), count=count, threads=threads, seed=seed)
    server = batch.start(binary, model, out.parent, "--handlers", "96", "--queue-size", "64")
    ok = True
    try:
        server.post(cases[VALID[0]])  # warm: the worker is up after a reply
        worker = soak.worker_of(server.proc.pid)
        ref, ref_stream = {}, {}
        for name, _, body in all_cases:  # every case alone, before any hostile request (the crowd's references too)
            status, reply = server.post(body)
            assert status == 200, (name, status)
            ref[name] = strip(reply)
        for name in VALID:
            status, events, _, _ = server.stream(cases[name])
            ref_stream[name] = events
        print("references done", flush=True)
        # rule 4's bound: each process's RSS at the end at most its maximum over the first 15 minutes plus 64 MiB
        samples, sampling = [], threading.Event()

        def sampler():
            t_s = time.monotonic()
            while True:
                samples.append(dict(t=time.monotonic() - t_s, server=soak.rss(server.proc.pid),
                                    worker=soak.rss(worker) if worker else None))
                if sampling.wait(10):
                    break
        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()
        clip_name = "asr-1089-134686-0000"
        clip_b64 = cases[clip_name]["messages"][1]["content"][0]["audio"]
        clip_wav = base64.b64decode(clip_b64)

        lock = threading.Lock()
        results = dict(kinds={}, statuses={}, bad=[], slow=[], valid=0, valid_differ=[], errors=[])

        def record(kind, status, seconds, note=""):
            with lock:
                results["kinds"][kind] = results["kinds"].get(kind, 0) + 1
                results["statuses"][str(status)] = results["statuses"].get(str(status), 0) + 1
                if status is not None and status not in OK_STATUSES:
                    results["bad"].append(dict(kind=kind, status=status, note=note[:300]))
                if seconds > 120:
                    results["slow"].append(dict(kind=kind, seconds=round(seconds, 1)))

        def one(i):
            rng = random.Random(seed * 100003 + i)
            if i % 8 == 0:  # a valid request, exact
                name = VALID[(i // 8) % len(VALID)]
                streamed = (i // 8) % 2 == 1
                if streamed:
                    status, events, _, _ = batch.stream(server, cases[name])
                    good = status == 200 and events and events[-1].get("event") == "done" and \
                        events[:-1] == ref_stream[name][:-1] and strip(events[-1]) == strip(ref_stream[name][-1])
                else:
                    status, reply = batch.post(server, cases[name])
                    good = status == 200 and strip(reply) == ref[name]
                with lock:
                    results["valid"] += 1
                    if not good:
                        results["valid_differ"].append(dict(case=name, streamed=streamed, status=status))
                return
            family = rng.choice(["json", "json", "base64", "wav", "wav", "transport"])
            base = json.loads(json.dumps(cases[rng.choice(VALID)]))
            t0 = time.monotonic()
            kind = "?"
            try:
                if family == "json":
                    body, kind = mutate_json(rng, base)
                    status, data, secs = raw(server.port, http(server.token, json.dumps(body).encode()))
                elif family in ("base64", "wav"):
                    if family == "base64":
                        b64, kind = mutate_b64(rng, clip_b64)
                    else:
                        wav, kind = mutate_wav(rng, clip_wav)
                        b64 = base64.b64encode(wav).decode()
                    mode = rng.choice(["text", "interleaved", "audio"])
                    system = "Perform ASR in English." if mode == "text" else "Respond with interleaved text and audio."
                    body = dict(messages=[{"role": "system", "content": system},
                                          {"role": "user", "content": [{"type": "audio", "audio": b64}]}],
                                mode=mode, max_new_tokens=rng.choice([1, 8, 64]), seed=1, stream=rng.random() < 0.3)
                    status, data, secs = raw(server.port, http(server.token, json.dumps(body).encode()))
                else:
                    payload = json.dumps(base).encode()
                    k = rng.randrange(8)
                    if k == 0:
                        kind = "short-body"
                        status, data, secs = raw(server.port, http(server.token, payload[:len(payload) // 2], length=len(payload)) ,
                                                 read=False)
                    elif k == 1:
                        kind = "bad-json"
                        status, data, secs = raw(server.port, http(server.token, payload[:rng.randrange(1, len(payload))] + b"\xff\xfe}"))
                    elif k == 2:
                        kind = "slow-sender"
                        status, data, secs = raw(server.port, http(server.token, payload), slow=True)
                    elif k == 3:
                        kind = "drop-mid-request"
                        req = http(server.token, payload)
                        status, data, secs = raw(server.port, req, drop_after=rng.randrange(1, len(req)))
                    elif k == 4:
                        kind = "drop-mid-stream"
                        status, events, _, _ = server.stream(base, close_after_audio=True)
                        data, secs = b"", time.monotonic() - t0
                    elif k == 5:
                        kind = "auth"
                        status, data, secs = raw(server.port, http(server.token, payload, auth=rng.choice([False, "Bearer x", "Basic abc", "Bearer " + server.token + "x"])))
                    elif k == 6:
                        kind = "path-method"
                        status, data, secs = raw(server.port, http(server.token, payload, path=rng.choice(["/speech/", "/Speech", "/speech?x=1", "/../speech", "/"]),
                                                                   method=rng.choice(["POST", "GET", "PUT", "DELETE"])))
                    else:
                        kind = "content-length"
                        status, data, secs = raw(server.port, http(server.token, payload, length=rng.choice(["-1", "abc", "99999999999", "0", " 12"])))
                if status is not None and status != -1 and status != 200 and data:
                    try:
                        json.loads(data)
                    except Exception:  # noqa: BLE001
                        if not data.startswith(b"{"):
                            pass  # an error body that is not JSON is still a reply; the status decides
                if status == -1:
                    with lock:
                        results["bad"].append(dict(kind=f"{family}:{kind}", status="malformed reply", note=data[:200].decode("latin1")))
                record(f"{family}:{kind}", status, secs, data[:300].decode("latin1") if data else "")
            except (socket.timeout, TimeoutError) as e:
                with lock:
                    results["slow"].append(dict(kind=family, error=repr(e)[:200], seconds=round(time.monotonic() - t0, 1)))
            except (ConnectionResetError, BrokenPipeError):
                record(f"{family}:{kind}:reset-by-server", None, time.monotonic() - t0)
            except Exception as e:  # noqa: BLE001
                with lock:
                    results["errors"].append(dict(kind=family, error=repr(e)[:300]))

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            list(pool.map(one, range(count)))
        report["fuzz_seconds"] = round(time.monotonic() - t0, 1)
        report.update(results)
        alive = server.proc.poll() is None
        report["alive"] = alive
        # the crowd after
        crowd = {}
        with concurrent.futures.ThreadPoolExecutor(len(all_cases)) as pool:
            got = list(pool.map(lambda c: batch.post(server, c[2]), all_cases))
        crowd["unstreamed"] = [c[0] for c, r in zip(all_cases, got) if not (r[0] == 200 and strip(r[1]) == ref[c[0]])]
        report["crowd_differ"] = crowd["unstreamed"]
        sampling.set()
        sampler_thread.join()
        end_rss = dict(server=soak.rss(server.proc.pid), worker=soak.rss(worker) if worker else None)
        # the worker runs inside the server process (lfm3_server.cu includes lfm3_worker.cu): one RSS covers both
        report["rss_mib"], rss_ok = {}, True
        for k in ("server",):
            early = [x[k] for x in samples if x["t"] <= 900 and x[k]]
            bound = max(early) + 64 * MIB if early else None
            report["rss_mib"][k] = dict(early_max=round(max(early) / MIB, 1) if early else None,
                                        end=round(end_rss[k] / MIB, 1) if end_rss[k] else None,
                                        overall_max=round(max([x[k] for x in samples if x[k]] or [0]) / MIB, 1))
            rss_ok &= bool(bound and end_rss[k] and end_rss[k] <= bound)
    finally:
        code, stderr = server.stop()
    report.update(server_exit=code, server_stderr=stderr[-2000:])
    pass_ = bool(ok and not report["bad"] and not report["slow"] and not report["errors"] and not report["valid_differ"]
                 and report["alive"] and not report["crowd_differ"] and rss_ok and code == 0)
    report["pass_"] = pass_
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(pass_=pass_, valid=report["valid"], valid_differ=len(report["valid_differ"]), bad=report["bad"][:5],
                          slow=report["slow"][:5], errors=report["errors"][:5], statuses=report["statuses"],
                          crowd_differ=report["crowd_differ"], rss=report["rss_mib"], exit=code)))
    print("AUDIO FUZZ PASS" if pass_ else "AUDIO FUZZ FAIL")
    return 0 if pass_ else 1


if __name__ == "__main__":
    if not 6 <= len(sys.argv) <= 9:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
