#!/usr/bin/env python3
"""Rule 1 of validation/text-wait-v1, over HTTP (black box): text and speech replies are the same whatever runs beside.

Text set: the soak's 3 text bodies, the text gate's control (128 tokens) and memory prompt (32), a sampled body
(temperature 0.528, seed 42) and a continuation on the control's session (the cache-reuse path). Compared fields:
generated_ids, text, reused_tokens, consumed_tokens, cancelled, finish_reason.
  1. the text set alone on the reference server (assure-s4);
  2. alone on the server under test;
  3. on the server under test while 8 streams of budget-truncated-s1 run (each stream's events must equal that
     request's stream alone);
  4. the 42 speech cases alone on the server under test (unstreamed and streamed), then all at once (a crowd of 42
     unstreamed, then 42 streamed) while a text client runs the text set back to back: every reply identical.
Run once as is and once with CUDA_LAUNCH_BLOCKING=1 in the environment (the servers inherit it).
Usage: check_lfm3_text_wait_exact.py SERVER REFERENCE_SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("soak", HERE / "check_lfm3_audio_soak.py")
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)
spec = importlib.util.spec_from_file_location("wait", HERE / "check_lfm3_text_wait.py")
wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wait)
check, batch, strip = soak.check, soak.batch, soak.strip

CONTROL = [dict(role="system", content="Respond with text."),
           dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
REMEMBERED = [dict(role="system", content="Respond with text."),
              dict(role="user", content="Remember the code word cobalt. Reply briefly.")]
KEYS = ("generated_ids", "text", "reused_tokens", "consumed_tokens", "cancelled", "finish_reason")


def start(binary, model, scratch, tag):
    """As check_lfm3_speech_batch_server.start, the server's stderr kept in SCRATCH/server-TAG.stderr."""
    import secrets
    import subprocess
    s = check.Server.__new__(check.Server)
    s.token = secrets.token_urlsafe(32)
    token_path = Path(scratch) / f"exact-{tag}.token"
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(s.token + "\n")
    s.stderr_path = Path(scratch) / f"server-{tag}.stderr"
    s.stderr_file = s.stderr_path.open("w")
    # a crowd of 42 at once beside a text client: 96 handlers and a queue of 64, as the batch-server check and the soak
    # give their crowd servers (with the defaults, 16 handlers, the crowd's surplus connections are refused or reset)
    s.proc = subprocess.Popen([binary, "--experimental", model, "--token-file", str(token_path), "--port", "0",
                               "--handlers", "96", "--queue-size", "64"],
                              stdout=subprocess.PIPE, stderr=s.stderr_file, text=True)
    s.port = json.loads(s.proc.stdout.readline())["port"]
    token_path.unlink()

    def stop():
        s.proc.terminate()  # SIGTERM only
        s.proc.wait(timeout=600)
        s.stderr_file.close()
        return s.proc.returncode, s.stderr_path.read_text()
    s.stop = stop
    return s


def text_set(server):
    """{name: the compared fields}; each body in a fresh session, the continuation on the control's."""
    out = {}
    singles = [(f"soak-{i}", b) for i, b in enumerate(soak.TEXTS)] + [
        ("gate-memory", dict(messages=REMEMBERED, max_new_tokens=32)),
        ("sampled", dict(messages=CONTROL, max_new_tokens=128,
                         sampling=dict(do_sample=True, temperature=0.528, seed=42)))]
    for name, body in singles:
        status, reply = generate(server, None, body)
        out[name] = pick(status, reply)
    status, created = retried_call(server, "/sessions", {"capacity": 2048})
    sid = created["session_id"]
    try:
        status, control = generate(server, sid, dict(messages=CONTROL, max_new_tokens=128))
        out["gate-control"] = pick(status, control)
        continued = CONTROL + [dict(role="assistant", content=control.get("text", "")),
                               dict(role="user", content="Summarize your previous answer in one short sentence.")]
        status, reply = generate(server, sid, dict(messages=continued, max_new_tokens=64))
        out["gate-continued"] = pick(status, reply)
    finally:
        wait.call(server, f"/sessions/{sid}/close", {})
    return out


def retried_call(server, path, body):
    status, reply = wait.call(server, path, body)
    while status == 429:
        time.sleep(0.01)
        status, reply = wait.call(server, path, body)
    return status, reply


def generate(server, sid, body):
    own = sid is None
    if own:
        status, created = retried_call(server, "/sessions", {"capacity": 2048})
        if status != 200:
            return status, created
        sid = created["session_id"]
    try:
        return retried_call(server, "/generate", dict(body, session_id=sid))
    finally:
        if own:
            wait.call(server, f"/sessions/{sid}/close", {})


def pick(status, reply):
    return dict(status=status, **{k: reply.get(k) for k in KEYS})


def same_stream(want, got):
    status, events = got[0], got[1]
    return status == 200 and bool(events) and events[-1].get("event") == "done" and events[:-1] == want[:-1] \
        and strip(events[-1]) == strip(want[-1])


def diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def main(binary, reference, model, build, libri, out_json):
    out = Path(out_json)
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                  cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"))
    cases = check.cases(build, libri)
    byname = {n: b for n, _, b in cases}
    ok = True

    tag = "blocking" if os.environ.get("CUDA_LAUNCH_BLOCKING") else "normal"
    ref = start(reference, model, out.parent, f"reference-{tag}")
    try:
        ref_text = text_set(ref)
    finally:
        code, _ = ref.stop()
        report["reference_exit"] = code
        ok &= code == 0

    server = start(binary, model, out.parent, f"test-{tag}")
    phase = "text alone"
    try:
        alone = text_set(server)
        report["text_alone_vs_reference"] = d = diff(ref_text, alone)
        ok &= not d and all(v["status"] == 200 for v in alone.values())
        print("text alone vs reference:", d or "identical", flush=True)

        phase = "text beside 8 streams"
        # 3. beside 8 streams
        long_body = byname["budget-truncated-s1"]
        _, long_alone, _, _ = batch.retried(lambda: server.stream(long_body, timeout=1800))
        streams = wait.start_streams(server, [long_body] * 8, 2.0)
        loaded = text_set(server)
        still = sum(s.is_alive() for s in streams)
        for s in streams:
            s.join()
        report["text_beside_8_streams"] = d = diff(alone, loaded)
        report["streams_alive_at_text_end"] = still
        streams_same = [same_stream(long_alone, (s.result[0], s.result[1])) for s in streams]
        report["streams_identical"] = sum(streams_same)
        ok &= not d and still == 8 and all(streams_same)
        print(f"text beside 8 streams: {d or 'identical'} (streams still running at the end: {still}; "
              f"identical {sum(streams_same)}/8)", flush=True)

        phase = "the 42 cases alone"
        # 4. the 42 cases alone, then as crowds beside a text client
        alone_reply, alone_stream = {}, {}
        for name, _, body in cases:
            status, reply = batch.post(server, body)
            alone_reply[name] = (status, strip(reply))
            r = batch.stream(server, body)
            alone_stream[name] = r[1]
        stop = threading.Event()
        texts = []

        def texter():
            while not stop.is_set():
                texts.append(text_set(server))
        tt = threading.Thread(target=texter)
        phase = "crowds beside a text client"
        tt.start()
        crowd = {}
        for streamed in (False, True):
            with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
                if streamed:
                    got = list(pool.map(lambda c: batch.stream(server, c[2]), cases))
                    bad = [c[0] for c, r in zip(cases, got) if not same_stream(alone_stream[c[0]], r)]
                else:
                    got = list(pool.map(lambda c: batch.post(server, c[2]), cases))
                    bad = [c[0] for c, r in zip(cases, got) if (r[0], strip(r[1])) != alone_reply[c[0]]]
            crowd["streamed" if streamed else "unstreamed"] = dict(identical=len(cases) - len(bad), differ=bad)
        stop.set()
        tt.join()
        text_bad = [diff(alone, t) for t in texts if diff(alone, t)]
        report["crowd_beside_text"] = crowd
        report["text_sets_beside_crowd"] = dict(runs=len(texts), differ=text_bad)
        ok &= all(not c["differ"] for c in crowd.values()) and not text_bad and len(texts) >= 1
        print("crowd beside text:", {k: v["identical"] for k, v in crowd.items()}, "text sets:", len(texts),
              "differ:", len(text_bad), flush=True)
    except Exception as e:  # noqa: BLE001  (recorded: where, and whether the server was still alive)
        report["error"] = dict(phase=phase, error=repr(e)[:500], server_returncode=server.proc.poll())
        ok = False
        print("ERROR during", phase, "server returncode", server.proc.poll(), repr(e)[:300], flush=True)
    finally:
        code, stderr = server.stop()
        report["server_exit"] = code
        report["server_stderr_tail"] = stderr[-3000:]
        ok &= code == 0
        out.write_text(json.dumps(report, indent=1))
    report["text"] = alone
    report["pass_"] = bool(ok)
    out.write_text(json.dumps(report, indent=1))
    print("TEXT WAIT EXACT PASS" if ok else "TEXT WAIT EXACT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
