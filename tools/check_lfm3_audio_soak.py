#!/usr/bin/env python3
"""Long mixed audio run of liquid-lfm3-server (validation/audio-assurance-v1 rule 4; Thor, lfm3-profile-v1). Black box.

1. References: the 42 cases of check_lfm3_speech_stream.py (TTS both voices, interleaved, split long-form, the budget
   cut, normalized numbers, spoken questions, LibriSpeech transcriptions), each alone, unstreamed and streamed, and
   three text /generate bodies, on a fresh server.
2. The soak: a second fresh server (8 speech rows, the default) under CLIENTS client threads for SECONDS. Each request
   is one of: a case unstreamed, a case streamed, a case streamed and dropped after its first audio event (about 1 in
   10 streamed; not compared), or a text /generate (about 1 in 10). 429s are retried and counted. Every completed
   reply is compared with its reference (timings aside; streamed: every event).
3. Every 30 s: VmRSS of the server (the worker runs inside it), and the host's memory in use (MemTotal - MemAvailable; on Thor the
   GPU allocates from system memory and reports no per-process figure), and this client's own RSS before and after
   glibc's malloc_trim(0) (the client is on the same host: its memory is in the host's figure).
4. After the soak: the 42 cases again, all at once, unstreamed and streamed (a crowd), identical to the references;
   then SIGTERM, exit 0.
Pass (rule 4): every completed reply identical, no 5xx and no transport error, both processes alive throughout, each
memory figure flat at idle, the crowd identical, exit 0. Memory figures: the server's RSS, and the host's memory in
use minus this client's RSS (the GPU's stand-in; text-wait-v1 found the client's own growth inside the host's figure).
Idle checkpoints (text-wait-v1, after twa-2's second soak showed that under load the GPU's share swings about 100 MiB
a sample, so a 30-sample median wobbles past 64 MiB): at 5 minutes, every 20 minutes after, and at the end, the
clients pause, the server drains (no request in flight), 10 s pass, then the same fixed sequence runs alone, one at
a time, each reply checked against its reference: a text /generate, a spoken question unstreamed (audio in,
interleaved out), a TTS case streamed, an ASR case unstreamed (the server sizes some buffers to the last request, so
the third soak's idle readings sat on one of two levels about 200 MiB apart, whichever the last request left); 10 s
pass, and 10 readings 1 s apart are taken; their median is the checkpoint. Flat: the last checkpoint at most the first plus 64 MiB, at least 4 checkpoints. Reported,
not gated: the load-time medians (the last 15 minutes against minutes 5 to 20) and the original endpoint test (the
last sample against the first 15 minutes' maximum plus 64 MiB).
The server's stdout and stderr are drained while it runs (a full pipe would block it).
Writes OUT_JSON (and OUT_JSON's .samples.jsonl) and exits 1 on any failure.
TEXT_SHARE (default 0.1): the share of text /generate requests; the speech kinds keep their proportions among the
rest (validation/text-wait-v1 rule 7 runs 0.25). At 0.1 every client draws exactly the requests it drew before.
Usage: check_lfm3_audio_soak.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [SECONDS [CLIENTS [TEXT_SHARE]]]
"""
import concurrent.futures
import ctypes
import hashlib
import importlib.util
import json
import os
import random
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
strip, start = batch.strip, batch.start
TEXTS = [dict(max_new_tokens=96, messages=[{"role": "user", "content": c}]) for c in (
    "Explain in one paragraph how a lighthouse warns ships at night.",
    "List three ways to keep bread fresh longer.",
    "What is the difference between weather and climate? Answer briefly.")]
MIB = 1 << 20
CHECKPOINT_FIRST, CHECKPOINT_EVERY = 300, 1200  # seconds into the soak
CHECKPOINT_SETTLE = 10  # seconds idle before the readings


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None


def children(pid):
    try:
        return [int(c) for c in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def worker_of(server_pid):
    """The worker process (the server's child; it starts after the server reports its port), or None."""
    kids = children(server_pid)
    return kids[0] if kids else None


LIBC = ctypes.CDLL("libc.so.6")


def client_rss():
    """(this process's RSS, its RSS after malloc_trim(0)): a drop is memory glibc held, not memory in use."""
    before = rss(os.getpid())
    LIBC.malloc_trim(0)
    return before, rss(os.getpid())


def drain(server):
    """Read the server's stdout and stderr while it runs (a full pipe would block its writes); stop() returns them."""
    got = {"out": [], "err": []}
    def pump(f, into):
        for line in f:
            into.append(line)
    threads = [threading.Thread(target=pump, args=(server.proc.stdout, got["out"]), daemon=True),
               threading.Thread(target=pump, args=(server.proc.stderr, got["err"]), daemon=True)]
    for t in threads:
        t.start()

    def stop():
        server.proc.terminate()  # SIGTERM only
        server.proc.wait(timeout=600)
        for t in threads:
            t.join(timeout=30)
        return server.proc.returncode, "".join(got["err"])
    server.stop = stop
    return server


def host_used():
    m = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()}
    return m["MemTotal"] - m["MemAvailable"]


def memory_report(samples):
    """(flat test per figure, the old endpoint test per figure, the gated figures)."""
    # flat (the last 15 minutes' median against minutes 5-20's, plus 64 MiB); the old endpoint test reported
    memory, endpoint = {}, {}
    early = [s for s in samples if s["t"] <= 900]
    last_t = samples[-1]["t"] if samples else 0
    for key in ("server_rss", "host_used", "host_used_less_client", "client_rss", "client_rss_trimmed"):
        vals = [s[key] for s in early if s.get(key) is not None]
        every = [s[key] for s in samples if s.get(key) is not None]
        end = samples[-1].get(key) if samples else None
        bound = max(vals) + 64 * MIB if vals else None
        endpoint[key] = dict(early_max_mib=round(max(vals) / MIB, 1) if vals else None,
                             end_mib=round(end / MIB, 1) if end else None,
                             overall_max_mib=round(max(every) / MIB, 1) if every else None,
                             pass_=end is not None and bound is not None and end <= bound)
        base = [s[key] for s in samples if 300 <= s["t"] <= 1200 and s.get(key) is not None]
        late = [s[key] for s in samples if s["t"] >= last_t - 900 and s.get(key) is not None]
        flat = dict(base_median_mib=round(statistics.median(base) / MIB, 1) if base else None,
                    late_median_mib=round(statistics.median(late) / MIB, 1) if late else None)
        flat["growth_mib"] = round(flat["late_median_mib"] - flat["base_median_mib"], 1) if base and late else None
        flat["pass_"] = flat["growth_mib"] is not None and flat["growth_mib"] <= 64 and last_t >= 2100
        memory[key] = flat
    gated = ("server_rss", "host_used_less_client")
    return memory, endpoint, gated


def idle_report(checkpoints):
    """Per gated figure: the first and last idle checkpoints (MiB), the growth, every checkpoint, pass."""
    out = {}
    for key in ("server_rss", "host_used_less_client"):
        vals = [c[key] for c in checkpoints]
        growth = round((vals[-1] - vals[0]) / MIB, 1) if len(vals) >= 2 else None
        out[key] = dict(first_mib=round(vals[0] / MIB, 1) if vals else None,
                        last_mib=round(vals[-1] / MIB, 1) if vals else None, growth_mib=growth,
                        every_mib=[round(v / MIB, 1) for v in vals],
                        pass_=len(vals) >= 4 and growth is not None and growth <= 64)
    return out


def generate(server, body):
    """(status, reply) of /generate in a fresh session; 429 retried by the caller."""
    def post(path, b):
        req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=json.dumps(b).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    status, created = post("/sessions", {"capacity": 1024})
    if status != 200:
        return status, created
    try:
        return post("/generate", dict(body, session_id=created["session_id"]))
    finally:
        post(f"/sessions/{created['session_id']}/close", {})


def describe(want, got):
    """Which top-level keys differ; for lists, the lengths and the first index that differs."""
    if not isinstance(got, dict) or not isinstance(want, dict):
        return dict(got_type=type(got).__name__)
    out = {}
    for k in sorted(set(want) | set(got)):
        a, b = want.get(k), got.get(k)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            out[k] = dict(want_len=len(a), got_len=len(b), first_differ=i)
        else:
            out[k] = dict(want=str(a)[:200], got=str(b)[:200])
    return out


def crowd_round(server, cases, alone, same_stream, differ_path):
    """The cases all at once, unstreamed then streamed. Per kind: identical count, differing names, and for each
    difference its status, the retries spent and which fields differ; differing replies go to differ_path."""
    crowd = {}
    for streamed in (False, True):
        tries = {}

        def call(c):
            n = 0
            while True:
                r = server.stream(c[2]) if streamed else server.post(c[2])
                n += 1
                if r[0] != 429 or n >= 3000:
                    tries[c[0]] = n
                    return r
                time.sleep(0.01)

        with concurrent.futures.ThreadPoolExecutor(len(cases)) as pool:
            got = list(pool.map(call, cases))
        detail = []
        for c, r in zip(cases, got):
            name = c[0]
            if streamed:
                if same_stream(name, r[0], r[1]):
                    continue
                d = dict(name=name, status=r[0], tries=tries[name], events=len(r[1] or []))
                reply = r[1]
            else:
                if r[0] == 200 and strip(r[1]) == alone[name]:
                    continue
                d = dict(name=name, status=r[0], tries=tries[name],
                         fields=describe(alone[name], strip(r[1]) if isinstance(r[1], dict) else r[1]))
                reply = r[1]
            detail.append(d)
            with open(differ_path, "a") as f:
                f.write(json.dumps(dict(d, streamed=streamed, reply=reply)) + "\n")
        kind = "streamed" if streamed else "unstreamed"
        crowd[kind] = dict(identical=len(cases) - len(detail), differ=[d["name"] for d in detail], detail=detail)
        print(f"crowd {kind}: {crowd[kind]['identical']}/{len(cases)} identical {[(d['name'], d['status'], d['tries']) for d in detail]}", flush=True)
    return crowd


def text_key(reply):
    return reply.get("generated_ids"), reply.get("text")


def main(binary, model, build, libri, out_json, seconds=7200, clients=12, text_share=0.1):
    seconds, clients, text_share = int(seconds), int(clients), float(text_share)
    assert 0 <= text_share < 1
    out = Path(out_json)
    samples_path = out.with_suffix(".samples.jsonl")
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), seconds=seconds, clients=clients,
                  text_share=text_share)
    cases = check.cases(build, libri)
    ok = True

    # 1. references, alone, on a fresh server
    ref = drain(start(binary, model, out.parent))
    alone, alone_stream, text_ref = {}, {}, []
    try:
        ref.post(cases[0][2])  # warm
        for name, _, body in cases:
            status, reply = ref.post(body)
            assert status == 200, (name, status, reply)
            alone[name] = strip(reply)
            status, events, _, _ = ref.stream(body)
            assert status == 200, (name, status)
            alone_stream[name] = events
        for body in TEXTS:
            status, reply = generate(ref, body)
            assert status == 200, (status, reply)
            text_ref.append(text_key(reply))
    finally:
        code, _ = ref.stop()
        ok &= code == 0
        report["reference_server_exit"] = code
    print("references done", flush=True)

    def same_stream(name, status, events):
        want = alone_stream[name]
        return status == 200 and bool(events) and events[-1].get("event") == "done" and events[:-1] == want[:-1] \
            and strip(events[-1]) == strip(want[-1])

    # 2. the soak
    server = drain(start(binary, model, out.parent, "--handlers", "96", "--queue-size", "64"))
    server.post(cases[0][2])  # warm: the worker is up after a reply
    worker = worker_of(server.proc.pid)
    report["worker_found"] = worker is not None
    counts = dict(unstreamed=0, streamed=0, dropped=0, text=0, retries_429=0)
    differ, errors, statuses = [], [], {}
    lock = threading.Lock()
    stop = threading.Event()
    go = threading.Event()  # cleared for an idle checkpoint
    go.set()
    active = [0]  # requests in flight (a client counts itself in under the lock only while go is set)
    checkpoints = []
    t_start = time.monotonic()
    samples = []

    def sampler():
        with samples_path.open("w") as f:
            while True:
                used = host_used()
                c_before, c_after = client_rss()
                s = dict(t=round(time.monotonic() - t_start, 1), server_rss=rss(server.proc.pid),
                         worker_rss=rss(worker) if worker else None, host_used=used,
                         client_rss=c_before, client_rss_trimmed=c_after, host_used_less_client=used - c_before,
                         server_alive=server.proc.poll() is None)
                samples.append(s)
                f.write(json.dumps(s) + "\n")
                f.flush()
                if stop.wait(30):
                    break

    def retry(call):
        for _ in range(30000):
            r = call()
            if r[0] != 429:
                return r
            with lock:
                counts["retries_429"] += 1
            time.sleep(0.02)
        return r

    def client(seed):
        rng = random.Random(seed)
        while not stop.is_set():
            with lock:
                admitted = go.is_set()
                if admitted:
                    active[0] += 1
            if not admitted:
                go.wait(1)
                continue
            try:
                one(rng)
            finally:
                with lock:
                    active[0] -= 1

    def checkpoint():
        """Pause the clients, wait for no request in flight, 10 s, then 10 readings 1 s apart (medians)."""
        t_pause = time.monotonic()
        with lock:
            go.clear()
        while True:
            with lock:
                if active[0] == 0:
                    break
            time.sleep(0.2)
        drained = time.monotonic() - t_pause
        time.sleep(CHECKPOINT_SETTLE)
        fixed_ok = fixed_sequence()
        time.sleep(CHECKPOINT_SETTLE)
        reads = []
        for _ in range(10):
            used = host_used()
            c, _ = client_rss()
            reads.append(dict(server_rss=rss(server.proc.pid), host_used=used, client_rss=c,
                              host_used_less_client=used - c))
            time.sleep(1)
        cp = dict(t=round(t_pause - t_start, 1), drain_seconds=round(drained, 1), fixed_identical=fixed_ok)
        for k in reads[0]:
            vals = sorted(r[k] for r in reads)
            cp[k] = statistics.median(vals)
            cp[k + "_spread_mib"] = round((vals[-1] - vals[0]) / MIB, 1)
        checkpoints.append(cp)
        print(f"checkpoint t={cp['t']:.0f} s: drained {cp['drain_seconds']} s, server RSS {cp['server_rss'] / MIB:.1f} MiB, "
              f"host less client {cp['host_used_less_client'] / MIB:.1f} MiB (spread {cp['host_used_less_client_spread_mib']})",
              flush=True)
        go.set()

    by_name = {name: body for name, _, body in cases}
    first_tts = next(name for name, kind, _ in cases if kind == "tts")

    def fixed_sequence():
        """The checkpoint's fixed requests, alone and in order; each compared with its reference (a miss is a differ)."""
        results = []
        status, reply = retry(lambda: generate(server, TEXTS[0]))
        results.append(("fixed-text-0", status == 200 and text_key(reply) == text_ref[0], status))
        status, reply = retry(lambda: server.post(by_name["spoken-q0"]))
        results.append(("fixed-spoken-q0", status == 200 and strip(reply) == alone["spoken-q0"], status))
        status, events, _, _ = retry(lambda: server.stream(by_name[first_tts]))
        results.append((f"fixed-{first_tts}", same_stream(first_tts, status, events), status))
        status, reply = retry(lambda: server.post(by_name["asr-1089-134686-0000"]))
        results.append(("fixed-asr-1089-134686-0000", status == 200 and strip(reply) == alone["asr-1089-134686-0000"],
                        status))
        with lock:
            for label, good, status in results:
                statuses[str(status)] = statuses.get(str(status), 0) + 1
                if not good:
                    differ.append(dict(kind="checkpoint", case=label, status=status,
                                       t=round(time.monotonic() - t_start, 1)))
        return all(good for _, good, _ in results)

    def one(rng):
        roll = rng.random()
        name, _, body = rng.choice(cases)
        label = name
        # the rest, rescaled to the 0.9 it was drawn from at the default share (unstreamed 0.4, dropped 0.05,
        # streamed 0.45 of the whole)
        rest = 0.1 + (roll - text_share) / (1 - text_share) * 0.9
        try:
            if roll < text_share:
                i = rng.randrange(len(TEXTS))
                status, reply = retry(lambda: generate(server, TEXTS[i]))
                kind, good = "text", status == 200 and text_key(reply) == text_ref[i]
                label = f"text-{i}"
            elif rest < 0.5:
                status, reply = retry(lambda: server.post(body))
                kind, good, label = "unstreamed", status == 200 and strip(reply) == alone[name], name
            elif rest < 0.55:
                status, events, _, _ = retry(lambda: server.stream(body, close_after_audio=True))
                kind, good, label = "dropped", status == 200, name
            else:
                status, events, _, _ = retry(lambda: server.stream(body))
                kind, good, label = "streamed", same_stream(name, status, events), name
            with lock:
                counts[kind] += 1
                statuses[str(status)] = statuses.get(str(status), 0) + 1
                if not good:
                    differ.append(dict(kind=kind, case=label, status=status, t=round(time.monotonic() - t_start, 1)))
        except Exception as e:  # noqa: BLE001  (a transport error is a failure, recorded)
            with lock:
                errors.append(dict(case=label, error=repr(e)[:300], t=round(time.monotonic() - t_start, 1)))

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    with concurrent.futures.ThreadPoolExecutor(clients) as pool:
        futures = [pool.submit(client, 1000 + i) for i in range(clients)]
        next_cp = CHECKPOINT_FIRST
        while time.monotonic() - t_start < seconds and server.proc.poll() is None:
            time.sleep(5)
            if time.monotonic() - t_start >= next_cp:
                checkpoint()
                next_cp += CHECKPOINT_EVERY
            done = sum(counts[k] for k in ("unstreamed", "streamed", "dropped", "text"))
            if int(time.monotonic() - t_start) % 600 < 5:
                print(f"t={int(time.monotonic() - t_start)} s: {done} requests, {len(differ)} differ, {len(errors)} errors",
                      flush=True)
        if server.proc.poll() is None:
            checkpoint()  # the end, still under the soak's clients (paused)
        stop.set()
        for f in futures:
            f.result()
    sampler_thread.join()
    report.update(counts=counts, statuses=statuses, differ=differ[:200], differ_count=len(differ), errors=errors[:50],
                  error_count=len(errors), soak_seconds=round(time.monotonic() - t_start, 1))
    bad_status = {k: v for k, v in statuses.items() if k != "200"}

    memory, endpoint, _ = memory_report(samples)
    idle = idle_report(checkpoints)
    report["memory_idle"], report["memory_load"], report["memory_endpoint"] = idle, memory, endpoint
    report["checkpoints"] = checkpoints
    alive = all(s["server_alive"] for s in samples)
    report["alive_throughout"] = alive

    # 4. the crowd after the soak (CROWD_ROUNDS, default 1; every round is gated)
    rounds = []
    for _ in range(int(os.environ.get("CROWD_ROUNDS", "1"))):
        rounds.append(crowd_round(server, cases, alone, same_stream, out.with_suffix(".crowd-differ.jsonl")))
    crowd = {kind: dict(identical=min(r[kind]["identical"] for r in rounds),
                        differ=sorted({n for r in rounds for n in r[kind]["differ"]}),
                        detail=[d for r in rounds for d in r[kind]["detail"]]) for kind in rounds[0]}
    report["crowd_after"], report["crowd_rounds"] = crowd, len(rounds)
    code, stderr = server.stop()
    report.update(server_exit=code, server_stderr=stderr[-2000:])
    report["pass_"] = pass_ = bool(
        ok and not differ and not errors and not bad_status and alive
        and all(m["pass_"] for m in idle.values()) and all(not c["differ"] for c in crowd.values()) and code == 0)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(pass_=pass_, counts=counts, statuses=statuses, differ=len(differ), errors=len(errors),
                          memory_idle=idle, client_load=memory["client_rss"], crowd={k: v["identical"] for k, v in crowd.items()}, exit=code)))
    print("AUDIO SOAK PASS" if pass_ else "AUDIO SOAK FAIL")
    return 0 if pass_ else 1


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7, 8, 9):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
