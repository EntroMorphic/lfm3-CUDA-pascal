#!/usr/bin/env python3
"""Where does liquid-lfm3-server's memory grow? (audio-assurance-v1, after the soak's rule-4 memory failure.)

Diagnosis, not a gate. One fresh server, then phases, each one kind of traffic; after each phase (and every 30 s) the
server's /proc/PID/smaps is summed by mapping (anonymous, [heap], each device or file), with VmRSS and the thread count.
  idle      60 s of nothing (the baseline's noise)
  churn     N create/close session pairs (no GPU)
  bad       N requests refused 400 (connection and parse path only)
  storm     long speech on 8 threads while 4 threads hammer text /generate with 429 retries (the soak's pattern)
  unstreamed / streamed / dropped   M speech requests of that kind on 8 threads (cases drawn as the soak draws them)
  idle2     60 s of nothing
Writes OUT_JSON (phases, each with the per-mapping totals) and prints the per-phase growth.
Usage: diag_lfm3_server_memory.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [N [M [STORM_SECONDS [PHASES]]]]
"""
import concurrent.futures
import importlib.util
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check", HERE / "check_lfm3_speech_stream.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
spec = importlib.util.spec_from_file_location("batch", HERE / "check_lfm3_speech_batch_server.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
MIB = 1 << 20
TEXT = dict(max_new_tokens=96, messages=[{"role": "user", "content": "List three ways to keep bread fresh longer."}])


def smaps(pid):
    """{mapping name: Rss bytes} plus totals."""
    out, name = defaultdict(int), None
    anon_maps = 0
    for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
        parts = line.split()
        if parts and "-" in parts[0] and len(parts) >= 5 and not parts[0].endswith(":"):
            name = parts[5] if len(parts) >= 6 else "[anon]"
            if name == "[anon]":
                anon_maps += 1
            if name.startswith("/") and not name.startswith("/dev"):
                name = "file:" + Path(name).name
        elif parts and parts[0] == "Rss:":
            out[name] += int(parts[1]) * 1024
    status = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in Path(f"/proc/{pid}/status").read_text().splitlines()}
    return dict(by_mapping=dict(out), anon_maps=anon_maps, vmrss=int(status["VmRSS"].split()[0]) * 1024,
                threads=int(status["Threads"]), fds=len(os.listdir(f"/proc/{pid}/fd")))


def call(server, method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data, method=method,
                                 headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main(binary, model, build, libri, out_json, n=20000, m=200, storm_seconds=900,
         phases="idle churn bad storm unstreamed streamed dropped idle2"):
    n, m, storm_seconds = int(n), int(m), int(storm_seconds)
    out = Path(out_json)
    cases = check.cases(build, libri)
    server = batch.start(binary, model, out.parent, "--handlers", "96", "--queue-size", "64")
    pid = server.proc.pid
    server.post(cases[0][2])  # warm
    t0 = time.monotonic()
    log, stop_sampler = [], threading.Event()
    phase_now = ["warm"]

    def sample(tag):
        s = smaps(pid)
        s.update(t=round(time.monotonic() - t0, 1), phase=phase_now[0], tag=tag)
        log.append(s)
        return s

    def sampler():
        while not stop_sampler.wait(30):
            sample("tick")

    threading.Thread(target=sampler, daemon=True).start()
    results = []
    before = sample("start")
    counts = {}
    rng = random.Random(7)
    long_cases = sorted(cases, key=lambda c: -len(json.dumps(c[2])))[:8]

    def pool_run(k, fn, threads=8):
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            return list(pool.map(fn, range(k)))

    for phase in phases.split():
        phase_now[0] = phase
        t = time.monotonic()
        if phase.startswith("idle"):
            time.sleep(60)
            got = []
        elif phase == "churn":
            def one(_):
                st, b = call(server, "POST", "/sessions", {"capacity": 512})
                if st != 200:
                    return st
                return call(server, "POST", f"/sessions/{json.loads(b)['session_id']}/close", {})[0]
            got = pool_run(n, one)
        elif phase == "bad":
            got = pool_run(n, lambda _: call(server, "POST", "/speech", raw=b'{"messages": 7}')[0])
        elif phase == "storm":
            stop = threading.Event()
            tally = defaultdict(int)

            def speaker(i):
                while not stop.is_set():
                    st, _ = batch.retried(lambda: server.post(long_cases[i % len(long_cases)][2]))
                    tally[f"speech:{st}"] += 1

            def texter(_):
                while not stop.is_set():
                    st, b = call(server, "POST", "/sessions", {"capacity": 1024})
                    if st != 200:
                        tally[f"session:{st}"] += 1
                        continue
                    sid = json.loads(b)["session_id"]
                    st, _ = call(server, "POST", "/generate", dict(TEXT, session_id=sid))
                    tally[f"generate:{st}"] += 1
                    call(server, "POST", f"/sessions/{sid}/close", {})
                    time.sleep(0.02)
            ths = [threading.Thread(target=speaker, args=(i,)) for i in range(8)] + \
                  [threading.Thread(target=texter, args=(i,)) for i in range(4)]
            for th in ths:
                th.start()
            time.sleep(storm_seconds)
            stop.set()
            for th in ths:
                th.join()
            got = dict(tally)
        else:
            picks = [rng.choice(cases)[2] for _ in range(m)]
            if phase == "unstreamed":
                got = pool_run(m, lambda i: batch.retried(lambda: server.post(picks[i]))[0])
            elif phase == "streamed":
                got = pool_run(m, lambda i: batch.retried(lambda: server.stream(picks[i]))[0])
            else:
                got = pool_run(m, lambda i: batch.retried(lambda: server.stream(picks[i], close_after_audio=True))[0])
            time.sleep(5)
        after = sample("end")
        if isinstance(got, list):
            tally = defaultdict(int)
            for st in got:
                tally[str(st)] += 1
            got = dict(tally)
        keys = set(before["by_mapping"]) | set(after["by_mapping"])
        delta = {k: round((after["by_mapping"].get(k, 0) - before["by_mapping"].get(k, 0)) / MIB, 2) for k in keys}
        delta = {k: v for k, v in sorted(delta.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 0.5}
        r = dict(phase=phase, seconds=round(time.monotonic() - t, 1), statuses=got,
                 vmrss_mib=round(after["vmrss"] / MIB, 1), vmrss_delta_mib=round((after["vmrss"] - before["vmrss"]) / MIB, 2),
                 anon_maps=after["anon_maps"], threads=after["threads"], fds=after["fds"], delta_by_mapping_mib=delta)
        results.append(r)
        print(json.dumps(r), flush=True)
        before = after
        out.write_text(json.dumps(dict(results=results, samples=log), indent=1))
    stop_sampler.set()
    code, _ = server.stop()
    out.write_text(json.dumps(dict(results=results, samples=log, exit=code), indent=1))
    print("exit", code)


if __name__ == "__main__":
    if not 6 <= len(sys.argv) <= 10:
        sys.exit(__doc__)
    main(*sys.argv[1:])
