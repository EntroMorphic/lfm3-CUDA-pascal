#!/usr/bin/env python3
"""validation/prefill-stall-v1: catch the intermittent 3-4 s stall in the act. Runs lfm3-speech at full speed on a
request list with native/probes/ptracer_any.c preloaded (the process opts in to being traced: Thor's ptrace_scope 1);
a monitor samples every engine thread's minor faults from /proc every 50 ms, and when one thread faults faster than
RATE per second it records the thread and attaches gdb once (every thread's backtrace, then detach), at most CATCHES
times, 10 s apart. (A first version ran the engine under gdb from the start: 6x slower, and no stall in 145 requests.)
Usage (Thor host): stall_catch.py LFM3_SPEECH MODEL_DIR LIST OUT_DIR PRELOAD_SO [CATCHES] [RATE]
LD_LIBRARY_PATH as for nsys (the container's pinned CUDA libraries).
STALL_PREFIX (e.g. "docker exec -u 1000 -e LD_PRELOAD=... lfm3-profile-v1"): run the engine inside the container
instead (paths as the container sees them; gdb on the host attaches to its host PID, the same uid 1000 on both sides,
the preload opting it in to tracing). The engine's PID is then found by its command line."""
import os
import subprocess
import sys
import time
from pathlib import Path

binary, model, lst, out, preload = sys.argv[1:6]
catches = int(sys.argv[6]) if len(sys.argv) > 6 else 4
rate = float(sys.argv[7]) if len(sys.argv) > 7 else 1e6
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, LD_PRELOAD=preload)
prefix = os.environ.get("STALL_PREFIX", "").split()
proc = subprocess.Popen(prefix + [binary, model, "--batch", lst], env=env if not prefix else None,
                        stdout=open(out / "run.out", "w"), stderr=subprocess.STDOUT)
log = open(out / "monitor.txt", "w")
pid = proc.pid
if prefix:  # the docker client is proc; the engine is its own host process
    pid = None
    for _ in range(600):
        r = subprocess.run(["pgrep", "-f", "^" + binary + " .*--batch " + lst], capture_output=True, text=True)
        if r.stdout.split():
            pid = int(r.stdout.split()[0])
            break
        time.sleep(0.1)
    if pid is None:
        print("engine not found", file=log, flush=True)
        sys.exit(1)
print(f"engine {pid}", file=log, flush=True)


def threads():
    res = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            s = open(f"/proc/{pid}/task/{tid}/stat").read()
        except OSError:
            continue
        f = s.rsplit(")", 1)[1].split()
        res[int(tid)] = (s[s.index("(") + 1:s.rindex(")")], int(f[7]), int(f[12]))
    return res


caught, last, prev, prev_t = 0, 0.0, {}, time.time()
started = time.time()  # the model's load faults too: no catch in its first 90 s
while proc.poll() is None:
    try:
        now_threads, now = threads(), time.time()
    except FileNotFoundError:
        break
    dt = now - prev_t
    for tid, (name, minflt, st) in now_threads.items():
        if tid in prev and dt > 0:
            r = (minflt - prev[tid][1]) / dt
            if r > rate / 4:
                print(f"{now:.3f} tid {tid} {name!r} minflt/s {r:.0f} stime+{st - prev[tid][2]}", file=log, flush=True)
            if r > rate and caught < catches and now - last > 10 and now - started > 90:
                caught, last = caught + 1, now
                print(f"{now:.3f} CATCH {caught}: tid {tid} {name!r}", file=log, flush=True)
                g = subprocess.run(["gdb", "-q", "-batch", "-p", str(pid), "-ex", "set pagination off",
                                    "-ex", "thread apply all bt 25", "-ex", "detach"],
                                   capture_output=True, text=True, timeout=300)
                state = "".join(f"--- {p}\n" + open(p).read() for p in
                                ("/proc/meminfo", "/proc/buddyinfo", "/proc/vmstat", f"/proc/{pid}/status"))
                (out / f"catch-{caught}.txt").write_text(f"faulting tid {tid} {name}\n" + g.stdout + g.stderr + state)
                print(f"{time.time():.3f} CATCH {caught} done in {time.time() - now:.1f} s", file=log, flush=True)
                now_threads = threads()
    prev, prev_t = now_threads, time.time()
    time.sleep(0.05)
proc.wait()
print(f"done, exit {proc.returncode}, {caught} catches", file=log, flush=True)
print(f"stall_catch: {caught} catches; engine exit {proc.returncode}")
