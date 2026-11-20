"""serve-v1 rules 11 and 12, through liquid-lfm3-server over HTTP (started as the reference's native-server check starts it).

Rule 11: each admissible configuration at the server's limits, every session filled to its capacity (a prompt of
capacity - 33 tokens, 32 new tokens): one of 32,768; 4,096 + 28,672; 2,048 + 30,720; 16,384 + 16,384; and
/generate_batch on 4,096 + 28,672. Whole-card memory sampled every 0.2 s (nvidia-smi); every request must succeed
and the peak of each configuration must be at most LIMIT MiB.
Rule 12: past the limits, the reference's refusal (status and error), and the control still gives its ids: a third
session (two of 4,096 open, so only the session count refuses it), a capacity over the total, a prompt plus
max_new_tokens over the session's capacity, and a full queue (--queue-size 1: resets of the two idle sessions sent at
the same instant, until one finds the other's job still waiting and gets 429 queue_full; at most ATTEMPTS pairs).
Each configuration's memory must have at least 5 samples and a peak above the idle card's, or it fails.
Prompts are a fixed paragraph repeated, sized by the server's own count (consumed_tokens) from two calibration runs.
    python3 tools/pascal/serve_memory.py SERVER CHECKPOINT CONTROL_OUTPUT OUTPUT [LIMIT=7740] [ATTEMPTS=300]
"""
import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import socket

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402

server_path, checkpoint, control_path, out = map(Path, sys.argv[1:5])
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 7740
out.mkdir(parents=True, exist_ok=False)
reference = json.loads(control_path.read_text())["generated_ids"]
CONTROL = [dict(role="system", content="Respond with text."),
           dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
PARAGRAPH = ("The river leaves the mountains in early spring, carrying silt and snowmelt past the old mill, the orchards "
             "and the long fields of the valley floor, until it widens into the slow brown estuary where the herons wait. ")
records, report = [], dict(limit_mib=limit, configurations=[], refusals=[], status="failed")

class Sampler:
    def __init__(self):
        self.peak, self.stop, self.rows, self.errors = 0, threading.Event(), [], 0
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()
    def run(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                v = int(r.stdout.split()[0]) if r.returncode == 0 else None
            except (OSError, ValueError, IndexError): v = None
            if v is None: self.errors += 1
            else: self.rows.append(v); self.peak = max(self.peak, v)
            time.sleep(0.2)
    def done(self):
        self.stop.set(); self.thread.join(); return self.peak

def text(repeats): return [dict(role="user", content=PARAGRAPH * repeats)]
server = NativeServer(server_path, checkpoint, out, "server", None, "--queue-size", "1")
client = Client(server.port, server.token, records)
failed = False
try:
    idle = Sampler(); time.sleep(1); report["before_mib"] = idle.done()
    if len(idle.rows) < 3: raise AssertionError(f"memory sampler took {len(idle.rows)} samples of the idle card ({idle.errors} errors)")
    # Calibration: prompt tokens = a + b * repeats.
    sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]
    n = {}
    for r in (1, 11):
        client.call("POST", f"/sessions/{sid}/reset", {})
        n[r] = client.call("POST", "/generate", dict(session_id=sid, messages=text(r), max_new_tokens=1))["consumed_tokens"]
    b = (n[11] - n[1]) / 10; a = n[1] - b
    client.call("POST", f"/sessions/{sid}/close", {})
    report["calibration"] = dict(tokens_at_1=n[1], tokens_at_11=n[11])
    def fill(capacity):  # repeats whose prompt is at most capacity - 33 tokens
        return max(1, int((capacity - 33 - a) // b))
    def run_configuration(name, capacities, batch=False):
        global failed
        sampler = Sampler(); entry = dict(name=name, capacities=capacities, prompts=[], statuses=[])
        try:
            sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in capacities]
            bodies = [dict(session_id=s, messages=text(fill(c)), max_new_tokens=32) for s, c in zip(sids, capacities)]
            if batch:
                status, reply = client.call("POST", "/generate_batch", dict(requests=bodies), expected=(200, 400, 429, 500, 503))
                entry["statuses"].append(status)
                if status == 200:
                    entry["prompts"] = [x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in reply["results"]]
                    entry["gpu_batching"] = reply.get("gpu_batching")
                else: entry["error"] = reply
            else:
                for body in bodies:
                    status, reply = client.call("POST", "/generate", body, expected=(200, 400, 429, 500, 503))
                    entry["statuses"].append(status)
                    if status == 200: entry["prompts"].append(reply["consumed_tokens"] - len(reply["generated_ids"]) + 1)
                    else: entry["error"] = reply
            for s in sids: client.call("POST", f"/sessions/{s}/close", {})
        finally:
            entry["peak_mib"] = sampler.done(); entry["samples"] = len(sampler.rows); entry["sampler_errors"] = sampler.errors
        ok = all(s == 200 for s in entry["statuses"]) and entry["peak_mib"] <= limit and \
             entry["samples"] >= 5 and entry["peak_mib"] > report["before_mib"] and \
             all(p >= c - 33 - 2 * b for p, c in zip(entry["prompts"], capacities)) and len(entry["prompts"]) == len(capacities)
        if batch: ok = ok and entry.get("gpu_batching") is True
        entry["ok"] = ok; failed |= not ok; report["configurations"].append(entry)
        print(f"rule 11 {name}: prompts {entry['prompts']} of {capacities}, statuses {entry['statuses']}, peak {entry['peak_mib']} MiB ({entry['samples']} samples): {'ok' if ok else 'FAIL'}", flush=True)
    run_configuration("one-32768", [32768])
    run_configuration("4096+28672", [4096, 28672])
    run_configuration("2048+30720", [2048, 30720])
    run_configuration("16384+16384", [16384, 16384])
    run_configuration("batch-4096+28672", [4096, 28672], batch=True)

    def control(sid):
        client.call("POST", f"/sessions/{sid}/reset", {})
        return client.call("POST", "/generate", dict(session_id=sid, messages=CONTROL, max_new_tokens=128))["generated_ids"] == reference
    def refusal(name, expect, call):
        global failed
        status, reply = call()
        entry = dict(name=name, status=status, error=reply.get("error") if isinstance(reply, dict) else None)
        entry["control_after"] = control(a_sid)
        entry["ok"] = (status, entry["error"]) == expect and entry["control_after"]; failed |= not entry["ok"]; report["refusals"].append(entry)
        print(f"rule 12 {name}: {status} {entry['error']}, control after {'same ids' if entry['control_after'] else 'DIFFERENT'}: {'ok' if entry['ok'] else 'FAIL'}", flush=True)
    a_sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]
    if not control(a_sid): failed = True; print("control before the refusals: DIFFERENT ids: FAIL")
    b_sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]  # 8,192 of 32,768 held: only the count refuses
    refusal("third session", (429, "session_quota"), lambda: client.call("POST", "/sessions", dict(capacity=16), expected=(200, 429)))
    client.call("POST", f"/sessions/{b_sid}/close", {})
    refusal("capacity over the total", (429, "session_quota"), lambda: client.call("POST", "/sessions", dict(capacity=28673), expected=(200, 400, 429)))
    b_sid = client.call("POST", "/sessions", dict(capacity=1024))["session_id"]
    refusal("prompt plus max_new_tokens over capacity", (400, "invalid_request"), lambda: client.call(
        "POST", "/generate", dict(session_id=b_sid, messages=text(fill(1024)), max_new_tokens=100), expected=(200, 400)))
    # A full queue: both sessions idle, a reset of each sent at the same instant (every byte but the last sent first, then
    # the last bytes together), so that one is queued while the other still waits to be taken (--queue-size 1).
    def raw(sid):
        body = b"{}"
        return (f"POST /sessions/{sid}/reset HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                f"Authorization: Bearer {server.token}\r\nConnection: close\r\n\r\n").encode() + body
    def exchange(sock, last, out, i, barrier):
        barrier.wait(); sock.sendall(last); chunks = []
        while True:
            b = sock.recv(65536)
            if not b: break
            chunks.append(b)
        head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
        out[i] = (int(head.split()[1]), json.loads(body))
    def full_queue(attempts=int(sys.argv[6]) if len(sys.argv) > 6 else 300):
        seen = {}
        for n in range(1, attempts + 1):
            socks = [socket.create_connection(("127.0.0.1", server.port)) for _ in range(2)]
            datas = [raw(a_sid), raw(b_sid)]
            for so, d in zip(socks, datas): so.sendall(d[:-1])
            out, barrier = [None, None], threading.Barrier(2)
            ts = [threading.Thread(target=exchange, args=(socks[i], datas[i][-1:], out, i, barrier)) for i in range(2)]
            [t.start() for t in ts]; [t.join() for t in ts]; [so.close() for so in socks]
            for st, reply in out:
                key = f"{st} {reply.get('error', 'ok') if isinstance(reply, dict) else '?'}"; seen[key] = seen.get(key, 0) + 1
            hit = [x for x in out if x[0] == 429 and x[1].get("error") == "queue_full"]
            other = [x for x in out if x[0] != 200 and not (x[0] == 429 and x[1].get("error") == "queue_full")]
            if other: report["full_queue_attempts"] = dict(pairs=n, outcomes=seen); return other[0]
            if hit: report["full_queue_attempts"] = dict(pairs=n, outcomes=seen); return hit[0]
        report["full_queue_attempts"] = dict(pairs=attempts, outcomes=seen)
        return (200, {})
    refusal("full queue", (429, "queue_full"), full_queue)
    for s in (a_sid, b_sid): client.call("POST", f"/sessions/{s}/close", {})
    report["status"] = "failed" if failed else "passed"
finally:
    report["server_exit"] = server.stop()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1)[:2_000_000] + "\n")
print("SERVE MEMORY (rules 11 and 12)", "PASS" if report["status"] == "passed" and report.get("server_exit") == 0 else "FAIL")
sys.exit(0 if report["status"] == "passed" and report.get("server_exit") == 0 else 1)
