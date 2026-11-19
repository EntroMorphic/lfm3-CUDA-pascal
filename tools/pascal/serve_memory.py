"""serve-v1 rules 11 and 12, through liquid-lfm3-server over HTTP (started as the reference's native-server check starts it).

Rule 11: each admissible configuration at the server's limits, every session filled to its capacity (a prompt of
capacity - 33 tokens, 32 new tokens): one of 32,768; 4,096 + 28,672; 2,048 + 30,720; 16,384 + 16,384; and
/generate_batch on 4,096 + 28,672. Whole-card memory sampled every 0.2 s (nvidia-smi); every request must succeed
and the peak of each configuration must be at most LIMIT MiB.
Rule 12: past the limits, the reference's refusal, and the control still gives its ids: a third session, a capacity
over the total, a prompt plus max_new_tokens over the session's capacity, a full queue (--queue-size 1).
Prompts are a fixed paragraph repeated, sized by the server's own count (consumed_tokens) from two calibration runs.
    python3 tools/pascal/serve_memory.py SERVER CHECKPOINT CONTROL_OUTPUT OUTPUT [LIMIT=7740]
"""
import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time

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
        self.peak, self.stop, self.rows = 0, threading.Event(), []
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()
    def run(self):
        while not self.stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if r.returncode == 0: v = int(r.stdout.split()[0]); self.rows.append(v); self.peak = max(self.peak, v)
            time.sleep(0.2)
    def done(self):
        self.stop.set(); self.thread.join(); return self.peak

def text(repeats): return [dict(role="user", content=PARAGRAPH * repeats)]
server = NativeServer(server_path, checkpoint, out, "server", None, "--queue-size", "1")
client = Client(server.port, server.token, records)
failed = False
try:
    idle = Sampler(); time.sleep(1); report["before_mib"] = idle.done()
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
            entry["peak_mib"] = sampler.done()
        ok = all(s == 200 for s in entry["statuses"]) and entry["peak_mib"] <= limit and \
             all(p >= c - 33 - 2 * b for p, c in zip(entry["prompts"], capacities)) and len(entry["prompts"]) == len(capacities)
        if batch: ok = ok and entry.get("gpu_batching") is True
        entry["ok"] = ok; failed |= not ok; report["configurations"].append(entry)
        print(f"rule 11 {name}: prompts {entry['prompts']} of {capacities}, statuses {entry['statuses']}, peak {entry['peak_mib']} MiB: {'ok' if ok else 'FAIL'}", flush=True)
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
        entry["ok"] = status in expect and entry["control_after"]; failed |= not entry["ok"]; report["refusals"].append(entry)
        print(f"rule 12 {name}: {status} {entry['error']}, control after {'same ids' if entry['control_after'] else 'DIFFERENT'}: {'ok' if entry['ok'] else 'FAIL'}", flush=True)
    a_sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]
    if not control(a_sid): failed = True; print("control before the refusals: DIFFERENT ids: FAIL")
    b_sid = client.call("POST", "/sessions", dict(capacity=28672))["session_id"]
    refusal("third session", (429,), lambda: client.call("POST", "/sessions", dict(capacity=16), expected=(200, 429)))
    client.call("POST", f"/sessions/{b_sid}/close", {})
    refusal("capacity over the total", (429,), lambda: client.call("POST", "/sessions", dict(capacity=28673), expected=(200, 400, 429)))
    b_sid = client.call("POST", "/sessions", dict(capacity=1024))["session_id"]
    refusal("prompt plus max_new_tokens over capacity", (400,), lambda: client.call(
        "POST", "/generate", dict(session_id=b_sid, messages=text(fill(1024)), max_new_tokens=100), expected=(200, 400)))
    # A full queue: with --queue-size 1, while A generates and B waits, try every further text request there is.
    def full_queue():
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(lambda: client.call("POST", "/generate", dict(session_id=a_sid, messages=text(fill(4096)), max_new_tokens=512), expected=(200, 429)))
            time.sleep(1.0)
            second = pool.submit(lambda: client.call("POST", "/generate", dict(session_id=b_sid, messages=CONTROL, max_new_tokens=128), expected=(200, 429)))
            time.sleep(0.5)
            third = client.call("POST", "/generate_batch", dict(requests=[dict(session_id=a_sid, messages=CONTROL, max_new_tokens=8),
                                                                        dict(session_id=b_sid, messages=CONTROL, max_new_tokens=8)]), expected=(200, 409, 429))
            results = [first.result(), second.result()]
        report["full_queue_attempt"] = dict(running=results[0][0], waiting=results[1][0], third=third[0], third_error=third[1].get("error"))
        return third
    refusal("full queue (or the session lease first)", (409, 429), full_queue)
    for s in (a_sid, b_sid): client.call("POST", f"/sessions/{s}/close", {})
    report["status"] = "failed" if failed else "passed"
finally:
    report["server_exit"] = server.stop()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1)[:2_000_000] + "\n")
print("SERVE MEMORY (rules 11 and 12)", "PASS" if report["status"] == "passed" and report.get("server_exit") == 0 else "FAIL")
sys.exit(0 if report["status"] == "passed" and report.get("server_exit") == 0 else 1)
