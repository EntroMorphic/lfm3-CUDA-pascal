"""red-team-2 B1: /generate_batch is one engine call, bounded by the engine timeout (300 s), so a lane's max_new_tokens
is bounded (lfm3_http.hpp kBatchMaxNewTokens, 4,096). This measures the worst case through liquid-lfm3-server with its
defaults: two sessions filling the total capacity (TOTAL // 16 and the rest), the larger one's prompt filled so that
4,096 new tokens just fit; first the batch with 1 new token a lane (both prefills), then with 4,096 (greedy). The batch
may stop before 4,096 at an end-of-turn token; the time per paired step is then taken from the steps it ran, and the
worst case is the prefill plus 4,095 paired steps at that rate. Also checks the bound: 4,097 is refused (400), and the
server is still serving afterwards (/health ready). Passes if the worst case is at most 80% of the engine timeout (the
margin the reference set for /speech's worst case, lfm3_http.hpp), and the refusal is as stated.
    python3 tools/pascal/batch_budget.py SERVER CHECKPOINT OUTPUT [TOTAL=30720]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402

server_path, checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
total = int(sys.argv[4]) if len(sys.argv) > 4 else 30720
LIMIT, ENGINE_TIMEOUT, STEPS = 4096, 300.0, 4095
out.mkdir(parents=True, exist_ok=False)
PARAGRAPH = ("The river leaves the mountains in early spring, carrying silt and snowmelt past the old mill, the orchards "
             "and the long fields of the valley floor, until it widens into the slow brown estuary where the herons wait. ")
records, report = [], dict(total_capacity=total, bound=LIMIT, engine_timeout=ENGINE_TIMEOUT)
server = NativeServer(server_path, checkpoint, out, "server", None)
client = Client(server.port, server.token, records)
ok = False
try:
    def text(k): return [dict(role="user", content=PARAGRAPH * k)]
    sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]
    n = {}
    for k in (1, 11):
        client.call("POST", f"/sessions/{sid}/reset", {})
        n[k] = client.call("POST", "/generate", dict(session_id=sid, messages=text(k), max_new_tokens=1))["consumed_tokens"]
    client.call("POST", f"/sessions/{sid}/close", {})
    per = (n[11] - n[1]) / 10
    small, large = total // 16, total - total // 16
    sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in (small, large)]
    k_large = int((large - LIMIT - 64 - (n[1] - per)) // per)  # the prompt and 4,096 new tokens within the session
    bodies = [dict(session_id=sids[0], messages=text(1), max_new_tokens=1),
              dict(session_id=sids[1], messages=text(k_large), max_new_tokens=1)]
    t0 = time.monotonic(); r1 = client.call("POST", "/generate_batch", dict(requests=bodies)); prefill = time.monotonic() - t0
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    bodies = [dict(b, max_new_tokens=LIMIT) for b in bodies]
    t0 = time.monotonic(); r2 = client.call("POST", "/generate_batch", dict(requests=bodies)); wall = time.monotonic() - t0
    paired = r2["paired_decode_calls"]
    prompts = [x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in r2["results"]]
    per_step = (wall - prefill) / max(paired, 1)
    worst = prefill + STEPS * per_step
    refused = client.call("POST", "/generate_batch", dict(requests=[bodies[0], dict(bodies[1], max_new_tokens=LIMIT + 1)]),
                          expected=(400,))[1]
    health = client.call("GET", "/health", auth=False)
    report.update(sessions=[small, large], prompts=prompts, prefill_seconds=round(prefill, 2), batch_seconds=round(wall, 2),
                  paired_decode_calls=paired, generated=[len(x["generated_ids"]) for x in r2["results"]],
                  ms_per_paired_step=round(per_step * 1000, 2), worst_case_seconds=round(worst, 1),
                  refusal_4097=refused, health_ready=health.get("ready"))
    ok = worst <= 0.8 * ENGINE_TIMEOUT and refused == {"error": "invalid_request"} and health.get("ready") is True and paired > 100
    print(f"batch budget: sessions {small} + {large}, prompts {prompts}; prefill {prefill:.1f} s; {paired} paired steps in "
          f"{wall - prefill:.1f} s ({per_step * 1000:.2f} ms each); worst case {prefill:.1f} + {STEPS} steps = {worst:.1f} s "
          f"(limit {0.8 * ENGINE_TIMEOUT:.0f} s); 4,097 refused {refused}; health ready {health.get('ready')}", flush=True)
finally:
    report["server_exit"] = server.stop()
    report["pass"] = bool(ok and report["server_exit"] == 0)
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
print("BATCH BUDGET", "PASS" if report["pass"] else "FAIL")
sys.exit(0 if report["pass"] else 1)
