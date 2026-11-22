"""red-team-2 B1: /generate_batch is one engine call, bounded by the engine timeout (300 s), so a lane's max_new_tokens
is bounded (lfm3_http.hpp kBatchMaxNewTokens, 4,096). This measures the worst case through liquid-lfm3-server with its
defaults: two sessions filling the total capacity (6,144 and the rest), each prompt filled so that 4,096 new tokens just
fit, so the batch ends with the capacity full; greedy, the prompts lists of numbers the model is asked to continue without stopping. The
worst case is the call's own wall time, measured, not extrapolated: both lanes must generate all 4,096 tokens (4,095
paired steps), or the run is inconclusive (exit 2). Before it, for the record, the same batch with 1 new token a lane
(the two prefills). Also checks the bound: on short prompts, where only the bound can refuse it, 4,097 is refused (400),
and the server is still serving afterwards (/health ready). Passes if the call takes at most 80% of the engine timeout
(the margin the reference set for /speech's worst case, lfm3_http.hpp) and the refusal is as stated. The parity build
is not measured here: it is the validation build, whose single prefill near capacity alone takes about 430 s
(serve-v1), past the default engine timeout for /generate as for the batch.
    python3 tools/pascal/batch_budget.py SERVER CHECKPOINT OUTPUT [TOTAL=29696]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402

server_path, checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
total = int(sys.argv[4]) if len(sys.argv) > 4 else 29696
LIMIT, ENGINE_TIMEOUT, STEPS = 4096, 300.0, 4095
out.mkdir(parents=True, exist_ok=False)
records, report = [], dict(total_capacity=total, bound=LIMIT, engine_timeout=ENGINE_TIMEOUT)
server = NativeServer(server_path, checkpoint, out, "server", None)
client = Client(server.port, server.token, records)
ok = False
try:
    # The prompts are a list of numbers, 1 to k, one a line, and an instruction to go on counting, so that greedy
    # decoding goes on (a long prose prompt with the same instruction stopped after 200 tokens, development 2).
    def text(k, count=True):  # k four-digit numbers from 1,000, so every line costs the same (8,999 at most)
        assert k <= 8999
        body = "\n".join(str(i) for i in range(1000, 1000 + k))
        return [dict(role="user", content=body + ("\nContinue this list: count upward from where it ends, one number per "
                                                  "line, and do not stop." if count else ""))]
    sid = client.call("POST", "/sessions", dict(capacity=8192))["session_id"]
    n = {}
    for k in (100, 1100):
        client.call("POST", f"/sessions/{sid}/reset", {})
        n[k] = client.call("POST", "/generate", dict(session_id=sid, messages=text(k), max_new_tokens=1))["consumed_tokens"]
    client.call("POST", f"/sessions/{sid}/close", {})
    per = (n[1100] - n[100]) / 1000  # tokens per line, every line alike
    small, large = 6144, total - 6144
    sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in (small, large)]
    fill = [int((c - LIMIT - 64 - (n[100] - 100 * per)) // per) for c in (small, large)]  # prompt + 4,096 just fit
    bodies = [dict(session_id=sid_, messages=text(k, True), max_new_tokens=1) for sid_, k in zip(sids, fill)]
    t0 = time.monotonic(); r1 = client.call("POST", "/generate_batch", dict(requests=bodies)); prefill = time.monotonic() - t0
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    bodies = [dict(b, max_new_tokens=LIMIT) for b in bodies]
    t0 = time.monotonic(); r2 = client.call("POST", "/generate_batch", dict(requests=bodies)); wall = time.monotonic() - t0
    paired = r2["paired_decode_calls"]
    generated = [len(x["generated_ids"]) for x in r2["results"]]
    prompts = [x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in r2["results"]]
    complete = generated == [LIMIT, LIMIT] and paired == STEPS  # the worst case itself, or no verdict
    # The bound alone refuses: short prompts on the reset sessions, where 4,097 new tokens fit the small one's capacity.
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    short = [dict(session_id=sids[0], messages=text(10), max_new_tokens=LIMIT + 1),
             dict(session_id=sids[1], messages=text(10), max_new_tokens=8)]
    assert all(n[100] + LIMIT + 1 <= c for c in (small, large))
    refused = client.call("POST", "/generate_batch", dict(requests=short), expected=(400,))[1]
    health = client.call("GET", "/health", auth=False)
    report.update(sessions=[small, large], prompts=prompts, prefill_seconds=round(prefill, 2), batch_seconds=round(wall, 2),
                  paired_decode_calls=paired, generated=generated, complete=complete,
                  refusal_4097=refused, health_ready=health.get("ready"))
    ok = complete and wall <= 0.8 * ENGINE_TIMEOUT and refused == {"error": "invalid_request"} and health.get("ready") is True
    print(f"batch budget: sessions {small} + {large}, prompts {prompts}; the two prefills {prefill:.1f} s; the worst case, "
          f"generated {generated} ({paired} paired steps), {wall:.1f} s (limit {0.8 * ENGINE_TIMEOUT:.0f} s); 4,097 refused "
          f"{refused}; health ready {health.get('ready')}", flush=True)
finally:
    report["server_exit"] = server.stop()
    report["pass"] = bool(ok and report["server_exit"] == 0)
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
verdict = "PASS" if report["pass"] else "INCONCLUSIVE (a lane stopped before 4,096)" if not report.get("complete", True) else "FAIL"
print("BATCH BUDGET", verdict)
sys.exit(0 if report["pass"] else 2 if verdict.startswith("INCONCLUSIVE") else 1)
