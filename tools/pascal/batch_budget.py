"""red-team-2 B1: /generate_batch is one engine call, bounded by the engine timeout (300 s), so a lane's max_new_tokens
is bounded (lfm3_http.hpp kBatchMaxNewTokens, 2,048). This measures the worst case through liquid-lfm3-server with its
defaults: two sessions filling the total capacity (one just large enough for a short prompt and 2,048 new tokens, the
other the rest), each prompt filled so that 2,048 new tokens just fit, so the batch ends with the capacity full; the
prompts are lists of numbers the model is asked to continue without stopping. Twice: greedy, and sampled by the host
sampler's slowest path. The worst case is each call's own wall time, measured, not extrapolated: both lanes must
generate all 2,048 tokens (2,047 paired steps), or the run is inconclusive (exit 2) unless it has already failed. Before it, for the record, the same batch with 1 new token a lane
(the two prefills). Also checks the bound: on short prompts, where only the bound can refuse it, 2,049 is refused (400),
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
LIMIT, ENGINE_TIMEOUT, STEPS = 2048, 300.0, 2047
out.mkdir(parents=True, exist_ok=False)
records, report = [], dict(total_capacity=total, bound=LIMIT, engine_timeout=ENGINE_TIMEOUT)
server = NativeServer(server_path, checkpoint, out, "server", None)
client = Client(server.port, server.token, records)
ok = False
try:
    # The prompts are a list of k numbers from 1,000, one a line, and an instruction to go on counting, so that greedy
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
    base = n[100] - 100 * per  # the template's tokens
    small = int(LIMIT + 64 + base + 20 * per) + 1  # a short prompt (20 lines) and 2,048 new tokens; the rest to the other
    large = total - small
    sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in (small, large)]
    fill = [int((c - LIMIT - 64 - base) // per) for c in (small, large)]  # prompt + 2,048 just fit
    bodies = [dict(session_id=sid_, messages=text(k), max_new_tokens=1) for sid_, k in zip(sids, fill)]
    t0 = time.monotonic(); client.call("POST", "/generate_batch", dict(requests=bodies)); prefill = time.monotonic() - t0
    # The host's token choice is part of the call: greedy, and the sampler's slowest path (do_sample with top_k 0 sorts
    # the whole vocabulary on the host at every step of every lane; top_p and a repetition penalty on top; red-team-2 R4-1).
    # near-greedy, so that the count goes on as greedily (temperature 0.5 ended a lane after 92 tokens, development 5); the
    # sort over the whole vocabulary costs the same at any temperature
    SAMPLED = dict(do_sample=True, temperature=0.05, top_k=0, top_p=0.98, repetition_penalty=1.01, seed=7)
    cases = {}
    for name, sampling in (("greedy", None), ("sampled", SAMPLED)):
        for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
        req = [dict(b, max_new_tokens=LIMIT, **({"sampling": sampling} if sampling else {})) for b in bodies]
        t0 = time.monotonic(); r = client.call("POST", "/generate_batch", dict(requests=req)); wall = time.monotonic() - t0
        generated = [len(x["generated_ids"]) for x in r["results"]]
        cases[name] = dict(seconds=round(wall, 1), generated=generated, paired=r["paired_decode_calls"],
                           prompts=[x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in r["results"]],
                           complete=generated == [LIMIT, LIMIT] and r["paired_decode_calls"] == STEPS)
        print(f"batch budget {name}: generated {generated} ({r['paired_decode_calls']} paired steps), {wall:.1f} s "
              f"(limit {0.8 * ENGINE_TIMEOUT:.0f} s)", flush=True)
    # The bound alone refuses: short prompts on the reset sessions, where 2,049 new tokens fit the small one's capacity.
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    short = [dict(session_id=sids[0], messages=text(10), max_new_tokens=LIMIT + 1),
             dict(session_id=sids[1], messages=text(10), max_new_tokens=8)]
    assert all(base + 10 * per + 64 + LIMIT + 1 <= c for c in (small, large))
    refused = client.call("POST", "/generate_batch", dict(requests=short), expected=(400,))[1]
    health = client.call("GET", "/health", auth=False)
    complete = all(c["complete"] for c in cases.values())
    definite_fail = any(c["seconds"] > 0.8 * ENGINE_TIMEOUT for c in cases.values()) or \
        refused != {"error": "invalid_request"} or health.get("ready") is not True
    report.update(sessions=[small, large], prefill_seconds=round(prefill, 2), cases=cases, complete=complete,
                  refusal_4097=refused, health_ready=health.get("ready"), definite_fail=definite_fail)
    ok = complete and not definite_fail
    print(f"batch budget: sessions {small} + {large}; the two prefills {prefill:.1f} s; 2,049 refused {refused}; "
          f"health ready {health.get('ready')}", flush=True)
finally:
    report["server_exit"] = server.stop()
    report["pass"] = bool(ok and report["server_exit"] == 0)
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
# A lane that stopped early leaves no verdict, unless a definite failure (over the limit even so, the refusal, health)
verdict = "PASS" if report["pass"] else "FAIL" if report.get("definite_fail", True) or report["server_exit"] != 0 \
    else "INCONCLUSIVE (a lane stopped before 2,048)"
print("BATCH BUDGET", verdict)
sys.exit(0 if report["pass"] else 2 if verdict.startswith("INCONCLUSIVE") else 1)
