"""red-team-2 B1: /generate_batch is one engine call, bounded by the engine timeout (300 s), so a lane's max_new_tokens
is bounded (lfm3_http.hpp kBatchMaxNewTokens, 2,048). This measures the worst case through liquid-lfm3-server with its
defaults: two sessions filling the total capacity (one just large enough for a short prompt, a continuation and 2,048
new tokens, the other the rest), each prompt filled so that 2,048 new tokens just fit, so the batch ends with the
capacity full; the prompts are lists of numbers the model is asked to continue without stopping. Three cases: greedy;
sampled (the host's sort and softmax over the whole vocabulary at every step of every lane, the sampler's costly
path); and sampled after each lane continues a cached prompt by nearly the most the worker continues (1,024 tokens,
forwarded a token at a time inside the call; a longer continuation is a full prefill, as the first two cases). The
worst case is each call's own wall time, measured, not extrapolated: both lanes must generate all 2,048 tokens (2,047
paired steps), or the run is inconclusive (exit 2) unless it has already failed. Before them, for the record, the
batch with 1 new token a lane (the two prefills). Also checks the bound: on short prompts, where only the bound can refuse it, 2,049 is refused (400),
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
    # The prompts are a list of k numbers from 10,000, one a line, and an instruction to go on counting, so that greedy
    # decoding goes on (a long prose prompt with the same instruction stopped after 200 tokens, development 2). Every
    # five-digit line costs the same 3 tokens (four-digit ones ran out at 8,999 lines, short of a 32,768 total: R5-2;
    # with six-digit ones the long lane stopped after 66 tokens, development 9).
    INSTRUCTION = "\nContinue this list: count upward from where it ends, one number per line, and do not stop."
    def numbers(start, k):
        assert 10000 <= start and start + k <= 100000
        return "\n".join(str(i) for i in range(start, start + k))
    def text(k, count=True): return [dict(role="user", content=numbers(10000, k) + (INSTRUCTION if count else ""))]
    sid = client.call("POST", "/sessions", dict(capacity=8192))["session_id"]
    n = {}
    for k in (100, 1100):
        client.call("POST", f"/sessions/{sid}/reset", {})
        n[k] = client.call("POST", "/generate", dict(session_id=sid, messages=text(k), max_new_tokens=1))["consumed_tokens"]
    client.call("POST", f"/sessions/{sid}/close", {})
    per = (n[1100] - n[100]) / 1000  # tokens per line, every line alike
    base = n[100] - 100 * per  # the template's tokens
    CONTINUE = 1024  # the worker's kMaxContinuation: the most a cached prefix is continued by (R5-1)
    # a short prompt (20 lines), room for a continuation and 2,048 new tokens; the rest to the other
    small = int(LIMIT + CONTINUE + 64 + base + 20 * per) + 1
    large = total - small
    sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in (small, large)]
    fill = [int((c - LIMIT - 64 - base) // per) for c in (small, large)]  # prompt + 2,048 just fit
    bodies = [dict(session_id=sid_, messages=text(k), max_new_tokens=1) for sid_, k in zip(sids, fill)]
    t0 = time.monotonic(); client.call("POST", "/generate_batch", dict(requests=bodies)); prefill = time.monotonic() - t0
    # The host's token choice is part of the call: greedy, and sampled (do_sample with top_k 0: the host sorts and
    # exponentiates the whole vocabulary at every step of every lane; red-team-2 R4-1). Its cost is about the same for
    # any parameters (a high temperature with top_p near 1 costs about 7% more on the host: R5-3); top_p 0.5 at
    # temperature 0.05 keeps the top token unless it holds less than half the mass, so the count goes on as greedily
    # (development 5, temperature 0.5 with a repetition penalty of 1.05, ended a lane at 92 tokens; developments 6 and
    # 7, temperature 0.05 and top_p 0.98, with a penalty of 1.01 and without, at 71: a near tie with the end of the turn).
    SAMPLED = dict(do_sample=True, temperature=0.05, top_k=0, top_p=0.5, seed=7)
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
    # The continued case: each lane's cached prompt continued by at most CONTINUE tokens (its reply, then 300 more
    # numbers and the instruction: about 950), forwarded a token at a time inside the call, then 2,048 sampled.
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    first = [int((c - LIMIT - CONTINUE - 64 - base) // per) for c in (small, large)]
    seeded = [dict(session_id=sid_, messages=text(k), max_new_tokens=1) for sid_, k in zip(sids, first)]
    r0 = client.call("POST", "/generate_batch", dict(requests=seeded))
    req = [dict(session_id=sid_, max_new_tokens=LIMIT, sampling=SAMPLED,
                messages=text(k) + [dict(role="assistant", content=x["text"]),
                                    dict(role="user", content=numbers(10000 + k, 300) + INSTRUCTION)])
           for sid_, k, x in zip(sids, first, r0["results"])]
    t0 = time.monotonic(); r = client.call("POST", "/generate_batch", dict(requests=req)); wall = time.monotonic() - t0
    generated = [len(x["generated_ids"]) for x in r["results"]]
    prompts = [x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in r["results"]]
    reused = [x["reused_tokens"] for x in r["results"]]
    continued = [p_ - u for p_, u in zip(prompts, reused)]
    cases["sampled-continued"] = dict(seconds=round(wall, 1), generated=generated, paired=r["paired_decode_calls"],
                                      prompts=prompts, reused=reused, continued=continued,
                                      complete=generated == [LIMIT, LIMIT] and r["paired_decode_calls"] == STEPS and
                                      all(u > 0 for u in reused) and all(800 <= c_ <= CONTINUE for c_ in continued))
    print(f"batch budget sampled-continued: reused {reused}, continued {continued} tokens; generated {generated} "
          f"({r['paired_decode_calls']} paired steps), {wall:.1f} s (limit {0.8 * ENGINE_TIMEOUT:.0f} s)", flush=True)
    # The bound alone refuses: short prompts on the reset sessions, where 2,049 new tokens fit the small one's capacity.
    for s in sids: client.call("POST", f"/sessions/{s}/reset", {})
    short = [dict(session_id=sids[0], messages=text(10), max_new_tokens=LIMIT + 1),
             dict(session_id=sids[1], messages=text(10), max_new_tokens=8)]
    assert all(base + 10 * per + 64 + LIMIT + 1 <= c for c in (small, large))  # only the bound can refuse it
    refused = client.call("POST", "/generate_batch", dict(requests=short), expected=(400,))[1]
    health = client.call("GET", "/health", auth=False)
    complete = all(c["complete"] for c in cases.values())
    definite_fail = any(c["seconds"] > 0.8 * ENGINE_TIMEOUT for c in cases.values()) or \
        refused != {"error": "invalid_request"} or health.get("ready") is not True
    report.update(sessions=[small, large], prefill_seconds=round(prefill, 2), cases=cases, complete=complete,
                  refusal_2049=refused, health_ready=health.get("ready"), definite_fail=definite_fail)
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
