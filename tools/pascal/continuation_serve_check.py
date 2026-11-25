"""continuation-v1 rule 4: a cached prefix is continued through the served paths.

  worker (WORKER: liquid-lfm3-worker, its stdin protocol, exact ids): a session holding a 1,000-token prompt and one
    generated token is continued by 1,024, 1,025 and 7,000 tokens, each from a fresh seed. EXPECT shipping: every one
    reuses the 1,000 consumed tokens. EXPECT parity (the Thor engine's path, red-team-2 R5-1's cap): 1,024 reuses
    them, 1,025 and 7,000 do not (full prefills).
  server (SERVER: liquid-lfm3-server, defaults, over HTTP; shipping only): red-team-2 R5-1's case, a session of
    29,696 seeded with about 1,000 tokens and continued by about 28,000, now 200 with the whole seed reused, within
    60 s, and /health ready; the session then continues again by a short turn (reusing everything before it); and a
    /generate_batch whose two lanes each continue a seeded session by about 5,000 tokens, both reusing their seeds.
    python3 tools/pascal/continuation_serve_check.py WORKER SERVER|- CHECKPOINT OUTPUT EXPECT(shipping|parity)
"""
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402
from lfm3_worker_client import WorkerClient  # noqa: E402

if len(sys.argv) != 6 or sys.argv[5] not in ("shipping", "parity"): sys.exit(__doc__)
worker_path, server_arg, checkpoint, out, expect = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
if (expect == "parity") != (server_arg == "-"): sys.exit("the server part runs for shipping only (SERVER '-' for parity)")
out.mkdir(parents=True, exist_ok=False)
report, ok = dict(expect=expect), True

raw = open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read(4 * 9000)
ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
w = WorkerClient(worker_path, checkpoint, timeout=900)
try:
    w.call(dict(op="create", session_id="A", capacity=8192))
    boundary = {}
    for n in (1024, 1025, 7000):
        want = 1000 if expect == "shipping" or n <= 1024 else 0
        w.call(dict(op="reset", session_id="A"))
        seed = w.call(dict(op="generate", session_id="A", input_ids=ids[:1000], max_new_tokens=1))
        prompt = ids[:1000] + seed["generated_ids"][:1] + ids[1000:1000 + n - 1]  # n tokens after the 1,000 consumed
        t0 = time.monotonic()
        r = w.call(dict(op="generate", session_id="A", input_ids=prompt, max_new_tokens=1))
        boundary[n] = dict(prompt=len(prompt), reused_tokens=r.get("reused_tokens"), expected=want,
                           seconds=round(time.monotonic() - t0, 2), ok=r.get("reused_tokens") == want and r.get("ok") is True)
        print(f"worker: {n} tokens after a cached 1,000: reused {r.get('reused_tokens')} (expected {want}) in "
              f"{boundary[n]['seconds']} s: {'ok' if boundary[n]['ok'] else 'FAIL'}", flush=True)
    report["worker"] = boundary; ok &= all(b["ok"] for b in boundary.values())
finally:
    w.close()

if expect == "shipping":
    records = []
    server = NativeServer(Path(server_arg), checkpoint, out, "server", None)
    client = Client(server.port, server.token, records)
    try:
        def numbers(start, k): return "\n".join(str(i) for i in range(start, start + k))  # five digits: 3 tokens a line
        sid = client.call("POST", "/sessions", dict(capacity=29696))["session_id"]
        seed_messages = [dict(role="user", content=numbers(10000, 320))]
        seed = client.call("POST", "/generate", dict(session_id=sid, messages=seed_messages, max_new_tokens=1))
        messages = seed_messages + [dict(role="assistant", content=seed["text"]), dict(role="user", content=numbers(10320, 9300))]
        t0 = time.monotonic()
        status, data = client.raw("POST", "/generate", json.dumps(dict(session_id=sid, messages=messages, max_new_tokens=1)).encode(),
                                  timeout=900)
        seconds = time.monotonic() - t0
        reply = json.loads(data) if data else {}
        hstatus, hdata = client.raw("GET", "/health", auth=False)
        health = json.loads(hdata)
        long = dict(seed_consumed=seed.get("consumed_tokens"), status=status, seconds=round(seconds, 1),
                    reused_tokens=reply.get("reused_tokens"), consumed_tokens=reply.get("consumed_tokens"),
                    error=reply.get("error"), health_status=hstatus, health=health)
        good = status == 200 and reply.get("reused_tokens") == seed.get("consumed_tokens") and seconds <= 60 and health.get("ready") is True
        long["ok"] = good; report["long"] = long; ok &= good
        print(f"server: seeded {seed.get('consumed_tokens')} tokens, continued to {reply.get('consumed_tokens')}: {status} in "
              f"{seconds:.1f} s, reused {reply.get('reused_tokens')}; /health {hstatus} ready {health.get('ready')}: "
              f"{'ok' if good else 'FAIL'}", flush=True)
        if status == 200:
            more = messages + [dict(role="assistant", content=reply["text"]), dict(role="user", content="Thanks.")]
            again = client.call("POST", "/generate", dict(session_id=sid, messages=more, max_new_tokens=4))
            good = again.get("reused_tokens") == reply.get("consumed_tokens")
            report["again"] = dict(reused_tokens=again.get("reused_tokens"), previous_consumed=reply.get("consumed_tokens"), ok=good)
            ok &= good
            print(f"server: a further turn reused {again.get('reused_tokens')} of {reply.get('consumed_tokens')}: "
                  f"{'ok' if good else 'FAIL'}", flush=True)
        client.call("POST", f"/sessions/{sid}/close", {})
        lanes = []
        for k in range(2):
            s = client.call("POST", "/sessions", dict(capacity=14848))["session_id"]
            m = [dict(role="user", content=numbers(20000 + 1000 * k, 320))]
            sd = client.call("POST", "/generate", dict(session_id=s, messages=m, max_new_tokens=1))
            lanes.append((s, m + [dict(role="assistant", content=sd["text"]), dict(role="user", content=numbers(30000 + 2000 * k, 1700))],
                          sd.get("consumed_tokens")))
        t0 = time.monotonic()
        batch = client.call("POST", "/generate_batch", dict(requests=[dict(session_id=s, messages=m, max_new_tokens=8) for s, m, _ in lanes]))
        seconds = time.monotonic() - t0
        results = batch.get("results", [])
        good = len(results) == 2 and all(r.get("reused_tokens") == c for r, (_, _, c) in zip(results, lanes))
        report["batch"] = dict(seconds=round(seconds, 1), reused=[r.get("reused_tokens") for r in results],
                               seeds=[c for _, _, c in lanes], consumed=[r.get("consumed_tokens") for r in results], ok=good)
        ok &= good
        print(f"server: batch lanes reused {report['batch']['reused']} of their seeds {report['batch']['seeds']}, consumed "
              f"{report['batch']['consumed']}, in {seconds:.1f} s: {'ok' if good else 'FAIL'}", flush=True)
    finally:
        report["server_exit"] = server.stop()
        (out / "records.json").write_text(json.dumps(records, indent=1)[:1_000_000] + "\n")
    ok &= report["server_exit"] == 0
(out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
print(f"CONTINUATION SERVE ({expect})", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
