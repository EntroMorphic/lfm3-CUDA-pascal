"""red-team-2 R5-1 (and R6-1): a cached prefix is continued by at most 1,024 tokens (the worker's kMaxContinuation);
a longer continuation is a full prefill. Two parts.

  boundary (WORKER: liquid-lfm3-worker, through its stdin protocol with exact ids): a session holding a 1,000-token
    prompt is continued by exactly 1,024 tokens (reused_tokens must be 1,000) and, seeded again, by 1,025 (reused_tokens
    must be 0, a full prefill). EXPECT new only; an old worker reuses both.
  watchdog (SERVER: liquid-lfm3-server with its defaults, over HTTP): a session of 29,696 seeded with about 1,000 tokens,
    then a /generate whose messages continue it by about 28,000 (the old worker forwards them a token at a time inside
    one engine call, about 17 ms each). EXPECT old: the call does not answer 200 and /health is then not ready (the
    server failed closed at the 300 s engine timeout). EXPECT new: 200, reused_tokens 0 (a full prefill), within
    240 s, and /health ready.
    python3 tools/pascal/continuation_check.py WORKER SERVER CHECKPOINT OUTPUT EXPECT(old|new)
"""
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402
from lfm3_worker_client import WorkerClient  # noqa: E402

if len(sys.argv) != 6 or sys.argv[5] not in ("old", "new"): sys.exit(__doc__)
worker_path, server_path, checkpoint, out, expect = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
out.mkdir(parents=True, exist_ok=False)
report, ok = dict(expect=expect), True

if expect == "new":
    raw = open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read(4 * 4096)
    ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    w = WorkerClient(worker_path, checkpoint, timeout=600)
    try:
        w.call(dict(op="create", session_id="A", capacity=8192))
        boundary = {}
        for n, want in ((1024, 1000), (1025, 0)):
            w.call(dict(op="reset", session_id="A"))
            seed = w.call(dict(op="generate", session_id="A", input_ids=ids[:1000], max_new_tokens=1))
            prompt = ids[:1000] + seed["generated_ids"][:1] + ids[1000:1000 + n - 1]  # n tokens after the 1,000 consumed
            r = w.call(dict(op="generate", session_id="A", input_ids=prompt, max_new_tokens=1))
            boundary[n] = dict(prompt=len(prompt), reused_tokens=r.get("reused_tokens"), expected=want,
                               ok=r.get("reused_tokens") == want)
            print(f"continuation boundary: {n} tokens after a cached 1,000: reused {r.get('reused_tokens')} "
                  f"(expected {want}): {'ok' if boundary[n]['ok'] else 'FAIL'}", flush=True)
        report["boundary"] = boundary; ok &= all(b["ok"] for b in boundary.values())
    finally:
        w.close()

records = []
server = NativeServer(server_path, checkpoint, out, "server", None)
client = Client(server.port, server.token, records)
try:
    def numbers(start, k): return "\n".join(str(i) for i in range(start, start + k))  # five digits: 3 tokens a line
    sid = client.call("POST", "/sessions", dict(capacity=29696))["session_id"]
    seed_messages = [dict(role="user", content=numbers(10000, 320))]
    seed = client.call("POST", "/generate", dict(session_id=sid, messages=seed_messages, max_new_tokens=1))
    messages = seed_messages + [dict(role="assistant", content=seed["text"]),
                                dict(role="user", content=numbers(10320, 9300))]
    t0 = time.monotonic()
    status, data = client.raw("POST", "/generate", json.dumps(dict(session_id=sid, messages=messages, max_new_tokens=1)).encode(),
                              timeout=900)
    seconds = time.monotonic() - t0
    reply = json.loads(data) if data else {}
    hstatus, hdata = client.raw("GET", "/health", auth=False)
    health = json.loads(hdata)
    report["watchdog"] = dict(seed_consumed=seed.get("consumed_tokens"), status=status, seconds=round(seconds, 1),
                              reused_tokens=reply.get("reused_tokens"), consumed_tokens=reply.get("consumed_tokens"),
                              error=reply.get("error"), health_status=hstatus, health=health)
    if expect == "old":
        good = status != 200 and health.get("ready") is False
    else:
        good = status == 200 and reply.get("reused_tokens") == 0 and seconds <= 240 and health.get("ready") is True
    ok &= good
    print(f"continuation watchdog ({expect}): seeded {seed.get('consumed_tokens')} tokens, then {status} in {seconds:.1f} s, "
          f"reused {reply.get('reused_tokens')}, consumed {reply.get('consumed_tokens')}, error {reply.get('error')}; "
          f"/health {hstatus} {health}: {'as expected' if good else 'NOT AS EXPECTED'}", flush=True)
finally:
    report["server_exit"] = server.stop()
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1)[:1_000_000] + "\n")
# the old server failed closed exits 1 (cleanup failed); only the new one must exit 0
ok &= expect == "old" or report["server_exit"] == 0
print(f"CONTINUATION ({expect})", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
