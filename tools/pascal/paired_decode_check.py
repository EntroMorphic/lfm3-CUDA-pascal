"""serve-v1 rule 10: paired decode (op "generate_batch") against serial decode on the compact window caches.

For each case a session of the given capacity (prompt: the first PROMPT tokens of window 0 of the 32,768-token
perplexity windows) is paired with one of capacity 4,096 (prompt: 257 tokens of window 1). Both are run serially
with logit traces, reset, then run as a pair: every step's logits (BF16 and FP32 traces) and every id must be equal.
Greedy, STEPS new tokens each.
    python3 tools/pascal/paired_decode_check.py WORKER CHECKPOINT WINDOWS OUTPUT [STEPS=600] [CASES=4097:3400,8192:7500,28672:28000]
Exit 0 only if every case is equal at every step and ran all its steps.
"""
from array import array
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfm3_worker_client import WorkerClient  # noqa: E402

worker_path, checkpoint, windows, out = map(Path, sys.argv[1:5])
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
cases = [tuple(map(int, c.split(":"))) for c in (sys.argv[6] if len(sys.argv) > 6 else "4097:3400,8192:7500,28672:28000").split(",")]
out.mkdir(parents=True, exist_ok=False)
traces = out / "traces"
ids = array("i"); ids.frombytes(windows.read_bytes())
w0, w1 = list(ids[:32768]), list(ids[32768:65536])
report = dict(steps=steps, cases=[], status="failed")  # "passed" only once every case has run
worker = WorkerClient(worker_path, checkpoint, timeout=600, cancellation=True, batching=True, sampling_trace_root=traces)
failed = False
def call(request):
    response = worker.call(request)
    if response.get("ok") is not True: raise AssertionError(f"rejected: {request.get('op')}: {response}")
    return response
try:
    for capacity, prompt in cases:
        assert prompt + steps <= capacity, (capacity, prompt)
        call(dict(op="create", session_id="A", capacity=4096)); call(dict(op="create", session_id="B", capacity=capacity))
        lanes = [dict(session_id="A", input_ids=w1[:257], max_new_tokens=steps, trace_sampling=True),
                 dict(session_id="B", input_ids=w0[:prompt], max_new_tokens=steps, trace_sampling=True)]
        serial = [call(dict(op="generate", **lane)) for lane in lanes]
        for sid in ("A", "B"): call(dict(op="reset", session_id=sid))
        paired = worker.call_batch_cancellable(lanes, [threading.Event(), threading.Event()])
        if paired.get("ok") is not True: raise AssertionError(f"generate_batch rejected: {paired}")
        case = dict(capacity=capacity, prompt=prompt, paired_decode_calls=paired.get("paired_decode_calls"), lanes=[])
        for lane, s, p in zip(lanes, serial, paired["results"]):
            n = len(s["generated_ids"]); unequal = []
            for i in range(n):
                for ext in ("logits.bf16", "logits.f32"):
                    a, b = traces / str(s["sampling_trace"]) / f"{i}.{ext}", traces / str(p["sampling_trace"]) / f"{i}.{ext}"
                    if a.exists() != b.exists() or (a.exists() and a.read_bytes() != b.read_bytes()): unequal.append(f"{i}.{ext}")
            same_ids = s["generated_ids"] == p["generated_ids"]
            ok = same_ids and not unequal and n == steps and bool((traces / str(s["sampling_trace"]) / "0.logits.f32").exists())
            failed |= not ok
            case["lanes"].append(dict(session=lane["session_id"], steps=n, ids_equal=same_ids, unequal_traces=unequal[:5],
                                      unequal_count=len(unequal), ok=ok))
            print(f"capacity {capacity} prompt {prompt}: lane {lane['session_id']}: {n} steps, ids {'equal' if same_ids else 'DIFFER'}, "
                  f"traces {'equal' if not unequal else f'{len(unequal)} DIFFER from {unequal[0]}'}: {'ok' if ok else 'FAIL'}", flush=True)
        if not case["paired_decode_calls"] or case["paired_decode_calls"] < steps - 1:
            failed = True; print(f"capacity {capacity}: only {case['paired_decode_calls']} paired calls: FAIL")
        report["cases"].append(case)
        for sid in ("A", "B"): call(dict(op="close", session_id=sid))
        for d in {str(x["sampling_trace"]) for x in serial + paired["results"]}:  # keep the record small
            for f in (traces / d).rglob("*.logits.*"): f.unlink()
    call(dict(op="shutdown"))
    report["status"] = "failed" if failed else "passed"
finally:
    worker.close()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print("PAIRED DECODE (rule 10)", "PASS" if report["status"] == "passed" else "FAIL")
sys.exit(0 if report["status"] == "passed" else 1)
