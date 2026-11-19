"""serve-v1 rule 10: paired decode (op "generate_batch") against serial decode on the compact window caches.

For each case a session of the given capacity (prompt: the first PROMPT tokens of window 0 of the 32,768-token
perplexity windows) is paired with one of capacity 4,096 (prompt: 257 tokens of window 1). Both are run serially
with logit traces (in continuations of up to 32 tokens, the worker's limit for traces), reset, then run as a pair
the same way: every step's logits (BF16 and FP32 traces) and every id must be equal.
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
        prompts = {"A": w1[:257], "B": w0[:prompt]}
        # The worker traces at most 32 tokens per request: each run is a chain of continuations of up to 32 tokens
        # (prompt + everything generated so far; the cache is reused, so every step after the first is a decode).
        def run(paired):
            gen = {"A": [], "B": []}; chunks = []; calls = 0
            while len(gen["A"]) < steps:
                k = min(32, steps - len(gen["A"]))
                lanes = [dict(session_id=sid, input_ids=prompts[sid] + gen[sid], max_new_tokens=k, trace_sampling=True) for sid in ("A", "B")]
                if paired:
                    r = worker.call_batch_cancellable(lanes, [threading.Event(), threading.Event()])
                    if r.get("ok") is not True: raise AssertionError(f"generate_batch rejected: {r}")
                    results = r["results"]; calls += r["paired_decode_calls"]
                else:
                    results = [call(dict(op="generate", **lane)) for lane in lanes]
                chunks.append([(str(x["sampling_trace"]), len(x["generated_ids"])) for x in results])
                for sid, x in zip(("A", "B"), results): gen[sid] += x["generated_ids"]
                if any(n < k for _, n in chunks[-1]): break  # a stop token
            return gen, chunks, calls
        serial = run(False)
        for sid in ("A", "B"): call(dict(op="reset", session_id=sid))
        paired = run(True)
        expected_calls = sum(min(n for _, n in c) - 1 for c in paired[1])
        case = dict(capacity=capacity, prompt=prompt, chunks=len(paired[1]), paired_decode_calls=paired[2],
                    expected_paired_decode_calls=expected_calls, lanes=[])
        for li, sid in enumerate(("A", "B")):
            unequal = []; steps_run = len(serial[0][sid])
            same_chunks = [c[li][1] for c in serial[1]] == [c[li][1] for c in paired[1]]
            for cs, cp in zip(serial[1], paired[1]):
                (ds, ns), (dp, _) = cs[li], cp[li]
                for i in range(ns):
                    for ext in ("logits.bf16", "logits.f32"):
                        a, b = traces / ds / f"{i}.{ext}", traces / dp / f"{i}.{ext}"
                        if not a.exists() or not b.exists() or a.read_bytes() != b.read_bytes(): unequal.append(f"{ds}/{i}.{ext}")
            same_ids = serial[0][sid] == paired[0][sid]
            ok = same_ids and same_chunks and not unequal and steps_run == steps
            failed |= not ok
            case["lanes"].append(dict(session=sid, steps=steps_run, ids_equal=same_ids, unequal_traces=unequal[:5],
                                      unequal_count=len(unequal), ok=ok))
            print(f"capacity {capacity} prompt {prompt}: lane {sid}: {steps_run} steps, ids {'equal' if same_ids else 'DIFFER'}, "
                  f"traces {'equal' if not unequal else f'{len(unequal)} DIFFER from {unequal[0]}'}: {'ok' if ok else 'FAIL'}", flush=True)
        if paired[2] != expected_calls or expected_calls < 1:
            failed = True; print(f"capacity {capacity}: {paired[2]} paired calls, expected {expected_calls}: FAIL")
        report["cases"].append(case)
        for sid in ("A", "B"): call(dict(op="close", session_id=sid))
        for f in traces.rglob("*.logits.*"): f.unlink()  # keep the record small
    call(dict(op="shutdown"))
    report["status"] = "failed" if failed else "passed"
finally:
    worker.close()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print("PAIRED DECODE (rule 10)", "PASS" if report["status"] == "passed" else "FAIL")
sys.exit(0 if report["status"] == "passed" else 1)
