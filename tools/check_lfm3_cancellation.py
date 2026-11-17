"""Cooperative native token-boundary cancellation gate, not GPU preemption.

The opt-in audit trace proves completed decode work before sending cancellation.
No timing sleep alone is accepted as evidence that a generation was running.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time

from lfm3_worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--control-request", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = json.loads(args.control_request.read_text())["input_ids"]
    reference = json.loads(args.control_result.read_text())["generated_ids"]
    if len(reference) < 33 or len(prompt) + 33 > 512:
        raise ValueError("fixture requires33 accepted predictions within512 capacity")
    worker_hash = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    records = []
    report = dict(status="running", worker_sha256=worker_hash, records=records,
                  scope="native completed-decode cancellation, pre-dispatch cancellation, stale IDs, cache abort, peer preservation and exact recovery; token boundaries only, no prefill/kernel preemption or latency SLA")
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    worker = None
    try:
        env = dict(os.environ)
        env.pop("LFM3_SERVER_TOKEN", None)
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        worker = WorkerClient(args.worker, args.checkpoint, timeout=60, env=env,
                              cancellation=True, sampling_trace_root=args.output / "sampling-traces")
        def call(request):
            result = worker.call(request)
            records.append(dict(request=request, response=result))
            save()
            if result.get("ok") is not True:
                raise AssertionError("native operation rejected")
            return result
        def generation(sid, **extra):
            return dict(op="generate", session_id=sid, input_ids=prompt, max_new_tokens=32, **extra)
        for sid in ("A", "B"):
            call(dict(op="create", session_id=sid, capacity=512))
            baseline = call(generation(sid))
            if baseline.get("generated_ids") != reference[:32]:
                raise AssertionError("warm native control IDs differ")
        before_b = call(dict(op="inspect", session_id="B"))
        before_a = call(dict(op="inspect", session_id="A"))
        if type(before_a.get("committed_consumed_tokens")) is not int:
            raise AssertionError("missing committed history count")
        target_rpc = worker.sequence + 1
        proof = args.output / "sampling-traces" / str(target_rpc) / "2.logits.bf16"
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(worker.call, generation("A", trace_sampling=True))
            deadline = time.monotonic() + 15
            while not proof.exists() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.001)
            if not proof.exists() or future.done():
                raise AssertionError("cancellation fixture did not observe live completed decode work")
            begin = time.monotonic()
            if not worker.cancel("A"):
                raise AssertionError("running cancellation was not admitted")
            cancelled = future.result(timeout=60)
            report["observed_cancel_completion_seconds"] = time.monotonic() - begin
        records.append(dict(case="running_cancel", response=cancelled))
        partial = cancelled.get("generated_ids")
        if (cancelled.get("ok") is not True or cancelled.get("cancelled") is not True or
            cancelled.get("cache_valid") is not False or cancelled.get("consumed_tokens") != 0 or
            not isinstance(partial, list) or not 2 <= len(partial) < 32 or partial != reference[:len(partial)]):
            raise AssertionError("running cancellation did not abort a correct partial generation")
        after_a = call(dict(op="inspect", session_id="A"))
        if (after_a.get("cache_valid") is not False or after_a.get("consumed_tokens") != 0 or
            after_a.get("inflight") is not False or after_a.get("transcript_tokens") != before_a["transcript_tokens"] or
            after_a.get("committed_consumed_tokens") != before_a["committed_consumed_tokens"]):
            raise AssertionError("cancelled cache/lease/transcript accounting mismatch")
        if call(dict(op="inspect", session_id="B")) != before_b:
            raise AssertionError("A cancellation mutated B")
        resumed_b = call(dict(op="generate", session_id="B", input_ids=prompt + reference[:32], max_new_tokens=1))
        if resumed_b.get("generated_ids") != reference[32:33] or resumed_b.get("reused_tokens") != len(prompt) + 31:
            raise AssertionError("B lost exact continuation/cache after A cancellation")
        recovered_a = call(generation("A"))
        if recovered_a.get("generated_ids") != reference[:32] or recovered_a.get("reused_tokens") != 0:
            raise AssertionError("cancelled A did not recover by exact fresh prefill")
        event = threading.Event()
        event.set()
        pre = worker.call_cancellable(generation("A"), event)
        records.append(dict(case="pre_dispatch_cancel", response=pre))
        if pre.get("cancelled") is not True or pre.get("generated_ids") != []:
            raise AssertionError("pre-dispatch cancellation executed generation")
        if worker.cancel("A"):
            raise AssertionError("idle generation accepted cancellation")
        # A late duplicate targets the old request, never the next generation.
        os.write(worker.cancel_write_fd, struct.pack("<Q", target_rpc))
        stale = call(generation("A"))
        if stale.get("cancelled") or stale.get("generated_ids") != reference[:32]:
            raise AssertionError("stale cancellation affected a later generation")
        for sid in ("A", "B"):
            call(dict(op="close", session_id=sid))
        call(dict(op="shutdown"))
        if worker.process.wait(timeout=10) != 0:
            raise AssertionError("unclean worker exit")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest() != worker_hash:
            raise AssertionError("worker changed during gate")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        except BaseException as exc:
            report["status"], report["cleanup_error"] = "failed", str(exc)
            save()
            raise
        save()
    print(json.dumps({key:value for key,value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
