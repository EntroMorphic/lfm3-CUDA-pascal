"""Explicit authenticated loopback/native cancellation gate; no kernel preemption."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time

from lfm3_server import LocalHTTPServer, Service, Settings
from lfm3_text_frontend import load_frontend
from lfm3_worker_client import WorkerClient


def validate_cancelled(response, reference):
    ids = response.get("generated_ids")
    if (response.get("cancelled") is not True or response.get("finish_reason") != "cancelled" or
        response.get("cache_valid") is not False or type(response.get("consumed_tokens")) is not int or response["consumed_tokens"] != 0 or
        not isinstance(ids, list) or not all(type(token) is int for token in ids) or
        not 2 <= len(ids) < 32 or ids != reference[:len(ids)]):
        raise AssertionError("HTTP did not return an exact aborted partial generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.control_result.read_text())["generated_ids"]
    if len(reference) < 32:
        raise ValueError("need32 accepted control IDs")
    worker_hash = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    sources = [Path(__file__), Path(__file__).with_name("lfm3_server.py"), Path(__file__).with_name("lfm3_worker_client.py")]
    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    records = []
    report = dict(status="running", worker_sha256=worker_hash, source_sha256=hashes, records=records,
                  scope="finite authenticated HTTP running cancellation, exact partial IDs, cache discard/history preservation, peer preservation and recovery; token-boundary only, no prefill/kernel preemption or latency guarantee")
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    worker = service = server = thread = None
    try:
        env = dict(os.environ)
        env.pop("LFM3_SERVER_TOKEN", None)
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        trace_root = args.output / "sampling-traces"
        worker = WorkerClient(args.worker, args.checkpoint, timeout=60, env=env,
                              cancellation=True, sampling_trace_root=trace_root)
        config = json.loads((args.checkpoint / "config.json").read_text())
        service = Service(worker, load_frontend(args.checkpoint), config,
                          Settings(request_timeout=90, socket_timeout=3))
        bearer = secrets.token_urlsafe(32)
        server = LocalHTTPServer(("127.0.0.1", 0), service, bearer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def call(path, body, expected=200, auth=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=120)
            headers = {"Content-Type":"application/json"}
            if auth:
                headers["Authorization"] = "Bearer " + bearer
            try:
                connection.request("POST", path, json.dumps(body).encode(), headers)
                response = connection.getresponse()
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise AssertionError("HTTP response too large")
                result = json.loads(raw)
                records.append(dict(path=path, status=response.status, response=result))
                save()
                if response.status != expected:
                    raise AssertionError((path, response.status, result))
                return result
            finally:
                connection.close()
        messages = [dict(role="system", content="Respond with text."), dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
        a, b = [call("/sessions", dict(capacity=512))["session_id"] for _ in range(2)]
        def generation(sid):
            return dict(session_id=sid, messages=messages, max_new_tokens=32)
        for sid in (a, b):
            if call("/generate", generation(sid)).get("generated_ids") != reference[:32]:
                raise AssertionError("warm HTTP control mismatch")
        def inspect(sid):
            return service._submit(lambda: worker.call(dict(op="inspect", session_id=sid)), lambda: None)
        before_a, before_b = inspect(a), inspect(b)
        audited = threading.Event()
        original = worker.call_cancellable
        def traced(request, event):
            # Gate-only instrumentation. HTTP request bodies cannot select trace
            # paths or enable this option; the startup audit root is explicit.
            return original(dict(request, trace_sampling=True) if audited.is_set() else request, event)
        worker.call_cancellable = traced
        target = worker.sequence + 1
        proof = trace_root / str(target) / "2.logits.bf16"
        audited.set()
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(call, "/generate", generation(a))
            deadline = time.monotonic() + 15
            while not proof.exists() and not pending.done() and time.monotonic() < deadline:
                time.sleep(0.001)
            if not proof.exists() or pending.done():
                raise AssertionError("HTTP cancellation fixture did not observe running decode")
            call(f"/sessions/{a}/cancel", {}, expected=401, auth=False)
            requested = call(f"/sessions/{a}/cancel", {}, expected=202)
            if requested.get("cancel_requested") is not True:
                raise AssertionError("HTTP did not acknowledge cancellation request")
            cancelled = pending.result(timeout=60)
        audited.clear()
        validate_cancelled(cancelled, reference)
        after_a, after_b = inspect(a), inspect(b)
        if (after_a.get("cache_valid") is not False or after_a.get("consumed_tokens") != 0 or
            type(after_a.get("committed_consumed_tokens")) is not int or
            after_a["committed_consumed_tokens"] != before_a.get("committed_consumed_tokens") or
            after_a.get("transcript_tokens") != before_a.get("transcript_tokens") or after_a.get("inflight") is not False):
            raise AssertionError("HTTP cancellation lost cache/history accounting")
        if before_b != after_b:
            raise AssertionError("HTTP cancellation mutated peer B")
        call(f"/sessions/{a}/cancel", {}, expected=409)
        for sid in (a, b):
            recovered = call("/generate", generation(sid))
            if recovered.get("generated_ids") != reference[:32] or recovered.get("cancelled") is not False:
                raise AssertionError("HTTP post-cancellation control mismatch")
        for sid in (a, b):
            call(f"/sessions/{sid}/close", {})
        if service.failed or service.sessions:
            raise AssertionError("HTTP cancellation failed service/lifecycle")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest() != worker_hash or any(
            hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name] for path in sources):
            raise AssertionError("cancellation gate source/binary changed")
        (args.output / "partial-response.txt").write_text(cancelled["text"] + "\n")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        try:
            if server is not None:
                server.shutdown()
                server.server_close()
            if thread is not None:
                thread.join(timeout=10)
            if service is not None:
                service.close()
                if service.cleanup_failed:
                    raise RuntimeError("HTTP cancellation cleanup failed")
            elif worker is not None:
                worker.close()
        except BaseException as exc:
            report["status"], report["cleanup_error"] = "failed", str(exc)
            save()
            raise
        save()
    print(json.dumps({key:value for key,value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
