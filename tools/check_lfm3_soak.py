"""Finite authenticated HTTP soak/resource/latency gate; not GPU batching.

Run only with exclusive GPU access. Default30 cycles is a bounded regression
fixture, not a claim of hours-long production stability. Thresholds are fixed
before execution; failures and intermediate samples are preserved.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import threading
import time

from lfm3_server import LocalHTTPServer, Service, Settings
from lfm3_text_frontend import load_frontend
from lfm3_worker_client import WorkerClient


def process_sample(pid):
    """Linux RSS and descriptor counts; not a GPU-memory measurement."""
    directory = Path(f"/proc/{pid}")
    rss = None
    for line in (directory / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    if rss is None:
        raise RuntimeError("missing process RSS")
    return dict(rss_bytes=rss, descriptors=len(list((directory / "fd").iterdir())))


def resource_gate(baseline, samples):
    if not samples:
        raise AssertionError("resource gate needs measured samples")
    for sample in samples:
        for name in ("server", "worker"):
            if sample[name]["rss_bytes"] > baseline[name]["rss_bytes"] + 32 * 1024 * 1024:
                raise AssertionError(name + " RSS grew more than fixed32MiB warm plateau budget")
            if sample[name]["descriptors"] > baseline[name]["descriptors"] + 2:
                raise AssertionError(name + " descriptors grew beyond fixed warm plateau budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.cycles <= 10000:
        raise ValueError("cycles must be5..10000")
    args.output.mkdir(parents=True, exist_ok=False)
    worker_hash = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    source_paths = [Path(__file__), Path(__file__).with_name("lfm3_server.py"),
                    Path(__file__).with_name("lfm3_worker_client.py")]
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    report = dict(status="running", worker_sha256=worker_hash, source_sha256=sources,
                  cycles_requested=args.cycles, samples=[], latencies=[],
                  limits=dict(warm_rss_growth_bytes=32 * 1024 * 1024, warm_fd_growth=2),
                  scope="finite serialized HTTP soak, host/worker RSS+FD plateau and end-to-end latency; NOT GPU-memory, batching, cancellation or universal performance acceptance")
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    worker = service = server = thread = None
    started = time.monotonic()
    try:
        config = json.loads((args.checkpoint / "config.json").read_text())
        reference = json.loads(args.control_result.read_text())["generated_ids"]
        if not reference:
            raise ValueError("empty accepted reference")
        frontend = load_frontend(args.checkpoint)
        env = dict(os.environ)
        env.pop("LFM3_SERVER_TOKEN", None)
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        worker = WorkerClient(args.worker, args.checkpoint, timeout=60, env=env)
        service = Service(worker, frontend, config, Settings(request_timeout=90, socket_timeout=3))
        bearer = secrets.token_urlsafe(32)
        server = LocalHTTPServer(("127.0.0.1", 0), service, bearer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def call(path, body, expected=200):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=120)
            try:
                begin = time.monotonic()
                connection.request("POST", path, json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + bearer})
                response = connection.getresponse()
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024 or response.status != expected:
                    raise AssertionError((path, response.status, raw[:128]))
                return json.loads(raw), time.monotonic() - begin
            finally:
                connection.close()
        messages = [dict(role="system", content="Respond with text."), dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
        a = call("/sessions", dict(capacity=1024))[0]["session_id"]
        b = call("/sessions", dict(capacity=1024))[0]["session_id"]
        def generate(sid, sampled):
            response, duration = call("/generate", dict(session_id=sid, messages=messages,
                max_new_tokens=128, sampling=dict(do_sample=sampled, top_k=1, seed=42)))
            if response.get("generated_ids") != reference or response.get("gpu_batching") is not False:
                raise AssertionError("soak control IDs changed")
            return response, duration
        # Warm both geometries/policies and lifecycle paths before the plateau.
        for sid, sampled in ((a, False), (b, True)):
            generate(sid, sampled)
            call(f"/sessions/{sid}/reset", {})
            generate(sid, sampled)
        call(f"/sessions/{b}/close", {})
        b = call("/sessions", dict(capacity=1024))[0]["session_id"]
        generate(b, True)
        def sample(cycle):
            return dict(cycle=cycle, elapsed_seconds=time.monotonic() - started,
                        server=process_sample(os.getpid()), worker=process_sample(worker.process.pid))
        report["baseline"] = sample(-1)
        save()
        for cycle in range(args.cycles):
            for sid, sampled in ((a, False), (b, True)):
                response, duration = generate(sid, sampled)
                report["latencies"].append(dict(cycle=cycle, sampled=sampled, seconds=duration,
                                                 generated_tokens=len(response["generated_ids"])))
            call("/generate", dict(session_id=a, messages=messages, max_new_tokens=128,
                                    sampling=dict(top_p=0)), expected=400)
            if cycle % 5 == 4:
                call(f"/sessions/{a}/reset", {})
                call(f"/sessions/{b}/close", {})
                b = call("/sessions", dict(capacity=1024))[0]["session_id"]
                generate(b, True)  # Observe plateau with both caches resident.
            report["samples"].append(sample(cycle))
            save()
            resource_gate(report["baseline"], report["samples"])
        for sid in (a, b):
            call(f"/sessions/{sid}/close", {})
        if service.sessions or service.failed or service.cleanup_failed:
            raise AssertionError("soak ended with leaked sessions or failed service")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest() != worker_hash:
            raise AssertionError("worker changed during soak")
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != sources[path.name] for path in source_paths):
            raise AssertionError("source changed during soak")
        report["elapsed_seconds"] = time.monotonic() - started
        report["latency_summary"] = {}
        for sampled in (False, True):
            times = sorted(row["seconds"] for row in report["latencies"] if row["sampled"] is sampled)
            report["latency_summary"]["topk1_sampled" if sampled else "greedy"] = dict(
                count=len(times), median_seconds=statistics.median(times),
                p95_seconds=times[min(len(times)-1, (95 * len(times) + 99)//100 - 1)],
                median_end_to_end_tokens_per_second=len(reference)/statistics.median(times))
        (args.output / "control-response.txt").write_text(response["text"] + "\n")
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
                    raise RuntimeError("soak cleanup did not complete")
            elif worker is not None:
                worker.close()
            if worker is not None and worker.process.poll() is None:
                raise RuntimeError("worker still running after soak cleanup")
        except BaseException as exc:
            report["status"], report["cleanup_error"] = "failed", str(exc)
            save()
            raise
        save()
    print(json.dumps({key:value for key,value in report.items() if key not in ("samples", "latencies")}, indent=2))


if __name__ == "__main__":
    main()
