"""Hash-pinned, paired end-to-end HTTP benchmark; no speed acceptance threshold."""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import sys
import threading
import time

from lfm3_owned_process import owned_main
from lfm3_server import LocalHTTPServer, Service, Settings
from lfm3_text_frontend import load_frontend
from lfm3_worker_client import WorkerClient


def mode_order(round_index):
    return ["serial-pair", "group"] if round_index % 2 == 0 else ["group", "serial-pair"]


def validate_reference(reference):
    if not isinstance(reference, list) or len(reference) != 32 or any(type(x) is not int or x < 0 for x in reference):
        raise AssertionError("need exactly 32 integer accepted control IDs")


def validate_numerical_evidence(evidence, worker_sha256):
    if (not isinstance(evidence, dict) or evidence.get("status") != "passed" or
            evidence.get("worker_sha256") != worker_sha256 or
            type(evidence.get("bitwise_logit_steps")) is not int or evidence["bitwise_logit_steps"] < 73 or
            evidence.get("independent_non_greedy_logit_replay") is not True or
            evidence.get("aggregate32k_checked") is not True):
        raise AssertionError("need passed hash-matched native grouped numerical evidence (73+ bitwise steps, sampling replay, 32K)")


def validate_response(response, mode, reference, consumed, finish):
    validate_reference(reference)
    if mode not in ("serial-pair", "group") or not isinstance(response, dict):
        raise AssertionError("invalid benchmark response/mode")
    grouped = mode == "group"
    if grouped and (type(response.get("paired_decode_calls")) is not int or
                    response["paired_decode_calls"] != 31 or response.get("gpu_batching") is not True):
        raise AssertionError("missing actual 31-call GPU grouping")
    lanes = response.get("results")
    if not isinstance(lanes, list) or len(lanes) != 2:
        raise AssertionError("need exactly two complete lanes")
    for index, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise AssertionError("invalid lane")
        ids = lane.get("generated_ids")
        if (not isinstance(ids, list) or ids != reference or any(type(x) is not int for x in ids) or
                lane.get("gpu_batching") is not grouped or lane.get("cancelled") is not False or
                lane.get("cache_valid") is not True or lane.get("finish_reason") != finish or
                lane.get("do_sample") is not bool(index) or type(lane.get("seed")) is not int or lane["seed"] != 42 or
                type(lane.get("consumed_tokens")) is not int or lane["consumed_tokens"] != consumed or
                type(lane.get("reused_tokens")) is not int or lane["reused_tokens"] != 0 or
                not isinstance(lane.get("text"), str)):
            raise AssertionError("lane IDs/stop/counts/fresh-prefill/sampling policy mismatch")
        if grouped and (type(lane.get("paired_decode_calls")) is not int or lane["paired_decode_calls"] != 31):
            raise AssertionError("lane did not participate in every paired decode")


def validate_coverage(records, samples):
    if type(samples) is not int or not 5 <= samples <= 100:
        raise AssertionError("samples must be 5..100")
    expected = [(phase, r, mode) for phase, count in (("warmup", 3), ("measured", samples))
                for r in range(count) for mode in mode_order(r)]
    actual = [(row.get("phase"), row.get("round"), row.get("mode")) for row in records]
    if actual != expected:
        raise AssertionError("missing, reordered or extra warmup/measured coverage")
    if any(type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or
           row["seconds"] <= 0 for row in records):
        raise AssertionError("latencies must be finite positive observations")


def validate_hashes(start, end):
    if (not start or start != end or not {"worker", "control_result", "numerical_evidence"} <= start.keys() or len(start) < 4 or
            any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                for v in start.values())):
        raise AssertionError("worker/source/artifact hashes missing or changed during run")


def summarize(records, samples):
    validate_coverage(records, samples)
    result = {}
    for mode in mode_order(0):
        times = [r["seconds"] for r in records if r["phase"] == "measured" and r["mode"] == mode]
        result[mode] = dict(samples=len(times), median_seconds=statistics.median(times),
                            p95_seconds=sorted(times)[math.ceil(.95 * len(times)) - 1],
                            aggregate_output_tokens_per_second=64 * len(times) / sum(times))
    ratios = []
    for r in range(samples):
        pair = {x["mode"]: x["seconds"] for x in records if x["phase"] == "measured" and x["round"] == r}
        ratios.append(dict(round=r, serial_pair_over_group_wall=pair["serial-pair"] / pair["group"]))
    result["paired_ratios"] = ratios
    result["median_paired_serial_over_group_wall"] = statistics.median(x["serial_pair_over_group_wall"] for x in ratios)
    return result


def hash_inputs(worker, control_result, numerical_evidence):
    root = Path(__file__).resolve().parents[1]
    # Include wrappers, imported local dependencies and native implementation, not build artifacts.
    paths = sorted(set((root / "tools").glob("*.py")) | set((root / "native").rglob("*.cu")) |
                   set((root / "native").rglob("*.cpp")) | set((root / "native").rglob("*.h")) |
                   set((root / "native").rglob("*.cuh")) | set((root / "native").rglob("*.hpp")))
    return dict(worker=hashlib.sha256(worker.read_bytes()).hexdigest(),
                control_result=hashlib.sha256(control_result.read_bytes()).hexdigest(),
                numerical_evidence=hashlib.sha256(numerical_evidence.read_bytes()).hexdigest(),
                **{str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


@owned_main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "worker", "control-result", "numerical-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.samples <= 100:
        parser.error("--samples must be 5..100")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", requested_samples=args.samples, records=[],
                  scope="HTTP/frontend/prefill/sampling/full responses; 64 output IDs per mode; no speed threshold, universal SLA or statistical significance claim")
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    worker = service = server = thread = None
    save()
    try:
        report["artifact_paths"] = dict(control_result=str(args.control_result.resolve()),
                                      numerical_evidence=str(args.numerical_evidence.resolve()), worker=str(args.worker.resolve()))
        report["start_sha256"] = hash_inputs(args.worker, args.control_result, args.numerical_evidence)
        report["numerical_evidence"] = json.loads(args.numerical_evidence.read_text())
        validate_numerical_evidence(report["numerical_evidence"], report["start_sha256"]["worker"])
        reference = json.loads(args.control_result.read_text())["generated_ids"][:32]
        validate_reference(reference)
        report["control_sha256"] = report["start_sha256"]["control_result"]
        report["python_executable"] = sys.executable
        report["loaded_local_modules"] = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
                                          ("lfm3_server", "lfm3_worker_client", "lfm3_text_frontend", "lfm3_owned_process", "lfm3_chat")}
        for name, path in report["loaded_local_modules"].items():
            if Path(path) != Path(__file__).resolve().with_name(name + ".py"):
                raise AssertionError("unexpected local module load path: " + name)
        env = dict(os.environ)
        env.pop("LFM3_SERVER_TOKEN", None)
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        report["worker_ld_library_path"] = env["LD_LIBRARY_PATH"]
        save()
        worker = WorkerClient(args.worker, args.checkpoint, timeout=180, env=env, cancellation=True, batching=True)
        frontend = load_frontend(args.checkpoint)
        service = Service(worker, frontend, json.loads((args.checkpoint / "config.json").read_text()),
                          Settings(request_timeout=180, socket_timeout=3))
        bearer = secrets.token_urlsafe(32)  # Deliberately never persisted.
        server = LocalHTTPServer(("127.0.0.1", 0), service, bearer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def call(path, body):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=200)
            try:
                connection.request("POST", path, json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + bearer})
                reply = connection.getresponse()
                raw = reply.read(4 * 1024 * 1024 + 1)
                if reply.status != 200 or len(raw) > 4 * 1024 * 1024:
                    raise AssertionError("HTTP benchmark request failed: " + str(reply.status))
                return json.loads(raw)
            finally:
                connection.close()
        sessions = [call("/sessions", dict(capacity=512))["session_id"] for _ in range(2)]
        messages = [dict(role="system", content="Respond with text."), dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
        requests = [dict(session_id=sid, messages=messages, max_new_tokens=32,
                         sampling=dict(do_sample=bool(i), top_k=1, seed=42)) for i, sid in enumerate(sessions)]
        consumed = len(frontend.encode_messages(messages, 512, 32)) + 31
        finish = "stop" if reference[-1] in service.stops else "length"
        report["policy"] = dict(requests=requests, expected_ids=reference, consumed_tokens=consumed,
                                reused_tokens=0, finish_reason=finish, session_capacity=512,
                                warmup_rounds_per_mode=3, lifecycle="same two sessions throughout; repeated shorter prompt forces fresh prefill")
        for phase, count in (("warmup", 3), ("measured", args.samples)):
            for r in range(count):
                for mode in mode_order(r):
                    started = time.perf_counter()
                    response = (call("/generate_batch", dict(requests=requests)) if mode == "group" else
                                dict(results=[call("/generate", request) for request in requests]))
                    seconds = time.perf_counter() - started
                    row = dict(phase=phase, round=r, mode=mode, order=mode_order(r), seconds=seconds, response=response)
                    report["records"].append(row)
                    save()  # Keep even a failing response and every completed intermediate observation.
                    validate_response(response, mode, reference, consumed, finish)
        report["latency_summary"] = summarize(report["records"], args.samples)
        for sid in sessions:
            call(f"/sessions/{sid}/close", {})
        if service.sessions or service.failed:
            raise AssertionError("leaked sessions or failed service")
        report["status"] = "validated-awaiting-cleanup"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        errors = []
        for cleanup in (lambda: server.shutdown() if server is not None and thread is not None and thread.is_alive() else None,
                        lambda: server.server_close() if server is not None else None,
                        lambda: thread.join(timeout=10) if thread is not None else None,
                        lambda: service.close() if service is not None else None,
                        lambda: service.frontend_owner.join(timeout=10) if service is not None else None,
                        lambda: worker.close() if worker is not None else None):
            try:
                cleanup()
            except BaseException as exc:
                errors.append(str(exc))
        if ((thread is not None and thread.is_alive()) or
                (service is not None and (service.cleanup_failed or service.owner.is_alive() or service.frontend_owner.is_alive())) or
                (worker is not None and (worker.cleanup_failed or worker.process.poll() is None))):
            errors.append("server/service/worker not quiescent")
        try:
            report["end_sha256"] = hash_inputs(args.worker, args.control_result, args.numerical_evidence)
            validate_hashes(report.get("start_sha256"), report["end_sha256"])
        except BaseException as exc:
            errors.append(str(exc))
        if errors:
            report.update(status="failed", cleanup_or_hash_errors=errors)
        elif report["status"] == "validated-awaiting-cleanup":
            report["status"] = "passed"
        save()
        if errors:
            raise RuntimeError("benchmark cleanup/hash validation failed: " + "; ".join(errors))
    print(json.dumps(dict(status=report["status"], latency_summary=report["latency_summary"]), indent=2))


if __name__ == "__main__":
    main()
