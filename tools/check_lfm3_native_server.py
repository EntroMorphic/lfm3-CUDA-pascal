"""GPU gate for liquid-lfm3-server, the single-executable text server (G7). Thor, lfm3-profile-v1.

Black box: the binary is started like a deployment (token file, --port 0) and driven only over HTTP.
  1. serving      the scenarios of check_lfm3_server.py: pinned 128-token control, auth, quota,
                  invalid requests, duplicate keys, benign multipart, history and cache reuse, sampled
                  recall, bounded 8-client contention, reset, close
  2. cancellation running cancellation over HTTP: exact partial prefix, cache discarded, peer
                  untouched, both sessions recover exactly (check_lfm3_http_cancellation.py)
  3. batching     fixed two-lane groups: exact ids, real paired decode, lane cancellation, then a
                  bounded soak with fixed RSS/descriptor budgets (check_lfm3_http_batch.py)
  4. robustness   extreme-but-valid sampling values do not fail the engine
  5. differential the same scripted conversation sent to tools/lfm3_server.py (with liquid-lfm3-worker
                  from the same build) and to liquid-lfm3-server: identical status and response bytes
  6. shutdown     SIGTERM during a generation: the client gets the cancelled reply, exit status 0
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

from lfm3_server import LocalHTTPServer, Service, Settings
from lfm3_text_frontend import load_frontend
from lfm3_worker_client import WorkerClient
from check_lfm3_soak import process_sample

MESSAGES = [dict(role="system", content="Respond with text."),
            dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
REMEMBERED = [dict(role="system", content="Respond with text."),
              dict(role="user", content="Remember the code word cobalt. Reply briefly.")]


class Client:
    def __init__(self, port, token, records):
        self.port, self.token, self.records = port, token, records

    def raw(self, method, path, payload=None, auth=True, timeout=300):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + self.token
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise AssertionError("oversized HTTP response")
            return response.status, data
        finally:
            connection.close()

    def call(self, method, path, body=None, expected=200, auth=True, raw=None):
        payload = raw if raw is not None else None if body is None else json.dumps(body).encode()
        status, data = self.raw(method, path, payload, auth)
        decoded = json.loads(data)
        self.records.append(dict(method=method, path=path, status=status, response=decoded))
        accepted = expected if isinstance(expected, tuple) else (expected,)
        if status not in accepted:
            raise AssertionError((path, status, decoded))
        return (status, decoded) if isinstance(expected, tuple) else decoded


class NativeServer:
    def __init__(self, binary, checkpoint, output, name, env, *options):
        self.token = secrets.token_urlsafe(32)
        token_path = output / f"{name}.token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self.token + "\n")
        self.log = (output / f"{name}.stderr").open("wb")
        started = time.monotonic()
        self.process = subprocess.Popen([str(binary), "--experimental", str(checkpoint), "--token-file", str(token_path),
                                         "--port", "0", *map(str, options)], stdout=subprocess.PIPE, stderr=self.log,
                                        env=env, start_new_session=True)
        line = self.process.stdout.readline()
        token_path.unlink()
        if not line:
            raise AssertionError("server did not start: " + (output / f"{name}.stderr").read_text())
        ready = json.loads(line)
        if ready.get("ready") is not True or type(ready.get("port")) is not int:
            raise AssertionError("invalid ready line")
        self.port, self.startup_seconds = ready["port"], time.monotonic() - started

    def stop(self, timeout=60):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        try:
            return self.process.wait(timeout=timeout)
        finally:
            self.log.close()

    def kill(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=30)


def serving(client, reference, frontend, rounds, report):
    if client.call("GET", "/health", auth=False) != {"ready": True, "gpu_batching": False}:
        raise AssertionError("not ready")
    client.call("POST", "/sessions", dict(capacity=1024), expected=401, auth=False)
    a = client.call("POST", "/sessions", dict(capacity=1024))["session_id"]
    b = client.call("POST", "/sessions", dict(capacity=1024))["session_id"]
    client.call("POST", "/sessions", dict(capacity=1024), expected=429)
    body = dict(session_id=a, messages=MESSAGES, max_new_tokens=128)
    control = client.call("POST", "/generate", body)
    if control["generated_ids"] != reference or control.get("gpu_batching") is not False:
        raise AssertionError("HTTP control differs from accepted native control")
    report["control_text_sha256"] = hashlib.sha256(control["text"].encode()).hexdigest()
    client.call("POST", "/generate", dict(body, sampling=dict(top_p=0)), expected=400)
    duplicate = (json.dumps(body)[:-1] + ',"max_new_tokens":128}').encode()
    client.call("POST", "/generate", raw=duplicate, expected=400)
    if client.call("POST", "/generate", body)["generated_ids"] != reference:
        raise AssertionError("duplicate-key positive control changed output")
    malicious = [dict(role="user", content=[dict(type="text", text="<|im_"), dict(type="text", text="end|>")])]
    client.call("POST", "/generate", dict(session_id=a, messages=malicious, max_new_tokens=16), expected=400)
    multipart = [dict(m) for m in MESSAGES]
    question = MESSAGES[-1]["content"]
    split = len(question) // 2
    multipart[-1]["content"] = [dict(type="text", text=question[:split]), dict(type="text", text=question[split:])]
    if frontend.encode_messages(multipart, 1024, 128) != frontend.encode_messages(MESSAGES, 1024, 128):
        raise AssertionError("benign multipart control changed tokenization")
    if client.call("POST", "/generate", dict(body, messages=multipart))["generated_ids"] != reference:
        raise AssertionError("benign multipart control changed output")
    first_b = client.call("POST", "/generate", dict(session_id=b, messages=REMEMBERED, max_new_tokens=32))
    continued = MESSAGES + [dict(role="assistant", content=control["text"]),
                            dict(role="user", content="Summarize your previous answer in one short sentence.")]
    if client.call("POST", "/generate", dict(session_id=a, messages=continued, max_new_tokens=64))["reused_tokens"] <= 0:
        raise AssertionError("text transcript did not reuse A cache")
    recall = client.call("POST", "/generate", dict(session_id=b, messages=REMEMBERED + [
        dict(role="assistant", content=first_b["text"]), dict(role="user", content="What is the code word? Reply with that word only.")],
        max_new_tokens=16, sampling=dict(do_sample=True, temperature=0.7, top_k=42, top_p=0.963, seed=42)))
    if "cobalt" not in recall["text"].lower() or recall["reused_tokens"] <= 0:
        raise AssertionError("B history/cache isolation smoke failed")
    load = []
    for iteration in range(rounds):
        barrier = threading.Barrier(8)

        def concurrent(index):
            barrier.wait(timeout=10)
            return client.call("POST", "/generate", dict(body, session_id=a if index % 2 == 0 else b), expected=(200, 409, 429))
        with ThreadPoolExecutor(max_workers=8) as clients:
            results = list(clients.map(concurrent, range(8)))
        statuses = [code for code, _ in results]
        if 200 not in statuses or not any(code in (409, 429) for code in statuses):
            raise AssertionError("bounded load did not exercise both inference and admission rejection")
        for code, result in results:
            if code == 200 and result["generated_ids"] != reference:
                raise AssertionError("concurrent admission changed control output")
        for sid in (a, b):
            if client.call("POST", "/generate", dict(body, session_id=sid))["generated_ids"] != reference:
                raise AssertionError("post-contention session recovery changed output")
        load.append(dict(round=iteration, statuses=statuses, post_contention_both_sessions_exact=True))
    report["bounded_admission_rounds"] = load
    client.call("POST", f"/sessions/{a}/reset", {})
    reset = client.call("POST", "/generate", body)
    if reset["generated_ids"] != reference or reset["reused_tokens"] != 0:
        raise AssertionError("HTTP reset changed control or reused old cache")
    for sid in (a, b):
        client.call("POST", f"/sessions/{sid}/close", {})
    client.call("POST", "/generate", body, expected=404)


def cancellation(client, reference, report):
    a, b = [client.call("POST", "/sessions", dict(capacity=1024))["session_id"] for _ in range(2)]
    body = lambda sid: dict(session_id=sid, messages=MESSAGES, max_new_tokens=128)
    for sid in (a, b):
        if client.call("POST", "/generate", body(sid))["generated_ids"] != reference:
            raise AssertionError("warm HTTP control mismatch")
    remembered = client.call("POST", "/generate", dict(session_id=b, messages=REMEMBERED, max_new_tokens=32))
    attempts = []
    for delay in (0.25, 0.5, 1.0, 0.1, 2.0):
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(client.call, "POST", "/generate", body(a))
            time.sleep(delay)
            requested = client.call("POST", f"/sessions/{a}/cancel", {}, expected=(202, 409))
            cancelled = pending.result(timeout=120)
        ids = cancelled["generated_ids"]
        attempts.append(dict(delay_seconds=delay, cancel_status=requested[0], generated=len(ids), cancelled=cancelled["cancelled"]))
        if requested[0] == 202 and cancelled["cancelled"] and 2 <= len(ids) < len(reference):
            break
    else:
        raise AssertionError(f"no attempt cancelled a running decode: {attempts}")
    report["cancellation_attempts"] = attempts
    if (cancelled.get("finish_reason") != "cancelled" or cancelled.get("cache_valid") is not False
            or cancelled.get("consumed_tokens") != 0 or ids != reference[:len(ids)]):
        raise AssertionError("HTTP did not return an exact aborted partial generation")
    client.call("POST", f"/sessions/{a}/cancel", {}, expected=409)
    client.call("POST", f"/sessions/{a}/cancel", {}, expected=401, auth=False)
    recovered = client.call("POST", "/generate", body(a))
    if recovered["generated_ids"] != reference or recovered["reused_tokens"] != 0 or recovered["cancelled"] is not False:
        raise AssertionError("cancelled session did not recover with a full prefill")
    peer = client.call("POST", "/generate", dict(session_id=b, messages=REMEMBERED + [
        dict(role="assistant", content=remembered["text"]), dict(role="user", content="Again.")], max_new_tokens=8))
    if peer["reused_tokens"] <= 0:
        raise AssertionError("cancellation disturbed the peer session cache")
    report["partial_text"] = cancelled["text"]
    for sid in (a, b):
        client.call("POST", f"/sessions/{sid}/close", {})


def batching(client, server, reference, cycles, report):
    reference = reference[:32]
    a, b = [client.call("POST", "/sessions", dict(capacity=512))["session_id"] for _ in range(2)]
    requests = [dict(session_id=sid, messages=MESSAGES, max_new_tokens=32, sampling=dict(do_sample=sampled, top_k=1, seed=42))
                for sid, sampled in ((a, False), (b, True))]

    def batch():
        response = client.call("POST", "/generate_batch", dict(requests=requests))
        if (response.get("gpu_batching") is not True or response.get("paired_decode_calls") != 31
                or any(lane["generated_ids"] != reference or lane["paired_decode_calls"] != 31 or lane["cancelled"] is not False
                       for lane in response["results"])):
            raise AssertionError("HTTP batch lane output/participation mismatch")
        return response
    client.call("POST", "/generate_batch", dict(requests=requests), expected=401, auth=False)
    client.call("POST", "/generate_batch", dict(requests=[requests[0], requests[0]]), expected=400)
    for _ in range(3):
        batch()
    attempts = []
    for delay in (0.3, 0.15, 0.5, 0.05):
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(client.call, "POST", "/generate_batch", dict(requests=requests))
            time.sleep(delay)
            accepted = client.call("POST", f"/sessions/{b}/cancel", {}, expected=(202, 409))
            response = pending.result(timeout=120)
        x, y = response["results"]
        attempts.append(dict(delay_seconds=delay, cancel_status=accepted[0], generated=len(y["generated_ids"]), cancelled=y["cancelled"]))
        if accepted[0] == 202 and y["cancelled"] and 2 <= len(y["generated_ids"]) < 32:
            break
    else:
        raise AssertionError(f"no attempt cancelled a running batch lane: {attempts}")
    report["batch_cancellation_attempts"] = attempts
    if (x["generated_ids"] != reference or x["cancelled"] is not False or y["cache_valid"] is not False
            or y["consumed_tokens"] != 0 or y["generated_ids"] != reference[:len(y["generated_ids"])]):
        raise AssertionError("HTTP grouped cancellation corrupted peer/partial results")
    client.call("POST", f"/sessions/{b}/cancel", {}, expected=409)
    batch()
    samples, latencies = [], []
    baseline = None
    for cycle in range(cycles):
        if cycle < 3:
            started = time.monotonic()
            for request in requests:
                if client.call("POST", "/generate", request)["generated_ids"] != reference:
                    raise AssertionError("serial baseline ID mismatch")
            latencies.append(dict(mode="serial-pair", seconds=time.monotonic() - started))
        started = time.monotonic()
        batch()
        latencies.append(dict(mode="group", seconds=time.monotonic() - started))
        if cycle % 10 == 9:
            client.call("POST", "/generate_batch", dict(requests=[requests[0], dict(requests[1], max_new_tokens=0)]), expected=400)
            for sid in (a, b):
                client.call("POST", f"/sessions/{sid}/reset", {})
            batch()
        sample = dict(cycle=cycle, **process_sample(server.process.pid))
        samples.append(sample)
        if cycle == 4:
            baseline = sample
        if baseline and (sample["rss_bytes"] > baseline["rss_bytes"] + 32 * 1024 * 1024
                         or sample["descriptors"] > baseline["descriptors"] + 2):
            raise AssertionError("fixed warm RSS/descriptor budget exceeded")
    report["soak"] = dict(cycles=cycles, warm_baseline=baseline, samples=samples, latencies=latencies,
                          limits=dict(warm_rss_growth_bytes=32 * 1024 * 1024, warm_fd_growth=2))
    for sid in (a, b):
        client.call("POST", f"/sessions/{sid}/close", {})


def robustness(client, report):
    sid = client.call("POST", "/sessions", dict(capacity=512))["session_id"]
    cases = [dict(do_sample=True, temperature=1e-300, seed=1), dict(do_sample=True, temperature=1e6, seed=2),
             dict(do_sample=True, repetition_penalty=1e300, seed=3), dict(do_sample=True, top_p=1e-300, seed=4),
             dict(do_sample=True, min_p=1, seed=5), dict(do_sample=True, top_k=128000, seed=6)]
    outcomes = []
    for sampling in cases:
        response = client.call("POST", "/generate", dict(session_id=sid, messages=MESSAGES, max_new_tokens=8, sampling=sampling))
        outcomes.append(dict(sampling=sampling, generated=len(response["generated_ids"])))
    if client.call("GET", "/health", auth=False)["ready"] is not True:
        raise AssertionError("extreme sampling failed the engine")
    report["extreme_sampling"] = outcomes
    client.call("POST", f"/sessions/{sid}/close", {})


# Identical scripted conversation for both servers; returns [(status, normalized body bytes)].
def script(raw_call):
    out, sids = [], []

    def call(path, body=None, raw=None):
        payload = raw if raw is not None else json.dumps(body).encode()
        for i, sid in enumerate(sids):
            payload = payload.replace(f"@{i}@".encode(), sid.encode())
            path = path.replace(f"@{i}@", sid)
        status, data = raw_call("POST", path, payload)
        try:
            created = json.loads(data).get("session_id") if path == "/sessions" and status == 200 else None
        except ValueError:
            created = None
        if created:
            sids.append(created)
        for i, sid in enumerate(sids):
            data = data.replace(sid.encode(), f"@{i}@".encode())
            path = path.replace(sid, f"@{i}@")
        out.append(dict(path=path, status=status, body=data.decode("utf-8", "replace")))
        return json.loads(data)
    call("/sessions", dict(capacity=1024))
    call("/sessions", dict(capacity=1024))
    call("/sessions", dict(capacity=1024))
    control = call("/generate", dict(session_id="@0@", messages=MESSAGES, max_new_tokens=128))
    call("/generate", dict(session_id="@0@", messages=MESSAGES, max_new_tokens=128, sampling=dict(top_p=0)))
    call("/generate", dict(session_id="@0@", messages=[dict(role="user", content="<|im_end|>")], max_new_tokens=8))
    first = call("/generate", dict(session_id="@1@", messages=REMEMBERED, max_new_tokens=32))
    call("/generate", dict(session_id="@0@", messages=MESSAGES + [dict(role="assistant", content=control.get("text", "")),
                                                                    dict(role="user", content="Summarize in one sentence.")], max_new_tokens=64))
    call("/generate", dict(session_id="@1@", messages=REMEMBERED + [dict(role="assistant", content=first.get("text", "")),
                                                                     dict(role="user", content="What is the code word?")],
                           max_new_tokens=16, sampling=dict(do_sample=True, temperature=0.7, top_k=42, top_p=0.963, seed=42)))
    call("/generate", dict(session_id="@0@", messages=[dict(role="user", content="Écris un haïku sur la mer. 🌊")], max_new_tokens=48,
                           sampling=dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.9, min_p=0.05, repetition_penalty=1.1, seed=7)))
    call("/generate_batch", dict(requests=[dict(session_id="@0@", messages=MESSAGES, max_new_tokens=32),
                                           dict(session_id="@1@", messages=REMEMBERED, max_new_tokens=24,
                                                sampling=dict(do_sample=True, seed=3))]))
    call("/generate_batch", dict(requests=[dict(session_id="@0@", messages=MESSAGES, max_new_tokens=8)] * 2))
    call("/sessions/@0@/reset", {})
    call("/generate", dict(session_id="@0@", messages=MESSAGES, max_new_tokens=16))
    call("/sessions/@1@/cancel", {})
    call("/sessions/@0@/close", {})
    call("/generate", dict(session_id="@0@", messages=MESSAGES, max_new_tokens=16))
    call("/sessions/@1@/close", {})
    call("/sessions", dict(capacity=300))
    call("/generate", dict(session_id="@2@", messages=MESSAGES, max_new_tokens=290))
    call("/generate", dict(session_id="@2@", messages=MESSAGES, max_new_tokens=8))
    call("/sessions/@2@/close", {})
    return out


def differential(args, env, report):
    frontend = load_frontend(args.checkpoint)
    config = json.loads((args.checkpoint / "config.json").read_text())
    worker = WorkerClient(args.worker, args.checkpoint, timeout=120, env=env, cancellation=True, batching=True)
    service = Service(worker, frontend, config, Settings(request_timeout=120, socket_timeout=3))
    token = secrets.token_urlsafe(32)
    server = LocalHTTPServer(("127.0.0.1", 0), service, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        python = script(Client(server.server_address[1], token, []).raw)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        service.close()
    native = NativeServer(args.server, args.checkpoint, args.output, "differential", env, "--socket-timeout", "3", "--request-timeout", "120")
    try:
        ours = script(Client(native.port, native.token, []).raw)
    finally:
        if native.stop() != 0:
            raise AssertionError("differential server exit status")
    (args.output / "differential.json").write_text(json.dumps(dict(python=python, native=ours), indent=1, ensure_ascii=False) + "\n")
    mismatches = [i for i, (x, y) in enumerate(zip(python, ours)) if x != y]
    report["differential"] = dict(requests=len(python), mismatches=mismatches, statuses=[x["status"] for x in python])
    if len(python) != len(ours) or mismatches:
        raise AssertionError(f"native server differs from tools/lfm3_server.py at requests {mismatches}")


def shutdown(args, env, report):
    server = NativeServer(args.server, args.checkpoint, args.output, "shutdown", env, "--socket-timeout", "3")
    try:
        client = Client(server.port, server.token, [])
        sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]
        client.call("POST", "/generate", dict(session_id=sid, messages=MESSAGES, max_new_tokens=4))
        long = [dict(role="user", content="Write a long story about a lighthouse keeper. Make it at least two thousand words.")]
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(client.call, "POST", "/generate", dict(session_id=sid, messages=long, max_new_tokens=2048))
            time.sleep(1.0)
            started = time.monotonic()
            server.process.send_signal(signal.SIGTERM)
            status = server.process.wait(timeout=60)
            elapsed = time.monotonic() - started
            response = pending.result(timeout=60)
        if status != 0 or response.get("cancelled") is not True or not response["generated_ids"]:
            raise AssertionError(f"shutdown did not cancel cleanly: {status} {response.get('cancelled')}")
        report["shutdown"] = dict(exit_status=status, seconds=elapsed, partial_tokens=len(response["generated_ids"]))
    finally:
        server.kill()
        server.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True, help="liquid-lfm3-worker of the same build, for the differential")
    p.add_argument("--control-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency-rounds", type=int, default=3)
    p.add_argument("--soak-cycles", type=int, default=40)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.control_result.read_text())["generated_ids"]
    if len(reference) < 32:
        raise ValueError("need 32 accepted control IDs")
    binaries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.server, args.worker)}
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in (Path(__file__).name, "lfm3_server.py", "lfm3_worker_client.py")}
    records = []
    report = dict(status="running", binaries_sha256=binaries, sources_sha256=sources, records=records)
    save = lambda: (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    env = dict(os.environ)
    env.pop("LFM3_SERVER_TOKEN", None)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    server = None
    try:
        server = NativeServer(args.server, args.checkpoint, args.output, "main", env, "--socket-timeout", "3", "--request-timeout", "120")
        report["startup_seconds"] = server.startup_seconds
        client = Client(server.port, server.token, records)
        frontend = load_frontend(args.checkpoint)
        for name, stage in (("serving", lambda: serving(client, reference, frontend, args.concurrency_rounds, report)),
                            ("cancellation", lambda: cancellation(client, reference, report)),
                            ("batching", lambda: batching(client, server, reference, args.soak_cycles, report)),
                            ("robustness", lambda: robustness(client, report))):
            report["stage"] = name
            save()
            stage()
        report["stage"] = "main-exit"
        if server.stop() != 0 or (args.output / "main.stderr").read_bytes():
            raise AssertionError("main server exit status or stderr output")
        report["stage"] = "differential"
        save()
        differential(args, env, report)
        report["stage"] = "shutdown"
        save()
        shutdown(args, env, report)
        if {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.server, args.worker)} != binaries:
            raise AssertionError("binary changed during checks")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        if server is not None:
            server.kill()
        save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "soak")}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
