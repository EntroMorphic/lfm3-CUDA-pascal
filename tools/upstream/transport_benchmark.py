#!/usr/bin/env python3
"""Measure what the transport's batching is worth.

Sends the same requests two ways through the loopback transport: one at a
time, and as a burst. The replies are identical either way; the wall time is
not.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("LIQUID_TEST_MODEL", ROOT / "models/LFM2.5-350M"))
RUNTIME = Path(os.environ.get("LIQUID_TEST_NATIVE", ROOT / "native/build/liquid-native"))
EVIDENCE = Path("/tmp/transport-benchmark")
PROMPTS = [
    "Hypothetically, might reflective recursion be a function of cognition?",
    "Explain what machine learning is, and why it matters.",
    "What is the difference between a process and a thread?",
    "Write a haiku about a copper kettle.",
    "Describe the water cycle.",
    "What makes a good API design?",
    "Summarize the theory of relativity.",
    "Give three tips for debugging.",
]


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def post(port, prompt, timeout=900):
    body = json.dumps(dict(model="lfm350", messages=[dict(role="user", content=prompt)])).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-ms", type=int, default=None)
    parser.add_argument("--burst", type=int, default=len(PROMPTS))
    options = parser.parse_args()
    if EVIDENCE.exists():
        subprocess.run(["rm", "-rf", str(EVIDENCE)], check=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/native_openai.py"),
         "--runtime", str(RUNTIME), "--model-dir", str(MODEL),
         "--evidence", str(EVIDENCE), "--port", str(port), "--max-requests", "512"]
        + (["--batch-window-ms", str(options.window_ms)] if options.window_ms is not None else []),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    else:
        server.terminate()
        raise SystemExit("the transport did not start: " + server.stderr.read())

    try:
        # Warm the weights up so neither measurement pays the first-call cost.
        post(port, PROMPTS[0])

        burst = options.burst
        start = time.monotonic()
        for index in range(burst):
            post(port, PROMPTS[index % len(PROMPTS)])
        sequential = time.monotonic() - start

        replies = [None] * burst
        errors = []

        def send(index):
            try:
                replies[index] = post(port, PROMPTS[index % len(PROMPTS)])
            except Exception as error:
                errors.append(error)

        start = time.monotonic()
        threads = [threading.Thread(target=send, args=(index,))
                   for index in range(burst)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent = time.monotonic() - start
        if errors:
            raise SystemExit(f"a request failed: {errors}")

        tokens = sum(reply["usage"]["completion_tokens"] for reply in replies if reply)
        batches = sorted(EVIDENCE.glob("batch-*"))
        sizes = []
        for directory in batches:
            command = json.loads((directory / "native-command.json").read_text())
            sizes.append(int(command[command.index("--batch-size") + 1])
                         if "--batch-size" in command else 1)
        print(f"requests              : {burst}")
        print(f"completion tokens     : {tokens}")
        print(f"sequential wall time  : {sequential:.2f} s  "
              f"({tokens / sequential:.0f} tok/s)")
        print(f"concurrent wall time  : {concurrent:.2f} s  "
              f"({tokens / concurrent:.0f} tok/s)")
        print(f"speed-up              : {sequential / concurrent:.2f}x")
        print(f"batch sizes           : {sizes}")
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    main()
