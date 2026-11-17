"""Strict evidence helpers shared by native benchmark and profiling tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


CONTROL_PROMPT = "Hypothetically, might reflective recursion be a function of cognition?"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_saved(command: list[str], directory: Path, timeout: float | None, *, env=None, on_start=None) -> dict:
    """Fresh evidence; None omits the deadline. Kill/reap the child group on failure."""
    directory.mkdir(exist_ok=False)
    write_json(directory / "command.json", command)
    result = dict(returncode=None, timed_out=False, complete=False)
    started = time.monotonic()
    try:
        with (directory / "stdout.txt").open("x") as stdout, (directory / "stderr.txt").open("x") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       start_new_session=True, env=env)
            try:
                result["pid"] = process.pid
                if on_start is not None: on_start(process.pid)
                result["returncode"] = process.wait(timeout=timeout)
            except BaseException as error:
                result.update(error=repr(error), timed_out=isinstance(error, subprocess.TimeoutExpired))
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                result["returncode"] = process.wait()
                raise
        if result["returncode"]:
            raise RuntimeError(f"Process exited {result['returncode']}; see {directory / 'stderr.txt'}")
        result["complete"] = True
        return result
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(directory / "process.json", result)


def header_fields(stdout: str) -> dict[str, str]:
    header = stdout.split("\ntext_begin\n", 1)[0].split("\nbatch_text_begin\n", 1)[0]
    result = {}
    for line in header.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in result:
            raise ValueError(f"Duplicate native header: {key}")
        result[key] = value
    return result


def fnv1a(tokens: list[int]) -> str:
    value = 1469598103934665603
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def validate_native(directory: Path, batch: int, vocabulary: int, cap: int) -> dict:
    fields = header_fields((directory / "stdout.txt").read_text())
    counts = list(map(int, fields["generated_tokens" if batch == 1 else "batch_sequence_tokens"].split()))
    hashes = fields["generated_fnv1a" if batch == 1 else "batch_generated_fnv1a"].split()
    reasons = fields["generation_end" if batch == 1 else "batch_sequence_end"].split()
    if not (len(counts) == len(hashes) == len(reasons) == batch):
        raise ValueError("Incomplete native per-row metadata")
    token_digests = []
    for row in range(batch):
        path = directory / ("tokens" if batch == 1 else f"tokens.{row}.txt")
        tokens = list(map(int, path.read_text().split()))
        if not 0 < len(tokens) == counts[row] <= cap:
            raise ValueError("Invalid or incomplete native token count")
        if any(token < 0 or token >= vocabulary for token in tokens):
            raise ValueError("Native token outside checkpoint vocabulary")
        if fnv1a(tokens) != hashes[row]:
            raise ValueError("Saved tokens disagree with native hash")
        if reasons[row] not in ("stop_token", "token_limit", "context_limit"):
            raise ValueError("Unknown native termination reason")
        if reasons[row] == "token_limit" and len(tokens) != cap:
            raise ValueError("Token-limit result has wrong length")
        token_digests.append(hashlib.sha256(b"".join(t.to_bytes(4, "little") for t in tokens)).hexdigest())
    rate = float(fields["generated_tokens_per_second" if batch == 1 else "batch_tokens_per_second"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Native throughput must be positive and finite")
    if batch > 1 and int(fields["batch_generated_tokens"]) != sum(counts):
        raise ValueError("Native aggregate token count disagrees with rows")
    result = dict(rate=rate, counts=counts, hashes=hashes, token_sha256=token_digests,
                  reasons=reasons, total_tokens=sum(counts), attention_mode=fields["attention_mode"])
    if "batch_graph_mode" in fields:
        if fields["batch_graph_mode"] not in ("forward", "eager"):
            raise ValueError("Unknown batch graph mode")
        setup = float(fields["batch_graph_setup_seconds"])
        if not math.isfinite(setup) or setup < 0:
            raise ValueError("Invalid batch graph setup time")
        result.update(batch_graph_mode=fields["batch_graph_mode"], batch_graph_setup_seconds=setup)
    return result


def assert_same_output(first: dict, second: dict) -> None:
    for key in ("counts", "hashes", "token_sha256", "reasons"):
        if first[key] != second[key]:
            raise ValueError(f"Native output mismatch: {key}")
