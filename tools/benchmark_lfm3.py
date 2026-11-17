"""Matched batch-one greedy measurements; exact gates, no adaptive tolerances.

Reference: initial oracle generation, one discarded full greedy warmup, then
three samples in one process. Native:
three fresh processes, each with one full greedy warmup then one timed sample.
These are backend-specific synchronized loop timings, not isolated kernel times.
Gate reports/checkpoint/executable are trusted local inputs, not attestations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys


MAX_JSON_BYTES = 4 * 1024 * 1024
SAMPLES = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, low=0, high=4096):
    require(type(value) is int and low <= value <= high, f"invalid {name}")
    return value


def finite(value, name, positive=False):
    require(type(value) in (int, float), f"invalid {name}")
    require(0 <= value <= sys.float_info.max and math.isfinite(value), f"invalid {name}")
    require(not positive or value > 0, f"{name} must be positive")
    return value


def read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_JSON_BYTES,
                f"{path}: expected bounded regular JSON file")
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "JSON file too large")
    # Explicit depth limit independent of interpreter recursion behavior.
    depth = 0
    quoted = escaped = False
    for char in raw.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            require(depth <= 64, "excessive JSON nesting")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(result, dict), "expected JSON object")
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gate(gate, binary_hash):
    require(gate.get("binary_sha256") == binary_hash, "binary differs from accepted gates")
    cases = gate.get("cases")
    require(isinstance(cases, list) and bool(cases), "gate cases must be nonempty")
    names = set()
    for case in cases:
        require(isinstance(case, dict), "invalid gate case")
        name = case.get("case")
        require(isinstance(name, str) and bool(name) and name not in names, "missing/duplicate gate case name")
        names.add(name)
        for key in ("greedy_exact", "teacher_argmax_exact", "reset_exact_logits"):
            require(case.get(key) is True, f"gate case {name}: {key} must be true")
        integer(case.get("tokens"), "gate tokens", 1)
        integer(case.get("activation_checks"), "gate activation_checks", 1, 2**31 - 1)
        for key in ("max_relative_rms", "max_logit_abs"):
            require(finite(case.get(key), key) == 0, f"gate case {name}: {key} is not exact")
    for key in ("stop_check", "suppression_check", "final_cache_slot_checked"):
        require(gate.get(key) is True, f"gate {key} must be true")
    integer(gate.get("invalid_requests_rejected"), "invalid_requests_rejected", 1, 2**31 - 1)


def ids(value, name, vocabulary, maximum=4096, nonempty=False):
    require(isinstance(value, list) and len(value) <= maximum, f"invalid {name} list")
    require(not nonempty or bool(value), f"empty {name}")
    for token in value:
        integer(token, name, 0, vocabulary - 1)
    return value


def make_request(source, reference, tokens, vocabulary):
    require(isinstance(source, dict) and isinstance(reference, dict), "invalid reference artifacts")
    input_ids = ids(source.get("input_ids"), "input_ids", vocabulary, nonempty=True)
    require(ids(reference.get("input_ids"), "reference input_ids", vocabulary) == input_ids,
            "reference/request input mismatch")
    capacity = integer(source.get("capacity"), "capacity", 1)
    require(len(input_ids) + tokens - 1 <= capacity, "request exceeds native cache capacity")
    suppress = ids(source.get("suppress_tokens"), "suppress_tokens", vocabulary, vocabulary)
    stop = ids(source.get("stop_tokens"), "stop_tokens", vocabulary, vocabulary)
    require(bool(stop) and not set(stop) & set(suppress) and len(set(suppress)) < vocabulary,
            "invalid stop/suppression policy")
    for key in ("suppress_tokens", "stop_tokens"):
        require(ids(reference.get(key), key, vocabulary, vocabulary) == source[key],
                f"reference/request {key} mismatch")
    expected = ids(reference.get("generated_ids"), "reference generated_ids", vocabulary, tokens, True)
    teacher = ids(source.get("teacher_ids"), "teacher_ids", vocabulary)
    require(teacher == expected[:-1], "reference teacher request mismatch")
    # Construct a whitelist instead of forwarding trace/teacher/unknown controls.
    return dict(input_ids=input_ids, capacity=capacity, suppress_tokens=suppress,
                stop_tokens=stop, max_new_tokens=tokens, warmup_runs=1, check_reset=False)


def validate_sequence(value, request, vocabulary):
    tokens = ids(value, "generated_ids", vocabulary, request["max_new_tokens"], True)
    require(not set(tokens) & set(request["suppress_tokens"]), "suppressed token generated")
    require(not set(tokens[:-1]) & set(request["stop_tokens"]), "tokens after stop")
    require(len(tokens) == request["max_new_tokens"] or tokens[-1] in request["stop_tokens"],
            "generation ended early without stop")
    return tokens


def validate_sample(sample, request, vocabulary, native=False):
    require(isinstance(sample, dict), "sample must be an object")
    tokens = validate_sequence(sample.get("generated_ids"), request, vocabulary)
    calls = integer(sample.get("decode_forward_calls"), "decode_forward_calls")
    require(calls == len(tokens) - 1, "decode_forward_calls must equal generated tokens minus one")
    finite(sample.get("prefill_seconds"), "prefill_seconds", positive=True)
    seconds = finite(sample.get("decode_seconds"), "decode_seconds", positive=bool(calls))
    if calls:
        require(math.isfinite(calls / seconds), "nonfinite decode throughput")
    if native:
        require(sample.get("backend") == "native-cuda-experimental", "unexpected native backend")
        # eager-bf16: the cuBLAS engine; eager-fp32-scores: the native-GEMM engine (G4), whose greedy parity with
        # Transformers is measured here as for any engine, not assumed.
        require(sample.get("attention") in ("eager-bf16", "eager-fp32-scores"), "unexpected native attention")
        require(ids(sample.get("input_ids"), "native input_ids", vocabulary) == request["input_ids"],
                "native input mismatch")
        require(sample.get("warmup_runs") == 1 and type(sample.get("warmup_runs")) is int,
                "native must report one warmup")
        require(sample.get("tracing") is False and sample.get("reset_checked") is False,
                "native timing includes tracing/reset")
        require(sample.get("reset_exact_logits_ok") is None, "unexpected reset check")


def mismatch(actual, expected):
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            break
    else:
        if len(actual) == len(expected):
            return None
        index = min(len(actual), len(expected))
    return dict(first_mismatch=index, expected_length=len(expected), actual_length=len(actual),
                expected_token=expected[index] if index < len(expected) else None,
                actual_token=actual[index] if index < len(actual) else None)


def stats(rows):
    rates = [row["decode_forward_calls"] / row["decode_seconds"]
             if row["decode_forward_calls"] else None for row in rows]
    return dict(prefill_seconds_median=statistics.median(row["prefill_seconds"] for row in rows),
                decode_seconds_median=statistics.median(row["decode_seconds"] for row in rows),
                decode_tok_s_median=statistics.median(rates) if all(rate is not None for rate in rates) else None,
                generated_tokens_per_sample=[len(row["generated_ids"]) for row in rows],
                decode_forward_calls_per_sample=[row["decode_forward_calls"] for row in rows])


def validate_reference(reference, request, vocabulary, attention):
    require(reference.get("backend") == "Transformers" and reference.get("attention") == attention,
            "unexpected reference backend/attention")
    expected = validate_sequence(reference.get("generated_ids"), request, vocabulary)
    require(isinstance(reference.get("text"), str), "reference text missing")
    benchmark = reference.get("benchmark")
    require(isinstance(benchmark, dict), "reference benchmark missing")
    require(type(benchmark.get("warmup_runs")) is int and benchmark["warmup_runs"] == 1,
            "reference must report one warmup")
    ref_samples = benchmark.get("samples")
    require(isinstance(ref_samples, list) and len(ref_samples) == SAMPLES,
            "expected three reference samples")
    for sample in ref_samples:
        validate_sample(sample, request, vocabulary)
    return expected, ref_samples


def summarize(reference, samples, request, vocabulary, attention, binary_hash):
    expected, ref_samples = validate_reference(reference, request, vocabulary, attention)
    require(isinstance(samples, list) and len(samples) == SAMPLES, "expected three native samples")
    require(len({sample.get("attention") for sample in samples}) == 1, "native samples disagree on the engine label")
    mismatches = []
    for backend, rows in (("native", samples), ("transformers", ref_samples)):
        for index, sample in enumerate(rows):
            validate_sample(sample, request, vocabulary, native=backend == "native")
            difference = mismatch(sample["generated_ids"], expected)
            if difference is not None:
                mismatches.append(dict(backend=backend, sample=index, **difference))
    exact = not mismatches
    return dict(greedy_exact=exact, accepted_parity_benchmark=exact, mismatches=mismatches,
                first_mismatch=min((row["first_mismatch"] for row in mismatches), default=None),
                input_tokens=len(request["input_ids"]), max_new_tokens=request["max_new_tokens"],
                reference_attention=attention, native_attention=samples[0].get("attention"),
                eager_parity_claim=("applies" if samples[0].get("attention") == "eager-bf16" else
                                    "does not apply (native-GEMM engine, G4); greedy agreement is measured, not assumed"),
                native=stats(samples), transformers=stats(ref_samples), samples=SAMPLES,
                native_warmup_runs_per_process=1, native_processes=SAMPLES,
                reference_warmup_runs_total=1, reference_initial_oracle_generation=True,
                reference_samples_same_process=True,
                warmup_scope="full greedy generation, same requested stopping/suppression policy; reference additionally runs initial oracle generation before its discarded warmup; native resets cache after each process warmup",
                binary_sha256=binary_hash,
                timing_scope="synchronized batch-1 backend-specific manual greedy loops; prefill separate; decode forward calls/sec; no traces; excludes loading and text decoding; zero decode calls => null throughput",
                limitations="native host logits/argmax and contiguous cache; reference Python token allocations/argmax and dynamic cache; different synchronization/loop boundaries; sequential workloads; no speedup or isolated-kernel claim; gate report not checkpoint-attested")


def run(args):
    integer(args.tokens, "tokens", 1)
    binary = args.binary.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    require(binary.is_file() and os.access(binary, os.X_OK), "binary must be executable")
    require(checkpoint.is_dir(), "checkpoint must be a directory")
    require(not args.output.exists() and not args.output.is_symlink(), "output must be a fresh directory")
    binary_hash = sha256(binary)
    validate_gate(read_json(args.gates), binary_hash)
    config = read_json(checkpoint / "config.json")
    text = config.get("text_config", config)
    require(isinstance(text, dict), "invalid text config")
    vocabulary = integer(text.get("vocab_size"), "vocab_size", 1, 2**31 - 1)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = args.output.resolve()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    def command(argv, log):
        with log.open("x") as stream:
            subprocess.run([str(x) for x in argv], env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=3600)

    command([sys.executable, Path(__file__).with_name("lfm3_reference.py"),
             "--checkpoint", checkpoint, "--output", output / "oracle", "--tokens", args.tokens,
             "--attention", args.attention, "--no-trace", "--benchmark"], output / "reference.log")
    reference = read_json(output / "oracle/reference.json")
    request = make_request(read_json(output / "oracle/request.json"), reference, args.tokens, vocabulary)
    # Fail malformed reference evidence before launching native GPU work.
    validate_reference(reference, request, vocabulary, args.attention)
    write_json(output / "request.json", request)
    samples = []
    for index in range(SAMPLES):
        command([binary, "--experimental", checkpoint, output / "request.json", output / f"native-{index}.json"],
                output / f"native-{index}.log")
        sample = read_json(output / f"native-{index}.json")
        validate_sample(sample, request, vocabulary, native=True)
        samples.append(sample)
    require(sha256(binary) == binary_hash, "binary changed during benchmark")
    report = summarize(reference, samples, request, vocabulary, args.attention, binary_hash)
    # Decoding is ancillary; persist parity/timing evidence before importing HF.
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["greedy_exact"]:
        raise ValueError("Greedy sequence mismatch; timings retained but not an accepted parity benchmark")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    with (output / "native-response.txt").open("x", encoding="utf-8") as stream:
        stream.write(tokenizer.decode(samples[0]["generated_ids"], skip_special_tokens=True) + "\n")
    with (output / "reference-response.txt").open("x", encoding="utf-8") as stream:
        stream.write(reference["text"] + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        print(f"benchmark_lfm3: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
