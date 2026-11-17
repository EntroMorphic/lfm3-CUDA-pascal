"""Unprofiled native decode benchmark with paired exact-output evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import threading
import time

from benchmark_common import CONTROL_PROMPT, assert_same_output, run_saved, sha256, validate_native, write_json


def environment_overrides(assignments: list[str]) -> dict[str, str]:
    result = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Expected an environment override NAME=VALUE: {assignment}")
        if name in result:
            raise ValueError(f"Duplicate environment override: {name}")
        result[name] = value
    return result


def summarize_pairs(runs: list[dict]) -> dict:
    variants = sorted({r["variant"] for r in runs})
    result = dict(median_rates={v: statistics.median(r["rate"] for r in runs if r["variant"] == v) for v in variants})
    if "baseline" in variants:
        ratios = []
        for repeat in sorted({r["repeat"] for r in runs}):
            pair = {r["variant"]: r["rate"] for r in runs if r["repeat"] == repeat}
            if len(pair) != 2 or sum(r["repeat"] == repeat for r in runs) != 2:
                raise ValueError("Incomplete/duplicate measured pair")
            ratios.append(pair["candidate"] / pair["baseline"])
        result.update(paired_ratios=ratios, median_paired_gain_percent=100 * (statistics.median(ratios) - 1))
    return result


def host_observation(pid, policies: list[Path]) -> dict:
    """Read only: policy clocks are kHz; affinity/current CPU refer to the child main thread."""
    row = dict(cpu_policies={})
    for policy in policies:
        values = {}
        for field in ("scaling_cur_freq", "scaling_governor", "affected_cpus"):
            try: values[field] = (policy / field).read_text().strip()
            except OSError: values[field] = None
        row["cpu_policies"][policy.name] = values
    row.update(process_pid=pid, process_affinity=None, process_cpu=None)
    if pid is not None:
        try: row["process_affinity"] = sorted(os.sched_getaffinity(pid))
        except (OSError, AttributeError): pass
        try:
            # comm may contain whitespace/parentheses. Field 39 follows the last ')'.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            row["process_cpu"] = int(fields[36])
        except (OSError, ValueError, IndexError): pass
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True, help="Uninstrumented candidate executable.")
    p.add_argument("--baseline", type=Path, help="Optional frozen executable; every row must match exactly.")
    p.add_argument("--model", type=Path, required=True, help="Complete local checkpoint directory.")
    p.add_argument("--output", type=Path, required=True, help="Fresh evidence directory; existing paths fail.")
    p.add_argument("--prompt", default=CONTROL_PROMPT)
    p.add_argument("--prompt-tokens", type=Path, help="Use existing IDs, preserving the exact historical input.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.528)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--pairs", type=int, default=5, help="Measured repetitions after full-length warmups.")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--reference-math", action="store_true")
    p.add_argument("--no-cuda-graph", action="store_true", help="Use eager decode for both variants.")
    p.add_argument("--baseline-no-cuda-graph", action="store_true", help="Use eager decode only for the paired baseline.")
    p.add_argument("--baseline-env", action="append", default=[], metavar="NAME=VALUE",
                   help="Repeatable baseline-only environment override; recorded in evidence.")
    p.add_argument("--runtime-env", action="append", default=[], metavar="NAME=VALUE",
                   help="Repeatable candidate-only environment override; recorded in evidence.")
    args = p.parse_args(argv)
    try:
        variant_environment = dict(baseline=environment_overrides(args.baseline_env),
                                   candidate=environment_overrides(args.runtime_env))
    except ValueError as error:
        p.error(str(error))
    if args.baseline_env and args.baseline is None:
        p.error("--baseline-env requires --baseline")
    if args.baseline_no_cuda_graph and args.baseline is None:
        p.error("--baseline-no-cuda-graph requires --baseline")
    if (not 1 <= args.batch_size <= 65535 or args.pairs < 1 or args.max_new_tokens < 1 or
            args.context_length < 1 or not 0 <= args.seed < 2**64 or
            not math.isfinite(args.temperature) or args.temperature < 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("Invalid batch, repetition count, cap, context, seed, temperature or timeout")
    args.runtime = args.runtime.resolve(); args.model = args.model.resolve()
    variants = [("candidate", args.runtime)]
    if args.baseline:
        variants.insert(0, ("baseline", args.baseline.resolve()))
    for _, runtime in variants:
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            p.error(f"Runtime is not executable: {runtime}")
    config = json.loads((args.model / "config.json").read_text())
    vocabulary = config["vocab_size"]
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(complete=False, scope="Unprofiled aggregate native decode; loading/prefill excluded from rate; subprocess wall time separately retained.",
                   platform=platform.platform(), settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   binary_sha256={name: sha256(binary) for name,binary in variants}, warmups=[], runs=[])
    try:
        if args.prompt_tokens:
            ids = list(map(int, args.prompt_tokens.read_text().split()))
            mode = "provided_ids"
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
            # Match the public frontend's known-base convention.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from liquid_inference.cli import model_name
            known = model_name(args.model.name)
            chat = bool(tokenizer.chat_template) and not (known and known.endswith("-Base"))
            text = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)], tokenize=False,
                                                 add_generation_prompt=True) if chat else args.prompt
            ids = tokenizer.encode(text, add_special_tokens=not chat); mode = "chat" if chat else "raw"
        if not ids or any(i < 0 or i >= vocabulary for i in ids):
            raise ValueError("Prompt must contain in-vocabulary token IDs")
        tokens = args.output / "prompt.tokens"; tokens.write_text("\n".join(map(str,ids)) + "\n")
        summary.update(prompt_mode=mode, prompt_token_sha256=sha256(tokens), prompt_tokens=len(ids),
                       config_sha256=sha256(args.model / "config.json"))
        write_json(args.output / "summary.json", summary)
        reference = None
        def run(label, variant):
            nonlocal reference
            name, runtime = variant
            directory = args.output / label
            command = [str(runtime), "--model", str(args.model), "--prompt-tokens", str(tokens),
                       "--tokens-output", str(directory / "tokens"), "--batch-size", str(args.batch_size),
                       "--temperature", str(args.temperature), "--seed", str(args.seed),
                       "--top-k", "0", "--top-p", "1", "--min-p", "0", "--repetition-penalty", "1",
                       "--context-length", str(args.context_length), "--max-new-tokens", str(args.max_new_tokens)]
            if args.reference_math: command.append("--attention-math-dynamic")
            if args.no_cuda_graph or (name == "baseline" and args.baseline_no_cuda_graph):
                command.append("--no-cuda-graph")
            stop = threading.Event(); samples=[]; child = {}; observation_errors=[]
            policies = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))
            def monitor():
                try:
                    while not stop.is_set():
                        row = dict(monotonic_ns=time.monotonic_ns())
                        for clock in ("gpu-gpc-0", "gpu-nvd-0"):
                            try: row[clock] = int((Path("/sys/class/devfreq") / clock / "cur_freq").read_text())
                            except OSError: row[clock] = None
                        row.update(host_observation(child.get("pid"), policies))
                        samples.append(row); stop.wait(.1)
                except Exception as error:
                    observation_errors.append(repr(error))
            thread = threading.Thread(target=monitor); thread.start()
            try: process = run_saved(command, directory, args.timeout,
                                     env={**os.environ, **variant_environment[name]},
                                     on_start=lambda pid: child.update(pid=pid))
            finally:
                stop.set(); thread.join()
                if directory.exists():
                    write_json(directory / "frequencies.json", samples)
                    write_json(directory / "observation-errors.json", observation_errors)
            if observation_errors:
                raise ValueError(f"Frequency observer failed: {observation_errors}")
            result = validate_native(directory, args.batch_size, vocabulary, args.max_new_tokens)
            if reference is None: reference = result
            assert_same_output(reference, result)
            result.update(label=label, variant=name, wall_seconds=process["wall_seconds"])
            print(f"{label}: {result['rate']:.3f} decode tok/s; exact saved rows", flush=True)
            return result
        for variant in variants:
            summary["warmups"].append(run("warmup-" + variant[0], variant)); write_json(args.output / "summary.json", summary)
        for repeat in range(1, args.pairs + 1):
            for variant in (variants if repeat % 2 else variants[::-1]):
                row = run(f"r{repeat}-" + variant[0], variant); row["repeat"] = repeat
                summary["runs"].append(row); write_json(args.output / "summary.json", summary)
        summary.update(summarize_pairs(summary["runs"]), complete=True)
        print(json.dumps({k:summary[k] for k in ("median_rates", "median_paired_gain_percent") if k in summary}, indent=2))
        return 0
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally: write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
