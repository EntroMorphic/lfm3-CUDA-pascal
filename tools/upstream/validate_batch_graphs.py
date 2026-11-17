"""Sequential Thor graph quality gates and unprofiled paired controls (fresh output)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from benchmark_common import assert_same_output, run_saved, sha256, validate_native, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True, help="Previous full-suite benchmarks directory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    models = args.models.resolve(); historical = args.historical.resolve()
    build = output / "build"
    binary = build / "liquid-native"
    env = dict(os.environ, LIQUID_BUILD_DIR=str(build), LIQUID_PROFILE_NVTX="0",
               LIQUID_TEST_MODEL=str(models / "LFM2.5-350M"), LIQUID_TEST_NATIVE=str(binary))
    report = dict(complete=False, gates=[], benchmarks=[])
    def save(): write_json(output / "campaign.json", report)
    def run(name, command, timeout=1800):
        result = run_saved(command, output / name, timeout, env=env)
        report["gates"].append(dict(name=name, **result)); save()
        print(name, "PASS", flush=True)
    try:
        run("build-runtime", ["bash", str(root / "native/build.sh")], 600)
        report["binary_sha256"] = sha256(binary); save()
        # pytest paths are absolute so the runner is independent of cwd.
        run("quality", [sys.executable, "-m", "pytest", "-q", *[
            str(root / "tests" / name) for name in (
                "test_batch_graphs.py", "test_batch_coherency.py", "test_dynamic_native.py")]], 3600)
        fixture = build / "batch-graphs"
        run("build-fixture", ["nvcc", "-O3", "-std=c++20", "--expt-relaxed-constexpr",
            "--default-stream", "per-thread", "-arch=sm_110", "-I" + str(root / "native/include"),
            str(root / "native/tests/batch_graphs.cu"), str(root / "native/src/safetensors.cpp"),
            str(root / "native/src/tokenizer.cpp"), "-lcublas", "-lcublasLt", "-lpcre2-8", "-o", str(fixture)], 600)
        for tool in ("memcheck", "initcheck"):
            name = "fixture-" + tool
            run(name, ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", str(fixture),
                "--model", str(models / "LFM2.5-350M"), "--batch-size", "2", "--context-length", "96",
                "--max-new-tokens", "34", "--temperature", ".528"])
            log = (output / name / "stdout.txt").read_text() + (output / name / "stderr.txt").read_text()
            if "ERROR SUMMARY: 0 errors" not in log or "Internal Sanitizer Error" in log:
                raise ValueError("Invalid sanitizer gate: " + name)
        # Full outputs against the unchanged original campaign, including MoE fallbacks.
        previous_summaries = sorted(historical.glob("*/summary.json"))
        if len(previous_summaries) != 15:
            raise ValueError("Expected all fifteen historical checkpoint summaries")
        for previous in previous_summaries:
            name = previous.parent.name
            old = json.loads(previous.read_text())
            directory = output / ("suite-" + name)
            run_saved([str(binary), "--model", str(models / name), "--prompt-tokens", str(previous.parent / "prompt.tokens"),
                "--tokens-output", str(directory / "tokens"), "--batch-size", "32", "--context-length", "8192",
                "--max-new-tokens", "512", "--temperature", ".528", "--seed", "42"], directory, 1800, env=env)
            config = json.loads((models / name / "config.json").read_text())
            result = validate_native(directory, 32, config["vocab_size"], 512)
            assert_same_output(old["runs"][0], result)
            report["gates"].append(dict(name="suite-" + name, exact_historical_output=True)); save()
            print("suite-" + name, "EXACT", flush=True)
        cases = [("230M", 32), ("350M", 32), ("350M", 64), ("1.2B-Instruct", 32), ("2.6B", 32)]
        for suffix, batch in cases:
            name = f"paired-{suffix}-b{batch}"
            run(name, [sys.executable, str(root / "tools/upstream/benchmark_decode.py"),
                "--runtime", str(binary), "--baseline", str(binary), "--baseline-no-cuda-graph",
                "--model", str(models / ("LFM2.5-" + suffix)), "--prompt-tokens",
                str(historical / ("LFM2.5-" + suffix) / "prompt.tokens"),
                "--output", str(output / (name + "-evidence")), "--batch-size", str(batch),
                "--temperature", ".528", "--seed", "42", "--context-length", "8192",
                "--max-new-tokens", "512", "--pairs", "5"], 7200)
            summary = json.loads((output / (name + "-evidence") / "summary.json").read_text())
            report["benchmarks"].append(dict(name=name, median_rates=summary["median_rates"],
                paired_ratios=summary["paired_ratios"], median_paired_gain_percent=summary["median_paired_gain_percent"]))
            save()
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally: save()


if __name__ == "__main__": main()
