"""Checkpoint integration gates. Run in the pinned PyTorch CUDA container on Thor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_trace_exact import exact_logit_steps  # noqa: E402


def comparator_args(args):
    """compare_lfm3_traces flags: native tiers record eager distances without judging them (teacher argmax stays
    exact); --activation-tolerance uses its default tolerances; otherwise bitwise equality with eager."""
    if args.native_tiers:
        return ["--max-relative-rms", "inf", "--max-logit-error", "inf"]
    return [] if args.activation_tolerance else ["--require-exact"]


def trace_manifest(native, generated_ids):
    """Tier 1: SHA-256 of every native trace file, and the free-running ids."""
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native.iterdir()) if p.is_file()}
    return dict(generated_ids=generated_ids, files=files)


def compare_manifest(golden, actual):
    golden = {k: v for k, v in golden.items() if not k.startswith("_")}  # provenance is not a fixture
    if set(golden) != set(actual):
        raise AssertionError(f"tier 1: fixture set differs from the pin: {sorted(set(golden) ^ set(actual))}")
    for case in golden:
        g, a = golden[case], actual[case]
        if g["generated_ids"] != a["generated_ids"]:
            raise AssertionError(f"tier 1: {case}: generated ids differ from the pin")
        if set(g["files"]) != set(a["files"]):
            raise AssertionError(f"tier 1: {case}: trace files differ from the pin: {sorted(set(g['files']) ^ set(a['files']))[:8]}")
        changed = [n for n in g["files"] if g["files"][n] != a["files"][n]]
        if changed:
            raise AssertionError(f"tier 1: {case}: {len(changed)} trace files differ from the pin, e.g. {changed[:4]}")


def finish_tiers(pin, golden, binary, manifest, fidelity):
    """Tier 2a must pass before tier 1 pins anything; then pin (with provenance) or compare with the pin."""
    failed = [f["case"] for f in fidelity if not f["passed"]]
    if failed:
        raise AssertionError(f"tier 2a: native farther from FP32 than eager and SDPA in {failed}")
    if pin:
        pinned = dict(manifest)
        pinned["_provenance"] = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                     tier2a=fidelity, plan="validation/g4-gates-v2/PLAN.md")
        pin.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")
    else:
        compare_manifest(json.loads(golden.read_text()), manifest)


def tier2a(tools, output, name, run):
    """Tier 2a: native's median activation distance from the FP32 anchor ≤ the worse of eager's and SDPA's."""
    out = output / f"{name}.fidelity.json"
    # compare_trace_spread reads a candidate ROOT/SUBDIR as ROOT/CASE/SUBDIR: pass roots, not case directories.
    anchor = json.loads((output / "fp32" / name / "reference.json").read_text())
    precision = anchor.get("precision", {})
    if anchor.get("dtype") != "float32" or precision.get("matmul_allow_tf32") or precision.get("cudnn_allow_tf32"):
        raise AssertionError(f"tier 2a: {name}: the anchor is not recorded as FP32 with TF32 disabled")
    run([sys.executable, tools / "compare_trace_spread.py", output / "fp32", out, "--cases", name,
         "--candidate", "eager", output / "reference", output,
         "--candidate", "sdpa", output / "sdpa" / "reference", output / "sdpa",
         "--candidate", "native", output / "native", output], output / f"{name}.fidelity.log")
    row = json.loads(out.read_text())["cases"][0]
    c = row["candidates"]
    median = {k: c[k]["activation_rel_rms_median"] for k in ("eager", "sdpa", "native")}
    if any(v is None for v in median.values()):
        raise AssertionError(f"tier 2a: {name}: no common activation files")
    return dict(case=name, common_files=row["common_files"], compared_through_step=row["compared_through_step"],
                median_rel_rms=median, passed=median["native"] <= max(median["eager"], median["sdpa"]))


def check_exact_logits(native, activation_tolerance, name):
    """Native-GEMM traces carry the exact FP32 logits the engine used: all steps or none, and each .bf16 trace must
    be their rounding (lfm3_trace_exact). Tolerance mode exists for that engine, so there they are required."""
    steps = sorted(int(p.name.split(".")[0]) for p in native.glob("*.logits.bf16"))
    exact = exact_logit_steps(native, steps)
    if activation_tolerance and not exact:
        raise AssertionError(f"{name}: --activation-tolerance requires the engine's exact .logits.f32 traces")
    return exact

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--activation-tolerance", action="store_true",
                   help="Judge activations by compare_lfm3_traces tolerances instead of bitwise BF16 equality with eager "
                        "(the native-GEMM build, G4). Teacher argmax, free-running greedy, reset, stop and suppression "
                        "stay exact.")
    p.add_argument("--native-tiers", action="store_true",
                   help="G4 gates for the native-GEMM engine (validation/g4-gates-v2/PLAN.md): tier 1, bitwise against a "
                        "pinned manifest of its own traces (--golden or --pin); tier 2a, median activation distance from an "
                        "FP32 anchor no larger than the worse of HF eager and HF SDPA, and tokens exact against eager. "
                        "Distances from eager are recorded, not judged.")
    pin = p.add_mutually_exclusive_group()
    pin.add_argument("--golden", type=Path, help="tier 1: the pinned manifest to compare against")
    pin.add_argument("--pin", type=Path, help="tier 1: write the manifest of this run (only if tier 2a passes)")
    args = p.parse_args()
    if args.native_tiers:
        args.activation_tolerance = True
        if not (args.golden or args.pin):
            p.error("--native-tiers needs --golden or --pin")
        if args.pin and args.pin.exists():
            p.error(f"--pin {args.pin} exists; re-pinning is deliberate: remove it first")
        if args.golden:
            golden = json.loads(args.golden.read_text())  # fail now, not after hours
            if not golden:
                p.error(f"--golden {args.golden} is empty")
    elif args.golden or args.pin:
        p.error("--golden and --pin need --native-tiers")
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    env = dict(os.environ)
    # Use precisely the reference container's cuBLAS/runtime, not another image's version.
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    def run(command, log):
        with log.open("w") as stream:
            result = subprocess.run([str(x) for x in command], env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): see {log}")

    reports, manifest, fidelity = [], {}, []
    for name, extra, steps in [
        ("baseline", [], 16),
        ("arithmetic", ["--prompt", "What is 17 plus 25? Answer briefly."], 16),
        ("window257", ["--length", "257"], 4),
        ("single", ["--length", "1"], 4),
    ]:
        directory = args.output / name
        run([sys.executable, tools / "lfm3_reference.py", "--checkpoint", args.checkpoint,
             "--output", directory, "--tokens", steps, *extra], args.output / (name + ".reference.log"))
        run([args.binary, "--experimental", args.checkpoint, directory / "request.json", directory / "native.json"],
            directory / "native.log")
        run([sys.executable, tools / "compare_lfm3_traces.py", directory, *comparator_args(args)], directory / "comparison.log")
        request = json.loads((directory / "request.json").read_text())
        expected = json.loads((directory / "reference.json").read_text())["generated_ids"]
        request.pop("teacher_ids")
        request.pop("trace_directory")
        request["max_new_tokens"] = len(expected)
        # Fill the last legal slot on the final decode (no spare cache capacity).
        request["capacity"] = len(request["input_ids"]) + len(expected) - 1
        (directory / "generation-request.json").write_text(json.dumps(request))
        run([args.binary, "--experimental", args.checkpoint, directory / "generation-request.json", directory / "generation.json"],
            directory / "generation.log")
        generation = json.loads((directory / "generation.json").read_text())
        assert generation["generated_ids"] == expected, f"free-running token mismatch: {name}"
        assert generation["reset_exact_logits_ok"], name
        assert not set(expected) & set(request["suppress_tokens"])
        exact_logits = check_exact_logits(directory / "native", args.activation_tolerance, name)
        if args.native_tiers:
            manifest[name] = trace_manifest(directory / "native", generation["generated_ids"])
            for attention, flags in (("sdpa", ["--attention", "sdpa"]), ("fp32", ["--dtype", "float32"])):
                run([sys.executable, tools / "lfm3_reference.py", "--checkpoint", args.checkpoint,
                     "--output", args.output / attention / name, "--tokens", steps, *flags, *extra],
                    args.output / f"{name}.{attention}.reference.log")
            fidelity.append(tier2a(tools, args.output, name, run))
        comparison = json.loads((directory / "comparison.json").read_text())
        reports.append(dict(case=name, activation_checks=comparison["checks"], tokens=len(expected),
                            max_relative_rms=max(x["relative_rms"] for x in comparison["results"]),
                            max_logit_abs=max(x["max_abs"] for x in comparison["results"] if ".logits." in x["name"]),
                            teacher_argmax_exact=True, greedy_exact=True, reset_exact_logits=True,
                            exact_fp32_logit_traces=exact_logits))
        print(name, "PASS", flush=True)
    # Deterministic stopping and suppression: use known first argmax as stop.
    directory = args.output / "baseline"
    request = json.loads((directory / "generation-request.json").read_text())
    first = json.loads((directory / "generation.json").read_text())["generated_ids"][0]
    request["stop_tokens"] = [first]
    (directory / "stop-request.json").write_text(json.dumps(request))
    run([args.binary, "--experimental", args.checkpoint, directory / "stop-request.json", directory / "stop.json"], directory / "stop.log")
    assert json.loads((directory / "stop.json").read_text())["generated_ids"] == [first]
    request["suppress_tokens"].append(first)
    request["max_new_tokens"] = 1
    (directory / "suppress-request.json").write_text(json.dumps(request))
    run([args.binary, "--experimental", args.checkpoint, directory / "suppress-request.json", directory / "suppress.json"], directory / "suppress.log")
    assert json.loads((directory / "suppress.json").read_text())["generated_ids"][0] != first

    # Request contract failures must happen before loading checkpoint weights.
    invalid = [dict(input_ids=[]), dict(input_ids=[-1]), dict(input_ids=[128000]),
               dict(input_ids=[1], capacity=0), dict(input_ids=[1], capacity=4097),
               dict(input_ids=[1], max_new_tokens=True), dict(input_ids=[1], capacity=1, teacher_ids=[2]),
               dict(input_ids=[1], max_new_tokens=1, teacher_ids=[2])]
    for i, req in enumerate(invalid):
        path = args.output / f"invalid-{i}.json"
        path.write_text(json.dumps(req))
        result = subprocess.run([str(args.binary), "--experimental", str(args.checkpoint), str(path), str(args.output / f"invalid-{i}-output.json")],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 1, (i, result.stderr)
        assert not (args.output / f"invalid-{i}-output.json").exists()
    if args.native_tiers:
        finish_tiers(args.pin, args.golden, args.binary, manifest, fidelity)
    report = dict(cases=reports, stop_check=True, suppression_check=True, invalid_requests_rejected=len(invalid),
                  native_tiers=dict(tier1=("pinned " + str(args.pin)) if args.pin else ("matches " + str(args.golden)),
                                    tier2a=fidelity) if args.native_tiers else None,
                  final_cache_slot_checked=True,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  activations=("native tiers: eager distances recorded, not judged; tier 2a against FP32" if args.native_tiers
                               else "tolerance (compare_lfm3_traces defaults)" if args.activation_tolerance else "bitwise BF16"),
                  scope="actual BF16 checkpoint activations/logits, teacher forcing and free-running greedy; eager attention only")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
