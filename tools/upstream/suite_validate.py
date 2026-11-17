#!/usr/bin/env python3
"""Offline validation matrix for a pinned suite_stage.py inventory.

Preserves every failed subprocess/report. Never downloads checkpoints or
weakens acceptance thresholds. Generation is bounded: a cap is not natural EOS.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from coherency_test import field_of, read_tokens, token_hash, valid_termination
from suite_stage import digest


def cases(name):
    if name.endswith("-Base"):
        return [("continuation", "raw", "The scientific method is a systematic approach to understanding the world. Its main steps are"),
                ("unicode", "raw", "日本の四季について説明します。春は")]
    return [("english", "chat", "Hypothetically, might reflective recursion be a function of cognition?"),
            ("japanese", "chat", "日本の四季の特徴を、それぞれ一文で説明してください。")]


def verify_checkpoint(checkpoint, model):
    """Verify every staged file, not just the large weight payloads."""
    if not model["files"]:
        raise RuntimeError("empty checkpoint file inventory")
    for item in model["files"]:
        if Path(item["name"]).name != item["name"]:
            raise RuntimeError("non-sibling checkpoint filename")
        path = checkpoint / item["name"]
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"checkpoint size mismatch: {path}")
        staged_sha, upstream_sha = item.get("sha256"), item["upstream_sha256"]
        if staged_sha and upstream_sha and staged_sha != upstream_sha:
            raise RuntimeError(f"staged and upstream digests disagree: {path}")
        expected_sha = staged_sha or upstream_sha
        if not expected_sha:
            raise RuntimeError(f"missing staged file digest: {path}")
        if digest(path) != expected_sha:
            raise RuntimeError(f"checkpoint digest mismatch: {path}")
    if json.loads((checkpoint / "config.json").read_text()) != model["config"]:
        raise RuntimeError("checkpoint config does not match pinned inventory")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="Checkpoint basenames; default all standalone models")
    p.add_argument("--generated-prefix", type=int, default=64)
    p.add_argument("--generation-cap", type=int, default=512)
    p.add_argument("--attention-math-dynamic", action="store_true")
    args = p.parse_args()
    if args.generated_prefix < 1 or args.generation_cap < 1:
        p.error("generation limits must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.inventory.read_text())
    selected = [m for m in inventory["models"] if m["category"] == "standalone" and
                (not args.only or m["id"].split("/")[1] in args.only)]
    if not selected or (args.only and set(args.only) != {m["id"].split("/")[1] for m in selected}):
        p.error("empty or incomplete requested inventory")
    runtime = (args.output / "liquid-native").resolve()
    shutil.copy2(args.runtime, runtime)
    sha = digest(runtime)
    report = dict(runtime_sha256=sha, all_gates_passed=False, models=[],
                  attention_math_dynamic=args.attention_math_dynamic,
                  scope="Pinned BF16 checkpoints; two prompts, bounded generation, context capacity 8192. "
                        "Default dense reference uses existing numerical bounds; MoE and opt-in dynamic math require byte parity. "
                        "This does not certify full advertised context or all possible inputs.")
    tool = Path(__file__).with_name("model_validate.py")

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def run(directory, name, command):
        (directory / (name + ".command.json")).write_text(json.dumps(list(map(str, command)), indent=2) + "\n")
        try:
            r = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired as exc:
            (directory / (name + ".timeout.txt")).write_text(str(exc))
            raise
        (directory / (name + ".stdout")).write_text(r.stdout)
        (directory / (name + ".stderr")).write_text(r.stderr)
        (directory / (name + ".returncode")).write_text(str(r.returncode) + "\n")
        r.check_returncode()
        return r.stdout

    for model in selected:
        name = model["id"].split("/")[1]
        directory = args.output / name
        directory.mkdir()
        checkpoint = args.models / name
        row = dict(id=model["id"], revision=model["revision"], passed=False, stages={}, cases=[])
        report["models"].append(row)
        save()
        try:
            # Recheck files on the offline host rather than trusting SCP alone.
            verify_checkpoint(checkpoint, model)
            row["stages"]["file_integrity"] = True
            text = run(directory, "upload", [args.probe, checkpoint, "--verify-all"])
            count, payload, allocated = map(int, text.splitlines()[0].split())
            if count <= 0 or payload <= 0 or allocated < payload or (
                    f"all_tensor_uploads_byte_identical: {count} tensors, {payload} bytes" not in text):
                raise RuntimeError("missing complete upload byte oracle")
            row["stages"]["upload_byte_identical"] = True
            row["payload_bytes"] = payload
            for tag, mode, prompt in cases(name):
                case = dict(name=tag, prompt_mode=mode, passed=False)
                row["cases"].append(case)
                validation = directory / (tag + "-reference")
                options = ["--require-byte-parity"] if model["model_type"] == "lfm2_moe" or args.attention_math_dynamic else []
                if args.attention_math_dynamic:
                    options += ["--attention-math-dynamic"]
                run(directory, tag + "-reference", [sys.executable, tool, "--runtime", runtime,
                    "--model", checkpoint, "--output", validation, "--prompt", prompt,
                    "--prompt-mode", mode, "--context-length", "8192", "--reference-batch-size", "32",
                    "--generated-prefix", str(args.generated_prefix), *options])
                reference = json.loads((validation / "report.json").read_text())
                if not reference["passed"] or not reference["native_single_batch_byte_identical"]:
                    raise RuntimeError("reference report did not pass")
                case["reference_byte_identical"] = reference["reference_bf16_byte_identical"]
                case["prefix_agreement"] = reference["generated_prefix_agreement"]
                agreement = case["prefix_agreement"]
                if (agreement["cases"] < 1 or agreement["native_matches"] != agreement["cases"] or
                        agreement["reference_matches"] != agreement["cases"]):
                    raise RuntimeError("generated prefix disagrees with native/reference teacher choices")
                case["max_relative_l2"] = max(m["max_relative_l2"] for m in reference["metrics"])
                case["max_scale_normalized_error"] = max(m["max_scale_normalized_error"] for m in reference["metrics"])
                prompt_options = (["--prompt-tokens", validation / "raw-prompt.tokens"] if mode == "raw"
                                  else ["--prompt", prompt])
                base = [runtime, "--model", checkpoint, "--context-length", "8192", "--temperature", "0",
                        "--max-new-tokens", str(args.generation_cap), *prompt_options]
                if args.attention_math_dynamic:
                    base += ["--attention-math-dynamic"]
                expected = None
                for batch in (1, 32, 33):
                    token_file = directory / f"{tag}-b{batch}.tokens"
                    stdout = run(directory, f"{tag}-b{batch}", [*base, "--batch-size", str(batch),
                                 "--tokens-output", token_file])
                    paths = [token_file] if batch == 1 else [Path(str(token_file) + f".{i}.txt") for i in range(batch)]
                    tokens = [read_tokens(path) for path in paths]
                    if expected is None:
                        expected = tokens[0]
                    eos = model["config"]["eos_token_id"]
                    eos = eos if isinstance(eos, list) else [eos]
                    if (not expected or any(t < 0 or t >= model["config"]["vocab_size"] for t in expected) or
                            not valid_termination(expected, eos, args.generation_cap) or
                            tokens != [expected] * batch):
                        raise RuntimeError("bounded generation differs across rows/batches or has invalid termination")
                    key = "generated_fnv1a" if batch == 1 else "batch_generated_fnv1a"
                    if field_of(stdout, key) != [token_hash(expected)] * batch:
                        raise RuntimeError("generation hash metadata mismatch")
                    counts_key = "generated_tokens" if batch == 1 else "batch_sequence_tokens"
                    if field_of(stdout, counts_key) != [str(len(expected))] * batch:
                        raise RuntimeError("generation count metadata mismatch")
                    if batch > 1 and field_of(stdout, "batch_generated_tokens") != [str(len(expected) * batch)]:
                        raise RuntimeError("generation total metadata mismatch")
                case.update(passed=True, generated_tokens=len(expected), generated_hash=token_hash(expected),
                            natural_eos=expected[-1] in eos)
                print(name, tag, "PASS", len(expected), "tokens; reference bytes", case["reference_byte_identical"], flush=True)
                save()
            row["passed"] = True
        except Exception as exc:
            row["error"] = str(exc)
            print(name, "FAIL", str(exc), flush=True)
        save()
    if digest(runtime) != sha or digest(args.runtime) != sha:
        raise RuntimeError("runtime changed during suite validation")
    report["all_gates_passed"] = all(m["passed"] for m in report["models"])
    save()
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
