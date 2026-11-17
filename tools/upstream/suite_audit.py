#!/usr/bin/env python3
"""CPU-only, fail-closed audit of saved suite matrices (not GPU safety gates)."""
import argparse
import json
from pathlib import Path

from coherency_test import field_of, read_tokens, token_hash, valid_termination
from suite_stage import digest
from suite_validate import cases


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(inventory, roots, only=None):
    entries = [m for m in inventory["models"] if m["category"] == "standalone" and
               (not only or m["id"].split("/")[1] in only)]
    targets = {m["id"]: m for m in entries}
    require(len(entries) == len(targets), "duplicate inventory models")
    require(bool(targets) and bool(roots), "empty inventory or evidence roots")
    if only:
        require(set(only) == {m.split("/")[1] for m in targets}, "incomplete requested inventory")
    rows, seen, runtime_sha, versions = [], set(), None, set()
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        require(summary["all_gates_passed"] is True, f"matrix failed: {root}")
        require(summary["attention_math_dynamic"] is True, "audit requires strict dynamic math matrices")
        sha = digest(root / "liquid-native")
        require(sha == summary["runtime_sha256"], "pinned executable digest mismatch")
        require(runtime_sha is None or runtime_sha == sha, "matrix executables differ")
        runtime_sha = sha
        for model in summary["models"]:
            name = model["id"]
            if name not in targets:
                continue
            require(name not in seen, f"duplicate model evidence: {name}")
            seen.add(name)
            expected = targets[name]
            require(model["passed"] is True and model["revision"] == expected["revision"], f"model/revision failed: {name}")
            require(model["stages"] == dict(file_integrity=True, upload_byte_identical=True), "incomplete model stages")
            directory = root / name.split("/")[1]
            upload = (directory / "upload.stdout").read_text()
            count, payload, allocated = map(int, upload.splitlines()[0].split())
            require(count > 0 and payload == model["payload_bytes"] and allocated >= payload and
                    f"all_tensor_uploads_byte_identical: {count} tensors, {payload} bytes" in upload,
                    "incomplete upload readback")
            tags = cases(name.split("/")[1])
            require([c["name"] for c in model["cases"]] == [t[0] for t in tags], "incomplete prompt coverage")
            commands = {"upload"}
            positions, prefix_cases = 0, 0
            for case, (tag, mode, prompt) in zip(model["cases"], tags):
                require(case["passed"] is True and case["prompt_mode"] == mode, "failed prompt case")
                reference_dir = directory / (tag + "-reference")
                report = json.loads((reference_dir / "report.json").read_text())
                require(report["passed"] is True and report["acceptance"]["require_byte_parity"] is True and
                        report["attention_math_dynamic"] is True and report["prompt_mode"] == mode and
                        report["reference_batch_size"] == 32 and report["dtype"] == "BF16" and
                        report["reference_bf16_byte_identical"] is True and
                        report["native_single_batch_byte_identical"] is True, "reference gate failed")
                versions.add((report["torch"], report["transformers"]))
                require(report["reference_attention"] == "FP32 math SDPA", "unexpected reference backend")
                metrics = report["metrics"]
                layers = expected["config"]["num_hidden_layers"]
                require([m["layer"] for m in metrics] == list(range(layers + 1)) and
                        all(m["bf16_bit_mismatches"] == m["max_relative_l2"] == m["max_scale_normalized_error"] == 0
                            for m in metrics), "incomplete or inexact layer/logit metrics")
                agreement = report["generated_prefix_agreement"]
                require(agreement["cases"] > 0 and agreement["native_matches"] == agreement["reference_matches"] == agreement["cases"] and
                        agreement == case["prefix_agreement"], "generated-prefix disagreement")
                teacher = read_tokens(reference_dir / "teacher.tokens")
                require(report["teacher_positions"] == len(teacher) and bool(teacher), "incomplete teacher trace")
                route_cases = (len(teacher) * (layers - expected["config"]["num_dense_layers"])
                               if expected["config"]["model_type"] == "lfm2_moe" else 0)
                require(all(report["routing"][key] == route_cases for key in
                            ("cases", "identical_input_math_matches", "reference_expert_set_matches")),
                        "incomplete routing evidence")
                route_bytes = (route_cases * (expected["config"]["num_experts"] * 2 +
                               expected["config"]["num_experts_per_tok"] * 6) if route_cases else 0)
                route1, route32 = reference_dir / "b1.bf16.moe", reference_dir / "b32.bf16.moe"
                require(route1.stat().st_size == route32.stat().st_size == route_bytes and
                        digest(route1) == digest(route32), "routing trace bytes differ")
                stride = layers * expected["config"]["hidden_size"] + expected["config"]["vocab_size"]
                single, batch = reference_dir / "b1.bf16", reference_dir / "b32.bf16"
                require(single.stat().st_size == batch.stat().st_size == len(teacher) * stride * 2 and
                        digest(single) == digest(batch), "native trace bytes differ")
                positions += len(teacher)
                prefix_cases += agreement["cases"]
                commands.add(tag + "-reference")
                baseline = read_tokens(directory / f"{tag}-b1.tokens")
                eos = expected["config"]["eos_token_id"]
                eos = eos if isinstance(eos, list) else [eos]
                require(bool(baseline) and len(baseline) == case["generated_tokens"] and
                        token_hash(baseline) == case["generated_hash"] and
                        (baseline[-1] in eos) == case["natural_eos"] and
                        all(0 <= t < expected["config"]["vocab_size"] for t in baseline), "invalid saved sequence")
                for width in (1, 32, 33):
                    command_name = f"{tag}-b{width}"
                    commands.add(command_name)
                    command = json.loads((directory / (command_name + ".command.json")).read_text())
                    require("--attention-math-dynamic" in command, "generation math flag missing")
                    cap = int(command[command.index("--max-new-tokens") + 1])
                    require(valid_termination(baseline, eos, cap), "invalid EOS/cap termination")
                    stdout = (directory / (command_name + ".stdout")).read_text()
                    require(field_of(stdout, "attention_mode") == ["math_sdpa_fp32"], "generation backend differs")
                    files = [directory / f"{tag}-b1.tokens"] if width == 1 else [
                        directory / f"{tag}-b{width}.tokens.{i}.txt" for i in range(width)]
                    require(all(read_tokens(path) == baseline for path in files), "saved row tokens differ")
                    hash_key = "generated_fnv1a" if width == 1 else "batch_generated_fnv1a"
                    count_key = "generated_tokens" if width == 1 else "batch_sequence_tokens"
                    require(field_of(stdout, hash_key) == [token_hash(baseline)] * width and
                            field_of(stdout, count_key) == [str(len(baseline))] * width,
                            "saved hashes/counts differ")
                    if width > 1:
                        require(field_of(stdout, "batch_generated_tokens") == [str(len(baseline) * width)],
                                "saved total differs")
            require({p.stem for p in directory.glob("*.returncode")} == commands, "incomplete subprocess evidence")
            for command in commands:
                require((directory / (command + ".returncode")).read_text().strip() == "0", "failed subprocess")
                require(all((directory / (command + suffix)).is_file() for suffix in
                            (".command.json", ".stdout", ".stderr")), "missing subprocess evidence")
            rows.append(dict(id=name, revision=expected["revision"], passed=True, evidence=str(directory),
                             teacher_positions=positions, generated_prefix_choices=prefix_cases,
                             payload_bytes=payload, cases=model["cases"]))
    require(seen == set(targets), "incomplete checkpoint coverage")
    require(len(versions) == 1 and all(next(iter(versions))), "reference library versions differ or are missing")
    return dict(passed=True, checkpoint_count=len(rows), runtime_sha256=runtime_sha, models=rows,
                reference_versions=dict(zip(("torch", "transformers"), next(iter(versions)))),
                scope="Saved strict checkpoint matrices only; GPU safety and legacy regression gates are separate.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(json.loads(args.inventory.read_text()), args.roots, args.only)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {report['checkpoint_count']} strict checkpoint matrices; all saved subprocesses, traces and rows agree.")


if __name__ == "__main__":
    main()
