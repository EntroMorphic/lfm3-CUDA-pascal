"""Logits-only-on-disk context gates; run in the pinned CUDA reference container.

Default acceptance is bitwise BF16 equality, never an adaptive tolerance. No GPU
work is done until request rejection checks pass. The official eager BF16 model
is loaded once, then freed before starting native processes. This is correctness
validation, not a throughput benchmark.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


CONTENT_TEXT = "Context boundary verification: count 17, 25, 42; remember the earlier numbers."
SOURCE_FILES = ["config.json", "modeling_lfm3.py", "lfm3_conv.py"]


sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_trace_exact import exact_logit_steps  # noqa: E402

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_hashes(checkpoint):
    return {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def weight_identity(checkpoint):
    # Avoid reading the entire multi-GB checkpoint just to resume. This is an
    # explicit local-file identity check, not a cryptographic weight digest.
    path = (checkpoint / "model.safetensors").resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def cases(suite):
    # (name, prefill, single-token decodes, check free-running generation)
    result = [("capacity1", 1, 0, True), ("decode2", 1, 1, False),
              ("conv7", 6, 2, False), ("window255", 255, 2, True),
              ("window256", 256, 1, False), ("window257", 257, 1, False)]
    if suite == "full":
        result += [("long1024", 1024, 2, False),
                   ("final4096", 4095, 1, True), ("prefill4096", 4096, 0, False)]
    return result


def invalid_requests():
    return [
        ("capacity0", dict(input_ids=[1], capacity=0), "capacity must be 1..32768"),
        ("capacity4097", dict(input_ids=[1], capacity=4097), "capacity must be 1..4096"),
        ("prefill_overflow", dict(input_ids=[1, 2], capacity=1), "invalid or overflowing"),
        ("teacher_overflow", dict(input_ids=[1], teacher_ids=[2], capacity=1), "invalid or overflowing"),
        ("generation_overflow", dict(input_ids=[1], max_new_tokens=2, capacity=1), "invalid or overflowing"),
        ("max_prefill_overflow", dict(input_ids=[1] * 4097, capacity=4096), "invalid or overflowing"),
        ("max_decode_overflow", dict(input_ids=[1] * 4096, teacher_ids=[2], capacity=4096), "invalid or overflowing"),
        ("max_generation_overflow", dict(input_ids=[1] * 4096, max_new_tokens=2, capacity=4096), "invalid or overflowing"),
        ("warmup_bool", dict(input_ids=[1], warmup_runs=True), "configuration field must be a number"),
        ("warmup11", dict(input_ids=[1], warmup_runs=11), "at most 10 warmup runs"),
        ("reset_number", dict(input_ids=[1], check_reset=1), "configuration field must be boolean"),
        ("trace_logits_number", dict(input_ids=[1], trace_logits_only=1), "configuration field must be boolean"),
    ]


def reject_before_weights(binary, checkpoint, output, env, timeout, run=subprocess.run):
    # Real config, deliberately NO weights: return code alone cannot prove that
    # request validation happened before Safetensors/model allocation.
    stub = output / "config-only-checkpoint"
    stub.mkdir()
    shutil.copyfile(checkpoint / "config.json", stub / "config.json")
    results = []
    for name, request, diagnostic in invalid_requests():
        path = output / (name + ".request.json")
        destination = output / (name + ".output.json")
        path.write_text(json.dumps(request))
        result = run([str(binary), "--experimental", str(stub), str(path), str(destination)],
                     env=env, capture_output=True, text=True, timeout=timeout)
        (output / (name + ".log")).write_text(result.stdout + result.stderr)
        require(result.returncode == 1 and diagnostic in result.stderr and not destination.exists(),
                f"{name}: did not reject before weights with expected diagnostic; see {name}.log")
        results.append(name)
    return results


def compare_context_manifest(golden, manifest):
    """Tier 1: the case set, each case's free-running ids and every teacher logit trace must equal the pin."""
    golden = {k: v for k, v in golden.items() if not k.startswith("_")}
    require(set(golden) == set(manifest), f"tier 1: case set differs from the pin: {sorted(set(golden) ^ set(manifest))}")
    for case in golden:
        require(golden[case]["generated_ids"] == manifest[case]["generated_ids"], f"tier 1: {case}: ids differ from the pin")
        require(golden[case]["files"] == manifest[case]["files"], f"tier 1: {case}: logit traces differ from the pin")


def compare_logits(reference, native, vocab, atol=0.0, rtol=0.0):
    require(len(reference) == len(native) == vocab * 2, "missing or wrong-size BF16 logits")
    max_abs = 0.0
    failures = 0
    for (r,), (n,) in zip(struct.iter_unpack("<H", reference), struct.iter_unpack("<H", native)):
        r = struct.unpack("<f", struct.pack("<I", r << 16))[0]
        n = struct.unpack("<f", struct.pack("<I", n << 16))[0]
        require(math.isfinite(r) and math.isfinite(n), "nonfinite logits")
        error = abs(r - n)
        max_abs = max(max_abs, error)
        failures += error > atol + rtol * abs(r)
    exact = reference == native
    require(not failures and (atol != 0 or rtol != 0 or exact),
            f"logit mismatch: {failures}/{vocab} outside explicit tolerance; max_abs={max_abs}")
    return dict(bitwise_equal=exact, max_abs=max_abs, values=vocab)


def make_references(checkpoint, output, selected):
    # Lazy imports keep --help and host tests usable without torch/transformers.
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model = AutoModelForMultimodalLM.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    backbone = model.model.language_model
    suppress = sorted(set(model._placeholder_token_ids +
                          [model.config.mixed_start_token_id, model.config.audio_start_token_id]))
    stop = backbone.config.eos_token_id
    stop = [stop] if isinstance(stop, int) else list(stop)
    stop = sorted(set(stop + [model.config.text_end_token_id]))
    content = processor.tokenizer.encode(CONTENT_TEXT, add_special_tokens=False)
    require(bool(content), "empty tokenized content")
    metadata = dict(torch=torch.__version__, transformers=transformers.__version__,
                    attention="eager", dtype="bfloat16", suppress_tokens=suppress, stop_tokens=stop,
                    layer_types=list(backbone.config.layer_types), vocab=backbone.config.vocab_size)
    references = {}
    with torch.inference_mode():
        for name, length, decodes, free_running in selected:
            directory = output / name
            directory.mkdir()
            reference = directory / "reference"
            reference.mkdir()
            inputs = (content * (length // len(content) + 1))[:length]
            teacher, argmax, cache = [], [], None
            current = inputs
            for step in range(decodes + 1):
                out = backbone(input_ids=torch.tensor([current], device="cuda"),
                               past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                # Match the official Omni forward's last-token head; do not
                # materialize a prefill-length by vocabulary logit tensor.
                logits = model.lm_head(out.last_hidden_state[:, -1:, :]).detach()
                require(logits.dtype == torch.bfloat16, "reference head must produce BF16")
                raw = logits.contiguous().cpu().view(torch.uint16).numpy().tobytes()
                (reference / f"{step}.logits.bf16").write_bytes(raw)
                scores = logits[0, -1].clone()
                scores[suppress] = -float("inf")
                token = int(scores.argmax())
                argmax.append(token)
                if step < decodes:
                    teacher.append(token)
                    current = [token]
            request = dict(input_ids=inputs, teacher_ids=teacher, capacity=length + decodes,
                           suppress_tokens=suppress, stop_tokens=stop, check_reset=True)
            expected = dict(teacher_argmax=argmax)
            if free_running:
                generated = []
                for token in argmax:
                    generated.append(token)
                    if token in stop:
                        break
                expected["generated_ids"] = generated
            (directory / "reference.json").write_text(json.dumps(expected, indent=2) + "\n")
            references[name] = (request, expected)
            print(f"{name}: reference ready ({length}+{decodes})", flush=True)
            del out, cache, logits, scores
    del model, backbone, processor
    gc.collect()
    torch.cuda.empty_cache()
    return references, metadata


def save_reference_manifest(checkpoint, output, references, metadata):
    entries = {}
    for name, (request, expected) in references.items():
        paths = sorted((output / name / "reference").glob("*.logits.bf16"))
        entries[name] = dict(request=request, expected=expected,
                             logits_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    manifest = dict(format_version=1, source_sha256=source_hashes(checkpoint),
                    weight_identity=weight_identity(checkpoint), reference=metadata, cases=entries)
    write_json(output / "reference-manifest.json", manifest)


def adopt_legacy_reference(checkpoint, source, selected):
    # Old runs saved all oracle vectors before launching native, but did not
    # persist source fingerprints or the later requests. Explicit user opt-in is
    # mandatory: their original checkpoint provenance cannot be reconstructed.
    from transformers import AutoProcessor

    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    content = processor.tokenizer.encode(CONTENT_TEXT, add_special_tokens=False)
    require(bool(content), "empty legacy content")
    requests = sorted(source.glob("*/teacher-request.json"))
    require(bool(requests), "legacy reference needs at least one saved teacher request")
    sample = json.loads(requests[0].read_text())
    metadata = dict(attention="eager", dtype="bfloat16", layer_types=text["layer_types"],
                    vocab=text["vocab_size"], suppress_tokens=sample["suppress_tokens"],
                    stop_tokens=sample["stop_tokens"],
                    provenance="legacy-adopted: original source/weight identity unavailable; operator explicitly opted in")
    entries = {}
    for name, length, decodes, free_running in selected:
        expected = json.loads((source / name / "reference.json").read_text())
        require(len(expected["teacher_argmax"]) == decodes + 1, f"{name}: legacy step count mismatch")
        request = dict(input_ids=(content * (length // len(content) + 1))[:length],
                       teacher_ids=expected["teacher_argmax"][:-1], capacity=length + decodes,
                       suppress_tokens=metadata["suppress_tokens"], stop_tokens=metadata["stop_tokens"], check_reset=True)
        previous = source / name / "teacher-request.json"
        if previous.exists():
            saved = json.loads(previous.read_text())
            require(all(saved.get(k) == v for k, v in request.items()), f"{name}: reconstructed legacy request mismatch")
        if free_running:
            generated = []
            for token in expected["teacher_argmax"]:
                generated.append(token)
                if token in metadata["stop_tokens"]:
                    break
            require(expected.get("generated_ids") == generated, f"{name}: legacy greedy IDs mismatch")
        entries[name] = dict(request=request, expected=expected, logits_sha256={
            f"{step}.logits.bf16": hashlib.sha256((source / name / "reference" / f"{step}.logits.bf16").read_bytes()).hexdigest()
            for step in range(decodes + 1)})
    return dict(format_version=1, source_sha256=source_hashes(checkpoint),
                weight_identity=weight_identity(checkpoint), reference=metadata, cases=entries)


def reuse_references(checkpoint, source, output, selected, allow_legacy=False):
    path = source / "reference-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        require(allow_legacy, "reference manifest missing; --allow-legacy-reference explicitly trusts old oracle provenance")
        manifest = adopt_legacy_reference(checkpoint, source, selected)
    require(manifest.get("format_version") == 1, "unsupported reference manifest")
    require(manifest["source_sha256"] == source_hashes(checkpoint), "reference checkpoint source fingerprint mismatch")
    require(manifest["weight_identity"] == weight_identity(checkpoint), "reference weight file identity mismatch")
    metadata = manifest["reference"]
    require(metadata.get("attention") == "eager" and metadata.get("dtype") == "bfloat16", "wrong oracle mode")
    references = {}
    for name, length, decodes, free_running in selected:
        require(name in manifest["cases"], f"{name}: absent from reference manifest")
        entry = manifest["cases"][name]
        request, expected = entry["request"], entry["expected"]
        require(len(request["input_ids"]) == length and len(request["teacher_ids"]) == decodes and
                request["capacity"] == length + decodes and request.get("check_reset") is True,
                f"{name}: reference request shape mismatch")
        require(len(expected["teacher_argmax"]) == decodes + 1 and
                request["teacher_ids"] == expected["teacher_argmax"][:-1], f"{name}: inconsistent teacher oracle")
        if free_running:
            require(bool(expected.get("generated_ids")), f"{name}: missing greedy oracle")
        destination = output / name / "reference"
        destination.mkdir(parents=True)
        wanted = {f"{step}.logits.bf16" for step in range(decodes + 1)}
        require(set(entry["logits_sha256"]) == wanted, f"{name}: reference checkpoint set mismatch")
        for filename in sorted(wanted):
            raw = (source / name / "reference" / filename).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == entry["logits_sha256"][filename],
                    f"{name}/{filename}: corrupt cached reference")
            compare_logits(raw, raw, metadata["vocab"])
            (destination / filename).write_bytes(raw)
        write_json(output / name / "reference.json", expected)
        references[name] = (request, expected)
    return references, metadata


def run_native(binary, checkpoint, directory, request, label, env, timeout):
    path = directory / (label + "-request.json")
    destination = directory / (label + ".json")
    path.write_text(json.dumps(request, indent=2) + "\n")
    with (directory / (label + ".log")).open("w") as log:
        result = subprocess.run([str(binary), "--experimental", str(checkpoint), str(path), str(destination)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, f"native failed: see {directory / (label + '.log')}")
    result = json.loads(destination.read_text())
    require(result.get("input_ids") == request["input_ids"], "native input mismatch")
    require(result.get("reset_checked") is True and result.get("reset_exact_logits_ok") is True,
            "same-process exact-logit reset not verified")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["smoke", "full"], default="full")
    parser.add_argument("--reuse-reference", type=Path, help="Reuse oracle vectors from a previous output; always rerun native checks")
    parser.add_argument("--allow-legacy-reference", action="store_true",
                        help="Explicitly trust old manifest-less oracle provenance; reconstruct requests with the CPU tokenizer")
    parser.add_argument("--atol", type=float, default=0.0, help="Explicit opt-in absolute logit tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="Explicit opt-in relative logit tolerance")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds per native invocation")
    parser.add_argument("--native-tiers", action="store_true",
                        help="G4 gates for the native-GEMM engine (validation/g4-gates-v2/PLAN.md): teacher argmax and "
                             "free-running ids stay exact against eager; logit distances from eager are recorded, not "
                             "judged; tier 1 compares every teacher logit trace and the ids with a pinned manifest")
    pin = parser.add_mutually_exclusive_group()
    pin.add_argument("--golden", type=Path, help="tier 1: the pinned manifest to compare against")
    pin.add_argument("--pin", type=Path, help="tier 1: write the manifest of this run")
    args = parser.parse_args()
    if args.native_tiers != bool(args.golden or args.pin):
        parser.error("--native-tiers needs exactly one of --golden or --pin, and they need --native-tiers")
    if args.pin and args.pin.exists():
        parser.error(f"--pin {args.pin} exists; re-pinning is deliberate: remove it first")
    golden = json.loads(args.golden.read_text()) if args.golden else None  # fail now, not after hours
    if args.golden and not {k for k in golden if not k.startswith("_")}:
        parser.error(f"--golden {args.golden} is empty")
    require(all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)), "invalid tolerance")
    require(math.isfinite(args.timeout) and args.timeout > 0, "invalid timeout")
    require(not args.allow_legacy_reference or args.reuse_reference is not None,
            "--allow-legacy-reference requires --reuse-reference")
    args.checkpoint, args.binary, args.output = (p.resolve() for p in (args.checkpoint, args.binary, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    selected = cases(args.suite)
    manifest = {}
    # Native tiers: logit distances from eager are information; argmax stays exact (validation/g4-gates-v2/PLAN.md).
    logit_atol = 1e30 if args.native_tiers else args.atol
    summary = dict(status="running", suite=args.suite, cases=[], maximum4096_checked=False,
                   atol=args.atol, rtol=args.rtol, native_tiers=args.native_tiers,
                   logit_atol_used=1e30 if args.native_tiers else args.atol,
                   reuse_reference=str(args.reuse_reference.resolve()) if args.reuse_reference else None,
                   requested_cases=[name for name, _, _, _ in selected])

    def progress(phase, case=None, step=None):
        summary["location"] = dict(phase=phase, case=case, step=step)
        write_json(args.output / "summary.json", summary)

    try:
        progress("request_rejections")
        summary["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        summary["source_sha256"] = source_hashes(args.checkpoint)
        summary["invalid_requests_rejected_before_weights"] = reject_before_weights(
            args.binary, args.checkpoint, args.output, env, args.timeout)
        progress("reuse_reference" if args.reuse_reference else "reference_generation")
        if args.reuse_reference:
            references, metadata = reuse_references(args.checkpoint, args.reuse_reference.resolve(), args.output,
                                                   selected, args.allow_legacy_reference)
        else:
            references, metadata = make_references(args.checkpoint, args.output, selected)
        # Written before the first native launch, so failed native runs retain a
        # complete reusable oracle bundle. Original evidence is never mutated.
        save_reference_manifest(args.checkpoint, args.output, references, metadata)
        summary["reference"] = metadata
        for name, length, decodes, free_running in selected:
            directory = args.output / name
            request, expected = references[name]
            trace = directory / "native-logits"
            trace.mkdir()
            comparisons = []
            report = dict(case=name, status="running", prefill=length, decodes=decodes,
                          capacity=length + decodes, logits=comparisons)
            summary["cases"].append(report)
            progress("native_teacher", name)
            result = run_native(args.binary, args.checkpoint, directory,
                                dict(request, trace_directory=str(trace), trace_logits_only=True),
                                "teacher", env, args.timeout)
            require(result.get("decode_forward_calls") == decodes, f"{name}: decode count mismatch")
            require(result.get("generated_ids") == [], f"{name}: unexpected generation in teacher run")
            wanted = {f"{step}.logits.bf16" for step in range(decodes + 1)}
            actual = {p.name for p in trace.iterdir()}
            # G4: an FP32-logit engine (native GEMM) also writes STEP.logits.f32, the exact values it returns.
            # Accept them only for every step, and only if each .bf16 trace is their round-to-nearest-even.
            if any(n.endswith(".logits.f32") for n in actual):
                wanted |= {f"{step}.logits.f32" for step in range(decodes + 1)}
            require(actual == wanted and all(not p.is_symlink() for p in trace.iterdir()),
                    f"{name}: missing logits or unexpected full activation files: {actual ^ wanted}")
            try:
                exact = exact_logit_steps(trace, range(decodes + 1))
            except ValueError as error:
                raise RuntimeError(f"{name}: {error}") from error
            # Native tiers are for the FP32-logit engine: its exact logit traces are required, not just accepted.
            require(exact or not args.native_tiers, f"{name}: --native-tiers requires the engine's exact .logits.f32 traces")
            actual_argmax = result.get("teacher_argmax")
            require(isinstance(actual_argmax, list) and len(actual_argmax) == decodes + 1,
                    f"{name}: teacher argmax count mismatch")
            for step in range(decodes + 1):
                progress("compare_logits", name, step)
                filename = f"{step}.logits.bf16"
                try:
                    comparisons.append(compare_logits((directory / "reference" / filename).read_bytes(),
                                                      (trace / filename).read_bytes(), metadata["vocab"],
                                                      logit_atol, args.rtol))
                    require(actual_argmax[step] == expected["teacher_argmax"][step], "teacher argmax mismatch")
                except Exception as error:
                    raise RuntimeError(f"{name} step={step}: {error}") from error
            if free_running:
                progress("native_generation", name)
                free_request = dict(request)
                del free_request["teacher_ids"]
                free_request["max_new_tokens"] = decodes + 1
                generation = run_native(args.binary, args.checkpoint, directory, free_request, "generation", env, args.timeout)
                require(generation.get("generated_ids") == expected["generated_ids"], f"{name}: free-running token mismatch")
                require(generation.get("decode_forward_calls") == len(expected["generated_ids"]) - 1,
                        f"{name}: free-running decode count mismatch")
            if args.native_tiers:
                manifest[name] = dict(generated_ids=generation["generated_ids"] if free_running else None,
                                      files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(trace.iterdir())})
            report.update(status="passed", final_slot_consumed=True, reset_exact=True,
                          greedy_checked=free_running,
                          greedy_final_slot_consumed=(len(expected["generated_ids"]) == decodes + 1) if free_running else None)
            progress("case_complete", name)
            print(name, "PASS", flush=True)
        if args.native_tiers:
            progress("tier1")
            if args.pin:
                write_json(args.pin, dict(manifest, _provenance=dict(binary_sha256=summary["binary_sha256"],
                                                                     suite=args.suite, plan="validation/g4-gates-v2/PLAN.md")))
                summary["tier1"] = f"pinned {args.pin}"
            else:
                compare_context_manifest(golden, manifest)
                summary["tier1"] = f"matches {args.golden}"
        summary.update(status="passed", maximum4096_checked=args.suite == "full")
        progress("complete")
    except Exception as error:
        summary["status"] = "failed"
        summary["failure"] = dict(type=type(error).__name__, message=str(error), **summary["location"])
        if summary["cases"] and summary["cases"][-1]["status"] == "running":
            summary["cases"][-1]["status"] = "failed"
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary["failure"]), file=sys.stderr, flush=True)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
