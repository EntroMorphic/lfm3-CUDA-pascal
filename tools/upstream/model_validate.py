#!/usr/bin/env python3
"""Offline, metadata-driven teacher-forced native/batch/Transformers validation.

This records numerical evidence, NOT performance. The native prefix is fixed
once, then every engine ingests exactly those tokens. Artifacts are never
overwritten. Transformers is only an oracle, not the native execution path.

--prompt-tokens replays actual saved frontend input, including historical raw
inputs without BOS. With --prompt as well, official tokenizer identity is
required. --prompt-metadata optionally binds a benchmark_decode summary; it
does not establish that historical formatting was correct or replay sampling.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct


def parse_prompt_ids(text, vocabulary):
    """Accept only nonempty whitespace-separated decimal IDs, never coercions."""
    fields = text.split()
    if (not fields or any(re.fullmatch(r"[0-9]+", t) is None for t in fields)):
        raise RuntimeError("prompt IDs must be nonempty whitespace-separated decimal integers")
    ids = [int(t) for t in fields]
    if any(t >= vocabulary for t in ids):
        raise RuntimeError("prompt ID outside checkpoint vocabulary")
    return ids


def official_prompt(tokenizer, text, mode):
    # Match the public frontend once its raw/chat mode has been selected. Chat
    # templates contain special tokens already; raw uses the postprocessor.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True) if mode == "chat" else text
    return tokenizer.encode(rendered, add_special_tokens=mode == "raw")


def prepare_prompt(tokenizer, text, mode, tokens_path, metadata_path, model, output, vocabulary):
    """Snapshot exact input, optionally binding a benchmark_decode summary.

    Saved IDs alone deliberately make no claim about BOS/template correctness.
    Text plus saved IDs is stricter: both must describe the official same input.
    """
    if text is None and tokens_path is None:
        raise RuntimeError("supply --prompt or --prompt-tokens")
    if metadata_path is not None and tokens_path is None:
        raise RuntimeError("--prompt-metadata requires --prompt-tokens")
    config_sha = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    source = tokens_path.read_bytes() if tokens_path is not None else None
    prompt = parse_prompt_ids(source.decode("utf-8"), vocabulary) if source is not None else None
    expected = official_prompt(tokenizer, text, mode) if text is not None else None
    if expected is not None:
        if not expected or any(type(t) is not int or not 0 <= t < vocabulary for t in expected):
            raise RuntimeError("official tokenizer produced invalid prompt IDs")
        if prompt is not None and prompt != expected:
            raise RuntimeError("saved prompt IDs differ from official tokenizer input")
        prompt = expected
    metadata = None
    if metadata_path is not None:
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError("duplicate prompt metadata key")
                result[key] = value
            return result
        def reject_constant(value):
            raise RuntimeError("nonfinite prompt metadata")
        metadata_raw = metadata_path.read_bytes()
        metadata = json.loads(metadata_raw, object_pairs_hook=unique_keys,
                              parse_constant=reject_constant)
        if (not isinstance(metadata, dict) or
                type(metadata.get("prompt_tokens")) is not int or
                metadata["prompt_tokens"] != len(prompt) or
                metadata.get("prompt_token_sha256") != hashlib.sha256(source).hexdigest() or
                metadata.get("config_sha256") != config_sha or
                metadata.get("prompt_mode") not in ("raw", "chat", "provided_ids") or
                (text is not None and metadata["prompt_mode"] not in (mode, "provided_ids"))):
            raise RuntimeError("prompt metadata is malformed or disagrees with input/model")
    snapshot = output / "prompt.tokens"
    snapshot.write_text("".join(f"{t}\n" for t in prompt))
    provenance = dict(source="saved_ids" if source is not None else "official_tokenizer",
                      source_path=str(tokens_path.resolve()) if tokens_path is not None else None,
                      source_sha256=hashlib.sha256(source).hexdigest() if source is not None else None,
                      prompt_token_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                      prompt_tokens=len(prompt), config_sha256=config_sha,
                      prompt_mode=mode if text is not None else "provided_ids",
                      add_special_tokens=(mode == "raw") if text is not None else None,
                      official_input_verified=text is not None,
                      metadata_verified=metadata is not None)
    if metadata is not None:
        (output / "source-prompt-metadata.json").write_bytes(metadata_raw)
        provenance.update(metadata_path=str(metadata_path.resolve()),
                          metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
                          source_prompt_mode=metadata["prompt_mode"])
    return prompt, snapshot, provenance


def validate_attention_policy(stdout, requested_math):
    header = stdout.split("\ntext_begin\n", 1)[0].split("\nbatch_text_begin\n", 1)[0]
    modes = [line.removeprefix("attention_mode: ") for line in header.splitlines()
             if line.startswith("attention_mode: ")]
    if (len(modes) != 1 or modes[0] not in ("math_sdpa_fp32", "native_online_fp32") or
            (requested_math and modes[0] != "math_sdpa_fp32")):
        raise RuntimeError("missing, malformed or mismatched native attention policy metadata")
    return modes[0]


def reference_row_bytes(rows):
    """Canonical little-endian BF16 snapshot of one independent teacher step."""
    import torch
    values = torch.cat([row.detach().cpu().flatten() for row in rows])
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("nonfinite independent reference trace")
    bf16 = values.to(torch.bfloat16)
    if not bool(torch.isfinite(bf16).all()):
        raise RuntimeError("nonfinite independent reference trace after BF16 conversion")
    return bf16.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def read_trace(path, expected_bytes):
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise RuntimeError(f"unexpected trace length for {path}: {len(raw)} != {expected_bytes}")
    return raw


def validate_prefix(tokens, limit, vocabulary, eos):
    eos = eos if isinstance(eos, list) else [eos]
    if (not tokens or len(tokens) > limit or any(not 0 <= t < vocabulary for t in tokens) or
            any(t in eos for t in tokens[:-1]) or (len(tokens) < limit and tokens[-1] not in eos)):
        raise RuntimeError("invalid or incomplete native-generated validation prefix")


def error_metrics(actual, expected):
    if not actual.isfinite().all() or not expected.isfinite().all():
        raise RuntimeError("nonfinite native/reference output")
    # Finite BF16 values can still overflow FP32 squared norms. A NaN ratio
    # must never disappear through max(previous, NaN). FP64 safely covers the
    # full finite BF16 range and the supported vector lengths.
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    relative_l2 = float(error.norm() / expected.norm().clamp_min(1e-8))
    scaled = float(error.abs().max() / expected.abs().max().clamp_min(1e-8))
    if not math.isfinite(relative_l2) or not math.isfinite(scaled):
        raise RuntimeError("nonfinite validation error metric")
    return relative_l2, scaled


def prefix_agreement(prefix, prompt_tokens, native_choices, reference_choices):
    if len(native_choices) != len(prefix) or len(reference_choices) != len(prefix):
        raise RuntimeError("incomplete teacher-forced selection trace")
    positions = range(prompt_tokens - 1, len(prefix) - 1)
    return dict(cases=len(prefix) - prompt_tokens,
                native_matches=sum(native_choices[p] == prefix[p + 1] for p in positions),
                reference_matches=sum(reference_choices[p] == prefix[p + 1] for p in positions))


def acceptance_passed(metrics, routing, expected_cases, require_byte_parity=False):
    return (bool(metrics) and all(
        math.isfinite(m["max_relative_l2"]) and 0 <= m["max_relative_l2"] < 0.05 and
        math.isfinite(m["max_scale_normalized_error"]) and 0 <= m["max_scale_normalized_error"] < 0.15 and
        (not require_byte_parity or m["bf16_bit_mismatches"] == 0) for m in metrics) and
        routing["cases"] == expected_cases and
        routing["identical_input_math_matches"] == expected_cases and
        routing["reference_expert_set_matches"] == expected_cases)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", help="Text to tokenize officially; with saved IDs, require exact identity")
    p.add_argument("--prompt-tokens", type=Path,
                   help="Exact saved whitespace-separated prompt IDs; never retokenized or given extra BOS")
    p.add_argument("--prompt-metadata", type=Path,
                   help="Optional benchmark_decode summary binding prompt count/file SHA256/mode/config SHA256")
    p.add_argument("--prompt-mode", choices=("chat", "raw"), default="chat",
                   help="Text formatting (default chat); select raw explicitly for base/--raw CLI inputs")
    p.add_argument("--attention-math-dynamic", action="store_true",
                   help="Use the opt-in metadata-driven FP32 math attention path for dense models")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--generated-prefix", type=int, default=40)
    p.add_argument("--context-length", type=int, help="Explicit native KV capacity for memory budgeting")
    p.add_argument("--reference-batch-size", type=int, default=1,
                   help="Explicit oracle batch width; changes vendor reduction geometry")
    p.add_argument("--require-byte-parity", action="store_true",
                   help="Fail if any layer/logit BF16 bits differ, in addition to numerical/routing gates")
    args = p.parse_args(argv)
    if args.prompt is None and args.prompt_tokens is None:
        p.error("supply --prompt or --prompt-tokens")
    if args.prompt_metadata is not None and args.prompt_tokens is None:
        p.error("--prompt-metadata requires --prompt-tokens")
    if args.batch_size < 2 or args.generated_prefix < 1 or args.reference_batch_size < 1:
        p.error("batch size must be at least 2 and generated prefix must be positive")
    if args.context_length is not None and args.context_length < 1:
        p.error("context-length must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.nn.attention import SDPBackend, sdpa_kernel

    config = json.loads((args.model / "config.json").read_text())
    hidden, layers, vocab = (config[k] for k in ("hidden_size", "num_hidden_layers", "vocab_size"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True) if args.prompt is not None else None
    prompt, prompt_file, input_contract = prepare_prompt(
        tokenizer, args.prompt, args.prompt_mode, args.prompt_tokens, args.prompt_metadata,
        args.model, args.output, vocab)
    # Keep the existing native chat-template check for text-only chat validation.
    prompt_options = (["--prompt", args.prompt] if args.prompt_tokens is None and args.prompt_mode == "chat"
                      else ["--prompt-tokens", str(prompt_file)])
    command = [str(args.runtime), "--model", str(args.model)]
    if args.attention_math_dynamic:
        command += ["--attention-math-dynamic"]
    if args.context_length is not None:
        command += ["--context-length", str(args.context_length)]

    def run(name, options):
        result = subprocess.run([*command, *options], capture_output=True, text=True, timeout=1800)
        (args.output / f"{name}.stdout").write_text(result.stdout)
        (args.output / f"{name}.stderr").write_text(result.stderr)
        result.check_returncode()
        return result.stdout

    native_prompt = run("prompt", [*prompt_options, "--print-prompt-tokens"])
    if parse_prompt_ids(native_prompt, vocab) != prompt:
        raise RuntimeError("native prompt IDs differ from validation input")
    token_file = args.output / "generated.tokens"
    prefix_stdout = run("prefix", [*prompt_options, "--max-new-tokens", str(args.generated_prefix),
                                  "--tokens-output", str(token_file)])
    effective_attention = validate_attention_policy(prefix_stdout, args.attention_math_dynamic)
    input_contract["native_attention_mode"] = effective_attention
    (args.output / "input-contract.json").write_text(json.dumps(input_contract, indent=2, allow_nan=False) + "\n")
    generated = [int(t) for t in token_file.read_text().split()]
    validate_prefix(generated, args.generated_prefix, vocab, config["eos_token_id"])
    prefix = prompt + generated
    teacher = args.output / "teacher.tokens"
    teacher.write_text("".join(f"{t}\n" for t in prefix))
    kinds = config.get("layer_types")
    if kinds is None:
        indices = config.get("full_attn_idxs")
        kinds = ["full_attention" if indices is None or i in indices else "conv" for i in range(layers)]
    first_attention = next((i for i, kind in enumerate(kinds) if kind == "full_attention"), -1)
    trace_moe = config["model_type"] == "lfm2_moe" and config["num_dense_layers"] < layers
    first_moe = config["num_dense_layers"] if trace_moe else -1
    for batch in (1, args.batch_size):
        run(f"b{batch}", ["--prompt-tokens", str(teacher), "--parity-output", str(args.output / f"b{batch}.bf16"),
                          "--parity-layer", str(first_attention), "--batch-size", str(batch),
                          *(["--parity-mlp-layer", str(first_moe)] if trace_moe else [])])
    stride = layers * hidden + vocab
    raw = read_trace(args.output / "b1.bf16", len(prefix) * stride * 2)
    if raw != (args.output / f"b{args.batch_size}.bf16").read_bytes():
        raise RuntimeError("native single/batch teacher-forced layers or logits differ")
    actual = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().view(len(prefix), stride)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    expert_options = {"experts_implementation": "eager"} if config["model_type"] == "lfm2_moe" else {}
    reference = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", **expert_options,
    ).cuda().eval()
    captured = {}
    reference_routes = {}
    parts = {}

    def hook(index):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[index] = value[0].detach().float().cpu().flatten()
        return capture

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(reference.model.layers)]
    if trace_moe:
        def norm_hook(module, inputs, output):
            parts["input"] = inputs[0][0].detach().cpu().flatten()
            parts["normalized"] = output[0].detach().cpu().flatten()
        def ff_hook(module, inputs, output):
            parts["feed_forward"] = output[0].detach().cpu().flatten()
        handles.append(reference.model.layers[first_moe].ffn_norm.register_forward_hook(norm_hook))
        handles.append(reference.model.layers[first_moe].feed_forward.register_forward_hook(ff_hook))
        part_raw = read_trace(args.output / "b1.bf16.mlp", len(prefix) * 4 * hidden * 2)
        if part_raw != (args.output / f"b{args.batch_size}.bf16.mlp").read_bytes():
            raise RuntimeError("native single/batch MoE stage traces differ")
        native_parts = torch.frombuffer(bytearray(part_raw), dtype=torch.bfloat16).view(len(prefix), 4, hidden)
    part_errors = dict(input=0, normalized=0, feed_forward=0, layer_output=0)
    for i, layer in enumerate(reference.model.layers):
        if hasattr(layer.feed_forward, "experts"):
            def route_hook(module, inputs, output, index=i):
                reference_routes[index] = output[2][0].detach().cpu().tolist()
            handles.append(layer.feed_forward.gate.register_forward_hook(route_hook))
    expected_cases = len(prefix) * (layers - first_moe) if trace_moe else 0
    route_bytes = expected_cases * (config["num_experts"] * 2 + config["num_experts_per_tok"] * 6) if trace_moe else 0
    route_raw = read_trace(args.output / "b1.bf16.moe", route_bytes)
    if route_raw != (args.output / f"b{args.batch_size}.bf16.moe").read_bytes():
        raise RuntimeError("native single/batch routing traces differ")
    route_offset = 0
    routing = dict(cases=0, identical_input_math_matches=0, reference_expert_set_matches=0, first_math_failures=[])
    metrics = [{"layer": i, "max_relative_l2": 0.0, "max_scale_normalized_error": 0.0,
                "bf16_bit_mismatches": 0} for i in range(layers + 1)]
    argmax_matches = 0
    native_choices, reference_choices = [], []
    cache = None
    # Retain independent oracle bytes, not only aggregate error metrics, so a
    # separate read-only audit can reproduce strict numerical findings later.
    reference_path = args.output / "reference.bf16"
    with reference_path.open("xb") as reference_trace, torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for pos, token in enumerate(prefix):
            output = reference(torch.full((args.reference_batch_size, 1), token, device="cuda", dtype=torch.long), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            expected = [captured[i] for i in range(layers)] + [output.logits[0, -1].float().cpu()]
            reference_trace.write(reference_row_bytes(expected))
            if trace_moe:
                parts["layer_output"] = captured[first_moe].to(torch.bfloat16)
                for i, name in enumerate(part_errors):
                    if not torch.isfinite(native_parts[pos, i]).all() or not torch.isfinite(parts[name]).all():
                        raise RuntimeError("nonfinite native/reference MoE stage output")
                    part_errors[name] += int((native_parts[pos, i].view(torch.uint16) != parts[name].view(torch.uint16)).sum())
            if route_raw:
                experts, k = config["num_experts"], config["num_experts_per_tok"]
                for layer_index in range(config["num_dense_layers"], layers):
                    logits = torch.frombuffer(bytearray(route_raw[route_offset:route_offset + experts * 2]), dtype=torch.bfloat16).cuda()
                    route_offset += experts * 2
                    ids = list(struct.unpack_from(f"{k}i", route_raw, route_offset))
                    route_offset += k * 4
                    native_weights = torch.frombuffer(bytearray(route_raw[route_offset:route_offset + k * 2]), dtype=torch.bfloat16)
                    route_offset += k * 2
                    probability = logits.sigmoid()
                    ff = reference.model.layers[layer_index].feed_forward
                    scores = probability.float() + ff.expert_bias if config["use_expert_bias"] else probability
                    chosen = scores.argsort(descending=True, stable=True)[:k].tolist()
                    weights = probability[ids]
                    if config["norm_topk_prob"]:
                        weights = weights / (weights.sum() + 1e-6)
                    weights = weights * config["routed_scaling_factor"]
                    correct = ids == chosen and torch.equal(weights.cpu(), native_weights)
                    routing["cases"] += 1
                    routing["identical_input_math_matches"] += int(correct)
                    routing["reference_expert_set_matches"] += int(set(ids) == set(reference_routes[layer_index]))
                    if not correct and len(routing["first_math_failures"]) < 5:
                        routing["first_math_failures"].append(dict(position=pos, layer=layer_index, native=ids, expected=chosen))
            for i, expected_row in enumerate(expected):
                lo, hi = (i * hidden, (i + 1) * hidden) if i < layers else (layers * hidden, stride)
                row = actual[pos, lo:hi]
                relative_l2, scaled = error_metrics(row, expected_row)
                metrics[i]["max_relative_l2"] = max(metrics[i]["max_relative_l2"], relative_l2)
                metrics[i]["max_scale_normalized_error"] = max(metrics[i]["max_scale_normalized_error"], scaled)
                metrics[i]["bf16_bit_mismatches"] += int((row.to(torch.bfloat16).view(torch.uint16) !=
                    expected_row.to(torch.bfloat16).view(torch.uint16)).sum())
            native_choices.append(int(actual[pos, layers * hidden:].argmax()))
            reference_choices.append(int(expected[-1].argmax()))
            argmax_matches += int(native_choices[-1] == reference_choices[-1])
    for handle in handles:
        handle.remove()
    if route_offset != len(route_raw):
        raise RuntimeError("unexpected native routing trace length")
    passed = acceptance_passed(metrics, routing, expected_cases, args.require_byte_parity)
    prefix_checks = prefix_agreement(prefix, len(prompt), native_choices, reference_choices)
    if args.require_byte_parity:
        passed = (passed and not any(part_errors.values()) and
                  prefix_checks["native_matches"] == prefix_checks["cases"] and
                  prefix_checks["reference_matches"] == prefix_checks["cases"])
    report = dict(passed=passed, model=str(args.model), torch=torch.__version__, transformers=transformers.__version__,
                  dtype="BF16", teacher_positions=len(prefix), batch_size=args.batch_size,
                  native_single_batch_byte_identical=True, reference_attention="FP32 math SDPA",
                  reference_argmax_matches=argmax_matches, metrics=metrics,
                  acceptance=dict(max_relative_l2=0.05, max_scale_normalized_error=0.15,
                                  expected_routing_cases=expected_cases, require_byte_parity=args.require_byte_parity),
                  timing_comparison=False)
    report["reference_batch_size"] = args.reference_batch_size
    report["prompt_mode"] = input_contract["prompt_mode"]
    report["input_contract"] = input_contract
    report["native_attention_mode"] = effective_attention
    report["attention_math_dynamic"] = args.attention_math_dynamic
    report["generated_prefix_agreement"] = prefix_checks
    report["native_context_length"] = args.context_length
    report["routing"] = routing
    report["first_moe_part_bit_mismatches"] = part_errors
    report["first_moe_parts_layer"] = first_moe
    report["reference_bf16_byte_identical"] = all(m["bf16_bit_mismatches"] == 0 for m in metrics)
    if reference_path.stat().st_size != len(prefix) * stride * 2:
        raise RuntimeError("independent reference trace coverage differs from teacher inputs")
    report["reference_trace_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    report["reference_experts_implementation"] = getattr(reference.config, "_experts_implementation", None)
    report["reference_expert_bias_dtypes"] = sorted({str(layer.feed_forward.expert_bias.dtype)
        for layer in reference.model.layers if hasattr(layer.feed_forward, "expert_bias")})
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
