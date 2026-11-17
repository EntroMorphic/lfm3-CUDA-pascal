#!/usr/bin/env python3
"""Coherency test for the batched decode path.

Runs the batched path and the single-sequence path over the same prompt and
checks that the batch did not corrupt anything:

  1. every sequence is non-empty and free of degenerate repetition
  2. every sequence is identical to every other
  3. every sequence is token-for-token identical to the single-sequence run
  4. the aggregate rate is reported against the single-sequence rate

The checks are only meaningful when every sequence in the batch shares the
same prompt, because greedy decoding then makes the expected output unique.

Usage (inside the GPU container, where the runtime and model live):

    python3 tools/upstream/coherency_test.py \
        --runtime native/build/liquid-native \
        --model models/LFM2.5-350M/model.safetensors \
        --tokenizer models/LFM2.5-350M/tokenizer.json \
        --batch-size 32
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def engine(args, extra):
    if getattr(args, "attention_math_dynamic", False):
        extra = ["--attention-math-dynamic", *extra]
    if getattr(args, "context_length", None) is not None:
        extra = ["--context-length", str(args.context_length), *extra]
    prompt = (["--prompt-tokens", args.prompt_tokens] if getattr(args, "prompt_tokens", None)
              else ["--prompt", args.prompt])
    result = subprocess.run(
        [args.runtime, "--model", args.model, "--tokenizer", args.tokenizer,
         *prompt] + extra,
        capture_output=True, text=True, timeout=1800)
    if getattr(args, "work", None):
        tag = "batch" if "--batch-size" in extra else "single"
        Path(args.work, f"{tag}.stdout").write_text(result.stdout)
        Path(args.work, f"{tag}.stderr").write_text(result.stderr)
    result.check_returncode()
    return result.stdout


def read_tokens(path):
    try:
        with open(path) as handle:
            return [int(line) for line in handle if line.strip()]
    except FileNotFoundError:
        return []


def repetition_fraction(tokens, n=8):
    """Fraction of n-grams that repeat an earlier n-gram."""
    if len(tokens) < n + 1:
        return 0.0
    seen = set()
    repeats = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return repeats / (len(tokens) - n + 1)


def rate_of(output, key):
    values = field_of(output, key)
    try:
        return float(values[0]) if len(values) == 1 else 0.0
    except ValueError:
        return 0.0


def field_of(output, key):
    # Generated text is untrusted content, not timing/count metadata.
    header = re.split(r"(?m)^(?:batch_)?text_begin$", output, maxsplit=1)[0]
    matches = [line.split(":", 1)[1].split() for line in header.splitlines() if line.startswith(key + ":")]
    return matches[0] if len(matches) == 1 else []


def token_hash(tokens):
    if any(t < 0 or t >= 2**32 for t in tokens):
        return None
    value = 1469598103934665603
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def valid_termination(tokens, eos, limit):
    return (0 < len(tokens) <= limit and not any(t in eos for t in tokens[:-1]) and
            (len(tokens) == limit or tokens[-1] in eos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-tokens", help="Officially tokenized raw/base or externally rendered prompt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--context-length", type=int, help="Explicit native KV capacity")
    parser.add_argument("--attention-math-dynamic", action="store_true")
    parser.add_argument("--work", default=None)
    parser.add_argument("--json", action="store_true",
                        help="emit the result as JSON instead of a report")
    parser.add_argument("--reference", action="store_true",
                        help="also run the Transformers reference and compare")
    args = parser.parse_args()
    if args.reference and args.prompt_tokens:
        parser.error("--reference currently requires --prompt, not --prompt-tokens")
    if args.batch_size < 2 or args.max_new_tokens < 1:
        parser.error("batch-size must be at least 2 and max-new-tokens positive")
    if args.context_length is not None and args.context_length < 1:
        parser.error("context-length must be positive")
    model_path = Path(args.model)
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.with_name("config.json")
    config = json.loads(config_path.read_text())
    vocabulary = config["vocab_size"]
    eos = config["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    if (type(vocabulary) is not int or vocabulary <= 0 or not eos or
            any(type(t) is not int or not 0 <= t < vocabulary for t in eos)):
        parser.error("invalid checkpoint vocabulary/EOS metadata")

    work = args.work or tempfile.mkdtemp(prefix="coherency-")
    if args.work:
        os.makedirs(work, exist_ok=False)
    args.work = work

    reference = engine(args, ["--max-new-tokens", str(args.max_new_tokens),
                              "--tokens-output", os.path.join(work, "ref.txt")])
    ref_tokens = read_tokens(os.path.join(work, "ref.txt"))
    ref_rate = rate_of(reference, "generated_tokens_per_second")

    batch = engine(args, ["--batch-size", str(args.batch_size),
                          "--max-new-tokens", str(args.max_new_tokens),
                          "--tokens-output", os.path.join(work, "b.txt")])
    rate = rate_of(batch, "batch_tokens_per_second")
    seq_hashes = field_of(batch, "batch_generated_fnv1a")
    seq_counts = field_of(batch, "batch_sequence_tokens")
    sequences = [read_tokens(os.path.join(work, f"b.txt.{i}.txt"))
                 for i in range(args.batch_size)]

    empty = [i for i, s in enumerate(sequences) if len(s) == 0]
    worst_repetition = max((repetition_fraction(s) for s in sequences), default=0.0)
    in_range = all(0 <= t < vocabulary for s in [ref_tokens, *sequences] for t in s)
    all_identical = all(s == sequences[0] for s in sequences)
    matches_reference = all(s == ref_tokens for s in sequences)

    checks = {
        "no_empty_sequence": bool(ref_tokens) and not empty,
        "not_degenerate": worst_repetition < 0.35,
        "tokens_in_range": in_range,
        "sequences_identical": all_identical,
        "matches_single_sequence": matches_reference,
        "hashes_uniform": len(seq_hashes) == args.batch_size and len(set(seq_hashes)) == 1,
        "hashes_match_tokens": (seq_hashes == [token_hash(s) for s in sequences] and
                                field_of(reference, "generated_fnv1a") == [token_hash(ref_tokens)]),
        "counts_match_tokens": (seq_counts == [str(len(s)) for s in sequences] and
            field_of(batch, "batch_generated_tokens") == [str(sum(map(len, sequences)))] and
            field_of(reference, "generated_tokens") == [str(len(ref_tokens))]),
        "valid_termination": all(valid_termination(s, eos, args.max_new_tokens) for s in [ref_tokens, *sequences]),
        "valid_rates": all(math.isfinite(r) and r > 0 for r in (rate, ref_rate)),
    }
    passed = all(checks.values())

    comparison = None
    if args.reference:
        comparison = compare_with_reference(args, work, ref_tokens, sequences)
        checks["reference_completed"] = "error" not in comparison
        passed = all(checks.values())

    result = {
        "batch_size": args.batch_size,
        "reference_tokens": len(ref_tokens),
        "batch_tokens_per_sequence": sorted(set(len(s) for s in sequences)),
        "distinct_hashes": len(set(seq_hashes)),
        "single_sequence_tok_per_s": ref_rate if math.isfinite(ref_rate) else None,
        "batch_tok_per_s": rate if math.isfinite(rate) else None,
        "worst_repeated_8gram_fraction": worst_repetition,
        "checks": checks,
        "passed": passed,
        "reference_comparison": comparison,
    }

    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if passed else 1

    print(f"=== coherency: batch {args.batch_size} ===")
    print(f"  reference (single sequence) : {len(ref_tokens)} tokens, {ref_rate:.1f} tok/s")
    print(f"  batch aggregate             : {rate:.1f} tok/s "
          f"({rate / max(ref_rate, 1e-9):.1f}x)")
    print(f"  tokens per sequence         : {result['batch_tokens_per_sequence']}")
    print(f"  distinct hashes             : {result['distinct_hashes']}")
    print(f"  worst repeated-8-gram       : {worst_repetition:.4f}")
    print()
    for name, value in checks.items():
        print(f"  [{'OK ' if value else 'FAIL'}] {name}")
    if comparison:
        print()
        print("=== native vs the Transformers reference ===")
        if "error" in comparison:
            print(f"  reference run failed: {comparison['error']}")
        else:
            print(f"  {'metric':<26} {'native':>12} {'reference':>12}")
            print("  " + "-" * 50)
            for key in ("chars", "words", "sentences", "repeated_lines",
                        "repeated_trigram_fraction"):
                print(f"  {key:<26} {comparison['native'][key]:>12.4f} "
                      f"{comparison['reference'][key]:>12.4f}")
            for key in ("headings", "bold_bullets"):
                print(f"  {key + ' (count)':<26} {len(comparison['native'][key]):>12} "
                      f"{len(comparison['reference'][key]):>12}")
            print(f"  {'tokens':<26} {comparison['native_tokens']:>12} "
                  f"{comparison['reference_tokens']:>12}")
            print(f"  {'first token divergence':<26} {comparison['first_divergence']:>12}")
            print(f"  native headings    : {comparison['native']['headings']}")
            print(f"  reference headings : {comparison['reference']['headings']}")
            print(f"  native bullet leads   : {comparison['native']['bold_bullets']}")
            print(f"  reference bullet leads: {comparison['reference']['bold_bullets']}")

    print()
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def structure(text):
    """Structural description of a generated answer."""
    headings = re.findall(r"^#{1,6}\s+(.*)$", text, re.M)
    bullets = re.findall(r"^\s*[-*]\s+\*\*([^*]+)\*\*", text, re.M)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    words = text.split()
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 24]
    tri = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "headings": headings,
        "bold_bullets": bullets,
        "repeated_lines": len(lines) - len(set(lines)),
        "repeated_trigram_fraction": 1.0 - (len(set(tri)) / len(tri)) if tri else 0.0,
    }


def compare_with_reference(args, work, native_tokens, sequences):
    """Compare the native answer with the Transformers reference answer."""
    # tools/upstream/coherency_test.py -> repository root
    script = Path(__file__).resolve().parents[2] / "scripts" / "reference_generate.py"
    # The runtime takes the checkpoint file; the Transformers reference wants
    # the directory, so it can find the tokenizer and the config beside it.
    model_path = Path(args.model).resolve()
    model_directory = str(model_path if model_path.is_dir() else model_path.parent)
    out = subprocess.run(
        [sys.executable, str(script), "--model", model_directory, "--prompt", args.prompt,
         "--output", os.path.join(work, "ref.json")],
        capture_output=True, text=True, timeout=1800,
    )
    if out.returncode != 0:
        return {"error": out.stderr[-400:]}
    with open(os.path.join(work, "ref.json")) as handle:
        reference = json.load(handle)
    ref_tokens = reference["generated_token_ids"]
    limit = min(len(native_tokens), len(ref_tokens))
    divergence = next((i for i in range(limit) if native_tokens[i] != ref_tokens[i]),
                      limit if len(native_tokens) != len(ref_tokens) else None)
    return {
        "native": structure(_native_text(args, work)),
        "reference": structure(reference["text"]),
        "native_tokens": len(native_tokens),
        "reference_tokens": len(ref_tokens),
        "first_divergence": divergence,
    }


def _native_text(args, work):
    path = os.path.join(work, "native.txt")
    out = subprocess.run(
        [args.runtime, "--model", args.model, "--tokenizer", args.tokenizer,
         "--prompt", args.prompt, "--max-new-tokens", str(args.max_new_tokens),
         *(["--context-length", str(args.context_length)] if getattr(args, "context_length", None) else [])],
        capture_output=True, text=True, timeout=1800,
    )
    out.check_returncode()
    text = ""
    if "text_begin" in out.stdout:
        text = out.stdout.split("text_begin", 1)[1].split("text_end", 1)[0].strip()
    return text


if __name__ == "__main__":
    sys.exit(main())
