#!/usr/bin/env python3
"""Measure the batched decode rate table.

The batched decode rate depends on the batch size, on how long the batch
generates, and on the prompt length, because attention reads a per-sequence
KV cache and so does not amortize across the batch. This measures all three
on the device and prints the tables the documentation quotes.

usage: batch_table.py [--model DIR] [--runtime PATH] [--batches 1,2,4,8]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

PROMPT = "Hypothetically, might reflective recursion be a function of cognition?"


def rate(runtime, model, prompt, batch, generated):
    command = [str(runtime), "--model", str(model / "model.safetensors"),
               "--tokenizer", str(model / "tokenizer.json"), "--prompt", prompt,
               "--max-new-tokens", str(generated), "--batch-size", str(batch)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise SystemExit(f"the runtime failed: {result.stderr[-2000:]}")
    for line in result.stdout.splitlines():
        match = re.search(r"(?:batch_)?tokens_per_second: ([0-9.]+)", line)
        if match:
            return float(match.group(1))
    raise SystemExit("no rate reported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/LFM2.5-350M"))
    parser.add_argument("--runtime", type=Path, default=Path("native/build/liquid-native"))
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--generated", default="24,100,200,400")
    options = parser.parse_args()
    batches = [int(value) for value in options.batches.split(",")]
    generated_lengths = [int(value) for value in options.generated.split(",")]

    print("=== batch scaling, 200 new tokens ===")
    baseline = None
    for batch in batches:
        value = rate(options.runtime, options.model, PROMPT, batch, 200)
        if baseline is None:
            baseline = value
        print(f"  batch {batch:>3}: {value:8.0f} tok/s  ({value / baseline:4.1f}x)")

    print("=== generated length ===")
    header = "  tokens: " + "".join(f"  batch {batch:>3}" for batch in batches)
    print(header)
    for generated in generated_lengths:
        row = f"  {generated:>6}:"
        for batch in batches:
            row += f"  {rate(options.runtime, options.model, PROMPT, batch, generated):8.0f}"
        print(row)

    print("=== prompt length, 24 new tokens ===")
    for repeats in (1, 4, 12, 32):
        prompt = (PROMPT + " ") * repeats
        row = f"  x{repeats:>3}:"
        for batch in (1, 8, 32, 64):
            row += f"  batch {batch:>3}={rate(options.runtime, options.model, prompt, batch, 24):8.0f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
