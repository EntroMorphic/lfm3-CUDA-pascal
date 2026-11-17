"""CPU-only structural census of saved token histories; no model entropy estimate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from benchmark_common import sha256, write_json


def entropy(counts) -> float:
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n) if total else 0.


def analyze(streams: list[list[int]], prompts: list[list[int]]) -> dict:
    if not streams or any(not s for s in streams):
        raise ValueError("Every stream must contain at least one generated token")
    if len(prompts) == 1: prompts = prompts * len(streams)
    if len(prompts) != len(streams) or any(not p for p in prompts):
        raise ValueError("Supply one nonempty prompt or one per generated stream")
    if any(type(token) is not int or token < 0 for row in streams + prompts for token in row):
        raise ValueError("Token IDs must be nonnegative integers")
    roots = {}; edges = {}; states = []; next_node = 0
    for prompt in prompts:
        key = tuple(prompt)
        if key not in roots:
            roots[key] = next_node; next_node += 1
        states.append(roots[key])
    rows = len(streams); steps = []
    for step in range(max(map(len, streams))):
        live = [i for i, stream in enumerate(streams) if step < len(stream)]
        groups = Counter(states[i] for i in live)
        branches = Counter((states[i], streams[i][step]) for i in live)
        branch_counts = {}
        for (parent, token), count in branches.items():
            branch_counts.setdefault(parent, []).append(count)
        observed_branch_entropy = sum(groups[parent] / len(live) * entropy(counts)
                                      for parent, counts in branch_counts.items())
        steps.append(dict(generated_index=step, live_rows=len(live), unique_prefixes=len(groups),
            group_sizes=sorted(groups.values(), reverse=True),
            prefix_identity_entropy_bits=entropy(groups.values()),
            observed_conditional_branch_entropy_bits=observed_branch_entropy,
            live_canonical_tiles=(len(live)+31)//32,
            unique_prefix_canonical_tiles=(len(groups)+31)//32))
        for row in live:
            key = (states[row], streams[row][step])
            if key not in edges:
                edges[key] = next_node; next_node += 1
            states[row] = edges[key]
    decode = steps[1:]  # Distribution for generated index zero comes from prefill.
    fixed_tiles = ((rows+31)//32) * len(decode)
    live_tiles = sum(s["live_canonical_tiles"] for s in decode)
    unique_tiles = sum(s["unique_prefix_canonical_tiles"] for s in decode)
    live_predictions = sum(s["live_rows"] for s in decode)
    unique_predictions = sum(s["unique_prefixes"] for s in decode)
    return dict(scope="Saved-history equality under caller-supplied prompts. Prefix identity/observed branching statistics are not model predictive entropy, a state-equivalence certificate or a measured speedup.",
        rows=rows, prompt_groups=len(roots), generated_tokens=sum(map(len, streams)),
        decode_prediction_positions=live_predictions,
        unique_decode_prefix_positions=unique_predictions,
        duplicate_decode_prefix_fraction=(1-unique_predictions/live_predictions) if live_predictions else 0.,
        fixed_row_decode_slots=rows*len(decode),
        canonical_tiles=dict(fixed_rows=fixed_tiles, live_rows=live_tiles, unique_prefixes=unique_tiles,
            prefix_only_tile_reduction_fraction=(1-unique_tiles/live_tiles) if live_tiles else 0.,
            live_compaction_tile_reduction_fraction=(1-live_tiles/fixed_tiles) if fixed_tiles else 0.),
        steps=steps,
        qualifications=["Rows must share compatible checkpoint, precision, attention policy, context semantics and operator contracts before any engine state can be shared.",
                        "Different prompt histories stay separate even if later token suffixes coincide; divergent prefixes never merge based on a matching next token.",
                        "Every row retains its seed, sampling step, constraints and stopping state. A shared distribution still requires independent row draws.",
                        "Canonical tile counts are a structural scheduling model. N=32 rounding, row layout, grouped math attention and expert dispatch require independent parity validation.",
                        "Observed sample branching does not recover the full vocabulary distribution or certify safe probability-tail removal."])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=Path, nargs="+", required=True, help="Generated token files, one per row.")
    p.add_argument("--prompt-tokens", type=Path, nargs="+", required=True, help="One prompt file, or one per row in the same order.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists(): p.error("Output already exists")
    read = lambda path: list(map(int,path.read_text().split()))
    report = analyze([read(p) for p in args.tokens], [read(p) for p in args.prompt_tokens])
    report["inputs"] = {str(p):sha256(p) for p in args.tokens + args.prompt_tokens}
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("steps","inputs","qualifications")},indent=2))


if __name__ == "__main__": main()
