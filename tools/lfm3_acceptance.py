#!/usr/bin/env python3
"""Offline acceptance of zero-cost drafts (n-gram / prompt lookup) for tree speculation on LFM3's own greedy outputs
(journal/lfm3-single-stream_synth.md, step 1). Teacher forced: the generated sequence y is the ground truth the
verifier would produce.

Each pass, at position i: the drafts are the continuations that follow earlier occurrences, in prompt + y[:i], of the
longest suffix (n = 8 down to 1) of the context, most recent first. They are merged into a token tree of at most
R - 1 draft nodes (row R is the root, the current position). The verifier accepts the longest tree path equal to
y[i:], plus the model's own next token after it, so a pass commits accepted + 1 tokens. R = 1 is plain decoding.
Reported: tokens per pass by workload and row budget, and the implied tok/s at the measured pass time for that many
rows (batched decoder on Thor: 16.2 ms at M <= 8, 18.3 ms at M <= 32, 19.9 ms at M <= 64; validation/batch-decode-v1).
This models only free drafters; lookahead drafts (the verifier's own outputs at every node) are not simulated.
Usage: lfm3_acceptance.py OUT_JSON [...]   (liquid-lfm3 outputs: input_ids, generated_ids)
"""
import json
import sys
from pathlib import Path

PASS_MS = {1: 16.2, 8: 16.2, 16: 18.3, 32: 18.3, 64: 19.9}


def drafts(ctx, depth, max_n=8):
    """Continuations after earlier occurrences of the context's longest matching suffix, most recent first."""
    out = []
    for n in range(min(max_n, len(ctx) - 1), 0, -1):
        suffix = ctx[-n:]
        for j in range(len(ctx) - n - 1, -1, -1):
            if ctx[j:j + n] == suffix:
                cont = ctx[j + n:j + n + depth]
                if cont:
                    out.append(cont)
        if out:
            return out
    return out


def tree_accept(conts, budget, truth):
    """Insert continuations into a trie up to `budget` nodes; return the longest trie path that matches truth."""
    trie, nodes = {}, 0
    for c in conts:
        node = trie
        for t in c:
            if t not in node:
                if nodes >= budget:
                    break
                node[t] = {}
                nodes += 1
            node = node[t]
    a, node = 0, trie
    for t in truth:
        if t not in node:
            break
        node, a = node[t], a + 1
    return a


def simulate(prompt, y, rows):
    if rows == 1:
        return len(y)
    i, passes = 0, 0
    while i < len(y):
        ctx = prompt + y[:i]
        a = tree_accept(drafts(ctx, rows - 1), rows - 1, y[i:i + rows - 1])
        i += a + 1
        passes += 1
    return passes


def main(files):
    rows_set = (1, 8, 16, 32, 64)
    report = {}
    print("| workload | tokens | " + " | ".join(f"R={r} tok/pass (tok/s)" for r in rows_set) + " |")
    print("|---|---|" + "---|" * len(rows_set))
    tot = {r: [0, 0] for r in rows_set}
    for f in files:
        d = json.load(open(f))
        name = Path(f).stem.replace("out-", "")
        p, y = d["input_ids"], d["generated_ids"]
        cells = []
        report[name] = {}
        for r in rows_set:
            passes = simulate(p, y, r)
            tpp = len(y) / passes
            tot[r][0] += len(y); tot[r][1] += passes
            report[name][r] = dict(passes=passes, tokens_per_pass=tpp, tok_s=tpp * 1000 / PASS_MS[r])
            cells.append(f"{tpp:.2f} ({tpp * 1000 / PASS_MS[r]:.0f})")
        print(f"| {name} | {len(y)} | " + " | ".join(cells) + " |")
    cells = [f"{tot[r][0] / tot[r][1]:.2f} ({tot[r][0] / tot[r][1] * 1000 / PASS_MS[r]:.0f})" for r in rows_set]
    print(f"| **all** | {tot[1][0]} | " + " | ".join(cells) + " |")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
