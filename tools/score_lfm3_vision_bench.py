#!/usr/bin/env python3
"""Score MMStar and POPE runs (tools/run_lfm3_vision_bench_hf.py's JSONL, or the native engine's in the same form).
validation/images-v1 rule 3.
Usage: score_lfm3_vision_bench.py A.jsonl [B.jsonl]
One file: MMStar accuracy (with each parse rule's share and per-category accuracy) and POPE accuracy, F1 and
yes-ratio per split and overall. Two files (A = reference, B = candidate): both scores, then item-by-item agreement of
answers and of generated_ids, and the paired-bootstrap 95% interval of B minus A in points (10,000 resamples of items,
seed 0), per benchmark.
POPE: "yes" is the positive class; an unparsed reply counts as wrong and as not "yes".
"""
import json
import random
import sys
from collections import Counter, defaultdict


def load(path):
    rows = {}
    env = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "env" in r:
                env = r["env"]
                continue
            if r["key"] in rows:
                sys.exit(f"{path}: duplicate key {r['key']}")
            rows[r["key"]] = r
    return env, rows


def pope_stats(rows):
    tp = fp = fn = tn = yes = 0
    for r in rows:
        said_yes = r["answer"] == "yes"
        truth_yes = r["truth"] == "yes"
        yes += said_yes
        if said_yes and truth_yes: tp += 1
        elif said_yes: fp += 1
        elif truth_yes: fn += 1
        else: tn += 1
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = sum(r["correct"] for r in rows) / n
    return dict(n=n, acc=100 * acc, f1=100 * f1, precision=100 * prec, recall=100 * rec, yes_ratio=100 * yes / n,
                unparsed=sum(r["answer"] is None for r in rows))


def report(name, rows):
    print(f"== {name}")
    mm = [r for r in rows.values() if r["benchmark"] == "mmstar"]
    if mm:
        acc = 100 * sum(r["correct"] for r in mm) / len(mm)
        print(f"MMStar  n={len(mm)}  accuracy {acc:.2f}%  unanswered {sum(r['answer'] is None for r in mm)}")
        rules = Counter(r["parse_rule"] for r in mm)
        print("  parse rules: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
        cat = defaultdict(list)
        for r in mm:
            cat[r.get("category") or r["split"]].append(r["correct"])
        if len(cat) > 1:
            for k in sorted(cat):
                print(f"  {k:40s} {100 * sum(cat[k]) / len(cat[k]):6.2f}%  (n={len(cat[k])})")
    pope = [r for r in rows.values() if r["benchmark"] == "pope"]
    if pope:
        splits = defaultdict(list)
        for r in pope:
            splits[r["split"]].append(r)
        for k in sorted(splits) + ["all"]:
            s = pope_stats(pope if k == "all" else splits[k])
            print(f"POPE {k:12s} n={s['n']}  acc {s['acc']:.2f}  F1 {s['f1']:.2f}  P {s['precision']:.2f}  "
                  f"R {s['recall']:.2f}  yes {s['yes_ratio']:.1f}%  unparsed {s['unparsed']}")


def bootstrap(a, b, reps=10000, seed=0):
    """Paired bootstrap of mean(b - a) in points; a, b lists of 0/1."""
    d = [int(y) - int(x) for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return 100 * sum(d) / n, means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def compare(ra, rb):
    if set(ra) != set(rb):
        sys.exit(f"key sets differ: {len(set(ra) - set(rb))} only in A, {len(set(rb) - set(ra))} only in B")
    print("== B against A")
    for bench in ("mmstar", "pope"):
        keys = sorted(k for k in ra if ra[k]["benchmark"] == bench)
        if not keys:
            continue
        same_ans = sum(ra[k]["answer"] == rb[k]["answer"] for k in keys)
        same_ids = sum(ra[k]["generated_ids"] == rb[k]["generated_ids"] for k in keys)
        same_reply = sum(ra[k]["reply"] == rb[k]["reply"] for k in keys)
        flips = Counter((ra[k]["correct"], rb[k]["correct"]) for k in keys)
        mean, lo, hi = bootstrap([ra[k]["correct"] for k in keys], [rb[k]["correct"] for k in keys])
        print(f"{bench}: n={len(keys)}  same answer {same_ans}  same reply text {same_reply}  "
              f"same generated_ids {same_ids}")
        print(f"  right->wrong {flips[(True, False)]}  wrong->right {flips[(False, True)]}  "
              f"difference {mean:+.2f} points, 95% CI [{lo:+.2f}, {hi:+.2f}]")


def main(argv):
    if not 2 <= len(argv) <= 3:
        sys.exit(__doc__)
    ea, ra = load(argv[1])
    report(argv[1], ra)
    if len(argv) == 3:
        eb, rb = load(argv[2])
        report(argv[2], rb)
        if ea != eb:
            diff = {k for k in set(ea or {}) | set(eb or {}) if (ea or {}).get(k) != (eb or {}).get(k)}
            print("env differs in:", ", ".join(sorted(diff)))
        compare(ra, rb)


if __name__ == "__main__":
    main(sys.argv)
