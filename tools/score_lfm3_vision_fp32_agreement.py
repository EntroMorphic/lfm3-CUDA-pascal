#!/usr/bin/env python3
"""validation/images-v1 rule 3b: which engine's MMStar answers are closer to the model's own FP32 answers.

Inputs, in tools/run_lfm3_vision_bench_hf.py's JSONL row format (the native rows from tools/lfm3_vision_bench_native.py):
the FP32 reference (HF, FP32 on the GPU), HF BF16 (rule 3's reference run) and native. Per MMStar item, an engine
agrees with FP32 when its parsed answer equals FP32's (an unanswered item counts as the answer "none", so two
unanswered replies agree).
Primary, the pass rule (set before the FP32 run): native's agreement rate >= HF BF16's.
Reported, not gating: the paired-bootstrap 95% interval of native minus HF BF16 agreement in points (10,000 resamples
of items, seed 0); each engine's replies identical to FP32's; the median token position where each engine's reply
first departs from FP32's (over items where it does); FP32's own accuracy beside the two engines'.
Usage: score_lfm3_vision_fp32_agreement.py FP32.jsonl HF_BF16.jsonl NATIVE.jsonl [--json OUT.json]
"""
import json
import random
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "key" in r and r["benchmark"] == "mmstar":
                rows[r["key"]] = r
    return rows


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    if len(argv) != 4:
        sys.exit(__doc__)
    ref, bf16, nat = (load(p) for p in argv[1:])
    keys = sorted(ref)
    if set(bf16) != set(keys) or set(nat) != set(keys) or len(keys) != 1500:
        sys.exit(f"item sets differ or are incomplete: FP32 {len(ref)}, BF16 {len(bf16)}, native {len(nat)}")
    ans = lambda r: r["answer"] if r["answer"] is not None else "none"
    agree = {name: [ans(rows[k]) == ans(ref[k]) for k in keys] for name, rows in (("hf_bf16", bf16), ("native", nat))}
    rate = {name: 100 * sum(v) / len(v) for name, v in agree.items()}
    diff = rate["native"] - rate["hf_bf16"]
    rng = random.Random(0)
    n = len(keys)
    pairs = list(zip(agree["native"], agree["hf_bf16"]))
    boots = []
    for _ in range(10_000):
        s = sum(a - b for a, b in (pairs[rng.randrange(n)] for _ in range(n)))
        boots.append(100 * s / n)
    boots.sort()
    ci = (boots[249], boots[9_749])
    report = dict(items=n, agreement=rate, difference_points=diff, ci95=ci,
                  native_only_agrees=sum(a and not b for a, b in pairs), hf_bf16_only_agrees=sum(b and not a for a, b in pairs))
    for name, rows in (("hf_bf16", bf16), ("native", nat)):
        same = sum(rows[k]["generated_ids"] == ref[k]["generated_ids"] for k in keys)
        firsts = sorted(d for d in (first_difference(rows[k]["generated_ids"], ref[k]["generated_ids"]) for k in keys)
                        if d is not None)
        report[name + "_identical_replies"] = same
        report[name + "_median_first_difference"] = firsts[len(firsts) // 2] if firsts else None
    report["accuracy"] = {name: 100 * sum(rows[k]["correct"] for k in keys) / n
                          for name, rows in (("fp32", ref), ("hf_bf16", bf16), ("native", nat))}
    report["passed"] = rate["native"] >= rate["hf_bf16"]
    print(json.dumps(report, indent=1))
    print(f"rule 3b: native agrees with FP32 on {rate['native']:.2f}% of items, HF BF16 on {rate['hf_bf16']:.2f}% "
          f"({diff:+.2f} points, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]) -> {'PASS' if report['passed'] else 'FAIL'}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
