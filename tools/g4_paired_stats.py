#!/usr/bin/env python3
"""Paired per-window NLL differences between engines (G4 evidence). Host numpy; scipy optional.

Usage: g4_paired_stats.py OUT.json NAME=FILE_PATTERN ... -- A-B ...
  FILE_PATTERN contains {L} for the length, e.g. build/ppl-g4-v3/nll-native-gemm-{L}.f32.
  Each A-B pair is reported at 256, 257, 4096 and 32768 tokens: mean ΔNLL per token (windows are equal-sized,
  so the token mean equals the window mean), windows where A is higher / lower, a paired percentile bootstrap
  95% CI over windows (10,000 resamples, seed 0), and a Student-t 95% CI over windows. With few windows
  (9 at 32,768) the percentile bootstrap is too narrow; the t-interval is the conservative reading.
"""
import hashlib
import json
import math
import sys

import numpy as np

LENGTHS = (256, 257, 4096, 32768)


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df)), "scipy"
    except ImportError:  # Cornish-Fisher expansion; within 0.3% of the exact quantile for df >= 8
        z = 1.959963984540054
        return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2), "cornish-fisher"


def main():
    split = sys.argv.index("--")
    out, specs, pairs = sys.argv[1], sys.argv[2:split], sys.argv[split + 1:]
    files = dict(spec.split("=", 1) for spec in specs)
    report = dict(definition=__doc__.split("\n\n")[0], files={}, pairs=[])
    for pair in pairs:
        a, b = pair.split("-", 1)
        for length in LENGTHS:
            paths = files[a].format(L=length), files[b].format(L=length)
            x, y = (np.fromfile(p, np.float32).astype(np.float64) for p in paths)
            for name, p in zip((a, b), paths):
                report["files"].setdefault(p, hashlib.sha256(open(p, "rb").read()).hexdigest())
            if x.size != y.size or x.size % (length - 1):
                raise SystemExit(f"{pair} {length}: NLL files are not paired windows")
            d = (x - y).reshape(-1, length - 1).mean(1)
            n = d.size
            rng = np.random.default_rng(0)
            boot = d[rng.integers(0, n, (10000, n))].mean(1)
            q, source = t975(n - 1)
            half = q * d.std(ddof=1) / math.sqrt(n)
            report["pairs"].append(dict(
                pair=pair, length=length, windows=n, mean=float(d.mean()),
                a_higher=int((d > 0).sum()), a_lower=int((d < 0).sum()), equal=int((d == 0).sum()),
                bootstrap_ci=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                t_ci=[float(d.mean() - half), float(d.mean() + half)], t_quantile_source=source))
    open(out, "w").write(json.dumps(report, indent=2) + "\n")
    print(f"{'pair':<16}{'L':>6}{'n':>6}{'mean':>11}{'A>B':>6}{'A<B':>6}   bootstrap 95%              t 95%")
    for r in report["pairs"]:
        print(f"{r['pair']:<16}{r['length']:>6}{r['windows']:>6}{r['mean']:>+11.2e}{r['a_higher']:>6}{r['a_lower']:>6}"
              f"   [{r['bootstrap_ci'][0]:+.2e}, {r['bootstrap_ci'][1]:+.2e}]   [{r['t_ci'][0]:+.2e}, {r['t_ci'][1]:+.2e}]")


if __name__ == "__main__":
    main()
