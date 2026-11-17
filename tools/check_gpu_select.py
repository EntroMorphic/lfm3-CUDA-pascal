#!/usr/bin/env python3
"""Chi-squared test of the device sampler's draws (native/tests/lfm3_gpu_select_check.cu) against the host sampler's
exact distribution. Bins with expected count >= 5 are kept, the rest folded into one tail bin. PASS iff every case's
p > 1e-3 / cases (Bonferroni) and no zero-probability id was drawn.
Usage: check_gpu_select.py OUT_DIR CASES"""
import sys

import numpy as np
from scipy import stats


def main(d, cases):
    ok = True
    ps = []
    for k in range(int(cases)):
        c = np.fromfile(f"{d}/case-{k}.counts.i64", np.int64).astype(np.float64)
        p = np.fromfile(f"{d}/case-{k}.probs.f64", np.float64)
        n = c.sum()
        impossible = int(c[p == 0].sum())
        e = p * n
        keep = e >= 5
        obs = np.append(c[keep], c[~keep].sum())
        exp = np.append(e[keep], e[~keep].sum())
        if exp[-1] < 5:
            j = np.argmin(exp[:-1]); obs[j] += obs[-1]; exp[j] += exp[-1]; obs, exp = obs[:-1], exp[:-1]
        pv = stats.chisquare(obs, exp * obs.sum() / exp.sum()).pvalue
        ps.append(pv)
        ok &= impossible == 0 and pv > 1e-3 / int(cases)
        print(f"case {k}: draws {int(n)}, bins {len(obs)}, chi2 p = {pv:.3f}, zero-probability draws {impossible}, "
              f"entropy {-(p[p > 0] * np.log(p[p > 0])).sum():.2f} nats")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(0 if main(*sys.argv[1:]) else 1)
