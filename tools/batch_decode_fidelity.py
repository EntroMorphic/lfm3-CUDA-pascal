#!/usr/bin/env python3
"""Decode-path fidelity of the batched decoder (P0 speed gate), read like G4 tier 2b but on decode positions.

Inputs: lfm3-batch-teacher's STEM.batch.f32 and STEM.serial.f32 (ROWS x (LENGTH - PREFIX), the NLL of tokens
PREFIX..LENGTH-1 decoded one step at a time after a Runner prefill of PREFIX tokens), and ppl_reference outputs for the
same windows: the exact FP32 anchor (hf-eager_blocked-fp32), HF SDPA and HF eager (BF16). Their index t - 1 scores
token t.
Per token: A = |NLL_engine - NLL_fp32|; per window the mean; D = mean(A_engine) - mean(A_sdpa), 95% bootstrap CI over
windows. FAIL iff the whole CI of D lies above +1e-5 nats (tier 2b's rule). Reported for the batched decoder (the
verdict) and, for comparison, the accepted serial decode and HF eager.
Usage: batch_decode_fidelity.py STEM LENGTH PREFIX ANCHOR.f32 SDPA.f32 EAGER.f32 WINDOWS.i32 SUMMARY_SHA256 [OUT.json]
  SUMMARY_SHA256: the windows file's hash as the reference summary.json records it (checked).
"""
import hashlib
import json
import sys

import numpy as np

EPS = 1e-5


def ci(d, n=4000, seed=0):
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(stem, length, prefix, anchor, sdpa, eager, windows, windows_sha, out=None):
    length, prefix = int(length), int(prefix)
    if hashlib.sha256(open(windows, "rb").read()).hexdigest() != windows_sha:
        sys.exit("windows file does not match the reference summary")
    b = np.fromfile(f"{stem}.batch.f32", np.float32).astype(np.float64)
    s = np.fromfile(f"{stem}.serial.f32", np.float32).astype(np.float64)
    rows = b.size // (length - prefix)
    b, s = b.reshape(rows, -1), s.reshape(rows, -1)
    ref = {k: np.fromfile(p, np.float32).reshape(-1, length - 1)[:rows, prefix - 1:length - 1].astype(np.float64)
           for k, p in (("fp32", anchor), ("sdpa", sdpa), ("eager", eager))}
    err = {"batched": np.abs(b - ref["fp32"]), "serial": np.abs(s - ref["fp32"]),
           "sdpa": np.abs(ref["sdpa"] - ref["fp32"]), "eager": np.abs(ref["eager"] - ref["fp32"])}
    per = {k: v.mean(1) for k, v in err.items()}
    report = dict(rows=rows, positions_per_row=length - prefix, prefix=prefix, eps=EPS,
                  mean_abs_from_fp32={k: float(v.mean()) for k, v in err.items()},
                  perplexity={k: float(np.exp(v.mean())) for k, v in
                              (("batched", b), ("serial", s), ("fp32", ref["fp32"]), ("sdpa", ref["sdpa"]))},
                  batched_vs_serial_mean_abs=float(np.abs(b - s).mean()), comparisons={})
    for k in ("batched", "serial", "eager"):
        d = per[k] - per["sdpa"]
        c = ci(d)
        report["comparisons"][f"{k}_minus_sdpa"] = dict(mean=float(d.mean()), ci95=c,
                                                        verdict="worse" if c[0] > EPS else "better" if c[1] < -EPS else "equal")
    report["pass"] = report["comparisons"]["batched_minus_sdpa"]["verdict"] != "worse"
    print(json.dumps(report, indent=1))
    if out:
        open(out, "w").write(json.dumps(report, indent=1) + "\n")
    return report["pass"]


if __name__ == "__main__":
    if len(sys.argv) not in (9, 10):
        sys.exit(__doc__)
    sys.exit(0 if main(*sys.argv[1:]) else 1)
