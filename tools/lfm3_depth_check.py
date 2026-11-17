#!/usr/bin/env python3
"""A1/B1 check of the native depth decoder (liquid-depth) against lfm3_depth_anchor.py's anchors.

Positions: every (frame, pass, codebook) that the pass sampled, i.e. a pass's newly revealed codebooks (4 per pass).
A1  Logit fidelity: relative L2 distance of each position's 2017 logits from FP32 (and FP64). PASS iff native's
    median is at most the worse of HF BF16 SDPA's and eager's (the text engine's tier 2a, for logits).
B1  Distribution fidelity, at the sampling distribution: the repetition penalty (1.2) over the codebook's codes in
    the previous 64 frames, then temperature 0.4, as HF samples.
    - Primary: the exact KL(p_FP64 || p_m) at each position (no sampling). Paired per position against SDPA (and
      eager): 95% bootstrap CI of the mean difference. "worse" iff the CI lies entirely above +eps (FAIL), "better"
      iff entirely below -eps, else "equal" (eps = 1e-5 nats, as §9).
      The penalty history is HF's: the last 64 accepted frames, i.e. EOS frames excluded.
    - As written in the bar: NLL of the reference's own sampled code, native minus SDPA. The codes were sampled
      from SDPA, so this difference is biased toward SDPA by construction: its expectation is KL(p_SDPA || p_m) >= 0.
      Reported, not used as the verdict.
Regression tripwire (not part of the bar, but part of the exit code): native at the level it achieved, so a regression
to BF16-level error (cos/sin, activations, eps), which A1/B1 alone would still pass, fails: median relative L2 from
FP64 <= 1e-6 (achieved 1.6e-7), max <= 1e-5 (6.7e-7), mean KL from FP64 <= 1e-9 nats (1.5e-12; HF FP32 9e-11).
Usage (lfm3-omni-dev): lfm3_depth_check.py TRACE_DIR ANCHOR_DIR NATIVE_DIR OUT_JSON CASE [CASE ...]
  NATIVE_DIR/CASE.native.f32: liquid-depth's output.
"""
import json
import sys
from pathlib import Path

import numpy as np

PENALTY, TEMPERATURE, WINDOW, EPS, EOS_CODE = 1.2, 0.4, 64, 1e-5, 2016
TRIPWIRE = dict(median_vs_fp64=1e-6, max_vs_fp64=1e-5, mean_kl_from_fp64=1e-9)
VARIANTS = ("native", "sdpa", "eager", "fp32")


def sampled_positions(masked):
    """[frames, 3, 12] bool: the codebooks each pass sampled (revealed by it)."""
    picked = masked.copy()
    picked[:, :2] &= ~masked[:, 1:]
    return picked


def relative_l2(x, ref):
    return np.linalg.norm(x - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def log_sampling(logits, history):
    """log softmax(penalize(logits) / T); logits [n, 2017] float64; history [n, 2017] bool."""
    s = np.where(history, np.where(logits < 0, logits * PENALTY, logits / PENALTY), logits) / TEMPERATURE
    s = s - s.max(-1, keepdims=True)
    return s - np.log(np.exp(s).sum(-1, keepdims=True))


def penalty_histories(raw, frame_index, codebook_index):
    """[n, 2017] bool: the codes of each position's codebook in the last WINDOW accepted (non-EOS) frames before it."""
    accepted = np.nonzero(~(raw == EOS_CODE).any(1))[0]
    history = np.zeros((len(frame_index), 2017), bool)
    for n, (fr, c) in enumerate(zip(frame_index, codebook_index)):
        before = accepted[accepted < fr][-WINDOW:]
        history[n, raw[before, c]] = True
    return history


def bootstrap(d, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(ci):
    return "worse" if ci[0] > EPS else "better" if ci[1] < -EPS else "equal"


def main(trace_dir, anchor_dir, native_dir, out_json, cases):
    import torch
    rel = {v: {"fp32": [], "fp64": []} for v in VARIANTS}
    kl = {v: [] for v in VARIANTS}
    nll = {v: [] for v in VARIANTS + ("fp64",)}
    per_case = {}
    for case in cases:
        f = torch.load(Path(trace_dir) / case / "frames.pt")
        masked = f["pass_masked"].numpy().astype(bool)
        raw = f["raw_codes"].reshape(masked.shape[0], 12).numpy()
        frames = masked.shape[0]
        a = Path(anchor_dir)
        logits = dict(fp64=np.load(a / f"{case}.fp64.npy"), fp32=np.load(a / f"{case}.fp32.f32.npy"),
                      sdpa=np.load(a / f"{case}.sdpa.f32.npy"), eager=np.load(a / f"{case}.eager.f32.npy"),
                      native=np.fromfile(Path(native_dir) / f"{case}.native.f32", np.float32).reshape(frames, 3, 12, 2017))
        pick = sampled_positions(masked)
        fi, pi, ci = np.nonzero(pick)
        assert len(fi) == frames * 12, "every codebook is sampled exactly once per frame"
        history = penalty_histories(raw, fi, ci)
        code = raw[fi, ci]
        sel = {k: v[fi, pi, ci].astype(np.float64) for k, v in logits.items()}
        lp = {k: log_sampling(v, history) for k, v in sel.items()}
        p64 = np.exp(lp["fp64"])
        for v in VARIANTS:
            rel[v]["fp32"].append(relative_l2(sel[v], sel["fp32"]) if v != "fp32" else np.zeros(len(fi)))
            rel[v]["fp64"].append(relative_l2(sel[v], sel["fp64"]))
            kl[v].append((p64 * (lp["fp64"] - lp[v])).sum(-1))
        for v in nll:
            nll[v].append(-lp[v][np.arange(len(fi)), code])
        per_case[case] = dict(positions=int(len(fi)),
                              median_rel_vs_fp32={v: float(np.median(rel[v]["fp32"][-1])) for v in ("native", "sdpa", "eager")},
                              mean_kl_from_fp64={v: float(kl[v][-1].mean()) for v in VARIANTS})
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    a1 = {v: dict(median_vs_fp32=float(np.median(cat(rel[v]["fp32"]))), p99_vs_fp32=float(np.percentile(cat(rel[v]["fp32"]), 99)),
                  median_vs_fp64=float(np.median(cat(rel[v]["fp64"]))), max_vs_fp64=float(cat(rel[v]["fp64"]).max()))
          for v in VARIANTS}
    bar = max(a1["sdpa"]["median_vs_fp32"], a1["eager"]["median_vs_fp32"])
    a1_pass = a1["native"]["median_vs_fp32"] <= bar
    k = {v: cat(kl[v]) for v in VARIANTS}
    b1 = dict(mean_kl_from_fp64={v: float(k[v].mean()) for v in VARIANTS})
    for ref in ("sdpa", "eager"):
        ci = bootstrap(k["native"] - k[ref])
        b1[f"native_minus_{ref}"] = dict(mean=float((k["native"] - k[ref]).mean()), ci95=ci, verdict=verdict(ci))
    n = {v: cat(nll[v]) for v in nll}
    as_written = {}
    for v in ("native", "fp64", "fp32", "eager"):
        d = n[v] - n["sdpa"]
        ci = bootstrap(d)
        as_written[f"{v}_minus_sdpa"] = dict(mean=float(d.mean()), ci95=ci, verdict=verdict(ci))
    b1_pass = all(b1[f"native_minus_{r}"]["verdict"] != "worse" for r in ("sdpa", "eager"))
    achieved = dict(median_vs_fp64=a1["native"]["median_vs_fp64"], max_vs_fp64=a1["native"]["max_vs_fp64"],
                    mean_kl_from_fp64=b1["mean_kl_from_fp64"]["native"])
    tripwire = dict(limits=TRIPWIRE, native=achieved, pass_=all(achieved[k] <= v for k, v in TRIPWIRE.items()))
    result = dict(positions=int(len(k["native"])), a1=dict(variants=a1, bar_median=bar, pass_=a1_pass),
                  b1=dict(primary=b1, pass_=b1_pass, as_written_nll_vs_sdpa_biased=as_written), tripwire=tripwire,
                  cases=per_case)
    Path(out_json).write_text(json.dumps(result, indent=1) + "\n")
    print(f"positions {result['positions']}")
    for v in VARIANTS:
        print("A1 %-6s median rel-L2 vs fp32 %.3e (p99 %.3e)  vs fp64 %.3e (max %.3e)" % (
            v, a1[v]["median_vs_fp32"], a1[v]["p99_vs_fp32"], a1[v]["median_vs_fp64"], a1[v]["max_vs_fp64"]))
    print("A1 bar %.3e: %s" % (bar, "PASS" if a1_pass else "FAIL"))
    for v in VARIANTS:
        print("B1 mean KL(fp64 || %-6s) = %.3e nats" % (v, b1["mean_kl_from_fp64"][v]))
    for r in ("sdpa", "eager"):
        x = b1[f"native_minus_{r}"]
        print("B1 native - %-5s %.3e  CI [%.3e, %.3e]  %s" % (r, x["mean"], *x["ci95"], x["verdict"]))
    for key, x in as_written.items():
        print("B1 as written (biased) NLL %-17s %.3e  CI [%.3e, %.3e]  %s" % (key, x["mean"], *x["ci95"], x["verdict"]))
    print("B1:", "PASS" if b1_pass else "FAIL")
    print("tripwire", {k: "%.2e <= %.0e" % (achieved[k], v) for k, v in TRIPWIRE.items()}, "PASS" if tripwire["pass_"] else "FAIL")
    return a1_pass and b1_pass and tripwire["pass_"]


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]) else 1)
