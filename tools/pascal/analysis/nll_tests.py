"""Tests of two readings of the perplexity differences (CPU only; bootstrap over windows, 10,000 resamples).
A. "Flattening": does an engine's NLL minus the FP32 model's fall with the FP32 NLL (as a slightly higher softmax
   temperature would make it)? Slope of (engine - fp32) against fp32 NLL, per engine and length, with a CI.
B. Where in the window the differences sit, with CIs.
    python3 tools/pascal/analysis/nll_tests.py THOR_DIR PASCAL_DIR"""
import sys
from pathlib import Path
import numpy as np
T, P = Path(sys.argv[1]), Path(sys.argv[2])
W = {256: 1160, 257: 1155, 4096: 72, 32768: 9}
def load(path, L): return np.fromfile(path, dtype=np.float32).astype(np.float64).reshape(W[L], L - 1)
def engines(L):
    e = {"fp32": load(T / f"ppl-fp32-v2/nll-hf-eager_blocked-fp32-{L}.f32", L), "sdpa": load(T / f"ppl-v5/nll-hf-sdpa-{L}.f32", L),
         "thor": load(T / f"ppl-g4-v3/nll-native-gemm-{L}.f32", L), "pascal": load(P / f"nll-pascal-{L}.f32", L)}
    p = T / f"ppl-v5/nll-hf-eager-{L}.f32"
    if p.exists(): e["eager"] = load(p, L)
    return e
rng = np.random.default_rng(0)
def boot(stat_parts, combine, n=10000):
    """stat_parts: per-window sufficient statistics [W, k]; combine: function of their column sums."""
    Wn = stat_parts.shape[0]; idx = rng.integers(0, Wn, size=(n, Wn)); sums = stat_parts[idx].sum(axis=1)
    vals = combine(sums.T); return combine(stat_parts.sum(axis=0)), np.quantile(vals, 0.025), np.quantile(vals, 0.975)
def slope(x, d):  # per-window sums for an OLS slope of d on x
    parts = np.stack([np.ones_like(x).sum(1), x.sum(1), d.sum(1), (x * x).sum(1), (x * d).sum(1)], axis=1)
    return boot(parts, lambda s: (s[0] * s[4] - s[1] * s[2]) / (s[0] * s[3] - s[1] ** 2))
print("A. slope of (engine - FP32) against the FP32 NLL, per nat; negative = flatter than the FP32 model")
print("   length   " + "".join(f"{k:>34s}" for k in ("sdpa", "eager", "thor", "pascal", "pascal - thor")))
for L in (256, 257, 4096, 32768):
    e = engines(L); row = []
    for k in ("sdpa", "eager", "thor", "pascal"):
        if k not in e: row.append(" " * 34); continue
        v, lo, hi = slope(e["fp32"], e[k] - e["fp32"]); row.append(f"  {v:+.2e} [{lo:+.2e},{hi:+.2e}]")
    v, lo, hi = slope(e["fp32"], e["pascal"] - e["thor"]); row.append(f"  {v:+.2e} [{lo:+.2e},{hi:+.2e}]")
    print(f"   {L:6d} " + "".join(row))
print("\n   the same with the 1% most surprising tokens (by the FP32 model) left out:")
for L in (256, 257, 4096, 32768):
    e = engines(L); cut = np.quantile(e["fp32"], 0.99); m = (e["fp32"] <= cut).astype(float); row = []
    for k in ("sdpa", "eager", "thor", "pascal"):
        if k not in e: row.append(" " * 34); continue
        x = e["fp32"]; d = (e[k] - e["fp32"])
        parts = np.stack([m.sum(1), (x * m).sum(1), (d * m).sum(1), (x * x * m).sum(1), (x * d * m).sum(1)], axis=1)
        v, lo, hi = boot(parts, lambda s: (s[0] * s[4] - s[1] * s[2]) / (s[0] * s[3] - s[1] ** 2)); row.append(f"  {v:+.2e} [{lo:+.2e},{hi:+.2e}]")
    print(f"   {L:6d} " + "".join(row))
print("\nB. mean difference by position band, with 95% CI")
def band(d, a, b):
    parts = np.stack([d[:, a - 1:b].sum(1), np.full(d.shape[0], float(b - a + 1))], axis=1)
    return boot(parts, lambda s: s[0] / s[1])
for L, bands in ((257, [(1, 64), (65, 256)]), (256, [(1, 64), (65, 255)]), (4096, [(1, 64), (65, 1024), (1025, 4095)]), (32768, [(1, 1024), (1025, 8192), (8193, 32767)])):
    e = engines(L)
    for name, d in (("pascal - thor", e["pascal"] - e["thor"]), ("pascal - sdpa", e["pascal"] - e["sdpa"]), ("thor - fp32  ", e["thor"] - e["fp32"]), ("pascal - fp32", e["pascal"] - e["fp32"]), ("sdpa - fp32  ", e["sdpa"] - e["fp32"])):
        print(f"   {L:6d} {name}: " + "   ".join(f"{a}-{b}: {v:+.2e} [{lo:+.2e},{hi:+.2e}]" for (a, b) in bands for (v, lo, hi) in [band(d, a, b)]))
    print()
print("C. per-token spread around the FP32 model (rms of engine - fp32), and between engines")
for L in (256, 257, 4096, 32768):
    e = engines(L); r = lambda a, b: np.sqrt(((e[a] - e[b]) ** 2).mean())
    c = np.corrcoef((e["pascal"] - e["fp32"]).ravel(), (e["thor"] - e["fp32"]).ravel())[0, 1]
    print(f"   {L:6d}: sdpa {r('sdpa', 'fp32'):.4f}  thor {r('thor', 'fp32'):.4f}  pascal {r('pascal', 'fp32'):.4f}   pascal-thor {r('pascal', 'thor'):.4f}   correlation of the two engines' deviations {c:+.3f}")
