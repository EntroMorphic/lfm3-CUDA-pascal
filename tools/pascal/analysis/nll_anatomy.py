"""Where a perplexity difference comes from: per-position, per-confidence and per-token anatomy of the NLL files
of several engines on the same windows. CPU only.
    python3 tools/pascal/analysis/nll_anatomy.py THOR_DIR PASCAL_DIR
THOR_DIR holds ppl-v5, ppl-fp32-v2, ppl-g4-v3 (from Thor); PASCAL_DIR holds nll-pascal-L.f32."""
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
def ci(per_window, seed=0, n=10000):
    rng = np.random.default_rng(seed); idx = rng.integers(0, per_window.size, size=(n, per_window.size))
    m = per_window[idx].mean(axis=1); return np.quantile(m, 0.025), np.quantile(m, 0.975)

print("1. Each engine's mean NLL minus the FP32 model's, 256 against 257 (same corpus, windows one token longer)")
for L in (256, 257):
    e = engines(L); print(f"   {L}: " + "  ".join(f"{k} {(v - e['fp32']).mean():+.2e}" for k, v in e.items() if k != "fp32"))
e6, e7 = engines(256), engines(257)
print("   change from 256 to 257: " + "  ".join(f"{k} {((e7[k] - e7['fp32']).mean() - (e6[k] - e6['fp32']).mean()):+.2e}" for k in e7 if k != "fp32"))

print("\n2. Length 257 by position in the window (position p is predicted from p tokens of context; mean over 1,155 windows)")
e = engines(257); pos = np.arange(1, 257)
bands = [(1, 16), (17, 64), (65, 128), (129, 192), (193, 240), (241, 255), (256, 256)]
print("   positions      pascal-sdpa   pascal-fp32   sdpa-fp32     thor-fp32     eager-fp32    pascal-thor")
for a, b in bands:
    s = slice(a - 1, b)
    row = [(e["pascal"] - e["sdpa"])[:, s].mean(), (e["pascal"] - e["fp32"])[:, s].mean(), (e["sdpa"] - e["fp32"])[:, s].mean(),
           (e["thor"] - e["fp32"])[:, s].mean(), (e["eager"] - e["fp32"])[:, s].mean(), (e["pascal"] - e["thor"])[:, s].mean()]
    print(f"   {a:3d}-{b:3d} ({b - a + 1:3d})  " + "  ".join(f"{x:+.2e}   " for x in row))
d = (e["pascal"] - e["sdpa"]); tot = d.sum()
print(f"   share of the total pascal-sdpa difference from position 256 alone: {d[:, 255].sum() / tot * 100:.1f}% (it is 1/256 = 0.4% of the tokens)")
print(f"   pascal-sdpa without position 256: {d[:, :255].mean():+.2e}; position 256 alone: {d[:, 255].mean():+.2e}")
lo, hi = ci(d[:, :255].mean(axis=1)); print(f"   pascal-sdpa on positions 1-255 only: CI [{lo:+.2e}, {hi:+.2e}]")
d6 = engines(256); x = (d6["pascal"] - d6["sdpa"]); print(f"   for comparison, length 256, all 255 positions: {x.mean():+.2e}")

print("\n3. Is it a few tokens or all of them? pascal minus X at 257")
for other in ("sdpa", "thor", "fp32"):
    d = (e["pascal"] - e[other]).ravel(); a = np.abs(d); order = np.argsort(-a)
    top = [d[order[:k]].sum() / d.sum() * 100 for k in (10, 100, 1000)]
    print(f"   vs {other:5s}: mean {d.mean():+.2e}, median {np.median(d):+.2e}, tokens higher {np.mean(d > 0) * 100:.2f}% lower {np.mean(d < 0) * 100:.2f}% equal {np.mean(d == 0) * 100:.2f}%, "
          f"rms {np.sqrt((d ** 2).mean()):.2e}; trimmed mean (drop top and bottom 1%) {np.sort(d)[len(d) // 100: -len(d) // 100].mean():+.2e}")

print("\n4. Does the difference depend on how confident the model is? Tokens binned by the FP32 model's NLL (length 257 and 256)")
for L in (256, 257):
    e = engines(L); ref = e["fp32"].ravel(); q = np.quantile(ref, [0, .25, .5, .75, .9, .99, 1])
    print(f"   {L}: FP32 NLL range      tokens   pascal-fp32   thor-fp32     sdpa-fp32     pascal-thor")
    for lo_, hi_ in zip(q[:-1], q[1:]):
        m = (ref >= lo_) & (ref <= hi_)
        print(f"        {lo_:6.3f}-{hi_:6.3f}  {m.sum():7d}   " + "   ".join(f"{(e[k] - e[j]).ravel()[m].mean():+.2e}" for k, j in (("pascal", "fp32"), ("thor", "fp32"), ("sdpa", "fp32"), ("pascal", "thor"))))
    for k, j in (("pascal", "thor"), ("pascal", "sdpa"), ("thor", "fp32"), ("pascal", "fp32"), ("sdpa", "fp32")):
        dd = (e[k] - e[j]).ravel(); slope = np.polyfit(ref, dd, 1)
        print(f"        {k}-{j}: slope against FP32 NLL {slope[0]:+.3e} per nat (intercept {slope[1]:+.2e}), correlation {np.corrcoef(ref, dd)[0, 1]:+.4f}")
