"""validation/prefill-arithmetic-v1: each variant's NLL minus plain's on the 65-token windows, paired, with a 95%
bootstrap interval over windows.   python3 tools/pascal/analysis/arith_report.py ARITH_DIR"""
import json
import sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); W, L = 4622, 65
def load(v):  # every value finite, and the windows of 65 tokens (red-team-2 C9)
    meta = D / v / "nll-pascal-65.meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("length") != L: sys.exit(f"{v}: not windows of {L} tokens")
    x = np.fromfile(D / v / "nll-pascal-65.f32", dtype=np.float32).astype(np.float64)
    if x.size != W * (L - 1) or not np.isfinite(x).all(): sys.exit(f"{v}: not {W} windows of {L - 1} finite values")
    return x.reshape(W, L - 1)
rng = np.random.default_rng(0)
def ci(per_window, n=10000):
    idx = rng.integers(0, per_window.size, size=(n, per_window.size)); m = per_window[idx].mean(axis=1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)
plain = load("plain"); print(f"plain: mean NLL {plain.mean():.9f} (perplexity {np.exp(plain.mean()):.4f}), {plain.size} tokens")
print("variant     minus plain   95% CI                       positions 1-16   17-64        slope vs plain NLL   tokens identical")
for v in ("exact", "rz", "away", "head", "nohead", "mlp", "conv", "attnproj", "attn"):
    if not (D / v / "nll-pascal-65.f32").exists(): continue
    x = load(v); d = x - plain; lo, hi = ci(d.mean(axis=1)); s = np.polyfit(plain.ravel(), d.ravel(), 1)[0]
    print(f"{v:10s}  {d.mean():+.2e}   [{lo:+.2e}, {hi:+.2e}]   {d[:, :16].mean():+.2e}      {d[:, 16:].mean():+.2e}    {s:+.2e}            {np.mean(d == 0) * 100:5.1f}%")
names = [v for v in ("head", "mlp", "conv", "attnproj", "attn") if (D / v / "nll-pascal-65.f32").exists()]
if len(names) == 5 and (D / "exact" / "nll-pascal-65.f32").exists():
    total = sum((load(v) - plain).mean() for v in names)
    print(f"\nsum of the five single-site differences {total:+.2e}; exact {(load('exact') - plain).mean():+.2e}")
if (D / "nohead" / "nll-pascal-65.f32").exists() and (D / "head" / "nll-pascal-65.f32").exists():
    print(f"head + nohead {(load('head') - plain).mean() + (load('nohead') - plain).mean():+.2e}")
# The README's tables (red-team-2 C10: its "Position 1 alone" and "Positions 2-64" columns, and the bosexact table, had
# no committed tool). Position 1 is column 0 (predicted from [BOS] alone); positions 2-64 are columns 1-63. Each
# interval here is a window bootstrap with its own generator, default_rng(0), so each reproduces on its own.
def ci0(x, n=10000):
    r = np.random.default_rng(0); idx = r.integers(0, x.size, size=(n, x.size)); m = x[idx].mean(axis=1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)
print("\nvariant     position 1 alone   positions 2-64   95% CI")
for v in ("exact", "rz", "away", "bosexact", "bosplain"):
    if not (D / v / "nll-pascal-65.f32").exists(): continue
    d = load(v) - plain; lo, hi = ci0(d[:, 1:].mean(axis=1))
    print(f"{v:10s}  {d[:, 0].mean():+.3e}         {d[:, 1:].mean():+.3e}       [{lo:+.3e}, {hi:+.3e}]")
if all((D / v / "nll-pascal-65.f32").exists() for v in ("exact", "bosexact")):
    e, b = load("exact"), load("bosexact"); d = (b - e)[:, 1:].mean(axis=1); lo, hi = ci0(d)
    print(f"bosexact - exact, positions 2-64: {d.mean():+.2e} [{lo:+.2e}, {hi:+.2e}]; position 1 equal value for value: "
          f"{bool(np.all(b[:, 0] == e[:, 0]))}")
