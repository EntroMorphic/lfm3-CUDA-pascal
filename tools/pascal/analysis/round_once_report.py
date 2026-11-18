"""validation/round-once-v1, rule 2: each engine's NLL minus the FP32 model's at position 1, over positions 1-64 and
over the whole window, and its mean |NLL - NLL_FP32| per token.
    python3 tools/pascal/analysis/round_once_report.py LENGTH NAME=FILE..."""
import sys
import numpy as np
L = int(sys.argv[1]); files = dict(a.split("=", 1) for a in sys.argv[2:])
def load(path): return np.fromfile(path, dtype=np.float32).astype(np.float64).reshape(-1, L - 1)
anchor = load(files.pop("fp32"))
print(f"length {L}: {anchor.shape[0]} windows; engine minus the FP32 model, mean NLL")
print(f"{'engine':12s} {'position 1':>12s} {'positions 1-64':>15s} {'65 on':>11s} {'all':>11s} {'mean |d|':>10s} {'tokens equal to plain':>22s}")
plain = load(files["plain"]) if "plain" in files else None
for name, path in files.items():
    d = load(path) - anchor
    same = "" if plain is None else f"{np.mean(load(path) == plain) * 100:21.1f}%"
    print(f"{name:12s} {d[:, 0].mean():+12.3e} {d[:, :64].mean():+15.3e} {d[:, 64:].mean():+11.3e} {d.mean():+11.3e} {np.abs(d).mean():10.4e} {same}")
