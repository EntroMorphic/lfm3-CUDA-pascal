"""validation/round-once-v1, rule 2: each engine's NLL minus the FP32 model's at position 1, over positions 1-64 and
over the whole window, and its mean |NLL - NLL_FP32| per token.
    python3 tools/pascal/analysis/round_once_report.py LENGTH NAME=FILE...     (one NAME must be fp32)
Each FILE needs its .meta.json beside it (as the scorers write it): the length it records must be LENGTH and every
file must have scored the same windows, or the positions would not be the ones the columns are named for."""
import hashlib
import json, sys
import numpy as np
L = int(sys.argv[1]); files = dict(a.split("=", 1) for a in sys.argv[2:])
if "fp32" not in files: sys.exit("one NAME must be fp32 (the anchor)")
windows = set()
def load(path):
    meta = json.load(open(path[:-len(".f32")] + ".meta.json"))
    if meta["length"] != L: sys.exit(f"{path}: scored windows of {meta['length']} tokens, not {L}")
    windows.add(meta["windows_sha256"])
    if "nll_sha256" in meta and hashlib.sha256(open(path, "rb").read()).hexdigest() != meta["nll_sha256"]:
        sys.exit(f"{path}: not the file its .meta.json records (nll_sha256)")  # red-team-2 C9
    x = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if x.size % (L - 1) or not np.isfinite(x).all(): sys.exit(f"{path}: not whole windows of {L - 1} finite values")
    return x.reshape(-1, L - 1)
data = {name: load(path) for name, path in files.items()}
if len(windows) != 1 or len({d.shape for d in data.values()}) != 1: sys.exit("the files did not score the same windows")
anchor = data.pop("fp32")
print(f"length {L}: {anchor.shape[0]} windows; engine minus the FP32 model, mean NLL")
print(f"{'engine':12s} {'position 1':>12s} {'positions 1-64':>15s} {'65 on':>11s} {'all':>11s} {'mean |d|':>10s} {'tokens equal to plain':>22s}")
plain = data.get("plain")
for name, x in data.items():
    d = x - anchor
    same = "" if plain is None else f"{np.mean(x == plain) * 100:21.1f}%"
    later = f"{d[:, 64:].mean():+11.3e}" if L - 1 > 64 else f"{'-':>11s}"
    print(f"{name:12s} {d[:, 0].mean():+12.3e} {d[:, :64].mean():+15.3e} {later} {d.mean():+11.3e} {np.abs(d).mean():10.4e} {same}")
