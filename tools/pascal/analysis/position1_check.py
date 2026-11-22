"""text-correct-v2 rule 4: the NLL at position 1 (computed from row 0 alone, prompt position 0) equals the Thor
engine's value for value in every window, at 256, 257, 4,096 and 32,768. Position 1 is column 0 of each window's
length - 1 NLL values (float32, window-major, as lfm3-score and the Thor engine's scorer write them).
    python3 tools/pascal/analysis/position1_check.py THOR_DIR PASCAL_DIR
THOR_DIR holds nll-native-gemm-L.f32 and .meta.json (the Thor engine's, hash checked), PASCAL_DIR nll-pascal-L.f32 or nll-pascal-L.f32.gz with
nll-pascal-L.meta.json (whose nll_sha256 must be the file's, and whose length must be L). Exit 0 only if every window
is equal at every length. (Red-team-2 C10: the rule's result had no committed tool.)
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

thor, pascal = Path(sys.argv[1]), Path(sys.argv[2])
ok = True
for L in (256, 257, 4096, 32768):
    p = pascal / f"nll-pascal-{L}.f32"
    raw = p.read_bytes() if p.exists() else gzip.decompress((pascal / f"nll-pascal-{L}.f32.gz").read_bytes())
    meta = json.loads((pascal / f"nll-pascal-{L}.meta.json").read_text())
    assert meta["length"] == L and hashlib.sha256(raw).hexdigest() == meta["nll_sha256"], f"{L}: not the recorded file"
    a = np.frombuffer(raw, dtype=np.float32)
    t = (thor / f"nll-native-gemm-{L}.f32").read_bytes()
    tmeta = json.loads((thor / f"nll-native-gemm-{L}.meta.json").read_text())
    assert hashlib.sha256(t).hexdigest() == tmeta["nll_sha256"], f"{L}: the Thor engine's file is not the one its meta records"
    assert tmeta.get("windows_sha256") == meta["windows_sha256"], f"{L}: the two scored different windows"
    b = np.frombuffer(t, dtype=np.float32)
    assert a.size == b.size and a.size % (L - 1) == 0, f"{L}: sizes {a.size}, {b.size}"
    a, b = a.reshape(-1, L - 1), b.reshape(-1, L - 1)
    equal = int(np.sum(a[:, 0].view(np.uint32) == b[:, 0].view(np.uint32)))  # bit for bit
    ok &= equal == a.shape[0]
    print(f"length {L}: position 1 equal to the Thor engine's in {equal} of {a.shape[0]} windows; "
          f"positions 2+ equal in {np.mean(a[:, 1:] == b[:, 1:]) * 100:.1f}% of values (information)")
print("POSITION 1 (text-correct-v2 rule 4)", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
