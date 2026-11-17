#!/usr/bin/env python3
"""validation/images-v1 rule 2: the native vision encoder's stages (lfm3-vision-check) against HF's FP32 dumps (the
gold reference), next to HF's BF16 dumps against the same (the error HF itself runs with on a GPU).
Per stage: relative L2 error ||a - ref|| / ||ref|| over the whole tensor, and the minimum per-row (per-token) cosine
to the reference, for native and for HF BF16, and the ratio native / BF16.
Rule 2, on the projected features (image_features, the native ones rounded to BF16 as the language model receives
them): ratio <= 1.0, and native minimum cosine >= HF BF16's.
Usage: compare_lfm3_vision_stages.py NATIVE_DIR FP32_CASE_DIR BF16_CASE_DIR [--json OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

STAGES = (["patchifier", "embedding_norm", "rotary.0", "rotary.1"] + [f"layer{i:02d}" for i in range(14)]
          + ["proj.layer_norm", "proj.linear_1", "proj.gelu", "proj.linear_2", "image_features"])


def hf(case_dir, stage):
    p = Path(case_dir) / ("image_features.npy" if stage == "image_features" else f"0.{stage}.npy")
    return np.load(p).astype(np.float64)


def metrics(a, ref):
    a, ref = a.reshape(ref.shape[0], -1), ref.reshape(ref.shape[0], -1)
    rel = float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
    num = (a * ref).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(ref, axis=1)
    cos = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
    return rel, float(cos.min())


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    if len(argv) != 4:
        sys.exit(__doc__)
    nat_dir, f32_dir, b16_dir = argv[1:]
    meta = json.loads((Path(nat_dir) / "stages.json").read_text())
    rows = {}
    print(f"{'stage':16s} {'native rel':>11s} {'bf16 rel':>11s} {'ratio':>7s} {'native mincos':>14s} {'bf16 mincos':>12s}")
    for s in STAGES:
        ref = hf(f32_dir, s)
        nat = np.fromfile(Path(nat_dir) / f"{s}.f32", dtype=np.float32).astype(np.float64)
        if nat.size != ref.size:
            sys.exit(f"{s}: native has {nat.size} values, HF {ref.size}")
        nat = nat.reshape(ref.shape)
        b16 = hf(b16_dir, s)
        nr, nc = metrics(nat, ref)
        br, bc = metrics(b16, ref)
        ratio = nr / br if br > 0 else float("inf")
        rows[s] = dict(native_rel=nr, bf16_rel=br, ratio=ratio, native_min_cos=nc, bf16_min_cos=bc)
        print(f"{s:16s} {nr:11.3e} {br:11.3e} {ratio:7.3f} {nc:14.8f} {bc:12.8f}")
    f = rows["image_features"]
    passed = f["ratio"] <= 1.0 and f["native_min_cos"] >= f["bf16_min_cos"]
    print(f"rule 2 on image_features: ratio {f['ratio']:.3f} (<= 1), min cos {f['native_min_cos']:.6f} vs BF16 "
          f"{f['bf16_min_cos']:.6f} -> {'PASS' if passed else 'FAIL'}; encode seconds {meta['seconds']}")
    if out_json:
        Path(out_json).write_text(json.dumps(dict(stages=rows, passed=passed, seconds=meta["seconds"]), indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
