#!/usr/bin/env python3
"""torchvision's uint8 resize as HF's image processor calls it (BICUBIC, antialias=True, CPU), hashed per case, so the
native port (lfm::image::resize_bicubic_aa) can be compared exactly (validation/images-v1 rule 1).
Each case: an input of in_w x in_h RGB noise (noise_rgb below, which the native check regenerates), resized to out_w x
out_h. Sizes: a third shrinking, a third growing, a sixth one dimension only, a sixth extreme shrinks (to 1..64 from
up to 3000); all random from SEED.
Writes CASES.txt ("in_w in_h out_w out_h key" per line, for the native check) and OUT.jsonl (the case, the sha256 of
the input and of the output, both as H x W x 3 bytes).
Usage: hash_torch_resize.py N SEED CASES.txt OUT.jsonl
       hash_torch_resize.py --cases CASES.txt OUT.jsonl     (cases given: "in_w in_h out_w out_h key" lines)
"""
import hashlib
import json
import random
import sys

import numpy as np

GOLDEN, C1, C2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def noise_bytes(key, n):
    """SplitMix64 as a counter: word i is the generator's output for state key + (i + 1) * golden; bytes little-endian."""
    words = (n + 7) // 8
    with np.errstate(over="ignore"):
        z = np.uint64(key) + np.arange(1, words + 1, dtype=np.uint64) * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
        z = z ^ (z >> np.uint64(31))
    return z.astype("<u8").view(np.uint8)[:n]


def noise_rgb(key, w, h):
    return noise_bytes(key, w * h * 3).reshape(h, w, 3)


def cases(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 6
        iw, ih = rng.randint(1, 1200), rng.randint(1, 1200)
        if kind in (0, 1):    # shrink
            ow, oh = rng.randint(1, iw), rng.randint(1, ih)
        elif kind in (2, 3):  # grow
            ow, oh = rng.randint(iw, 1600), rng.randint(ih, 1600)
        elif kind == 4:       # one dimension
            ow, oh = (iw, rng.randint(1, 1600)) if rng.random() < 0.5 else (rng.randint(1, 1600), ih)
        else:                 # extreme shrink
            iw, ih = rng.randint(1000, 3000), rng.randint(1, 3000)
            ow, oh = rng.randint(1, 64), rng.randint(1, max(1, min(ih, 64)))
        out.append((iw, ih, ow, oh, (seed << 32) | i))
    return out


def main(n, seed, cases_txt, out_jsonl):
    import torch
    import torchvision
    from torchvision.transforms.v2 import functional as tvF
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if n == "--cases":
        cs = [tuple(int(v) for v in line.split()) for line in open(cases_txt) if line.strip()]
    else:
        cs = cases(int(n), int(seed))
        with open(cases_txt, "w") as f:
            for c in cs:
                f.write(" ".join(map(str, c)) + "\n")
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(dict(env=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                         cpu=torch.backends.cpu.get_cpu_capability())), ) + "\n")
        for i, (iw, ih, ow, oh, key) in enumerate(cs):
            a = noise_rgb(key, iw, ih)
            t = torch.from_numpy(a).permute(2, 0, 1).contiguous()
            r = tvF.resize(t, [oh, ow], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
            b = r.permute(1, 2, 0).contiguous().numpy().tobytes()
            f.write(json.dumps(dict(case=i, size=[iw, ih, ow, oh], in_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                                    out_sha256=hashlib.sha256(b).hexdigest())) + "\n")
    print(len(cs), "cases")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--cases":
        main("--cases", None, sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
