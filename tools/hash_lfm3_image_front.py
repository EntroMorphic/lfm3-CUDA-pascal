#!/usr/bin/env python3
"""HF's image front end, hashed per image (validation/images-v1 rule 1): what the model is given, so two machines
(and later the native front end) can be compared exactly.

Per image: the file's sha256; HF's `transformers.image_utils.load_image` (which applies `ImageOps.exif_transpose`,
then `convert("RGB")`) and the sha256 of the resulting RGB bytes and its size; then the checkpoint's image processor
(Lfm3VlImageProcessor, CPU) on that image: image_grid_thw, the token count (t*h*w/4), and the sha256, shape and dtype
of every tensor it returns. A failure is recorded as its exception (type and message), not skipped.
Sources (any number, each NAME=KIND:ARG):
  files:DIR        every *.jpg, *.jpeg, *.png in DIR (sorted)
  bench:DIR        DIR/manifest.jsonl's unique images (extract_vision_benchmarks.py)
  sweep:N:SEED     N synthetic sizes (below), uniform noise from numpy's PCG64 seeded per size (the same pixels on
                   every machine), made in memory as PIL images (no file; "load" is skipped, the RGB hash is kept)
  noise:N:SEED     the sweep's N sizes, but the pixels from SplitMix64 as a counter (noise_rgb; the same as
                   tools/hash_torch_resize.py), keyed (SEED << 32) | (w << 16) | h, so the native check can regenerate
                   them; named LABEL/iii-WxH
The sweep: a quarter uniform in 1..4096 for both sides; a quarter with a side at 32k+16 (rounding ties); a quarter
near the minimum and maximum pixel counts (65,536 and 2,097,152); a quarter near the aspect limit of 200, both sides
of it (HF refuses above 200).
Usage: hash_lfm3_image_front.py MODEL_DIR OUT_JSONL SOURCE...
"""
import hashlib
import io
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sweep_sizes(n, seed):
    rng = np.random.default_rng(seed)
    sizes = []
    q = n // 4
    for _ in range(q):
        sizes.append((int(rng.integers(1, 4097)), int(rng.integers(1, 4097))))
    for i in range(q):
        tie = 32 * int(rng.integers(1, 128)) + 16
        other = int(rng.integers(16, 4097))
        sizes.append((tie, other) if i % 2 else (other, tie))
    for i in range(q):
        target = 65536 if i % 2 else 2097152
        aspect = math.exp(rng.uniform(-math.log(8), math.log(8)))
        w = max(1, round(math.sqrt(target * aspect))) + int(rng.integers(-2, 3))
        h = max(1, round(target / max(w, 1))) + int(rng.integers(-2, 3))
        sizes.append((w, h))
    for i in range(n - 3 * q):
        short = int(rng.integers(1, 21))
        ratio = 200 + rng.uniform(-3, 3) if i % 3 else (200 if i % 2 else 201)
        long = max(1, min(4096, round(short * ratio)))
        sizes.append((long, short) if i % 2 else (short, long))
    return sizes


GOLDEN, C1, C2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def noise_rgb(key, w, h):
    """SplitMix64 as a counter: word i is the output for state key + (i + 1) * golden; bytes little-endian."""
    n = w * h * 3
    with np.errstate(over="ignore"):
        z = np.uint64(key) + np.arange(1, (n + 7) // 8 + 1, dtype=np.uint64) * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
        z = z ^ (z >> np.uint64(31))
    return z.astype("<u8").view(np.uint8)[:n].reshape(h, w, 3)


def main(model_dir, out_jsonl, *sources):
    import PIL
    import PIL.features
    import torch
    import torchvision
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.image_utils import load_image

    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True).image_processor
    env = dict(machine=platform.machine(), python=platform.python_version(), transformers=transformers.__version__,
               torch=torch.__version__, torchvision=torchvision.__version__, pillow=PIL.__version__,
               libjpeg_turbo=PIL.features.version("libjpeg_turbo"), processor=type(proc).__name__)
    out = Path(out_jsonl)
    t0 = time.time()
    n = 0
    with out.open("w") as f:
        f.write(json.dumps(dict(env=env, sources=sources)) + "\n")

        def emit(name, kind, file_bytes, loader):
            nonlocal n
            row = dict(name=name, kind=kind, file_sha256=sha(file_bytes) if file_bytes is not None else None)
            try:
                img = loader()
                row.update(rgb_size=list(img.size), rgb_sha256=sha(img.tobytes()))
            except Exception as e:  # noqa: BLE001  (recorded)
                row.update(load_error=f"{type(e).__name__}: {e}"[:300])
                f.write(json.dumps(row) + "\n")
                n += 1
                return
            try:
                got = proc(images=[img], return_tensors="pt")
                tensors = {}
                for k, v in got.items():
                    if isinstance(v, torch.Tensor):
                        a = v.detach().cpu().contiguous()
                        tensors[k] = dict(shape=list(a.shape), dtype=str(a.dtype), sha256=sha(a.numpy().tobytes()))
                row["tensors"] = tensors
                thw = got.get("image_grid_thw")
                if thw is not None:
                    t, h, w = (int(x) for x in thw[0])
                    row.update(grid_thw=[t, h, w], tokens=t * h * w // 4)
            except Exception as e:  # noqa: BLE001  (recorded)
                row.update(process_error=f"{type(e).__name__}: {e}"[:300])
            f.write(json.dumps(row) + "\n")
            n += 1

        for source in sources:
            label, _, spec = source.partition("=")
            kind, _, arg = spec.partition(":")
            if kind == "files":
                for p in sorted(Path(arg).iterdir()):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        emit(f"{label}/{p.name}", "file", p.read_bytes(), lambda p=p: load_image(str(p)))
            elif kind == "bench":
                seen = set()
                for line in (Path(arg) / "manifest.jsonl").read_text().splitlines():
                    r = json.loads(line)
                    if r["image"] in seen:
                        continue
                    seen.add(r["image"])
                    p = Path(arg) / r["image"]
                    emit(f"{label}/{r['image']}", "file", p.read_bytes(), lambda p=p: load_image(str(p)))
            elif kind == "sweep":
                count, _, seed = arg.partition(":")
                for w, h in sweep_sizes(int(count), int(seed)):
                    def make(w=w, h=h):
                        rng = np.random.default_rng([int(seed), w, h])
                        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
                    emit(f"{label}/{w}x{h}", "sweep", None, make)
            elif kind == "noise":
                count, _, seed = arg.partition(":")
                for i, (w, h) in enumerate(sweep_sizes(int(count), int(seed))):
                    def make(w=w, h=h):
                        return Image.fromarray(noise_rgb((int(seed) << 32) | (w << 16) | h, w, h), "RGB")
                    emit(f"{label}/{i:03d}-{w}x{h}", "noise", None, make)
            else:
                raise SystemExit(f"unknown source kind {kind!r}")
    print(json.dumps(dict(rows=n, seconds=round(time.time() - t0, 1), env=env)))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
