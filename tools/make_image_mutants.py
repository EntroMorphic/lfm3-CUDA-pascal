#!/usr/bin/env python3
"""Hostile inputs for the image front end (validation/images-v1 rules 1 and 5): COUNT files made from the variant
set by seeded mutations, so HF (Pillow) and the native decoder can be compared file by file. Every outcome must agree:
the same pixels, or both refuse.
Mutations (one or two per file): flip 1-8 random bits; overwrite 1-16 random bytes; truncate at a random point;
insert or delete 1-64 bytes; corrupt a JPEG segment length or a PNG chunk length; swap two marker/chunk regions;
append garbage after the end; zero a random run. Written as m00000-<source>.<ext>, deterministic for a SEED.
Usage: make_image_mutants.py VARIANTS_DIR OUT_DIR COUNT SEED
"""
import random
import sys
from pathlib import Path


def mutate(data, rng):
    b = bytearray(data)
    kind = rng.choice(["bits", "bytes", "truncate", "insert", "delete", "length", "swap", "append", "zero"])
    n = len(b)
    if kind == "bits":
        for _ in range(rng.randint(1, 8)):
            i = rng.randrange(n)
            b[i] ^= 1 << rng.randrange(8)
    elif kind == "bytes":
        for _ in range(rng.randint(1, 16)):
            b[rng.randrange(n)] = rng.randrange(256)
    elif kind == "truncate":
        b = b[: rng.randrange(1, n)]
    elif kind == "insert":
        i = rng.randrange(n)
        b[i:i] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
    elif kind == "delete":
        i = rng.randrange(n)
        del b[i:i + rng.randint(1, 64)]
    elif kind == "length":
        if b[:2] == b"\xff\xd8":  # a JPEG segment's length field
            marks = [i for i in range(2, n - 3) if b[i] == 0xFF and 0xC0 <= b[i + 1] <= 0xFE and b[i + 1] != 0xD8]
            if marks:
                i = rng.choice(marks) + 2
                b[i:i + 2] = rng.choice([b"\x00\x00", b"\x00\x01", b"\xff\xff", bytes([rng.randrange(256), rng.randrange(256)])])
        elif b[:4] == b"\x89PNG":  # a chunk's length field
            offs, i = [], 8
            while i + 8 <= n:
                offs.append(i)
                i += 12 + int.from_bytes(b[i:i + 4], "big")
            if offs:
                i = rng.choice(offs)
                b[i:i + 4] = rng.choice([b"\x00\x00\x00\x00", b"\x7f\xff\xff\xff", b"\xff\xff\xff\xff",
                                         rng.randrange(1 << 32).to_bytes(4, "big")])
    elif kind == "swap":
        i, j = sorted(rng.randrange(n) for _ in range(2))
        k = min(rng.randint(1, 64), n - j)
        b[i:i + k], b[j:j + k] = b[j:j + k], b[i:i + k]
    elif kind == "append":
        b += bytes(rng.randrange(256) for _ in range(rng.randint(1, 256)))
    elif kind == "zero":
        i, k = rng.randrange(n), rng.randint(1, 256)
        b[i:i + k] = bytes(len(b[i:i + k]))
    return bytes(b), kind


def main(variants, out_dir, count, seed):
    src = sorted(p for p in Path(variants).iterdir() if p.suffix in (".jpg", ".png") and "truncated" not in p.name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    kinds = {}
    for i in range(int(count)):
        p = rng.choice(src)
        data, kind = mutate(p.read_bytes(), rng)
        if rng.random() < 0.3:
            data, kind2 = mutate(data, rng)
            kind += "+" + kind2
        name = f"m{i:05d}-{p.stem}{p.suffix}"
        (out / name).write_bytes(data)
        kinds[kind] = kinds.get(kind, 0) + 1
    print(count, "mutants", dict(sorted(kinds.items())))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
