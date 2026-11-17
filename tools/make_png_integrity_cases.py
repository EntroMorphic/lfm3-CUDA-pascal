#!/usr/bin/env python3
"""PNGs whose chunk CRCs are all valid but whose zlib stream is cut, altered or unusually laid out, each named with the
outcome the native decoder must give (validation/images-v1, decision of 2026-11-15; the stb Adler-32 patch):
  ok-TAG-*  decodes, to the same pixels as ok-TAG-reference (TAG: noise, stored blocks; smooth, Huffman-coded)
  bad-*     refused (corrupt_image)
Usage: make_png_integrity_cases.py OUT_DIR
"""
import random
import sys
import zlib
from pathlib import Path


def chunk(kind, data):
    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")


def png(w, h, stream, split=None):
    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    parts = [stream] if split is None else [stream[i:i + split] for i in range(0, len(stream), split)]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + b"".join(chunk(b"IDAT", p) for p in parts)
            + chunk(b"IEND", b""))


def variants(w, h, raw, tag):
    """ok-<tag>-reference and the cases that must decode to it or be refused."""
    cases = {}
    full = zlib.compress(raw, 9)
    cases[f"ok-{tag}-reference"] = png(w, h, full)
    cases[f"ok-{tag}-level0-stored"] = png(w, h, zlib.compress(raw, 0))
    cases[f"ok-{tag}-level1"] = png(w, h, zlib.compress(raw, 1))
    fixed = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_FIXED)
    cases[f"ok-{tag}-fixed-huffman"] = png(w, h, fixed.compress(raw) + fixed.flush())
    cases[f"ok-{tag}-split-1-byte-idat"] = png(w, h, full, split=1)
    cases[f"ok-{tag}-split-7-byte-idat"] = png(w, h, full, split=7)
    cases[f"ok-{tag}-extra-after-stream"] = png(w, h, full + b"\x00\x01\x02 trailing")
    for k in range(1, 9):
        cases[f"bad-{tag}-trailer-cut-{k}"] = png(w, h, full[:-k])
    cases[f"bad-{tag}-adler-flipped-bit"] = png(w, h, full[:-1] + bytes([full[-1] ^ 1]))
    cases[f"bad-{tag}-adler-zero"] = png(w, h, full[:-4] + b"\x00\x00\x00\x00")
    stored = zlib.compress(raw, 0)
    cases[f"bad-{tag}-stored-adler"] = png(w, h, stored[:-2] + bytes([stored[-2] ^ 0x80, stored[-1]]))
    cases[f"bad-{tag}-stored-cut-2"] = png(w, h, stored[:-2])
    # an Adler-32 that is right for different data of the same length: one raw byte changed, its stream recompressed,
    # the original trailer kept
    raw2 = bytearray(raw)
    raw2[100] ^= 0x40
    other = zlib.compress(bytes(raw2), 9)
    cases[f"bad-{tag}-other-data-same-trailer"] = png(w, h, other[:-4] + full[-4:])
    return cases


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    w, h = 61, 37
    cases = {}
    # two images: noise (zlib stores it, so the stored-block path) and a gentle gradient (Huffman-coded)
    noise = b"".join(bytes([rng.choice([0, 1, 2, 3, 4])]) + bytes(rng.randrange(256) for _ in range(3 * w))
                     for _ in range(h))
    smooth = b"".join(bytes([y % 5]) + bytes((3 * x + 5 * y + rng.randrange(4)) & 255 for x in range(3 * w))
                      for y in range(h))
    for tag, raw in (("noise", noise), ("smooth", smooth)):
        cases.update(variants(w, h, raw, tag))
    for name, data in cases.items():
        (out / f"{name}.png").write_bytes(data)
    print(len(cases), "cases in", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
