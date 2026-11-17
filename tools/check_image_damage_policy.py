#!/usr/bin/env python3
"""An independent oracle for the native image decoder's departures from HF (validation/images-v1, decision of
2026-11-15). Written apart from native/src/image_decode.cpp, in Python, from the policy as the README states it.

  check_image_damage_policy.py expect LABEL=DIR...
      For every *.png in DIR (sorted), the outcome the policy requires, in lfm3-image-decode-check's JSON form:
      {"name", "rgb_size", "rgb_sha256"} or {"name", "load_error": "policy: ..."}.
        - Chunk CRCs and the zlib stream are checked with Python's zlib (the stream must end, which means its
          Adler-32 matched). IHDR/PLTE/IDAT with a bad CRC, an unknown critical chunk, or a damaged critical chunk:
          refused. A damaged ancillary chunk: dropped. A last chunk running past the file: an IDAT keeps the rest of
          the file; any other ends the reading, as does a chunk type that is not four letters. IEND ends it.
        - Pixels of an accepted file: 8-bit, Pillow's (load_image's steps: open, exif_transpose, RGB) on a sanitised
          copy (the kept chunks with fresh CRCs, the cut IDAT, IEND); 16-bit, a pure-Python decoder (unfiltering
          here, round(v / 257) per sample), oriented by Pillow's reading of the same copy's metadata. If that decode
          fails, the file is refused ("policy: decode: ...").
  check_image_damage_policy.py compare NATIVE.jsonl EXPECT.jsonl HF.jsonl
      PNGs against EXPECT, JPEGs against HF (lfm3 front records, first line the env header); prints every
      disagreement and a summary.
"""
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path

from PIL import Image, ImageOps

SIG = b"\x89PNG\r\n\x1a\n"


class Refuse(Exception):
    pass


def chunk(kind, data):
    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")


def sanitise(b):
    """The policy: returns (sanitised PNG bytes, IHDR fields) or raises Refuse."""
    if b[:8] != SIG:
        raise Refuse("not a PNG")
    out, i, ihdr, first = [SIG], 8, None, True
    while i + 8 <= len(b):
        length = int.from_bytes(b[i:i + 4], "big")
        kind = b[i + 4:i + 8]
        if first and kind == b"CgBI":
            raise Refuse("CgBI")
        first = False
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind):
            break
        avail = len(b) - i - 8
        if length + 4 > avail:
            if kind == b"IDAT":
                out.append(chunk(b"IDAT", b[i + 8:]))
            break
        if kind == b"IEND":
            break
        data = b[i + 8:i + 8 + length]
        ok = zlib.crc32(kind + data) == int.from_bytes(b[i + 8 + length:i + 12 + length], "big")
        critical = not kind[0] & 0x20
        if kind in (b"IHDR", b"PLTE", b"IDAT"):
            if not ok:
                raise Refuse(f"{kind.decode()} CRC")
            if kind == b"IHDR" and length >= 13:
                ihdr = data
            out.append(chunk(kind, data))
        elif critical:
            raise Refuse("unknown critical chunk" if ok else "damaged critical chunk")
        elif ok:
            out.append(chunk(kind, data))
        i += 12 + length
    out.append(chunk(b"IEND", b""))
    s = b"".join(out)
    idat = b"".join(d for k, d in chunks(s) if k == b"IDAT")
    z = zlib.decompressobj()
    try:
        z.decompress(idat)
    except zlib.error as e:
        raise Refuse(f"zlib: {e}")
    if not z.eof:
        raise Refuse("zlib stream does not end")
    if ihdr is None:
        raise Refuse("no IHDR")
    return s, ihdr


def chunks(s):
    i = 8
    while i + 12 <= len(s):
        n = int.from_bytes(s[i:i + 4], "big")
        yield s[i + 4:i + 8], s[i + 8:i + 8 + n]
        i += 12 + n


def decode16(s, ihdr):
    """A non-interlaced 16-bit PNG, to RGB by round(v / 257)."""
    w, h = int.from_bytes(ihdr[0:4], "big"), int.from_bytes(ihdr[4:8], "big")
    depth, color, interlace = ihdr[8], ihdr[9], ihdr[12]
    if depth != 16 or color not in (0, 2, 4, 6) or interlace != 0:
        raise Refuse(f"decode: oracle handles non-interlaced 16-bit only (depth {depth}, color {color})")
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    bpp = 2 * ch
    stride = w * bpp
    raw = zlib.decompress(b"".join(d for k, d in chunks(s) if k == b"IDAT"))
    if len(raw) < h * (stride + 1):
        raise Refuse("decode: not enough data")
    prev = bytearray(stride)
    rgb = bytearray(w * h * 3)
    for y in range(h):
        f = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if f == 0:
            pass
        elif f == 1:
            for x in range(bpp, stride):
                row[x] = (row[x] + row[x - bpp]) & 255
        elif f == 2:
            for x in range(stride):
                row[x] = (row[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                left = row[x - bpp] if x >= bpp else 0
                row[x] = (row[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = row[x - bpp] if x >= bpp else 0
                b_, c = prev[x], (prev[x - bpp] if x >= bpp else 0)
                p = a + b_ - c
                pa, pb, pc = abs(p - a), abs(p - b_), abs(p - c)
                row[x] = (row[x] + (a if pa <= pb and pa <= pc else b_ if pb <= pc else c)) & 255
        else:
            raise Refuse(f"decode: filter type {f}")
        for x in range(w):
            v = [round((row[x * bpp + 2 * k] << 8 | row[x * bpp + 2 * k + 1]) / 257) for k in range(ch)]
            r = v[:1] * 3 if ch <= 2 else v[:3]
            rgb[3 * (y * w + x):3 * (y * w + x) + 3] = bytes(r)
        prev = row
    return w, h, bytes(rgb)


def expect(path):
    b = path.read_bytes()
    s, ihdr = sanitise(b)
    if ihdr[8] == 16:
        w, h, rgb = decode16(s, ihdr)
        meta = Image.open(io.BytesIO(s))
        img = Image.frombytes("RGB", (w, h), rgb)
        img.info = dict(meta.info)
        img = ImageOps.exif_transpose(img)
    else:
        try:
            with Image.open(io.BytesIO(s)) as im:
                im.load()
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:  # noqa: BLE001 - any Pillow refusal is a refusal here
            raise Refuse(f"decode: {type(e).__name__}: {e}")
    return list(img.size), hashlib.sha256(img.tobytes()).hexdigest()


def cmd_expect(args):
    for arg in args:
        label, d = arg.split("=", 1)
        for p in sorted(q for q in Path(d).iterdir() if q.suffix.lower() == ".png"):
            name = f"{label}/{p.name}"
            try:
                size, sha = expect(p)
                print(json.dumps({"name": name, "rgb_size": size, "rgb_sha256": sha}))
            except Refuse as e:
                print(json.dumps({"name": name, "load_error": f"policy: {e}"}))


def rows(path, header):
    lines = Path(path).read_text().splitlines()[1 if header else 0:]
    return {r["name"]: r for r in map(json.loads, lines)}


def same(a, b):
    if bool(a.get("load_error")) != bool(b.get("load_error")):
        return False
    return bool(a.get("load_error")) or (a["rgb_size"] == b["rgb_size"] and a["rgb_sha256"] == b["rgb_sha256"])


def cmd_compare(native, expected, hf):
    nat, exp, ref = rows(native, False), rows(expected, False), rows(hf, True)
    counts = {"png same": 0, "png differ": 0, "jpeg same": 0, "jpeg differ": 0, "png refused": 0, "png shown": 0}
    for k, n in sorted(nat.items()):
        if k.lower().endswith(".png"):
            e = exp[k]
            counts["png refused" if e.get("load_error") else "png shown"] += 1
            key = "png same" if same(n, e) else "png differ"
        else:
            e = ref[k]
            key = "jpeg same" if same(n, e) else "jpeg differ"
        counts[key] += 1
        if key.endswith("differ"):
            print("DIFFER", k, "| expected:", e.get("load_error") or e.get("rgb_size"), "| native:",
                  n.get("load_error") or n.get("rgb_size"))
    print(json.dumps(counts))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "expect":
        cmd_expect(sys.argv[2:])
    elif len(sys.argv) == 5 and sys.argv[1] == "compare":
        cmd_compare(*sys.argv[2:])
    else:
        sys.exit(__doc__)
