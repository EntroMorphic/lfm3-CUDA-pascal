#!/usr/bin/env python3
"""JPEGs whose EXIF Orientation entry is well formed or not, to compare the native reading with Pillow's
(validation/images-v1; the m02528 finding of 2026-11-15). One small base JPEG, re-wrapped with an APP1 "Exif" block
whose IFD0 holds a single 0x0112 entry, over a grid:
  type      1 BYTE, 2 ASCII, 3 SHORT, 4 LONG, 5 RATIONAL, 6 SBYTE, 7 UNDEFINED, 8 SSHORT, 9 SLONG, 11 FLOAT, 99
  count     0, 1, 2, 513
  layout    inline (value field), or at an offset inside the block, or at an offset past its end
  value     6 (each stored value), little- and big-endian
  xmp       none, or an XMP packet saying 8 after the EXIF segment (Pillow consults XMP only when EXIF lacks the tag)
Plus entries with a preceding tag whose data lies out of bounds, single cases (s-*: TIFF headers good, swapped, bad,
short and BigTIFF; a doubled "Exif" prefix; IFDs out of range or cut short; duplicate tags; rational, float, double,
LONG8 and out-of-range values), and the bad-header case where Pillow raises: a JPEG whose JFIF gives a DPI, and a PNG
eXIf (s-png-*.png). Named g-<type>-<count>-<layout>-<order>-<xmp>.jpg. Needs Pillow (the PNG cases).
Usage: make_exif_orientation_grid.py BASE_JPEG OUT_DIR
"""
import struct
import sys
from pathlib import Path

UNIT = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 11: 4, 99: 1}


def values(typ, count, big):
    e = ">" if big else "<"
    if typ in (1, 6, 7):
        return bytes([6]) * count
    if typ == 2:
        return (b"6" * max(count - 1, 0) + b"\0")[:count]
    if typ in (3, 8):
        return struct.pack(e + "h" * count, *([6] * count))
    if typ in (4, 9):
        return struct.pack(e + "i" * count, *([6] * count))
    if typ == 5:
        return struct.pack(e + "I" * (2 * count), *([6, 1] * count))
    if typ == 11:
        return struct.pack(e + "f" * count, *([6.0] * count))
    return bytes([6]) * count


def tiff(entries, big, extra=b""):
    """entries: list of (tag, type, count, value_bytes, layout). TIFF header, IFD0 at 8, data after it."""
    e = ">" if big else "<"
    head = (b"MM\0*" if big else b"II*\0") + struct.pack(e + "I", 8)
    ifd_len = 2 + 12 * len(entries) + 4
    data_at = 8 + ifd_len
    ifd, data = struct.pack(e + "H", len(entries)), b""
    for tag, typ, count, val, layout in entries:
        if layout == "inline":
            field = (val + b"\0\0\0\0")[:4]
        elif layout == "offset":
            field = struct.pack(e + "I", data_at + len(data))
            data += val
        else:  # past the end
            field = struct.pack(e + "I", 4096)
        ifd += struct.pack(e + "HHI", tag, typ, count) + field
    return head + ifd + struct.pack(e + "I", 0) + data + extra


def jpeg_with(base, exif_tiff, xmp):
    seg = b"Exif\0\0" + exif_tiff
    out = base[:2] + b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg
    if xmp:
        x = (b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><rdf:Description tiff:Orientation=\"8\"/></x:xmpmeta>")
        out += b"\xff\xe1" + struct.pack(">H", len(x) + 2) + x
    return out + base[2:]


def main(base_path, out_dir):
    base = Path(base_path).read_bytes()
    assert base[:2] == b"\xff\xd8"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for typ in UNIT:
        for count in (0, 1, 2, 513):
            for big in (False, True):
                val = values(typ, count, big)
                for layout in ("inline", "offset", "past"):
                    if layout == "inline" and len(val) > 4:
                        continue
                    if layout != "inline" and len(val) <= 4:
                        continue
                    for xmp in (False, True):
                        t = tiff([(0x0112, typ, count, val, layout)], big)
                        name = f"g-t{typ}-c{count}-{layout}-{'be' if big else 'le'}-{'xmp' if xmp else 'noxmp'}.jpg"
                        (out / name).write_bytes(jpeg_with(base, t, xmp))
                        n += 1
    # a well-formed orientation after a tag whose data is out of bounds, and after one larger than the block
    for big in (False, True):
        for first in ("past", "huge"):
            cnt = 4 if first == "past" else 100000
            ent = [(0x010F, 2, cnt, b"x" * cnt if first == "huge" else b"abcd", "past"),
                   (0x0112, 3, 1, values(3, 1, big), "inline")]
            name = f"g-after-{first}-{'be' if big else 'le'}.jpg"
            (out / name).write_bytes(jpeg_with(base, tiff(ent, big), False))
            n += 1
    # single cases: (name, the APP1 payload after "Exif\0\0", xmp)
    le6 = tiff([(0x0112, 3, 1, values(3, 1, False), "inline")], False)
    be6 = tiff([(0x0112, 3, 1, values(3, 1, True), "inline")], True)
    special = {
        "hdr-swapped-II": b"II\0*" + le6[4:],
        "hdr-swapped-MM": b"MM*\0" + be6[4:],
        "hdr-bad": b"XX*\0" + le6[4:],
        "hdr-bad-xmp": b"XX*\0" + le6[4:],
        "hdr-short": le6[:6],
        "hdr-bigtiff": b"II+\0" + le6[4:],
        "double-prefix": b"Exif\0\0" + le6,
        "ifd-offset-past": le6[:4] + struct.pack("<I", 4096) + le6[8:],
        "ifd-cut-before-entry": le6[:8] + struct.pack("<H", 1) + le6[10:14],
        "ifd-count-2-cut": le6[:8] + struct.pack("<H", 2) + le6[10:22],
        "dup-first-unsupported": tiff([(0x0112, 99, 1, b"\x08", "inline"), (0x0112, 3, 1, values(3, 1, False),
                                                                             "inline")], False),
        "dup-first-empty": tiff([(0x0112, 3, 0, b"", "inline"), (0x0112, 3, 1, values(3, 1, False), "inline")], False),
        "dup-first-loaded-7": tiff([(0x0112, 3, 1, struct.pack("<H", 7), "inline"),
                                    (0x0112, 3, 1, values(3, 1, False), "inline")], False),
        "dup-first-bytes": tiff([(0x0112, 1, 1, b"\x06", "inline"), (0x0112, 3, 1, values(3, 1, False), "inline")],
                                False),
        "rational-12-2": tiff([(0x0112, 5, 1, struct.pack("<II", 12, 2), "offset")], False),
        "rational-13-2": tiff([(0x0112, 5, 1, struct.pack("<II", 13, 2), "offset")], False),
        "rational-6-0": tiff([(0x0112, 5, 1, struct.pack("<II", 6, 0), "offset")], False),
        "srational-neg": tiff([(0x0112, 10, 1, struct.pack("<ii", -12, -2), "offset")], False),
        "float-6.5": tiff([(0x0112, 11, 1, struct.pack("<f", 6.5), "inline")], False),
        "double-3": tiff([(0x0112, 12, 1, struct.pack("<d", 3.0), "offset")], False),
        "long8-5": tiff([(0x0112, 16, 1, struct.pack("<Q", 5), "offset")], False),
        "ifd-type-13-7": tiff([(0x0112, 13, 1, struct.pack("<I", 7), "inline")], False),
        "short-9": tiff([(0x0112, 3, 1, struct.pack("<H", 9), "inline")], False),
        "short-9-xmp": tiff([(0x0112, 3, 1, struct.pack("<H", 9), "inline")], False),
        "short-0": tiff([(0x0112, 3, 1, struct.pack("<H", 0), "inline")], False),
        "long-65542": tiff([(0x0112, 4, 1, struct.pack("<I", 65542), "inline")], False),
        "sshort-minus6": tiff([(0x0112, 8, 1, struct.pack("<h", -6), "inline")], False),
    }
    for name, payload in special.items():
        (out / f"s-{name}.jpg").write_bytes(jpeg_with(base, payload, name.endswith("xmp")))
        n += 1
    # Pillow reads a JPEG's EXIF early, guarded, only when JFIF gives no DPI; with JFIF density units set (1: dots per
    # inch) its first EXIF read is exif_transpose's, unguarded. The same for PNG eXIf.
    j = base.find(b"JFIF\0")
    if j > 0:
        dpi_base = base[:j + 7] + b"\x01" + base[j + 8:]
        for name in ("hdr-bad", "hdr-good"):
            payload = (b"XX*\0" + le6[4:]) if name == "hdr-bad" else le6
            (out / f"s-jfif-dpi-{name}.jpg").write_bytes(jpeg_with(dpi_base, payload, False))
            n += 1
    import io
    import zlib
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (5, 3), (10, 200, 30)).save(buf, "PNG")
    png = buf.getvalue()
    for name in ("hdr-bad", "hdr-good"):
        payload = (b"XX*\0" + le6[4:]) if name == "hdr-bad" else le6
        chunk = len(payload).to_bytes(4, "big") + b"eXIf" + payload + zlib.crc32(b"eXIf" + payload).to_bytes(4, "big")
        (out / f"s-png-exif-{name}.png").write_bytes(png[:33] + chunk + png[33:])
        n += 1
    print(n, "files in", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
