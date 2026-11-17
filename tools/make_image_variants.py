#!/usr/bin/env python3
"""The image variants of validation/images-v1 rule 1, written once as fixed files so every machine (and the native
front end) decodes the same bytes. Each targets a path in Pillow 13.0.0 that HF's `load_image` takes
(`ImageOps.exif_transpose`, then `convert("RGB")`), read from Pillow's source:

JPEG: chroma subsampling 4:2:0, 4:2:2, 4:4:4; progressive; optimized Huffman tables; restart markers; quality 50
  and 100; grayscale; CMYK (Pillow always reads 4 components as Adobe-inverted "CMYK;I", then cmyk2rgb); EXIF
  orientations 1-8 (little-endian TIFF), 6 big-endian, 0 and 9 (no transform); orientation only in XMP, attribute
  form (6) and element form (8) (Pillow's getexif falls back to XMP's tiff:Orientation when EXIF has none); EXIF 1
  with XMP 6 (EXIF wins: no transform); odd sizes ending mid-MCU; a truncated file (refused).
PNG: RGB, RGBA (partial alpha: dropped, not composited), LA, L, 1-bit, P, P with tRNS, 16-bit RGB (Pillow keeps the
  high byte), 16-bit RGBA, 16-bit LA (Pillow opens it as RGBA), 16-bit L (Pillow opens it as I;16 and saturates
  every value over 255 to white), Adam7 interlaced, a gAMA chunk (ignored), an eXIf orientation 6, an XMP iTXt
  orientation 8.
Source: scikit-learn's sample image china.jpg (public), cropped to 320x240.
Usage: make_image_variants.py SOURCE_JPG OUT_DIR
"""
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


def tiff_orientation(value, big_endian=False):
    """A minimal EXIF block (Exif header, TIFF, IFD0 with one SHORT Orientation entry)."""
    e = ">" if big_endian else "<"
    head = b"MM\x00\x2a" if big_endian else b"II\x2a\x00"
    ifd = struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0x0112, 3, 1) + struct.pack(e + "H", value) + b"\x00\x00"
    return b"Exif\x00\x00" + head + struct.pack(e + "I", 8) + ifd + struct.pack(e + "I", 0)


def xmp(orientation, element=False):
    tag = (f"<tiff:Orientation>{orientation}</tiff:Orientation>" if element else "")
    attr = ("" if element else f' tiff:Orientation="{orientation}"')
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/"{attr}>{tag}'
            f"</rdf:Description></rdf:RDF></x:xmpmeta>").encode()


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def raw_png(width, height, bit_depth, color_type, rows):
    """A PNG written by hand (16-bit LA and L, which Pillow does not write from those modes)."""
    raw = b"".join(b"\x00" + r for r in rows)
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(raw, 9)) + png_chunk(b"IEND", b""))


def main(source, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Image.open(source).convert("RGB").crop((100, 80, 420, 320))  # 320x240
    small = base.crop((0, 0, 200, 120))
    made = {}

    def put(name, data, note):
        (out / name).write_bytes(data)
        made[name] = note

    def jpeg(name, img, note="", **kw):
        buf = io.BytesIO()
        img.save(buf, "JPEG", **kw)
        put(f"{name}.jpg", buf.getvalue(), note or {k: (v if not isinstance(v, bytes) else len(v)) for k, v in kw.items()})
        return buf.getvalue()

    def png(name, img, note="", **kw):
        buf = io.BytesIO()
        img.save(buf, "PNG", **kw)
        put(f"{name}.png", buf.getvalue(), note or img.mode)

    # JPEG
    jpeg("sub-420-q90", base, quality=90, subsampling=2)
    jpeg("sub-422-q90", base, quality=90, subsampling=1)
    jpeg("sub-444-q90", base, quality=90, subsampling=0)
    jpeg("progressive-420", base, quality=90, subsampling=2, progressive=True)
    jpeg("progressive-444", base, quality=90, subsampling=0, progressive=True)
    jpeg("optimized-420", base, quality=90, subsampling=2, optimize=True)
    jpeg("restart-420", base, quality=90, subsampling=2, restart_marker_blocks=4)
    jpeg("q50-420", base, quality=50, subsampling=2)
    jpeg("q100-444", base, quality=100, subsampling=0)
    jpeg("gray", base.convert("L"), quality=90)
    jpeg("cmyk", base.convert("CMYK"), quality=90)
    for o in range(1, 9):
        jpeg(f"exif-orientation-{o}", small, f"EXIF orientation {o}", quality=90, subsampling=2,
             exif=tiff_orientation(o))
    jpeg("exif-orientation-6-bigendian", small, "EXIF orientation 6, big-endian TIFF", quality=90, subsampling=2,
         exif=tiff_orientation(6, True))
    for o in (0, 9):
        jpeg(f"exif-orientation-{o}", small, f"EXIF orientation {o}: no transform", quality=90, subsampling=2,
             exif=tiff_orientation(o))
    jpeg("xmp-orientation-6-attr", small, "XMP only, attribute form", quality=90, subsampling=2, xmp=xmp(6))
    jpeg("xmp-orientation-8-element", small, "XMP only, element form", quality=90, subsampling=2, xmp=xmp(8, True))
    jpeg("exif-1-xmp-6", small, "EXIF 1 and XMP 6: EXIF wins", quality=90, subsampling=2,
         exif=tiff_orientation(1), xmp=xmp(6))
    for w, h in ((1, 1), (3, 5), (17, 9), (33, 31), (1001, 7)):
        img = base.resize((w, h), Image.BICUBIC) if w <= 320 else Image.new("RGB", (w, h))
        if w > 320:  # wider than the source: tile it
            for x in range(0, w, 320):
                img.paste(base.crop((0, 0, 320, h)), (x, 0))
        jpeg(f"odd-{w}x{h}-420", img, quality=90, subsampling=2)
    full = (out / "sub-420-q90.jpg").read_bytes()
    put("truncated-420.jpg", full[: len(full) // 2], "the first half of sub-420-q90.jpg (refused)")

    # PNG
    arr = np.asarray(base)
    alpha = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    png("png-rgb", base)
    png("png-rgba", Image.fromarray(np.dstack([arr, alpha]), "RGBA"), "RGBA, alpha 0-255 across")
    png("png-la", Image.fromarray(np.dstack([np.asarray(base.convert("L")), alpha]), "LA"))
    png("png-l", base.convert("L"))
    png("png-1bit", base.convert("1"))
    pal = base.convert("P", palette=Image.ADAPTIVE, colors=64)
    png("png-p", pal, "P, 64 colors")
    png("png-p-trns", pal, "P with tRNS (index 0 transparent)", transparency=0)
    rgb16 = (arr.astype(np.uint16) * 257 + np.arange(320, dtype=np.uint16)[None, :, None] % 257)
    put("png-rgb16.png", raw_png(320, 240, 16, 2, [r.astype(">u2").tobytes() for r in rgb16]),
        "16-bit RGB (Pillow keeps the high byte)")
    rgba16 = np.dstack([rgb16, (alpha.astype(np.uint16) * 257)])
    put("png-rgba16.png", raw_png(320, 240, 16, 6, [r.astype(">u2").tobytes() for r in rgba16]), "16-bit RGBA")
    l16 = np.asarray(base.convert("L")).astype(np.uint16) * 257
    la16 = np.dstack([l16, alpha.astype(np.uint16) * 257])
    put("png-la16.png", raw_png(320, 240, 16, 4, [r.astype(">u2").tobytes() for r in la16]),
        "16-bit LA (Pillow opens it as RGBA)")
    put("png-l16.png", raw_png(320, 240, 16, 0, [r.astype(">u2").tobytes() for r in l16]),
        "16-bit L (Pillow: I;16, values over 255 saturate to white)")
    l16_low = np.minimum(np.asarray(base.convert("L")).astype(np.uint16), 255)
    put("png-l16-low.png", raw_png(320, 240, 16, 0, [r.astype(">u2").tobytes() for r in l16_low]),
        "16-bit L with every value under 256 (Pillow keeps the low byte)")
    png("png-interlaced", base, "Adam7")
    buf = io.BytesIO()
    base.save(buf, "PNG")
    data = buf.getvalue()
    interlaced = Image.open(io.BytesIO(data))
    raw = np.asarray(interlaced)
    # Adam7 by hand: Pillow does not write interlaced PNGs
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    idat = b""
    for x0, y0, dx, dy in passes:
        sub = raw[y0::dy, x0::dx]
        if sub.size:
            idat += b"".join(b"\x00" + r.tobytes() for r in sub)
    put("png-interlaced.png", b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 320, 240, 8, 2, 0, 0, 1))
        + png_chunk(b"IDAT", zlib.compress(idat, 9)) + png_chunk(b"IEND", b""), "Adam7 interlaced RGB")
    gamma = data[:33] + png_chunk(b"gAMA", struct.pack(">I", 22727)) + data[33:]
    put("png-gamma.png", gamma, "gAMA 0.22727 (Pillow ignores it)")
    put("png-exif-6.png", data[:33] + png_chunk(b"eXIf", tiff_orientation(6)[6:]) + data[33:], "eXIf orientation 6")
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xmp(8).decode())
    png("png-xmp-8", base, "iTXt XMP orientation 8", pnginfo=info)
    (out / "variants.json").write_text(json.dumps(made, indent=1))
    print(len(made), "variants")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
