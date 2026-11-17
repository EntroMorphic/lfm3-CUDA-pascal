# Third-party code vendored into the native engine

## libjpeg-turbo 3.2.0 (`libjpeg-turbo/`)
- Why: validation/images-v1, decision 1 (Tripp, 2026-11-14). The release Thor's Pillow 13.0.0 bundles, so the native
  front end decodes JPEG to exactly the pixels HF gives the model.
- Source: https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz
  - sha256 `6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e`, equal to GitHub's release digest.
  - GPG signature (`.tar.gz.sig`) good, by "The libjpeg-turbo Project (Signing key for official binaries)", key
    fingerprint `0338 C8D8 D9FD A62C F9C4 21BD 7EC2 DBB6 F4DB F434` (from libjpeg-turbo.org/key/LJT-GPG-KEY).
- Unmodified. Built as a static library with its own CMake build, Release, SIMD on, the libjpeg 6.2 API (Pillow's
  wheels: multibuild's `build_libjpeg_turbo`, a default Release build; Thor's Pillow loads `libjpeg.so.62`).
- License: `LICENSE.md` and `README.ijg` (IJG License, Modified BSD License, zlib License), copied from the release.

## stb_image.h (`stb/`)
- Why: PNG decoding (decision 1). PNG is lossless, so a correct decoder gives Pillow's pixels; the conversions to RGB
  are done by the engine itself, as Pillow does them.
- Source: https://github.com/nothings/stb at commit `de234272d56f` (committer date 2026-10-31), the
  most recent commit to `stb_image.h` at least two weeks old when fetched (2026-11-14); upstream was in a burst of
  changes. Version string v2.30. sha256 of the file as fetched: `8eebe78c8c17a23d34e72a1dc19c001ae62d7ed39c5d41cc457126dd7acecf46`.
- Local patches (each marked "liquid-CUDA" in the file):
  1. `stbi__getn` from upstream commit 51c22cc ("two small UB fixes in getn", 2026-11-01): returns at once for
     `n <= 0`, and compares lengths instead of forming `img_buffer + n` (undefined for a large `n`). The PNG decoder
     reads chunk data through it.
  2. `stbi__mad4sizes_valid` unguarded, as upstream 98a8f2a (2026-11-12) does: the pinned file does not compile with
     only PNG enabled (the PNG path calls it under a guard that excludes it).
  3. Ours, not upstream's (2026-11-15, validation/images-v1, decision of that date): `stbi__parse_zlib` checks the
     zlib trailer's Adler-32 (RFC 1950) when it parses a zlib header (every PNG but CgBI, which the engine refuses).
     The four trailer bytes must be real data (a new `zfake` count of the zero bytes `stbi__fill_bits` invents past
     the end; a stream that used `hit_zeof_once`'s padding is truncated), and the sum over the output must match.
     Upstream checks neither the Adler-32 nor PNG CRCs, so damaged image data decoded silently into wrong pixels; the
     CRCs are checked by the engine before stb sees the file (`image_decode.cpp`, `png_chunks`).
  The later upstream commits were read: the rest touch BMP, TGA, HDR, PNM, GIF, the info API and warnings, none
  compiled here. sha256 after patches 1 and 2: `8089b668482daac1a0ce505cb9c4f8245c1931621775270b3a2f225553522257`;
  after patch 3: `d49dac4db0ada8b5721439bd59cf310c17bb617d46895672197436d8c6fc5a65`.
- Known in the pinned file, worked around rather than patched: `stbi_load*` refuses `req_comp = 0` (an upstream
  regression of 2026-10-28, reverted 2026-11-12); the engine always asks for 4 components and drops alpha.
- Compiled with `STBI_ONLY_PNG` (every other format's decoder left out) and `STBI_NO_STDIO`.
- License: public domain or MIT, the engine's choice (the file's end); MIT is chosen.

## PyTorch (`PYTORCH_LICENSE`)
- Earlier rounds; see the file.

## Pillow (`PILLOW_LICENSE`)
- Why: validation/images-v1, decision of 2026-11-15. The EXIF orientation reading in `src/image_decode.cpp`
  (`tiff_orientation` and the XMP fallback) reproduces Pillow 13.0.0's rules (`Image.Exif.load`,
  `TiffImagePlugin.ImageFileDirectory_v2`, `ImageOps.exif_transpose`), written from reading that code, so the engine
  orients images exactly as HF's loader does.
- License: MIT-CMU (the PIL/Pillow notice, copied from Pillow's `LICENSE`).
