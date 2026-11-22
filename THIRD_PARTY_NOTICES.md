# Third-party notices

This repository, lfm3-CUDA-pascal, is a port of liquid-CUDA (the Thor engine, imported at its commit `e2f8b7f`) and
keeps its MIT license and copyright ([LICENSE](LICENSE): "Liquid Blackwell Inference Engine contributors"; this port's
changes are under the same terms). It vendors or ports the following code, each under its own license.
Provenance (sources, hashes, signatures, local patches) is recorded in
[native/third_party/README.md](native/third_party/README.md).

| Component | Where | Use | License |
|---|---|---|---|
| libjpeg-turbo 3.2.0 | `native/third_party/libjpeg-turbo/` (release tarball, unmodified) | JPEG decoding | IJG, Modified BSD, zlib — [LICENSE.md](native/third_party/libjpeg-turbo/LICENSE.md), [README.ijg](native/third_party/libjpeg-turbo/README.ijg) |
| stb_image v2.30 | `native/third_party/stb/stb_image.h` (patched, marked "liquid-CUDA") | PNG decoding | MIT (chosen from public domain / MIT) |
| PyTorch | ported into `native/src/image_preprocess.cpp` (antialiased bicubic resize) and native kernels citing ATen | Bit-identical reproduction of HF's numerics | BSD-3-Clause — [PYTORCH_LICENSE](native/third_party/PYTORCH_LICENSE) |
| Pillow 13.0.0 | rules reproduced in `native/src/image_decode.cpp` (EXIF orientation) | Orienting images as HF's loader does | MIT-CMU — [PILLOW_LICENSE](native/third_party/PILLOW_LICENSE) |

## Not included

- **Model weights.** LFM3-Omni-2B (and LFM3-Audio-2B, used in one comparison) are published by Liquid AI under their
  own license. They are downloaded separately and never committed. The checkpoint's `config.json` (its metadata, no
  weights) is copied into a few validation records, by checks that record the configuration they ran with
  (`validation/serve-v1/records/*/sampling/config-only/`, `validation/speech-v1/records/sp-v1/carried/serve-*/`), as
  the Thor engine's records do.
- **Benchmark and evaluation data** (LibriSpeech, FLEURS, PG-19, MMStar, POPE, …) are fetched by the tools at pinned
  revisions and checked against recorded checksums; only derived results are committed.
- **Reference stacks** (Transformers, PyTorch, torchvision) run in pinned containers to produce references; they are
  not linked into the engine.
