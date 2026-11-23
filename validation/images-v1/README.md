# images-v1

Phase 5: image input. The Thor engine's vision encoder and projector, its image front end and its image path through
the speech runner, built for the GTX 1080 and judged by the reference's image round
([the Thor engine's images-v1](../../reference/), rules 1, 2, 3, 3b, 5, 6 and 7; its listening and reading rule, 4,
is not repeated here: rule 4 below measures understanding directly).

## What it will be

- **Encoder in FP32 on the CUDA cores.** The reference's default (`Precision::kX2`, validation/images-speed-v1) runs
  the encoder's matrix multiplies and attention on BF16 tensor cores, which the GTX 1080 does not have. Its CUDA-core
  path, `kFp32` (FP32 activations over the stored BF16 weights, the path that passed the reference's rule 2 before
  the tensor-core work), is this port's: the tensor-core paths are compiled only with `LFM3_VISION_TENSOR_CORES=1`
  (`native/include/lfm/vision_encoder.cuh`). Nothing else in the encoder changes. Like the rest of the engine, it is
  built with `--fmad=false`, as the Thor engine builds it.
- **Builds.** `tools/pascal/build_speech.sh` builds with `-DLFM3_VISION=1` (`VISION=0` gives phase 4's builds) and
  adds `lfm3-vision-check` (the encoder's stages, for rule 3). Tags: shipping `text-v40`, parity `parity-v40`. The
  text-only server (`build_serve.sh`) has no image input and is not rebuilt.
- **Image cache.** The server keeps encoded images on the GPU (validation/image-cache-v1), 512 MiB by default on Thor.
  Here 32 MiB (four full-size images): with the encoder's scratch at the largest image, about 436 MiB (computed: nine
  FP32 buffers of 8,192 patches), the card has about 497 MiB free at the speech server's worst case
  ([memory-v2](../memory-v2/)).
- **Where images go.** As on Thor: `/speech` image parts (and `lfm3-speech`'s), in the 4,096-token speech runner.
  `/generate` sessions take no images.

The parity build gets the same encoder. There are no Thor pins for the vision path (the Thor engine's own encoder
is now on tensor cores), so the parity build's vision is judged only by rule 2: its text and speech must stay the
Thor engine's.

Baseline: [memory-v2](../memory-v2/)'s `text-v39` and `parity-v39`. HF's records, from the Thor engine's images-v1
([inputs/reference/](inputs/reference/)): `hf-bench-run1.jsonl.gz` (HF BF16 on Thor's GPU, transformers 5.20.2, all
10,500 items; its run 2 identical), `hf-fp32-mmstar.jsonl.gz` (HF FP32, MMStar), `front-aarch64.jsonl.gz` (HF's
front end, per image), and `thor-fp32-compare/` (the Thor engine's `kFp32` encoder against HF, the 11 cases of
rule 3, for comparison). HF's stage dumps (the 11 stage-0 cases, FP32 and BF16, 4.1 GB) and the benchmark images
(MMStar and POPE: 1,930 images, manifest sha256 `6257d1e1...`) are copied to the card's `/workspace/data`, the
sha256 over every file's sha256 equal on both ends: `vision-bench` `e4fd6199...`, `vision-ref/hf-5.20.2/float32`
`b347cc48...`, `bfloat16` `d4e8cba0...`, the case images and `report.json` `4237393b...`. The reference's 20
held-out MMStar images' dumps are on Thor only and are not used.

## Rules (committed before the v40 builds have run on the card)

1. **Build.** `text-v40` and `parity-v40` build with 0 errors and 0 warnings.
2. **Nothing else changes.** Device code against v39: `liquid-lfm3` and `lfm3-score` identical; `lfm3-speech`,
   `liquid-lfm3-worker` and `liquid-lfm3-server` identical function for function but for the vision encoder's
   kernels (`lfm3_vision::kernels`, present only in v40). Outputs identical to v39's, as in memory-v2 rule 3: the
   parity gate (7,826 pins), the fixtures (compact windows unset and 1, memcheck), the scorer at 256, 257 and 4,096,
   the logits above 4,096, speech-v1 rule 3 (16 Thor clips) and its rendering, the shipping and LibriSpeech speech,
   paired decode.
3. **The encoder is at least as accurate as HF's** (the reference's rule 2). `lfm3-vision-check` (shipping build) on
   HF's own pixel values for each of the 11 stage-0 cases, compared by `tools/compare_lfm3_vision_stages.py`: the
   projected features' relative L2 error to HF FP32 at most HF BF16's (ratio <= 1.0), and the minimum per-token
   cosine to HF FP32 at least HF BF16's. The Thor engine's `kFp32` figures are reported beside.
4. **Understanding is as good as HF's** (the reference's rules 3 and 3b). The shipping `lfm3-speech`, all 10,500
   items with HF's requests (`tools/lfm3_vision_bench_native.py`: greedy, text mode, 512 new tokens for MMStar, 16 for
   POPE), the images through this build's own front end:
   - POPE, 9,000 items: native accuracy minus HF BF16's, paired-bootstrap 95% lower bound at least -0.5 points
     (`tools/score_lfm3_vision_bench.py`). F1 and yes-ratio reported per split.
   - MMStar, 1,500 items: agreement with HF FP32's answers at least HF BF16's
     (`tools/score_lfm3_vision_fp32_agreement.py`). Accuracy reported. (The reference's rule 3 on MMStar was mis-sized:
     with 1,500 items an engine as good as HF fails it most of the time; it is not repeated.)
5. **The front end is HF's.** `lfm3-image-decode-check --preprocess` built on the card's host (its compiler and the
   vendored libjpeg-turbo, as the engine links them) on the 1,930 benchmark images: decoded pixels, grid, token count
   and pixel values equal HF's record for every image (`tools/compare_lfm3_image_front.py`).
6. **Refusals and the server path.** `tools/check_lfm3_image_server.py` on the shipping server (the 11 stage-0
   conversations through the server, unstreamed, streamed and concurrent, equal to `lfm3-speech`'s replies) passes;
   the host suite's image tests pass (refusals: aspect ratio, undecodable, truncated, unsupported, zero-size,
   oversized, placeholder mismatch, images where a mode cannot take them).
7. **Memory.** speech-v1 rule 12 at the defaults (one speech row, 29,696, image cache 32 MiB) with the image
   configuration (`IMAGES=1 tools/pascal/speech_memory.py`: the filled text session open, the audio clip at its limit,
   then five distinct full-size images): every configuration's peak at most 7,740 MiB. If it fails, the image cache
   default is lowered (to 0 if need be) before anything else, and rule 7 run again.
8. **Sanitizers.** `lfm3-vision-check` under memcheck on the smallest case, the two-image case and the largest:
   0 errors, outputs identical to the plain run.

Reported, not gated: encode time per image by size, the benchmark's wall time, and how many replies equal HF BF16's.

## Results (2026-11-22/23; jobs `img-v1`, tree `242b17b`, and `img-v1b`, tree `499dc40`)

**All eight rules met.** Three were completed by `job2.sh` after errors of mine in `job.sh`, recorded below. Records:
[records/](records/) (`img-v1.log` and `img-v1b.log` are the two jobs whole).

| Rule | Result |
|---|---|
| 1. Build | `text-v40` and `parity-v40`: every binary 0 warnings |
| 2. Nothing else changes | Device code identical to v39 in all ten binaries, `lfm3-speech`, the worker and the server with exactly the eight `lfm3_vision::kernels` added (`e1/`). Parity gate 7,826 of 7,826; fixtures (compact unset and 1, memcheck); scorer byte-identical at 256, 257 and 4,096 and the logits above 4,096, both builds; speech-v1 rule 3 16 of 16 Thor clips and the rendering 15 of 15; shipping, split and LibriSpeech speech identical to speech-v1's; paired decode passes |
| 3. Encoder accuracy | **Pass on all 11.** Relative error to HF FP32 1.65e-3 to 1.67e-3 (the BF16 rounding of the output), 0.028 to 0.076 of HF BF16's; minimum token cosine 0.9999983 to 0.9999985 (HF BF16's 0.633 to 0.992). Every figure equal, to the digits printed, to the Thor engine's `kFp32` encoder's (`thor-fp32-compare/`) |
| 4. Understanding | **Pass.** POPE: native 89.03 (F1 88.93) against HF BF16's 89.11 (89.00), difference -0.08 points, 95% CI [-0.18, +0.01]; 8,981 of 9,000 replies identical. MMStar: agreement with HF FP32's answers **96.00%** against HF BF16's 93.87% (+2.13 points, 95% CI [+0.87, +3.40]); 1,264 replies identical to FP32's (HF BF16: 1,169). Accuracy 58.93% (HF BF16 59.27, FP32 59.07); unanswered 23 (HF BF16 14) |
| 5. Front end | **1,930 of 1,930** images: decoded pixels, grid, token count and pixel values equal HF's record |
| 6. Server and refusals | The 11 conversations: prompts equal HF's and pixel values byte-identical to HF's (all 11, `e2e/`); through the server, unstreamed, streamed and concurrent, every reply equal to `lfm3-speech`'s (`image-server-h16.json`). Host image tests: 93 passed (`host-image-tests.txt`) |
| 7. Memory | **Pass:** highest peak **7,689 MiB** (limit 7,740): the clip at its limit after five full-size images beside the filled 29,696-token session. Every other configuration as memory-v2 (7,243 at most) |
| 8. Sanitizers | memcheck on `small`, `two` and `cherry`: 0 errors, every stage identical to the plain run |

POPE per split (native / HF BF16): adversarial 86.23 / 86.40, popular 88.77 / 88.80, random 92.10 / 92.13; yes-ratio
49.1% / 49.0%.

**The errors and what was run again** (`job2.sh`, after `job.sh`; `job.sh` unchanged as the record of what ran):
- Rule 3: `job.sh` ran the stage comparison with the container's `python3`, which has no numpy, so nothing was compared
  (the encoder's runs had completed and written every stage). Compared again on those outputs with the venv's Python.
- Rule 6: `job.sh`'s server check sent the 11 requests at once to a server that takes 8 connections (`--handlers`,
  default 8) and closes any beyond them at once; three were cut off and the check stopped on them. Its sequential part
  (each case unstreamed and streamed) had passed. Run again with `--handlers 16`: every concurrent reply equal (front
  end refusals retried as before: 429 `frontend_busy`, up to 120 times, as the Thor engine's record also shows).
- Rule 7: `job.sh`'s image configuration sent the audio clip before the images and peaked at 7,243 MiB, the audio-in
  configuration's figure. Development 1 (`dev/dev1.sh`, `img-dev2.log`): the vision encoder's scratch and the cache,
  about 452 MiB, stay resident once a full-size image has been encoded, so the worst case is the clip after the
  images (estimated 7,695). Measured with the clip again after them: 7,689.

**Reported, not gated.** Encode time on the GTX 1080 (FP32): 280 patches 0.07 s, 1,040 patches 0.27 s, 2 images of
1,040 0.53 s, 7,920 patches (the largest case) 3.3 s. The 10,500 benchmark items in 5.1 hours (`lfm3-speech --batch`,
one at a time). Of the 11 conversations, the replies to `noise`, `tall` and `wide` are off the image (a football
season, an election, a statue); the Thor engine's replies to them are the same, nearly word for word: the model's.

**For the owner.**
1. The image cache's default is 32 MiB here, Thor's 512 (rule 7: 51 MiB to spare with it).
2. The server closes a connection beyond `--handlers` without a reply (`lfm3_http.hpp`, `Server::serve`): a client
   sending its ninth request at once sees a reset, not a 429 or 503. The Thor engine's code, unchanged here.
