# Building and running

## On the GTX 1080 (this port)

| | |
|---|---|
| Hardware | NVIDIA GeForce GTX 1080 (Pascal, `sm_61`), 8 GB |
| Toolchain | CUDA 12.9 (`nvcc` 12.9.86; CUDA 13 no longer targets `sm_61`), GCC 13.3, C++20, in an LXD container ([phase0-foundations-v1](../validation/phase0-foundations-v1/)) |
| Flags | `-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM` |

The builds are the scripts in `tools/pascal/`, not CMake: `build_text.sh TAG` (`liquid-lfm3`, `lfm3-score`),
`build_serve.sh TAG` (and the text-only `liquid-lfm3-worker` and `liquid-lfm3-server`, `-DLFM3_TEXT_ONLY=1`) and
`build_speech.sh TAG` (and `lfm3-speech`, with the worker and server built with speech, `-DLFM3_TEXT_ONLY=0
-DLFM3_VISION=0`). Each writes `/workspace/build/TAG/` with the binaries, their SHA-256, the flags, the toolchain and
the hashes of every source; `EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL` gives the parity build (PLAN decision 4; on the
Thor engine "parity" meant the cuBLAS build below, which this port does not build). The checkpoint is expected at
`/workspace/models/LFM3-Omni-2B`. `tools/pascal/remote.sh` syncs this repository to the card's machine and runs the
scripts there (`tools/pascal/target.env`, from `target.env.example`). `native/CMakeLists.txt` is the Thor engine's:
it sets `sm_110` per target and has no targets for `lfm3-score` or `lfm3-speech`, so it does not build for this card.

The rest of this page is the Thor engine's, as imported.

## Target (the Thor engine)

| | |
|---|---|
| Hardware | NVIDIA Jetson AGX Thor (Blackwell, `sm_110`), 128 GB unified memory |
| Toolchain | CUDA 13.0.88, GCC 13.3, C++20, built inside a pinned L4T r38.3 development image (aarch64 SBSA) |
| Numerics flags | `-O3 --fmad=false -std=c++20 -arch=sm_110`; `--fmad=false` keeps every multiply-add as the reference rounds it |

The compiler image is pinned by digest (`sha256:3c2b5b8e…109052f`), never by tag, and recorded with each build's
`sources.sha256`. Builds run with `--network none`.

## Executables

| Target | Source | What it is |
|---|---|---|
| `liquid-lfm3-server` | `native/src/lfm3_server.cu` | The loopback HTTP server: text sessions, `/speech` (speech in and out, streaming, concurrent rows) |
| `liquid-lfm3-worker` | `native/src/lfm3_worker.cu` | The engine as a resident worker process (the server's engine, out of process) |
| `liquid-lfm3` | `native/src/lfm3_main.cu` | The text engine CLI |
| `lfm3-speech` | `native/apps/lfm3_speech.cu` | Text, audio and image in → text and speech out, one request or a batch; the reference CLI for the speech and image rounds |
| `liquid-codec`, `liquid-depth` | `native/src/codec_main.cu`, `depth_main.cu` | The audio codec and depth decoder alone, for their checks |

## With CMake (the server, worker and text engine)

```bash
cmake -S native -B build/native -DLIQUID_LFM3_NATIVE_GEMM=ON   # ON is the default: no cuBLAS
cmake --build build/native --target liquid-lfm3-server liquid-lfm3-worker liquid-lfm3 -j4
```

Options: `LIQUID_LFM3_NATIVE_GEMM` (default ON; OFF links cuBLAS, the earlier parity build),
`LIQUID_LFM3_OPS_TESTS` (isolated operator test bridge and probes), `LIQUID_PROFILE_NVTX` (diagnostic NVTX ranges).

On Thor, `tools/thor/build_native.sh` runs this inside the pinned image and records each binary's sha256.

## lfm3-speech

`tools/thor/build_speech.sh TAG` builds `build/TAG/lfm3-speech` in the pinned image with the engine's flags and
writes `sources.sha256` next to it. Image input links the vendored libjpeg-turbo, built once by
`tools/build_image_decode_check.sh`.

## Host-only builds

Front-end code that needs no GPU (tokenizer, chat template, image decoding and preprocessing, speech text handling)
builds with plain `g++ -std=c++20` and is exercised by the host test suite:

```bash
PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q
```

## Reference environments

References come from Hugging Face Transformers 5.20.2 with PyTorch 2.10 and torchvision 0.25, in pinned containers
and virtual environments on Thor (GPU references) and on an x86 workstation (CPU references). Each validation round
records the exact versions it used.

## Machine layout

The job scripts in `tools/thor/` assume one directory (the *workspace*) that holds this repository next to the model
checkpoints and datasets, and that the containers mount it at `/workspace`:

```
<workspace>/
├── liquid-CUDA/          this repository (build outputs in liquid-CUDA/build/, not committed)
├── models/LFM3-Omni-2B/  the checkpoint
└── data/                 benchmark and evaluation data, fetched by the tools
```

| Variable | Default | Used by |
|---|---|---|
| `LFM3_WORKSPACE` | the directory above this repository (derived from each script's own location) | `tools/thor/*.sh` |
| `THOR_HOST` | none — required | `tools/thor_remote.py` (workstation-side remote jobs, `user@host`) |
| `THOR_REPO` | this repository's absolute path | `tools/thor_remote.py` (the repository's path on the target) |
| `THOR_CONTAINER` | `lfm3-profile-v1` | `tools/thor_remote.py` (the runtime container) |

The runtime container runs the binaries with the pinned CUDA libraries first on `LD_LIBRARY_PATH`
(`/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia`).
