# lfm3-CUDA-pascal

An inference engine for LFM3-Omni-2B (text, speech out, audio in) on an NVIDIA GeForce GTX 1080 (Pascal, `sm_61`,
8 GB), at the quality of the Hugging Face reference. It is a port of the Thor engine (liquid-CUDA, imported at its
commit `e2f8b7f`, written for the Jetson AGX Thor, `sm_110`); the port changes the kernels the older card lacks and
validates the result separately. [PLAN.md](PLAN.md) gives the goal, the owner's decisions and each phase's status.

## Status

| Phase | | |
|---|---|---|
| 1 | Text correct | passed ([text-correct-v2](validation/text-correct-v2/)) |
| 2 | Speed | closed with one of 14 rules failed ([speed-v1](validation/speed-v1/)) |
| 3 | Text serving | passed ([serve-v1](validation/serve-v1/)) |
| 4 | Speech | 13 of 14 rules pass; the owner's blind listening pending ([speech-v1](validation/speech-v1/)) |
| 5 | Images | not started: no build here has image input |

## Two builds

- **Shipping:** the native GEMM, the engine as it is meant to run.
- **Parity** (`EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL`): the prefill products with the arithmetic of Blackwell's
  tensor cores, emulated, so that its outputs equal the Thor engine's bit for bit (PLAN decision 4: 7,826 of 7,826
  pinned trace files). Several times slower; for validation.

## Building

The builds run in a container on the card's machine, with CUDA 12.9 (`nvcc` 12.9.86; CUDA 13 no longer targets
`sm_61`), GCC 13.3, the checkpoint at `/workspace/models/LFM3-Omni-2B` and outputs under `/workspace/build`
([phase0-foundations-v1](validation/phase0-foundations-v1/) records the setup):

```bash
tools/pascal/build_text.sh TAG     # liquid-lfm3 (text CLI), lfm3-score (perplexity scorer)
tools/pascal/build_serve.sh TAG    # and the text-only server: liquid-lfm3-worker, liquid-lfm3-server
tools/pascal/build_speech.sh TAG   # and lfm3-speech, with the server and worker built with speech
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh TAG-parity   # the parity build
```

Each writes `/workspace/build/TAG/` with the binaries, their SHA-256, the flags and the hashes of every source.
`tools/pascal/remote.sh` syncs this repository to that machine and runs commands there; it reads the machine's
addresses from `tools/pascal/target.env` (not committed; see `target.env.example`). `native/CMakeLists.txt` is the
Thor engine's and targets `sm_110` only.

## Running

```bash
liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B request.json result.json
liquid-lfm3-server --experimental /workspace/models/LFM3-Omni-2B --port 8080
```

The server's API, and its defaults on this card (one speech row, a total text capacity of 30,720 tokens when it
serves speech): [docs/server-api.md](docs/server-api.md).

## Layout

| Path | |
|---|---|
| `native/` | The engine (CUDA C++, header-only modules in `native/include/lfm/`) |
| `tools/pascal/` | This port's build, sync, measurement and judging tools |
| `tools/`, `tests/`, `audit/` | The Thor engine's tools, host tests and engineering notes, as imported (some changed; `git diff fc130a3`) |
| `validation/` | Every validation round: its rules (set before the data), job scripts and records ([index](validation/README.md)) |
| `docs/` | The Thor engine's documents, with notes where this port differs |
