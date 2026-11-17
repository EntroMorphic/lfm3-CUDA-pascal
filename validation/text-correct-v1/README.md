# text-correct-v1

Phase 1 of [PLAN.md](../../PLAN.md): the text backbone on the GTX 1080 with the simplest correct kernels. Speed is
not judged in this round.

## What it will be

`liquid-lfm3` (the text CLI) and `lfm3-score` (the teacher-forced scorer) built for `sm_61` from the imported engine
with one change of substance: the prefill GEMM (`lfm::gemm` `wide`), which on Thor uses tensor cores, becomes a plain
FP32 kernel. The decode GEMMs (`thin`, `thin_k128`, `split_kn`) and every other operator are used as imported.

Out of scope: speed, the server, batching, other modalities, the cuBLAS build.

## What was known before the rules (2026-11-17)

- Compiled for `sm_61`, `lfm3_main.cu` fails only in the assembler, on `mma`, `.bf16`, `.m16n8k16`, `cp.async`
  and `ldmatrix`: the `wide` kernel. The cuBLAS configuration compiles with 0 errors (not run).
- No engine code had run on the card when these rules were committed.

## Comparators

| Name | What | Where |
|---|---|---|
| Thor pins | The Thor engine's greedy ids on its fixtures (equal to HF eager's there) | `liquid-CUDA/validation/g4-gates-v2/pins/` (in the repository) |
| HF SDPA | Per-token NLL of Transformers SDPA, BF16, on the WikiText-2 windows | Thor, `build/ppl-v5` (not in the repository) |
| HF FP32 anchor | Per-token NLL of the FP32 model | Thor, the tier 2b anchor directory (not in the repository) |
| Thor native | Per-token NLL of the Thor engine | Thor, `build/ppl-g4-v3` (not in the repository) |

The windows are bound by SHA-256 in `g4-gates-v2/tier2b/summary.json` (256: `d1c90869…`, 257: `0dad54b2…`,
4,096: `c8a4407f…`, 32,768: `54df38e0…`); windows of any other hash are refused.

## Rules

1. **Build.** `liquid-lfm3` and `lfm3-score` build for `sm_61` with `-O3 --fmad=false -std=c++20`, 0 warnings other
   than nvcc's deprecated-target notice, and link `libcudart` and the C and C++ runtime only.
2. **Tokens.** Greedy ids identical to the Thor pins on the four forward fixtures (`baseline` 16 ids, `arithmetic` 2,
   `window257` 4, `single` 4). Inputs are built on the workstation by the reference's recipe
   (`tools/lfm3_reference.py`: the chat template for the two prompts, the prompt's content tokens repeated to length
   for the other two).
3. **Perplexity.** `tools/ppl_compare.py`, unchanged, against HF SDPA at 256, 257, 4,096 and 32,768 tokens: no length
   is "worse" (the whole paired 95% CI of dNLL above +1e-5 nats/token).
4. **Fidelity.** `tools/ppl_fidelity.py`, unchanged: mean |NLL − NLL_FP32| no larger than HF SDPA's at each length.
5. **Determinism.** Two runs of each fixture and of the 256-token score give byte-identical ids and NLL files.
6. **Reset.** `check_reset` passes on every fixture (prefill and decode repeat with exactly equal logits after
   `Runner::reset`).
7. **Memory.** Peak device memory is recorded per length. A length that cannot run in 8 GB is a FAIL of that length,
   reported as such (PLAN decision 3), not a reason to shorten it.
8. **Sanitizer.** `compute-sanitizer --tool memcheck` on the `arithmetic` fixture: 0 errors, ids identical to the
   plain run.
9. **Nothing else changes.** The host suite gives the result recorded before the change.

Informational, not rules: dNLL against Thor native; prefill and decode times.

Rules 3 and 4 need the three NLL directories from Thor. Until they are on the workstation those rules are open, not
passed.

## Results (2026-11-17)

Build `text-v1` at commit `ecb5f43`: `liquid-lfm3` `e30023b7…e7aa593`, `lfm3-score` `82681a43…91448cd`. Records in
`records/`: the job as run (`ppl-v1.sh`, `ppl-v1.log`), each NLL file's metadata and the NLL files (gzip).

| Rule | Result |
|---|---|
| 1. Build | **PASS**: 0 warnings; the binaries link the C and C++ runtime only (`libcudart` is linked statically, which the rule did not foresee) |
| 2. Tokens | **PASS**: the Thor pins' ids on all four fixtures (16, 2, 4, 4) |
| 3. Perplexity | **OPEN**: needs the per-token HF SDPA files from Thor |
| 4. Fidelity | **OPEN**: needs the per-token FP32 anchor files from Thor |
| 5. Determinism | **PASS**: fixtures identical across two runs; the 256-token NLL file byte-identical across two full runs (`95979e15…`) |
| 6. Reset | **PASS** on every fixture |
| 7. Memory | 256, 257, 4,096 recorded below. **32,768: FAIL** — the scorer stops with "out of memory" |
| 8. Sanitizer | **PASS**: `compute-sanitizer --tool memcheck`, 0 errors on all four fixtures, ids equal to the plain run |
| 9. Nothing else changes | Host suite with the checkpoint reachable: 2,397 passed, 5 skipped, 1 failed. See below |

### Scores

| Length | Windows | Mean NLL | Perplexity | Wall | Peak device memory | NLL sha256 |
|---|---:|---:|---:|---:|---:|---|
| 256 | 1,160 | 3.280306402 | 26.583917 | 1,920 s | 4,407 MiB | `95979e15…` |
| 257 | 1,155 | 3.277541415 | 26.510514 | 2,082 s | 4,413 MiB | `ec238637…` |
| 4,096 | 72 | 2.569474382 | 13.058959 | 2,014 s | 5,195 MiB | `2886713b…` |
| 32,768 | 9 | — | — | — | out of memory | — |

Informational, from the Thor records (means only; no paired comparison is possible without the per-token files):
the FP32 anchor's mean NLL is 3.280256632, 3.277467314 and 2.569224751 at these lengths, so this engine's mean is
5.0e-5, 7.4e-5 and 2.5e-4 nats per token above it. At 256, Thor's record gives HF SDPA perplexity 26.582084 and the
Thor engine 26.578000, with a resolution of 1.25e-4 nats per token at that length.

### 32,768 tokens

Expected from the allocation arithmetic ([docs/long-context-memory.md](../../docs/long-context-memory.md)): the
per-token buffers need about 3,881 MiB at that capacity and about 3,737 MiB is free. PLAN decision 3 applies: it is
dealt with in [memory-v1](../memory-v1/).

### The host suite

Before the checkpoint was reachable at the path the tests expect, 721 tests skipped (phase 0). With
`../models/LFM3-Omni-2B` linked: 2,391 passed, 5 skipped, 6 failed on the first run; 2,397 passed, 5 skipped, 1
failed at `1664d89`.

- 4 failures were caused here, by a change to `Runner::linear`'s signature in the memory work (Thor candidate tools
  patch the source by that line). The signature is restored; they pass.
- 1 was a Thor record the import had left out (`validation/speech-e2e-v1`); imported, passes.
- 1 cannot run on this workstation: `tests/test_speech_text.py::test_every_validation_prompt_matches_hf_ids` loads
  the checkpoint's own code through Transformers, which needs the pinned 5.x; every Python environment found here
  has 4.57. It fails the same way with an empty private module cache. **Not counted as a pass.**

### What is open

Rules 3 and 4, on the Thor per-token files (`build/ppl-v5`, the tier 2b FP32 anchor, `build/ppl-g4-v3`). Phase 1
does not pass until they do, and until 32,768 runs.
