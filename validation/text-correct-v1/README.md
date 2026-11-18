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
| 3. Perplexity | **FAIL at 257** (the plain kernel): worse than HF SDPA by the rule's reading; 256 and 4,096 equal, 32,768 better. See "Against Hugging Face" |
| 4. Fidelity | **PASS** at all four lengths: closer to the FP32 model than HF SDPA is |
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

### Against Hugging Face (2026-11-18)

The per-token files were copied from Thor's host file system, read-only (`liquid-CUDA/build/ppl-v5`, `ppl-fp32-v2`,
`ppl-g4-v3`; the owner gave the address). Every one of the 26 NLL files matches its own metadata, the hashes
committed in the Thor repository where one exists (19 of them; the 7 HF copies inside `ppl-g4-v3` have none), and
the windows rebuilt here. This engine's files: `text-v1` at 256, 257 and 4,096, and `text-v2` at 32,768 (the same
arithmetic; [memory-v1](../memory-v1/)). Reports: `records/compare/`.

**Rule 3, `tools/ppl_compare.py` unchanged, against HF SDPA:**

| Length | Windows | PPL SDPA | PPL eager | PPL here | dNLL | 95% CI | Resolution | Verdict |
|---|---:|---:|---:|---:|---:|---|---:|---|
| 256 | 1,160 | 26.5821 | 26.5812 | 26.5839 | +6.89e-05 | [−6.31e-05, +1.99e-04] | 1.31e-04 | equal: PASS |
| 257 | 1,155 | 26.5045 | 26.5066 | 26.5105 | +2.27e-04 | [+9.94e-05, +3.57e-04] | 1.29e-04 | **worse: FAIL** |
| 4,096 | 72 | 13.0596 | 13.0590 | 13.0590 | −5.29e-05 | [−1.76e-04, +7.19e-05] | 1.24e-04 | equal: PASS |
| 32,768 | 9 | 12.5771 | — | 12.5746 | −1.96e-04 | [−3.43e-04, −5.90e-05] | 1.42e-04 | better: PASS |

The bar fails: at 257 tokens the whole interval lies above +1e-5.

**Rule 4, `tools/ppl_fidelity.py` unchanged, mean |NLL − NLL_FP32| per token:**

| Length | This engine | HF SDPA | Difference, 95% CI | Verdict |
|---|---:|---:|---|---|
| 256 | 1.672e-02 | 1.940e-02 | [−2.77e-03, −2.60e-03] | closer than SDPA |
| 257 | 1.673e-02 | 1.948e-02 | [−2.84e-03, −2.66e-03] | closer |
| 4,096 | 1.505e-02 | 1.753e-02 | [−2.56e-03, −2.38e-03] | closer |
| 32,768 | 1.587e-02 | 1.818e-02 | [−2.42e-03, −2.21e-03] | closer |

PASS: 12.7% to 14.1% closer to the FP32 model than SDPA at every length (the Thor engine's record: 12.8-14.4%).

**Where each engine's mean sits relative to the FP32 model (nats per token):**

| Length | HF SDPA | HF eager | Thor engine | This engine (plain kernel) |
|---|---:|---:|---:|---:|
| 256 | −1.92e-05 | −5.10e-05 | −1.73e-04 | +4.98e-05 |
| 257 | −1.53e-04 | −7.40e-05 | −6.97e-05 | +7.41e-05 |
| 4,096 | +3.02e-04 | +2.55e-04 | +2.29e-04 | +2.50e-04 |
| 32,768 | +4.23e-04 | — | +3.51e-04 | +2.27e-04 |

At 257 SDPA's mean is 1.5e-4 below the FP32 model's and this engine's is 0.7e-4 above it: nearer to FP32, and on the
other side of it from SDPA, which is what the rule counts as worse. That describes the failure; it does not excuse
it. The rule is the owner's and stands as written.

**Informational, this engine minus the Thor engine, paired:** +2.23e-04 at 256 (CI [+1.02e-04, +3.41e-04]),
+1.44e-04 at 257 ([+2.8e-05, +2.63e-04]), +2.1e-05 at 4,096 (CI spans zero), −1.24e-04 at 32,768 (CI spans zero).
The two engines differ only in the prefill GEMM's summation (decode is bit-identical), and under 1.1% of tokens
have identical NLL. At the two short lengths the plain kernel's sums give a slightly higher NLL than the tensor
core's.

### What is open

- **Why rule 3 fails at 257 is known** ([prefill-arithmetic-v1](../prefill-arithmetic-v1/)): the difference is made
  in the [BOS] row, which every window shares, so it is one fixed perturbation that the rule's window bootstrap
  counts 1,155 times. With the tensor-core arithmetic in that row alone the verdict at 257 is "equal". What to ship
  and whether the rule's statistics should change are the owner's decisions and are not made.
- Rule 3 fails at 257 with the plain prefill kernel. The exact kernel
  ([tensor-core-model-v1](../tensor-core-model-v1/)) reproduces the Thor engine's arithmetic, and the Thor engine
  passes rule 3 at every length; the exact build's NLL file at 257 tokens equals Thor's byte for byte
  ([tensor-core-model-v1](../tensor-core-model-v1/)), so at that length the exact build has the Thor engine's
  verdict, "equal", by identity. The other lengths have not been run with it.
- Regenerated with the re-tiled kernels (`ppl-v2`, build `text-v5`, `lfm3-score` `cf0f6b3d…`, after the host's reboot):
  all four NLL files byte-identical to the ones compared above (`95979e15…`, `ec238637…`, `2886713b…`, `25df36d8…`).
  Wall time 737, 1,005, 785 and 917 s (256, 257, 4,096, 32,768): 57 minutes for the set, against about 2.3 hours
  with the first kernel; peaks 4,409, 4,415, 5,029 and 6,577 MiB.

Phase 1 is not passed in this round. It passed on 2026-11-18 with position 0 on the tensor-core arithmetic: [text-correct-v2](../text-correct-v2/).
