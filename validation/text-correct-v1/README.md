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

## Results

(none yet)
