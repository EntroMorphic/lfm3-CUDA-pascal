# red-team-1: the code this project wrote, end to end

Status: **passed** (2026-11-18). Rules and predictions committed at `862daec` before the first run. Every finding of
two rounds of review is fixed. The engine's code for the text path computes what phase 1 recorded: the NLL files at
all four lengths are byte-identical to text-correct-v2's. Records: [records/](records/) (`SHA256SUMS` verified after the copy).

## Scope and method

Everything written or changed since the import (`git diff fc130a3`): the GEMM and the tensor-core model
(`native/include/lfm/gemm.cuh`, `tensor_core_model.hpp`), the runner and scorer changes (`native/src/lfm3_main.cu`,
`window_slots.hpp`, `native/apps/lfm3_score.cu`), and the validation tooling (`tools/pascal/`). Three independent
reviews by reading, one per area, each told to find defects and given none of the project's conclusions; on the
card, a sweep of the compact window buffers against the full-length layout at every boundary prompt length; then a
second independent review of the fixes themselves.
The imported serving, speech and vision code was read only where the new code touches it.

## Findings and what was done

No finding changes a recorded result. F1 to F4 are engine defects; T1 on are ways a tool could report a pass it
should not, or a number other than the one its header names.

| # | Finding | Fix |
|---|---|---|
| F1 | Above 4,096 tokens of capacity prefill attention runs in tiles of 128 rows; a last tile of 1 to 8 rows took the decode kernels, so a prompt of 129, 257, 513, … tokens had rows with a different arithmetic than at capacity 4,096 or below. The MLP blocks had been given the prefill kernel for exactly this; attention had not | The tiles of a prefill of more than 8 rows take the prefill kernel (`head_scores`, `head_values`: `Problem::wide`) |
| F2 | The scorer's last logit tile had the same defect when (length − 1) mod 256 is 1 to 8 (lengths 258–265, …). The bar's lengths are not affected | `lm_head` takes `wide` |
| F3 | `Problem::row0` defaulted to 0, so position 0's tensor-core arithmetic was given to row 0 of any product of more than 8 rows that did not say otherwise: the vision encoder's, the batch decoder's, the bench's | The default is −1 (not a prompt); only the text runner sets a position |
| F4 | The tensor-core model does not cover NaN and infinity, so a non-finite value among position 0's operands could come out finite, while every other row propagates it | The position-0 launch keeps any output the plain kernel left non-finite |
| F5 | The batch, tree and paired-decode code read keys and values at absolute positions; with the compact buffers (capacity above 4,096) that is an out-of-bounds read or write. Not built by the Pascal scripts, reachable through CMake | They refuse a runner with compact buffers (`Runner::require_full_length_caches`) |
| F6 | Build switches: `=0` meant on for two of them, an unknown value silently meant a default | `#error` on any value that means nothing |
| F7 | `LFM3_PREFILL_HEADS` took spaces, a sign and leading zeros; `LFM3_COMPACT_WINDOWS` was checked only when a windowed layer was reached | Digits only; checked at any capacity |
| F8 | Comments and one document said things the code does not do (position 0 in short prefills, padding rows, the host check's reach, `docs/long-context-memory.md` still "an idea") | Corrected |
| T1 | `long_decode_report.py` accepted non-finite values (an infinity in the yardstick passes anything) and left an old PASS file behind on an input error | Finite or no verdict; the output says "invalid input" |
| T2 | `bar_amended.py` did not repeat three of `ppl_compare`'s checks: a byte copy of the reference passed as "equal" | Repeated; the anchor must be an HF run and not a copy of the engine's file |
| T3 | `verify_memory.sh` could compare a result directory with itself | Refused |
| T4 | `run_text_fixtures.sh` judged the entries of `expected.json`, not the requests it ran, and trusted a sanitizer's exit status | The two sets must match; under compute-sanitizer every log must say "0 errors" |
| T5 | `score.sh` ignored a failed record step for all but the last length, and recorded neither the engine's environment switches nor the card's memory before the run | Fails; both recorded |
| T6 | `arith_experiment.sh` leaked `LFM_TCM_ROWS` into later variants, ignored failed builds and always exited 0 | Each variant in its own environment; failures propagate |
| T7 | `build_text.sh` left the previous binary beside new source hashes when a build failed | Removed first |
| T8 | `round_once_report.py` accepted any length that divides the file | Reads each file's metadata; lengths and windows must agree |
| T9 | `decode_atoms.py` counted teardown as a decode step in its per-kernel table | Excluded |
| T10 | `long_decode.sh` did not record the binary or windows in its records, did not require an idle card for a rule that compares peak memory, and one assert had the wrong precedence | Recorded and checked |
| T11 | `remote.sh` excluded `reference`, `build` and `target.env` by name at any depth, and did not say which commit a tree was | Anchored to the top; the sync line names the commit and whether there are uncommitted changes |
| T12 | `compare_pins.py` printed `PINS EQUAL` for a subset of the fixtures | Says so; unknown fixture names are an error |

Found by the second review, in the fixes above:

| # | Finding | Fix |
|---|---|---|
| R1 | F6 was incomplete: the preprocessor reads an unknown word as 0, so `-DLFM_GEMM_FIRST_ROW=on` turned position 0's arithmetic off and `-DLFM_GEMM_PLAIN_ROUND=away` meant nearest | A value must be one of the listed digits |
| R2 | `boundary_sweep.sh` printed PASS when no pair under its rule had run | Says NOT RUN and exits 2 |
| R3 | The sweep's generation case accepted any number of steps, so an early stop token passed, and its header claimed it fills the capacity (the last generated token is never fed back: it reaches 4,096 of 4,097 positions) | Exactly `STEPS` steps required; the header says what the case reaches |
| R4 | `first_row_check` never tested attention's products: several heads per call, and a P·V whose values stop short of K. A launch of position 0 that reached only the first head would have passed | Rule 4 of the check, both products, every head. A deliberately broken kernel (position 0 for the first two of four heads) now fails it |

Not changed, by decision:

- The launch that recomputes position 0 computes 8 rows for one (the 0.155 s per prefill). Phase 2's first item.
- Peak memory is the whole card's (`nvidia-smi`); a per-process figure is not available in the container. The
  tools now record what the card held before the run, and `long_decode.sh` refuses to start unless it is idle.
- `tests/test_window_slots.py` skips without `g++`; `arith_report.py` and `nll_tests.py` share one random stream
  across variants (information only).

## Verification

Five jobs, in order: `job.sh` (tree `862daec`, stopped by hand during its last step, scoring, because `job2.sh` repeats
it), `job2.sh` (the full list on the fixes of the first review), `job3.sh`, `job4.sh` and `job5.sh` (the second review's
fixes). After `job2.sh` only preprocessor checks changed in the engine: with symbols stripped, the final `liquid-lfm3`
and `lfm3-score` are byte-identical to the builds `job2.sh` verified, in both the shipping and the parity configuration.
The unstripped files differ in 5 bytes, the temporary file name `nvcc` leaves in the symbol table.

| # | Rule | Result |
|---|---|---|
| 1 | The findings reproduce on the old binary | **Yes.** Across capacities 4,097 and 4,096, prompts of 136, 257, 513, 1,025, 2,049 and 3,329 tokens differ. Scorer rows 256–258 differ in 18 of 24 values between 260- and 520-token windows |
| 2 | And not on the new one | **Pass.** All 10 prompt lengths identical across the two capacities at all 601 steps; rows 256–258 differ in 0 of 24 values |
| 3 | `first_row_check` | **Pass,** 210 checks, 0 failed. The broken kernel fails 8 of them |
| 4 | Switches | **Pass.** 9 valid forms compile with 0 warnings; 9 invalid values stop the build |
| 5 | Parity gate, fixtures, sanitizer | **Pass.** 7,826 of 7,826 pins; fixtures with compact windows off and on; the sanitizer reports 0 errors on the fixtures and on the compact-window decode from 257- and 513-token prompts |
| 6 | Boundary sweep | **Pass,** 33 of 33 pairs, on the final build |
| 7 | Nothing recorded changes | **Pass.** NLL files at 256, 257, 4,096 and 32,768 byte-identical to text-correct-v2's (mean NLL 3.280151, 3.277353, 2.569422, 2.531827); all five long-decode-v1 cases identical in ids and logits |
| 8 | Round-once check | **Pass,** 3,168 of 3,168 |
| 9 | Host suite | **As recorded:** 2,397 passed, 5 skipped, 1 failed. The failure is `test_every_validation_prompt_matches_hf_ids`, which needs Transformers 5.x (4.57.6 is installed here) and fails the same way without these changes |

## Predictions against the measurements

- Rule 1: **one miss.** Prompt 129 was predicted to differ across capacities and did not. The others held: 136, 257,
  513, 1,025, 2,049 and 3,329 differ; 9, 137 and 300 are identical. 257 differs at only 7 of 601 steps.
- Rule 2: held from 9 to 1,025 tokens. The open case resolved: 2,049 and 3,329 are identical as well, so the softmax's
  choice of kernel by tile is not a source of difference between these capacities.
- Rule 3: **held in FP32 only.** Row 0 differs from the plain kernel in nearly every FP32 output of the 2,048-wide shapes
  (for example 96 of 96), and in none of their BF16 outputs.
- Rules 4 to 9: held.

## Limits

- The batch, tree and worker programs do not compile for sm_61: `lfm3_batch.cuh` needs the Thor engine's GEMM tile type
  and `gemm_lossless.cuh`. So the guard of F5 in `lfm3_batch.cuh` and `lfm3_tree.cuh` has not been compiled to completion
  (no compiler error names it); the same guard compiles and links through the decode-batch probe. `job2.sh`'s comment
  says the errors are all in `gemm_lossless.cuh`; 17 to 19 of them are in `lfm3_batch.cuh` and `vision_encoder.cuh`.
- Speech, vision and serving code was reviewed only where the new code touches it.
- "No defects remain" is not something a review can show. What is shown: the second review's four findings are fixed,
  and neither review found a defect in what the engine computes on the text path.
