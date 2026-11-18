# red-team-1: the code this project wrote, end to end

Status: **fixes written, rules and predictions below, nothing built or run** (2026-11-18).

## Scope and method

Everything written or changed since the import (`git diff fc130a3`): the GEMM and the tensor-core model
(`native/include/lfm/gemm.cuh`, `tensor_core_model.hpp`), the runner and scorer changes (`native/src/lfm3_main.cu`,
`window_slots.hpp`, `native/apps/lfm3_score.cu`), and the validation tooling (`tools/pascal/`). Three independent
reviews by reading, one per area, each told to find defects and given none of the project's conclusions; and, on
the card, a sweep of the compact window buffers against the full-length layout at every boundary prompt length.
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

Not changed, by decision:

- The launch that recomputes position 0 computes 8 rows for one (the 0.155 s per prefill). Phase 2's first item.
- Peak memory is the whole card's (`nvidia-smi`); a per-process figure is not available in the container. The
  tools now record what the card held before the run, and `long_decode.sh` refuses to start unless it is idle.
- `tests/test_window_slots.py` skips without `g++`; `arith_report.py` and `nll_tests.py` share one random stream
  across variants (information only).

## Verification (`job.sh`)

New builds: `text-v7` (shipping) and `parity-v7` (the gate's).

1. **The findings reproduce on the old binary** (`text-v6`): the same request at capacity 4,097 and 4,096 gives
   different logits for prompts of 129, 136, 257, 513 and 1,025 tokens, and the scorer's rows 256–258 differ
   between 260-token and 520-token windows of the same tokens (`tail_tile_check.sh` fails).
2. **And not on the new one:** those five prompts identical across the two capacities; `tail_tile_check.sh` passes.
3. **`first_row_check`** (new, on the card): position 0 equals the tensor-core model computed on the host, bit
   for bit; every other row, and every row of a product that is not a prompt, equals the plain kernel; non-finite
   plain results at position 0 are kept.
4. **Switches:** the five valid ones compile; the five invalid values stop at an `#error`.
5. **Parity gate** 7,826 of 7,826; **fixtures** with compact windows off and on; **sanitizer** 0 errors on the
   fixtures and on a compact-window decode from 257-token and 513-token prompts (a last attention tile of one row).
6. **Boundary sweep:** 33 pairs, compact against full, identical (as on the old binary before the fixes: 33 of 33).
7. **Nothing recorded changes:** NLL files at 256, 257, 4,096 and 32,768 byte-identical to text-correct-v2's;
   every long-decode-v1 case's ids and logits identical to its record.
8. **Round-once check** still 3,168 of 3,168, and its "plain kernel" column is now the plain kernel alone.
9. **Host suite** as recorded: 2,397 passed, 5 skipped.

## Predictions (before any run)

- Rule 1: as stated; also different at 2,049 and 3,329 (both end in a tile of one row); identical at 9, 137 and 300.
- Rule 2: identical from 9 to 1,025 tokens. At 2,049 and 3,329 not known: above 4,096 of capacity the softmax runs
  over each tile's keys, and which of its two kernels a row gets then depends on the tile. If they still differ,
  that is the mechanism behind long-decode-v1's "seen, not explained", and it predates the import.
- Rule 3: row 0 differs from the plain kernel in most outputs of the 2,048-wide shapes.
- Rules 4 to 9: pass. The fixes touch no arithmetic that a recorded result used.
