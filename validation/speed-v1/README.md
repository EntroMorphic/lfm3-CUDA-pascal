# speed-v1: phase 2's bar

Status: **bar set, nothing measured against it** (2026-11-18). Committed before any engine run of a phase 2 change.

Phase 2's exit gate (PLAN): "a speed bar set before measuring; phase 1 outputs unchanged". The bar is set from
[speed-baseline-v1](../speed-baseline-v1/): today's figures for the shipping engine and the card's measured ceilings.

## Rules

Every rule is judged on the final phase 2 build (shipping configuration) and, for speed, against `text-v10` re-measured
in the same job with `tools/pascal/speed.sh` (median of 3, warm, no tracing).

**Outputs unchanged** (every one must hold):

1. Parity gate: 7,826 of 7,826 trace files equal to the Thor pins.
2. Fixtures with the compact windows off and on; `compute-sanitizer` 0 errors on the fixtures and on the compact-window
   sweep of 257- and 513-token prompts.
3. NLL files at 256, 257, 4,096 and 32,768 tokens byte-identical to text-correct-v2's.
4. All five long-decode-v1 cases identical in ids and logits; boundary sweep 33 of 33.
5. `first_row_check`, `wide2_check` and `round_once_check` pass; host suite as recorded.
6. Every case of `speed.sh` gives the same ids as `text-v10`.

**Speed** (medians; every one must hold):

| # | Case | Baseline | Bar | Why this figure |
|---|---|---:|---:|---|
| 7 | prefill-4096 | 9.29 s | ≤ 4.6 s (half) | Prefill is mostly GEMM, which runs at a sixth of the card's measured FP32 rate |
| 8 | prefill-32768 | 89.4 s | ≤ 44.7 s (half) | The same, with the attention products larger |
| 9 | prefill-257 | 0.92 s | ≤ 0.60 s | Short prompts: GEMM, and position 0's second launch (0.155 s, carried from phase 1) |
| 10 | prefill-32 | 0.42 s | ≤ 0.30 s | Mostly position 0's second launch and fixed costs |
| 11 | decode-1k | 15.69 ms | ≤ 15.0 ms per step | The weight reads are about 13.4 ms; the 1.5 ms of small kernels and gaps between launches (decode-atoms-v1) is the room |
| 12 | decode-32k | 20.58 ms | ≤ 18.5 ms per step | Attention over 31,000 keys costs about 4.9 ms per step over decode-1k; the keys and values themselves are about 1.5 ms of reading |
| 13 | decode-8k, prefill-1024 | 16.25 ms, 2.28 s | no slower than the baseline (+1%) | No case may get slower |
| 14 | every case | | peak device memory no higher than the baseline's (+16 MiB) | The memory budget |

## Predictions

- Rules 1 to 6 hold: the phase 2 kernels give every output the same chain of operations as the kernels they replace.
- prefill-4096 about 3.5 s, prefill-32768 about 30 s, prefill-257 about 0.45 s.
- Rule 11 is the hardest to meet; rule 12 needs a faster decode attention over long caches.
