# speed-v1: phase 2's bar

Status: **closed by the owner with rule 12 failed** (2026-11-19). On the final rule run (`job3.sh`, builds `text-v30`
and `parity-v30`, tree `e6d9782`) 13 of 14 rules pass; rule 12, decode at 31,000 tokens, measures 17.34 ms against
17.16 ms. The owner chose to close the phase there after the engineering below. Records: [records/](records/) and
[records/final/](records/final/) (each `SHA256SUMS` verified after the copy).

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

## Amendment to rules 11 and 12 (owner, 2026-11-18)

The first figures assumed the card reads its weights at 258 GB/s (decode-atoms-v1). Measured on the day of the bar,
in decode's shape (`native/bench/read_rate.cu`): 245 GB/s for matrix-sized reads, 235 for 8 MB, 248 for 524 MB, at
the memory clock the driver keeps under compute load (4,513 MHz). The owner keeps that clock as it is ("85% cap
extends the life of the hardware") and set the decode bar to 245 GB/s, as: a step may take the time to read what it
must read at 245 GB/s, plus the 1.5 ms allowance for the small kernels and the gaps between launches that the first
figures used (decode-atoms-v1: 1.01 + 0.48 ms).

| # | Case | Must read per step | At 245 GB/s | Bar (was) |
|---|---|---:|---:|---:|
| 11 | decode-1k | weights, 3,456.8 MB | 14.11 ms | ≤ 15.61 ms (15.0) |
| 12 | decode-32k | weights and the 6 full-attention layers' keys and values at 31,000 positions, 380.9 MB | 15.66 ms | ≤ 17.16 ms (18.5) |

Corrected 2026-11-19 (phase 2's red-team review): this row first said 761.9 MB, 17.22 ms and ≤ 18.72 ms, counting
keys and values for 8 key/value heads. The model has 4 (`num_key_value_heads`; 31,000 × 4 × 128 × 2 bytes × 2 × 6
layers = 380.9 MB). The owner's formula is unchanged; only the arithmetic was wrong. The rule runs' report script
used 18.72.

The windowed layers' caches (256 positions, 5 MB) and the cache's growth over the 700 steps are left out. Known when
the amendment was made: single development runs of the phase 2 build gave 15.69 and 18.78 ms (both over these bars).
The owner chose this form over two others offered with those figures.

## Predictions

- Rules 1 to 6 hold: the phase 2 kernels give every output the same chain of operations as the kernels they replace.
- prefill-4096 about 3.5 s, prefill-32768 about 30 s, prefill-257 about 0.45 s.
- Rule 11 is the hardest to meet; rule 12 needs a faster decode attention over long caches.

## Results

Three rule runs.

1. `job.sh` (`text-v17`, tree `2c97d68`): every rule but 11 and 12.
2. `job2.sh` (`text-v19`, tree `2fa785e`), after the greedy choice moved to the card: every rule but 12 as corrected.
   Its report said rule 12 passed, against the miscomputed 18.72 ms; phase 2's red-team review found the error.
3. `job3.sh` (`text-v30`, tree `e6d9782`), after the red-team round and the long-context work below: the final.

The last review's fixes (`11a9897`: the greedy header included in both builds; the attention test closer to the
engine) left the native engine's device code unchanged, all 101 and 102 functions instruction for instruction
(`sass_compare.py`); the rebuilt engine (`text-v31`) gave `text-v30`'s ids in every speed case and passed the
fixtures (`job4.sh`, `records/final/speed-v1d.log`, `speed-v1e.log`).

| # | Rule | Final run |
|---|---|---|
| 1 | Parity gate | **Pass,** 7,826 of 7,826 |
| 2 | Fixtures, compact off and on; sanitizer | **Pass;** 0 errors on the fixtures and on the 257- and 513-token compact sweep |
| 3 | NLL files at 256, 257, 4,096, 32,768 | **Pass,** all four byte-identical to text-correct-v2's |
| 4 | Long decode; boundary sweep | **Pass,** all five cases identical to long-decode-v1; 33 of 33 pairs |
| 5 | On-card checks; host suite | **Pass:** `first_row_check` 210, `wide2_check` 143 (and 0 errors under the sanitizer), `decode_attention_check` 234 (352 after the last review's additions), `greedy_pick_check` 400, `round_once_check`; host suite as recorded (2,397 passed, 5 skipped, 1 needing Transformers 5.x) |
| 6 | Same ids as `text-v10` in every speed run | **Pass;** and a 600-token free generation without tracing gives long-decode-v1's ids |

Speed, medians of 3, `text-v10` measured in the same job:

| # | Case | Baseline | Phase 2 | Ratio | Bar | |
|---|---|---:|---:|---:|---:|---|
| 7 | prefill-4096 | 9.55 s | 2.80 s | 0.29 | ≤ 4.6 s | **pass** |
| 8 | prefill-32768 | 89.9 s | 28.9 s | 0.32 | ≤ 44.7 s | **pass** |
| 9 | prefill-257 | 0.932 s | 0.348 s | 0.37 | ≤ 0.60 s | **pass** |
| 10 | prefill-32 | 0.430 s | 0.215 s | 0.50 | ≤ 0.30 s | **pass** |
| 11 | decode-1k | 15.69 ms | 15.51 ms | 0.99 | ≤ 15.61 ms (amended) | **pass** |
| 12 | decode-32k | 20.58 ms | 17.34 ms | 0.84 | ≤ 17.16 ms (amended, corrected) | **FAIL** by 0.18 ms |
| 13 | decode-8k; prefill-1024 | 16.25 ms; 2.33 s | 15.95 ms; 0.713 s | 0.98; 0.31 | no slower | **pass** |
| 14 | peak memory, every case | | unchanged in every case | | ≤ +16 MiB | **pass** |

Scoring the four bar lengths took 23 minutes of wall time against 64 in red-team-1.

### What made the difference

- **Prefill GEMM** (`wide2_kernel`): 128-column tiles with 16 to 128 rows per block, 8 outputs per thread per row,
  16-byte loads, two shared buffers: 7.2 TFLOPS on the 4,096-row shapes against 1.6, about 75% of the card's measured
  multiply-add rate.
- **Position 0** in one-row blocks: from 8 rows computed for one to one. It still costs about 150 ms per prefill: one
  row is a few thousand serial chains of up to 480 steps each, too few to fill the card. Carried.
- **Decode attention**: the four query heads of each key/value head read its keys once (`thin_k128_kernel`, G = 4,
  the grid capped at 256 blocks with the next rows loaded ahead) and its values once (`split_kn_partial_wide`: 16-byte
  loads, four probabilities per load, from 16 splits up): per layer at 31,000 keys, 537 µs to 148 µs for the keys and
  274 µs to 154 µs for the values (development profiles), 205 to 220 GB/s of the 245.
- **The greedy choice on the card**, in two stages (`lfm/greedy_pick.cuh`): 0.17 ms per step.

### Tried and dropped (development jobs `dev*.sh`)

Fusing adjacent norms (0.05 ms); a one-warp norm with the same arithmetic (identical, but 20 µs per norm against
3.8); keeping the softmax's scores in registers (no change: its time is its reductions); the values' probabilities in
shared memory, 4 columns per thread, 8 keys ahead, 2 to 16 splits per block (none better than 4 ahead, 8 columns, 4
splits).

### Where rule 12's 0.18 ms is

At 31,000 keys a step spends about 1.85 ms on attention against the 1.55 ms its keys and values take at 245 GB/s: the
two products at 205 to 220 GB/s and a 0.18 ms softmax over 31,000 scores, whose reduction order is fixed. The 1.5 ms
allowance was measured at 1,000 tokens and does not include the softmax over a long cache. A fused long-context
attention kernel (scores, softmax and values in one, with the same arithmetic per output) is the next lever; parked.

### The red-team round

Four independent reviews: kernels, runner and tools, and the fixes twice.

| Finding | Fix |
|---|---|
| Rule 12's figure counted keys and values for 8 key/value heads; the model has 4 | Corrected (above); rule 12 fails |
| `decode_attention_check` compared the rewritten kernels with themselves | Compared with the kernels before phase 2, copied from `a5bcd4a`; with the engine's grid cap, a cleared workspace and `lfm::gemm::run` |
| `wide2_check` could not see a write past the end of C, and its FP32 cases had no padding columns | A guard area and padded rows; also run under the sanitizer |
| `speed.sh` turned a run with no memory sample into 0 MiB (a rule 14 pass) and ignored a failed summary | Both fail |
| The cuBLAS build had stopped compiling (the greedy header was included in the native build only) | Included in both |
| `LFM_GEMM_FIRST_TX` accepted 0; `LFM_GEMM_WIDE_RM`/`_RN`'s comment described the replaced kernel | Checked; comment corrected |
| Figures in the two READMEs without a record (the read rates, a TFLOPS range, the spread) | Records added (`records/final/read-rate/`) or the figures corrected |

No review found a defect in what the engine computes: a host simulation of `wide2_kernel` matched the old kernel in
10.2 million outputs (and caught the −0 case when the pad FMA was removed), and one of the greedy choice matched the
host in 2,895 trials.

### Predictions against the measurements

- Rules 1 to 6 held.
- prefill-4096 2.80 s (predicted about 3.5), prefill-32768 28.9 s (about 30), prefill-257 0.35 s (about 0.45).
- Rule 11 was the hardest of the first figures, as predicted; rule 12, once corrected, was harder.
