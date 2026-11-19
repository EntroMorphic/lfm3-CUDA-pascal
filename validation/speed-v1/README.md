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
layers = 380.9 MB). The owner's formula is unchanged; only the arithmetic was wrong. The first two rule runs' report
used 18.72; the third used 17.16.

The windowed layers' keys and values (256 positions: 2.6 MB read per step, in a 5 MB buffer) and the cache's growth
over the 700 steps (31,001 to 31,700 keys) are left out. Counting both would put the bar at about 17.19 ms for the
average step and 17.21 ms for the last; rule 12 fails either way. Rule 11 counts no keys or values (about 10 MB at 300
to 1,000 positions, 0.04 ms), which makes it stricter.

The 245 GB/s is the 2026-11-18 measurement (`records/final/read-rate/read_rate-2026-11-18.txt`, 244.6 to 245.9 GB/s
for 31.46 MB). A second run of the same benchmark in development on 2026-11-19 (`read_rate.txt`, `dev7.sh`) gave
246.7 to 247.0 GB/s, and a third in `job5.sh` 245.9 to 247.0 (`records/final/sv5/read_rate.txt`). At 247 GB/s rule
11's bar would be 15.50 ms and the 15.51 ms measured would miss it: rule 11 passes against the bar as the owner set
it, by less than the measurements of the card differ. The clock during that third run was sampled
(`read_rate.clocks.txt`): every sample under load 4,513 MHz in P2; the one 5,005 MHz sample is P0 at 0% before the
first launch. `speed.sh` counts P0 and P2 alike as busy, which is why its records list both. Known when
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
fixtures (`job4.sh`, `records/final/speed-v1d.log`, `speed-v1e.log`). That comparison ran from a copy of the script
outside the repository and its dumps were not kept; `job5.sh` repeats it from the committed scripts with the dumps
kept (see the second review).

| # | Rule | Final run |
|---|---|---|
| 1 | Parity gate | **Pass,** 7,826 of 7,826 |
| 2 | Fixtures, compact off and on; sanitizer | **Pass;** 0 errors on the fixtures and on the 257- and 513-token compact sweep |
| 3 | NLL files at 256, 257, 4,096, 32,768 | **Pass,** all four byte-identical to text-correct-v2's |
| 4 | Long decode; boundary sweep | **Pass,** all five cases identical to long-decode-v1; 33 of 33 pairs |
| 5 | On-card checks; host suite | **Pass:** `first_row_check` 210, `wide2_check` 143 (and 0 errors under the sanitizer), `decode_attention_check` 234 (352 after the last review's additions), `greedy_pick_check` 400, `round_once_check`; host suite as recorded (2,397 passed, 5 skipped, 1 needing Transformers 5.x; run on the development machine at `e6d9782`, `records/final/host-suite/`, added after the second review) |
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

### Where decode-32k's time goes

From development profiles (not in the records): at 31,000 keys the six full-attention layers spend about 1.8 ms per
step on the two products (148 and 154 µs per layer, 205 to 220 GB/s) and about 0.18 ms on the softmax over 31,000
scores, whose reduction order is fixed: about 2.0 ms against the 1.55 ms their keys and values take at 245 GB/s, about
0.45 ms over. The step misses its bar by less (0.18 ms) because the rest of it comes in under the formula's share (the
same build runs decode-1k 0.10 ms under its bar). The 1.5 ms allowance was measured at 1,000 tokens and does not
include the softmax over a long cache. A fused long-context
attention kernel (scores, softmax and values in one, with the same arithmetic per output) is the next lever; parked.

### The red-team round

Three independent reviews: the kernels, the runner with its tests and tools, then the fixes that followed.

| Finding | Fix |
|---|---|
| Rule 12's figure counted keys and values for 8 key/value heads; the model has 4 | Corrected (above); rule 12 fails |
| `decode_attention_check` compared the rewritten kernels with themselves | Compared with the kernels before phase 2, copied from `a5bcd4a`; with the engine's grid cap, a cleared workspace and `lfm::gemm::run` |
| `wide2_check` could not see a write past the end of C, and its FP32 cases had no padding columns | A guard area and padded rows; also run under the sanitizer |
| `speed.sh` turned a run with no memory sample into 0 MiB (a rule 14 pass) and ignored a failed summary | Both fail |
| The cuBLAS build had stopped compiling (the greedy header was included in the native build only) | Included in both |
| `LFM_GEMM_FIRST_TX` accepted 0; `LFM_GEMM_WIDE_RM`/`_RN`'s comment described the replaced kernel | Checked; comment corrected |
| Figures in the two READMEs without a record (the read rates, a TFLOPS range, the spread) | Records added (`records/final/read-rate/`) or the figures corrected; some remain from development only (see the second review) |

No review found a defect in what the engine computes: a host simulation of `wide2_kernel` matched the old kernel in
10.2 million outputs (and caught the −0 case when the pad FMA was removed), and one of the greedy choice matched the
host in 2,895 trials.

### The second red-team review (after the close)

Three more independent reviews of the closed phase, by reading: the engine's kernels, the tests and tools, and this
record against its records. None found a defect in what the engine computes; every median, ratio, checksum, build
hash and bar recomputes. Records: `records/final/sv5/` (`job5.sh`, tree `7d95df8`, builds `text-v32` and
`parity-v32`) and `records/final/host-suite/`.

| Finding | Fix |
|---|---|
| `decode_attention_check` ran 1 and 2 rows; the rewritten kernels serve every row count, and a prompt of 3 to 8 tokens runs them with that many rows (nothing else ran 3 to 7) | 3 to 8 rows added: 616 cases, 0 failed, and 0 errors under the sanitizer |
| `split_kn_partial<R, 4>`'s launch could never be chosen (`check()` makes one row at 16 splits or more take `split_kn_partial_wide`) but was compiled for 8 row counts | Removed. Every other function of both binaries is instruction for instruction `text-v30`'s; the 8 removed are the only difference (`*.sass-compare-all-rows.txt`; the dumps are kept, `sass.sha256`). The first comparison expected 1 removed function and failed; the 8 are that launch's 8 row counts |
| `LFM_GEMM_WIDE_SPLITS=0` divided by zero; `LFM_GEMM_K128_BLOCKS=0` launched an empty grid | Both stop the build, as does a block over 1,024 threads |
| `sass_compare.py` passed two empty dumps | Fails; removed functions must be named |
| `job4.sh`'s ids check skipped a missing case | `job5.sh` names all 8 cases; a missing one fails |
| Rule 5's host suite cited red-team-1's result | The phase 2 runs are in the records (`e6d9782` and `7d95df8`: 2,397 passed, 5 skipped, 1 failed, as recorded) |
| Rule 11 passes by less than the card's read-rate measurements differ | Disclosed in the amendment |
| PLAN said 3.4 times at 32,768 tokens | 3.1 |
| The breakdown of decode-32k's attention did not add up | Corrected; marked as from development profiles |
| The windowed layers' cache was given as 5 MB read; the report-script remark covered all three runs | Corrected (2.6 MB read; the first two runs) |

On `job5.sh`'s build: parity gate 7,826 of 7,826; fixtures pass; every speed case gives `text-v30`'s ids (all 701,
1,001 and 701 decode argmaxes, and each prefill's); the cuBLAS build compiles with 0 warnings; `first_row_check` 210,
`wide2_check` 143, `greedy_pick_check` 400.

Not changed: these figures come from development jobs and have no record: position 0's ~150 ms, the greedy choice's
0.17 ms per step, the norm timings and the fused norms' 0.05 ms, the host simulations (10.2 million outputs, 2,895
trials), the development decode figures (15.69, 18.78, 17.33 ms) and the attention profile above. Peak memory is
sampled every 0.2 s on the whole card (rule 14). Rule 6's speed cases compare one argmax per prefill and the
teacher-forced argmaxes of decode; free generation is covered by the 600-token run and the fixtures.

### Predictions against the measurements

- Rules 1 to 6 held.
- prefill-4096 2.80 s (predicted about 3.5), prefill-32768 28.9 s (about 30), prefill-257 0.35 s (about 0.45).
- Rule 11 was the hardest of the first figures, as predicted; rule 12, once corrected, was harder.
