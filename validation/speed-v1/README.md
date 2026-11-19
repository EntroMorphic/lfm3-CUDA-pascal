# speed-v1: phase 2's bar

Status: **not passed** (2026-11-19). The second rule run (`job2.sh`, builds `text-v19` `3312933c…` and `parity-v19`
`3126b0ba…`, tree `2fa785e`) passes every rule but 12. Rule 12's figure was computed wrong when the amendment was
written (keys and values counted for 8 key/value heads; the model has 4): by the owner's formula the bar is 17.16 ms,
not 18.72, and 18.60 ms fails it. Found by phase 2's red-team review; see the correction below. The bar was committed before any engine run of a phase 2 change; rules 11 and 12 were
amended by the owner during the first rule run, before its speed results (below). Records: [records/](records/)
(`SHA256SUMS` verified after the copy). Phase 2's red-team round follows.

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

Two rule runs. The first (`job.sh`, `text-v17`, tree `2c97d68`) passed every rule but 11 and 12: decode-1k 15.69 ms and
decode-32k 18.79 ms against the amended 15.61 and 18.72. The engine then took the greedy choice on the card for
greedy requests with no tracing (`greedy_pick` in `lfm3_main.cu`: same tie rule as the host, same refusal of
non-finite logits), which removes the 512 KB logits copy and the host's scan from every step; the second run
(`job2.sh`) repeated every rule on that build.

| # | Rule | Second run |
|---|---|---|
| 1 | Parity gate | **Pass,** 7,826 of 7,826 |
| 2 | Fixtures, compact off and on; sanitizer | **Pass;** 0 errors on the fixtures and on the 257- and 513-token compact sweep |
| 3 | NLL files at 256, 257, 4,096, 32,768 | **Pass,** all four byte-identical to text-correct-v2's |
| 4 | Long decode; boundary sweep | **Pass,** all five cases identical to long-decode-v1; 33 of 33 pairs |
| 5 | `first_row_check`, `wide2_check`, `decode_attention_check`, `round_once_check`; host suite | **Pass:** 210 checks, 143 cases, 54 cases, pass; host suite as recorded (2,397 passed, 5 skipped, 1 needing Transformers 5.x) |
| 6 | Same ids as `text-v10` in every speed run | **Pass;** and a 600-token free generation without tracing (the path the greedy choice on the card takes) gives long-decode-v1's ids |

Speed, medians of 3, `text-v10` measured in the same job:

| # | Case | Baseline | Phase 2 | Ratio | Bar | |
|---|---|---:|---:|---:|---:|---|
| 7 | prefill-4096 | 9.47 s | 2.79 s | 0.29 | ≤ 4.6 s | **pass** |
| 8 | prefill-32768 | 89.0 s | 28.8 s | 0.32 | ≤ 44.7 s | **pass** |
| 9 | prefill-257 | 0.932 s | 0.347 s | 0.37 | ≤ 0.60 s | **pass** |
| 10 | prefill-32 | 0.430 s | 0.215 s | 0.50 | ≤ 0.30 s | **pass** |
| 11 | decode-1k | 15.69 ms | 15.52 ms | 0.99 | ≤ 15.61 ms (amended) | **pass** |
| 12 | decode-32k | 20.58 ms | 18.60 ms | 0.90 | ≤ 17.16 ms (amended, corrected) | **FAIL** (the run's report said pass against the miscomputed 18.72) |
| 13 | decode-8k; prefill-1024 | 16.25 ms; 2.31 s | 16.05 ms; 0.697 s | 0.99; 0.30 | no slower | **pass** |
| 14 | peak memory, every case | | unchanged in every case | | ≤ +16 MiB | **pass** |

Scoring the four bar lengths took 23 minutes of wall time against 64 in red-team-1.

### What made the difference

- **Prefill GEMM** (`wide2_kernel`): 128-column tiles with 16 to 128 rows per block, 8 outputs per thread per row,
  16-byte loads, two shared buffers: 7.2 TFLOPS on the 4,096-row shapes against 1.6, about 75% of the card's measured
  multiply-add rate.
- **Position 0** in one-row blocks: from 8 rows computed for one to one. It still costs about 150 ms per prefill: one
  row is a few thousand serial chains of up to 480 steps each, too few to fill the card. Carried.
- **Decode attention**: pairs of query heads read the keys (and, from 16 splits up, the values) once per pair: 3.2 ms
  to 1.7 ms per step at 31,000 keys (development profiles, not in the records). Each key/value head serves four query
  heads, so every key is still read twice.
- **The greedy choice on the card**: 0.17 ms per step.
- Tried and dropped: fusing adjacent norms (0.05 ms; each norm's time is its own reduction, not the launch).

### Predictions against the measurements

- Rules 1 to 6 held.
- prefill-4096 2.79 s (predicted about 3.5), prefill-32768 28.8 s (about 30), prefill-257 0.35 s (about 0.45).
- Rule 11 was the hardest, as predicted: it failed the first run by 0.08 ms.
