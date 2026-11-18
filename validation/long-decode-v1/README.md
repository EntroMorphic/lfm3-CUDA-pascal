# long-decode-v1: decode above 4,096 tokens of capacity

Status: **passed** (2026-11-18). Rules and prediction committed at `05024b5` before the run.

## Why

Phase 1 passed ([text-correct-v2](../text-correct-v2/)) on fixtures at a capacity of 512 and on the scorer, which
only prefills. Above 4,096 tokens of capacity the windowed layers keep their keys and values in compact buffers
(2 × 256 positions per head, [memory-v1](../memory-v1/)), and that is the default. Token-by-token decode in that
default configuration has never been run: memory-v1's long-decode test forced the layout on at a capacity of 1,024.

There is no Thor pin at these lengths (its CLI calls capacity above 4,096 "unaccepted long-context bring-up"), so
the comparisons are the engine against itself, and its decode path against its own prefill path, which is the one
the perplexity bar validated at 32,768.

## Under test

Build `text-v6`, the binaries of text-correct-v2 (`liquid-lfm3` `7464f580…`). No rebuild, no source change.

## Runs

`tools/pascal/long_decode.sh text-v6 long-decode-v1`. Tokens come from the perplexity windows.

| Case | Capacity | Window | Prompt | Then | Positions decoded |
|---|---:|---|---:|---|---|
| control | 4,096 | 0 of `windows-4096` | 300 | 1,100 teacher-forced | 300–1,400 |
| short | 32,768 | 0 of `windows-32768` | 300 | 1,100 teacher-forced | 300–1,400 |
| middle | 32,768 | 0 | 4,000 | 1,100 teacher-forced | 4,000–5,100 (crosses 4,096) |
| end | 32,768 | 4 | 31,667 | 1,100 teacher-forced | 31,667–32,767 (to the last position) |
| free | 32,768 | first of 1, 2, 3, 5, 6, 7, 8 that generates 520 tokens | 6,000 | up to 600 greedy tokens | 6,000 on |

Each case at 32,768 is run three times: `LFM3_COMPACT_WINDOWS` unset (the default: compact), `=0` (the imported
full-length layout), and unset again. The control is run once (capacity 4,096 is the imported layout, the validated
configuration). 1,100 steps cross the 256-token window and compact the 512-slot buffers at least twice; so do 520.

## Rules

1. **Build.** The binary's SHA-256 on the card equals text-correct-v2's.
2. **Compact equals full.** In every 32,768 case the default run and the `=0` run give the same ids and
   byte-identical FP32 logits at every step (1,101 steps in the teacher-forced cases; in `free`, as many as were
   generated). The default run's peak device memory is below the `=0` run's, which shows the default is the
   compact layout.
3. **Deterministic.** The second default run is identical to the first in every case.
4. **Decode against prefill** (`tools/pascal/analysis/long_decode_report.py`). At each teacher-forced step, d is
   the NLL of the next window token from the decode step's logits minus the NLL our scorer gave that position
   (text-correct-v2's NLL files). The yardstick y is the Thor engine's scorer minus ours at the same positions
   (`build/thor/ppl-v5`): two prefill arithmetics that both pass the bar. A case passes if
   mean |d| ≤ 2 × mean |y|. The control is judged the same way; if the control fails, the rule is mis-set and
   that is what gets reported, not a pass by other means.
5. **Parity gate, sanitizer, host suite.** Not re-run: no source or binary changes.

A case that cannot allocate (the full-length layout at 32,768 has never been run in the CLI) fails rule 2 for
that case and is reported as such.

## Prediction (before any run)

- Rule 2: identical everywhere. The compact buffers keep key order, and the forced test at 1,024 was identical.
- Rule 3: identical.
- Rule 4: ratios near 1 (between 0.5 and 1.5), no trend with position. The decode kernels and the prefill kernel
  sum in different orders, as the Thor and Pascal prefill kernels do.
- The full-length layout fits at 32,768 in the CLI, with a peak about 250 MiB above the compact one.
- The control's logits equal the `short` case's step for step (same tokens; only the capacity differs). Not a
  rule: the prefill attention is tiled differently above 4,096, and whether that changes bits is not known.
- `free` reaches 520 tokens on window 1.

## Not covered

- The server, batch and tree code above 4,096 (phase 3).
- Sampling (the runs are greedy or teacher-forced), and prompts given as chat messages.
- Capacities between 4,096 and 32,768.

## Results

Run on the card 2026-11-18, tree `7734f25f…`, binary `7464f580…` (rule 1: equal to text-correct-v2's). Records in
`records/` (one JSON per run with the ids, every step's logits SHA-256 and NLL; `summary.json`; `rule4.json`;
`job.log`), fetched and checked against `SHA256SUMS`.

| Rule | Result |
|---|---|
| 1. Build | **PASS** |
| 2. Compact equals full | **PASS**: ids and every step's FP32 logits identical in all four cases |
| 3. Deterministic | **PASS**: the second default run identical in all four cases |
| 4. Decode against prefill | **PASS**: ratios 0.68 to 0.82 |

### Rules 2 and 3

| Case | Steps | Compact against full | Repeat | Peak, compact (MiB) | Peak, full (MiB) | Prefill (s) | Decode (s) |
|---|---:|---|---|---:|---:|---:|---:|
| short | 1,101 | identical | identical | 6,451 | 6,703 | 1.0 | 18.4 |
| middle | 1,101 | identical | identical | 6,451 | 6,703 | 9.4 | 18.6 |
| end | 1,101 | identical | identical | 6,451 | 6,703 | 84.2 | 24.0 |
| free (window 1, 600 tokens) | 600 | identical | identical | 6,451 | 6,703 | 12.9 | 10.3 |

Decode time is the same in the two layouts. Per token: 16.7 ms from position 300, 16.9 ms from 4,000, 21.8 ms at
the end of the context, 17.2 ms in `free` (these runs write every step's logits to disk, so they are not speed
measurements).

### Rule 4

d = decode minus our prefill; y = Thor's prefill minus ours; NLL in nats, 1,101 positions per case.

| Case | Positions | Mean d | Mean \|d\| | Max \|d\| | Mean \|y\| | Max \|y\| | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| control (4,096) | 300–1,400 | +3.77e-04 | 1.518e-02 | 0.323 | 1.855e-02 | 0.221 | 0.82 |
| short | 300–1,400 | +5.75e-04 | 1.364e-02 | 0.188 | 1.771e-02 | 0.302 | 0.77 |
| middle | 4,000–5,100 | −6.21e-04 | 1.872e-02 | 0.209 | 2.362e-02 | 0.244 | 0.79 |
| end | 31,667–32,767 | +1.27e-04 | 1.736e-02 | 0.214 | 2.571e-02 | 0.334 | 0.68 |

### Against the prediction

Every item held: identical under rules 2 and 3; ratios inside 0.5 to 1.5 with no trend toward the end of the
context; the full-length layout fits at 32,768 with a peak 252 MiB above the compact one; `free` ran its 600
tokens on window 1; and the control's logits equal the `short` case's at every step, so a 300-token prefill and
the decode after it give the same bits at a capacity of 4,096 and of 32,768.

### Seen, not explained

The control and `short` have identical decode logits but different d (1.518e-02 and 1.364e-02). The difference is
on the prefill side: our scorer's NLL at positions 300–1,400 of the same tokens is not the same in the 4,096-token
windows as in the 32,768-token ones. The two prefills tile attention differently (whole prompt at 4,096, 128-row
tiles above); that this is the cause was not tested.
