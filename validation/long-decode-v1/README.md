# long-decode-v1: decode above 4,096 tokens of capacity

Status: **rules written, nothing run** (2026-11-18).

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
