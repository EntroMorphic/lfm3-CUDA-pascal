# memory-v1

Two changes to what the text engine reserves per token of capacity, described in
[docs/long-context-memory.md](../../docs/long-context-memory.md). Both are meant to leave every output bit
unchanged. This round checks that, then measures what they free.

## What it will be

1. **MLP row blocks.** The three MLP-wide buffers hold 256 rows instead of the whole context; prefill runs the MLP a
   block at a time, with the prefill GEMM kernel for every block (`lfm::gemm::Problem::wide`).
2. **Windowed key/value buffers.** A windowed layer keeps 2 × 256 positions per head (`lfm::text::WindowSlots`)
   instead of the whole context; prefill goes through one shared full-length pair. On by default only above 4,096
   tokens of capacity; `LFM3_COMPACT_WINDOWS=1` turns it on at any capacity so it can be compared here.

The batch, tree and decode-batch code (phase 3) read `Runner`'s caches with the full-length layout. At 4,096 and
below that layout is unchanged; above it they are not yet ported.

Baseline: build `text-v1` (`e30023b7…` CLI, `82681a43…` scorer) and its results in
[text-correct-v1](../text-correct-v1/). Under test: build `text-v2`.

## Rules (committed before `text-v2` has run on the card)

1. **Build.** As text-correct-v1 rule 1.
2. **Bookkeeping.** `tests/test_window_slots.py` passes: at every decode step the slots hold exactly the positions
   the engine's own key-range rule reads, contiguous and in order.
3. **Fixtures.** The four forward fixtures give the pinned ids, twice, with reset passing — with
   `LFM3_COMPACT_WINDOWS` unset and with it set to 1.
4. **Long decode.** `long-decode.request.json` (300 prompt tokens, 700 teacher-forced steps, capacity 1,024; the
   windowed layers' slots fill and compact twice): every step's FP32 logits byte-identical between `text-v1` and
   `text-v2` with `LFM3_COMPACT_WINDOWS=1`, and the argmax ids equal.
5. **NLL.** `text-v2` with `LFM3_COMPACT_WINDOWS=1`: NLL files byte-identical to `text-v1`'s at 256, 257 and 4,096
   (all windows). With it unset: byte-identical on the first 64 windows at 256 and 257 and the first 2 at 4,096.
6. **Memory.** Peak device memory at 256, 257 and 4,096, and at 32,768 (compact by default), recorded next to
   `text-v1`'s. The computed figures in the document are compared with these; a disagreement above 16 MiB is
   explained or the document corrected.
7. **32,768.** With `text-v2` the 32,768-token score either completes or fails; either is recorded. Its NLL has no
   `text-v1` counterpart if `text-v1` cannot run that length, so it is judged by text-correct-v1's rules 3 and 4.
8. **Nothing else changes.** The host suite's result is the one recorded before the change, plus the new test.

## Results (2026-11-17)

Builds from commit `ad7948f`: `text-v2` `liquid-lfm3` `0dc31546…e4519a2`, `lfm3-score` `75d017f9…29cb9b7`. The job
as run and its log: `records/mem-v1.sh`, `records/mem-v1.log`; NLL metadata and the 32,768-token NLL file in
`records/`.

| Rule | Result |
|---|---|
| 1. Build | **PASS**: 0 warnings, same linkage |
| 2. Bookkeeping | **PASS** (`tests/test_window_slots.py`) |
| 3. Fixtures | **PASS** with `LFM3_COMPACT_WINDOWS` unset and with it 1: pinned ids, two identical runs, reset passing |
| 4. Long decode | **PASS**: 701 argmax ids equal; the FP32 logits of all 701 steps byte-identical between `text-v1` and `text-v2` with compact windows (digest `7b36221a…`); decode 11.0 s both |
| 5. NLL | **PASS**: with compact windows, all windows at 256, 257 and 4,096 byte-identical to `text-v1`'s (`95979e15…`, `ec238637…`, `2886713b…`); unset, the subsets byte-identical |
| 6. Memory | below; the computed figures hold within 13 MiB |
| 7. 32,768 | **Completes**: 9 windows, mean NLL 2.531681687, perplexity 12.574635, 2,334 s, peak 6,575 MiB. Judged by text-correct-v1's rules 3 and 4, which are open on the Thor files |
| 8. Nothing else changes | host suite at `1664d89`: 2,397 passed, 5 skipped, 1 not runnable here (text-correct-v1) |

### Peak device memory (MiB)

| Capacity | `text-v1` | `text-v2`, windows full-length | `text-v2`, compact windows | Computed for the last column |
|---|---:|---:|---:|---:|
| 256 | 4,407 | 4,407 | 4,407 | 4,400 |
| 257 | 4,413 | 4,413 | 4,413 | — |
| 4,096 | 5,195 | 5,027 | 4,999 | 4,987 |
| 32,768 | out of memory | — | 6,575 | 6,562 |

At 4,096 the MLP blocks free 168 MiB (computed 169) and the compact windows 28 MiB more (computed 27: 8 KiB per
token less the fixed slots). At 32,768 the engine now leaves about 1.5 GiB of the card free. The measured peaks sit
6 to 13 MiB above the computed ones at every capacity; that offset is not explained (allocation granularity is the
guess, unmeasured) and is under the rule's 16 MiB.

Informational: the FP32 anchor's mean NLL at 32,768 in the Thor record is 2.531454657; this engine's is 2.3e-4 nats
per token above it.

### Parity

The fixtures' traces under this tree with the tensor-core model equal all 7,826 Thor pins
([tensor-core-model-v1](../tensor-core-model-v1/)); the 257-row fixture runs the MLP in two blocks. The compact
windows are off at the fixtures' capacity (512), so the parity gate does not exercise them; rules 4 and 5 do.

### Not covered

- Decode above 4,096 tokens of capacity with compact windows on by default has been run only through the scorer
  (prefill). The long-decode test forces the layout at capacity 1,024.
- The batch, tree and decode-batch code is not ported to the compact layout (phase 3).
