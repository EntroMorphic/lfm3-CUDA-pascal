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

## Results

(none yet)
