# memory-v2

One change to what the text engine reserves per token of capacity, to make room for images (phase 5) on a card the
speech server already fills to within 33 MiB of its limit (7,707 of 7,740 MiB, [red-team-2](../red-team-2/)).

## What it will be

A layer's operator is a convolution or an attention, never both. Their per-row buffers were seven separate
whole-context allocations: `projected` (3 × 2,048) and `operated` (2,048) for a convolution; `q` (2,048), `k`, `v`
(512 each), `attended` and `gated` (2,048 each) for an attention: 15,360 BF16 values (30 KiB) per token. They now
share one buffer, `Runner::operand`, of the larger of the two (8,192 values, 16 KiB per token), each part on a
256-byte boundary (`native/src/lfm3_main.cu`). Nothing in either set is read after its layer ends, so no value is
overwritten before its last use. No kernel changes; only the addresses they are given.

Computed saving: 14 KiB per token of every `Runner`'s capacity: 406 MiB at the speech server's default text capacity
of 29,696, 56 MiB for each 4,096-token runner.

Baseline: builds `text-v38`, `parity-v38`, `text-v38s` and their results in [red-team-2](../red-team-2/). Under
test: `text-v39`, `parity-v39`, `text-v39s`, from the same tree but for this change.

## Rules (committed before the v39 builds have run on the card)

1. **Build.** All three build with 0 errors and 0 warnings.
2. **Device code.** Every binary's device code identical to v38's, function for function (`sass_compare.py`).
3. **Outputs unchanged.** Each identical to v38's:
   - the parity gate: 7,826 of 7,826 Thor pins;
   - the forward fixtures with `LFM3_COMPACT_WINDOWS` unset and 1, and under memcheck with 0 errors;
   - the scorer's NLL files byte for byte at 256, 257 and 4,096 (red-team-2's subset), both builds;
   - above 4,096: the logits of red-team-2's A1 prompts (129, 136, 137, 300 tokens at capacity 4,097), both builds;
   - speech: speech-v1 rule 3 (16 of 16 Thor clips) on `parity-v39`; the shipping seed-1 clips, the split
     lighthouse and the 19 LibriSpeech utterances on both builds, byte for byte against speech-v1's;
   - paired decode (serve-v1 rule 10) passes.
4. **Memory.** speech-v1 rule 12 at one speech row and 29,696 passes, its highest peak at most 7,357 MiB (350 below
   7,707). Each configuration's peak is recorded against red-team-2's; a saving that differs from the computed one by
   more than 32 MiB is explained.

The serving checks are not rerun: host code outside `Runner`'s allocation is unchanged, and rule 2 holds the
kernels.

## Results

Pending.
