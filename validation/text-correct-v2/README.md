# text-correct-v2

Phase 1 of [PLAN.md](../../PLAN.md) again, after the owner's two decisions of 2026-11-18 (PLAN decisions 5 and 6):

- **Decision 5, the prefill kernel.** Prompt position 0 is computed with the tensor-core arithmetic, every other
  row with the plain kernel (`native/include/lfm/gemm.cuh`). Before this round that existed only as an experiment
  switch in the validation build (`LFM_TCM_ROWS=1`, [prefill-arithmetic-v1](../prefill-arithmetic-v1/)); here it is
  the engine.
- **Decision 6, the bar.** `tools/pascal/bar_amended.py`: a length fails only if the engine is "worse" than HF SDPA
  and "worse" than the FP32 model, by `ppl_compare`'s own statistic. `ppl_compare` is still run, unchanged, and
  reported first.

Build under test: `text-v3`. [text-correct-v1](../text-correct-v1/) is the record of the plain kernel and is not
changed.

## What is already known (so that the rules are read with it)

- The experiment hybrid passed the original bar at 256 and 257. It was not run at 4,096 or 32,768.
- The amended rule was written after the data that motivated it. Applied to files already on hand it gives: plain
  kernel PASS (257: worse than SDPA, not worse than FP32); experiment hybrid PASS; round-once [BOS] row FAIL at 256
  and 257. Those are not results of this round.

## Rules (committed before `text-v3` has run on the card)

1. **Build.** As text-correct-v1 rule 1.
2. **Tokens, determinism, reset.** The four forward fixtures: the Thor pins' ids, two runs identical, reset passing;
   with `LFM3_COMPACT_WINDOWS` unset and set to 1.
3. **Identity with the experiment.** `text-v3`'s NLL files at 256 and 257 are byte-identical to the experiment
   hybrid's: `c47912b0…` and `75430bb6…`. If not, the engine is not computing what was decided, and nothing below
   counts.
4. **Position 0 is the Thor engine's.** At 256, 257, 4,096 and 32,768 the NLL at position 1 (computed from row 0
   alone) equals the Thor engine's (`nll-native-gemm-L.f32`) value for value in every window.
5. **Perplexity, original bar.** `tools/ppl_compare.py`, unchanged, against HF SDPA at all four lengths, with
   `text-v3`'s own files at every length. No length "worse".
6. **Perplexity, amended bar.** `tools/pascal/bar_amended.py` at all four lengths: PASS.
7. **Fidelity.** `tools/ppl_fidelity.py`, unchanged: PASS at all four lengths.
8. **Parity gate.** `tools/pascal/gate_pins.sh` on this tree: 7,826 of 7,826.
9. **Sanitizer.** `compute-sanitizer --tool memcheck` on the four fixtures: 0 errors, ids equal.
10. **Memory and time.** Peak device memory and wall time per length, next to the plain kernel's. Recorded, not
    judged; a length that does not fit is a FAIL of that length.
11. **Nothing else changes.** The host suite's result is the one recorded (2,397 passed, 5 skipped, 1 not runnable
    here), plus any test added.

Phase 1 passes if rules 1 to 9 and 11 pass.

## Results

(none yet)
