# Proposal for the Thor project: the perplexity bar

Status: **a proposal** (2026-11-18). Adopted in this project as PLAN decision 6. The Thor repository is read-only
from here; this is for its owner and team to take or leave.

## What was found

Measured on the GTX 1080 engine, with the Thor project's own windows, HF references, FP32 run and statistics.
Records: [prefill-arithmetic-v1](../validation/prefill-arithmetic-v1/), [round-once-v1](../validation/round-once-v1/).

1. **Every window shares its first row.** Windows are `[BOS]` + content. Row 0 of a prefill depends on nothing
   else, so it is identical in every window, and the NLL at position 1 is one fixed distribution evaluated once per
   window. Every later position attends to that row.
2. **Two arithmetics differ in that row once, and the difference is counted once per window.** Giving the
   tensor-core arithmetic to row 0 alone reproduces the whole difference between the plain kernel and the
   tensor-core kernel (−4.47e-04 against −4.25e-04 for every row, positions 2-64 of 65-token windows). Giving it to
   the other 64 rows alone changes 98% of the per-token values and leaves the mean where it was. The bootstrap over
   windows treats windows as independent; for this effect there is one observation.
3. **The reference is not the truth.** At position 1, relative to the FP32 run: HF SDPA −3.17e-02, the Thor engine
   −2.61e-02, FMA rounded to nearest −2.14e-02, exact sums rounded once −1.87e-02. The more accurately row 0 is
   summed, the further the result is from SDPA.
4. **The FP32 model does not pass the bar at 257 tokens.** FP32 minus SDPA: +1.53e-04, 95% CI [+4.08e-05,
   +2.64e-04]: "worse". (256: equal. 4,096 and 32,768: better.)

## What it means for the Thor engine

Its results stand. It passes at every length. But the bar can fail an engine for computing row 0 more accurately
than cuBLAS's tensor-core path does, and it would fail the FP32 model at 257.

## Proposed amendment

Keep `ppl_compare.py` and its statistic. Add, per length where an FP32 run exists: the same statistic with the FP32
run as the reference. A length fails only if the engine is "worse" than HF SDPA and "worse" than FP32. Report
position 1 separately, without an interval. Implementation here: `tools/pascal/bar_amended.py` (it imports
`ppl_compare`'s loaders and verdict unchanged).

## Limits of this proposal

- The amendment was written after the data that motivated it.
- It does not repair the interval: an effect that comes through the shared row is still one draw. A corpus whose
  windows did not share a first row would need a different window rule, which would break comparison with every
  record so far.
- Why rounding error in row 0 lowers the next token's NLL is not established.
