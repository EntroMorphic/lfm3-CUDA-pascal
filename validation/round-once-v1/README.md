# round-once-v1

The project owner's question (2026-11-18): is there a third arithmetic, more accurate than the tensor core's
truncation and the plain kernel's rounding to nearest? There is: sum each dot product exactly and round it once.
This round measures what that does when given to the [BOS] row, the one row
[prefill-arithmetic-v1](../prefill-arithmetic-v1/) found to carry the whole difference between the two arithmetics.

## The three arithmetics

| | Inside one dot product of K terms |
|---|---|
| Plain kernel (the engine) | K fused multiply-adds in FP32, each rounded to nearest |
| Tensor-core model (the Thor engine's arithmetic) | K/16 steps; each of 16 products cut 25 bits below the step's largest exponent; each step's total truncated to FP32 |
| Round once (this round) | every product summed exactly in a fixed-point accumulator; one rounding to nearest even, straight to the stored format (BF16, or FP32 for attention scores and logits) |

Round once applies to the GEMMs only. The norms, the activation, the softmax and the convolution's sums are as in
the engine, and every GEMM output is still stored as BF16 (or FP32 where the engine keeps FP32). It is not the FP32
model.

Build: `-DLFM_GEMM_ROUND_ONCE` (`wide_once_kernel` in `native/include/lfm/gemm.cuh`); `LFM_ONCE_ROWS=N` gives it to
prompt positions below N, the plain kernel to the rest. It equals neither the Thor engine nor Hugging Face bit for
bit, and is not meant to.

## Rules, written before any of it has run

**Rule 0: the kernel does what it says.** `tools/pascal/round_once_check.py` on the card: every output of the
round-once kernel equals the exact dot product (Python integers) rounded once to nearest even, in both operand
layouts and both output formats, on operands of the model's magnitudes, exponents spread over 100 binades,
subnormal and zero operands, a 32,768-term sum, and sums built to sit on a BF16 or FP32 rounding boundary and just
either side of it. Any mismatch: the kernel is wrong and nothing below is run.

**Runs.** The scorer with `LFM_ONCE_ROWS=1` (round once for the [BOS] row, plain for every other row) on the full
256- and 257-token windows, and on the 65-token windows of prefill-arithmetic-v1.

**Rule 1: the bar.** `tools/ppl_compare.py`, unchanged, against HF SDPA at 256 and 257 (the other lengths' files in
the report are the plain kernel's). Reported as it comes out. This variant "passes where the plain kernel fails"
only if 257 is not `worse`.

**Rule 2: is it nearer the FP32 model?** The anchor is the Thor project's FP32 run
(`ppl-fp32-v2`, `hf-eager_blocked-fp32`). Engine minus FP32, mean NLL, at 256 and at 257:

| Quantity (257 tokens) | HF SDPA | Thor engine | Plain kernel |
|---|---:|---:|---:|
| Position 1 | −3.17e-02 | −2.61e-02 | −2.14e-02 |
| Positions 1-64 | −9.77e-04 | −7.93e-04 | −3.06e-04 |

"Round once in the [BOS] row is nearer the FP32 model than the plain kernel" is accepted only if, at both lengths:

1. at position 1, |round once − FP32| is smaller than |plain − FP32|;
2. over positions 1-64, |round once − FP32| is smaller than |plain − FP32|;
3. `tools/ppl_fidelity.py`'s mean |NLL − NLL_FP32| per token is not larger than the plain kernel's.

If 1 fails, the accuracy of the sums in the [BOS] row is not what separates the BF16 engines from the FP32 model
there (storing every result as BF16 is the larger rounding by far), and round once is one more draw of that row.

Position 1 is one fixed distribution, shared by every window (prefill-arithmetic-v1). Its difference between two
arithmetics is a single number, not a sample: no interval is attached to it, and none is needed, because the same
row is used by every prompt the engine will ever read.

**Prediction.** Rounding every multiply-add toward zero, and away from zero, both lowered the NLL at position 1
(by 4.4e-02 and 3.5e-02), away from the FP32 model; the plain kernel, the most accurate of the three, is the nearest
to it. If that is accuracy at work, round once lands above the plain kernel at position 1, nearer FP32. Written as
the expectation; the rule above decides.

**Not judged here:** speed (the kernel is integer arithmetic at a few rows; in an engine the [BOS] row would be
computed once), and parity (this arithmetic is not the Thor engine's).

## Results

(none yet)
