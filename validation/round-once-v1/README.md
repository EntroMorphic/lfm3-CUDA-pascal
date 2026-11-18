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

## Results (2026-11-18)

Tree `8a1a7ba5…`, scorer `ar-once` (`da7facd9…`), `LFM_ONCE_ROWS=1`. Files and reports in `records/`.

**Rule 0: PASS.** 3,168 of 3,168 outputs equal the exact dot product rounded once (`records/round-once-check.txt`).
For information, from the same run: on operands of the model's magnitudes the plain kernel's BF16 outputs already
equal the correct rounding in every case tried (272 of 272); its FP32 outputs (the format of attention scores and
logits) differ from it in 262 of 272.

**Rule 1, the bar against HF SDPA: FAIL at both lengths.**

| Length | PPL SDPA | PPL round once | dNLL | 95% CI | Verdict | Plain kernel | Tensor-core [BOS] row |
|---|---:|---:|---:|---|---|---|---|
| 256 | 26.5821 | 26.5872 | +1.92e-04 | [+5.97e-05, +3.24e-04] | **worse** | +6.89e-05, equal | −8.68e-05, equal |
| 257 | 26.5045 | 26.5123 | +2.96e-04 | [+1.67e-04, +4.26e-04] | **worse** | +2.27e-04, worse | +3.87e-05, equal |

**Rule 2, nearer the FP32 model: not accepted.** Engine minus FP32, mean NLL
(`tools/pascal/analysis/round_once_report.py`):

| | Position 1 | | Positions 1-64 | | Mean abs. difference per token | |
|---|---:|---:|---:|---:|---:|---:|
| | 256 | 257 | 256 | 257 | 256 | 257 |
| HF SDPA | −3.139e-02 | −3.169e-02 | −8.28e-04 | −9.77e-04 | 1.9400e-02 | 1.9476e-02 |
| Thor engine | −2.572e-02 | −2.605e-02 | −7.33e-04 | −7.93e-04 | 1.6741e-02 | 1.6773e-02 |
| Tensor-core [BOS] row, plain elsewhere | −2.572e-02 | −2.605e-02 | −6.68e-04 | −9.26e-04 | 1.6825e-02 | 1.6750e-02 |
| Plain kernel | −2.110e-02 | −2.142e-02 | −1.69e-04 | −3.06e-04 | 1.6716e-02 | 1.6729e-02 |
| Round-once [BOS] row, plain elsewhere | −1.852e-02 | −1.868e-02 | −9.76e-05 | −3.30e-04 | 1.6707e-02 | 1.6713e-02 |

1. Position 1: nearer than the plain kernel at both lengths, and the nearest of every engine measured. Holds; the
   prediction holds.
2. Positions 1-64: nearer at 256, farther at 257 (−3.30e-04 against −3.06e-04). **Fails at 257.**
3. `ppl_fidelity.py`: 1.671e-02 at both lengths, not larger than the plain kernel's. Holds (and PASS against SDPA).

By the rule written beforehand the claim needed all three at both lengths. It is not accepted. What the data show
is narrower: round once moves the one distribution at position 1 about 13% of the way from the plain kernel's to the
FP32 model's, and the per-token closeness to FP32 improves by 0.05% to 0.1%, which is nothing.

### What was not looked for

**The more accurately the [BOS] row is summed, the higher the NLL at position 1, in every case measured.** Position 1
minus the FP32 model, 257 tokens (the last two from prefill-arithmetic-v1's 65-token windows, relative to the plain
kernel there):

| [BOS] row arithmetic | Position 1, minus FP32 |
|---|---:|
| FP32 model | 0 |
| Round once | −1.87e-02 |
| Plain kernel (nearest, every multiply-add) | −2.14e-02 |
| Tensor-core truncation (Thor engine) | −2.61e-02 |
| HF SDPA | −3.17e-02 |
| Every multiply-add rounded away from zero | about −5.6e-02 (plain − 3.45e-02) |
| Every multiply-add rounded toward zero | about −6.6e-02 (plain − 4.41e-02) |

Rounding error in that row lowers the NLL of the first scored token, and the reference of the bar, HF SDPA, is among
the lowest. So an engine is judged "worse" at short context in proportion to how accurately it computes the [BOS]
row.

**The FP32 model itself does not pass the bar at 257 tokens.** The Thor project's FP32 run against HF SDPA, with
`ppl_compare`'s own bootstrap (10,000 resamples over windows, seed 0):

| Length | FP32 − SDPA | 95% CI | Verdict by the bar's rule |
|---|---:|---|---|
| 256 | +1.92e-05 | [−9.60e-05, +1.31e-04] | equal |
| 257 | +1.53e-04 | [+4.08e-05, +2.64e-04] | **worse** |
| 4,096 | −3.02e-04 | [−4.21e-04, −1.80e-04] | better |
| 32,768 | −4.23e-04 | [−5.62e-04, −2.81e-04] | better |

At 257 tokens the bar fails the model computed at full precision, the thing every BF16 engine is an approximation
of. The 257-token failure of this engine's plain kernel is the same failure.

## Conclusion

- A third arithmetic exists and is built: exact sums, one rounding. Checked against integer arithmetic.
- In the [BOS] row it is the nearest to the FP32 model at position 1 of anything measured, as predicted. Past
  position 1 it is not distinguishable from the plain kernel, and the full claim fails its rule at 257.
- It does not help with the bar; it fails it at both short lengths, further than the plain kernel does. That is
  because the bar's reference sits on the far side: SDPA's short-context NLL is below the FP32 model's, and the FP32
  model itself is "worse" than SDPA at 257 by the bar's rule.
- For choosing what to ship: closeness to SDPA at short context and closeness to the FP32 model pull in opposite
  directions in this one row. The tensor-core [BOS] row passes the bar and matches the Thor engine; the plain kernel
  and round once are nearer the FP32 model and fail the bar at 257.

Not established: why rounding error in the [BOS] row lowers the NLL of the next token (which elements, at which
layer).

## The 65-token windows (added when the run finished, 2026-11-18)

4,622 windows, positions 1-64 only, minus the plain kernel, paired, 95% bootstrap over windows
(`records/nll-once-65.meta.json`; NLL sha256 `29b59940…`; 5,608 s). The first three rows are prefill-arithmetic-v1's.

| [BOS] row | Other rows | All 64 positions | 95% CI | Position 1 | Positions 2-64 | 95% CI |
|---|---|---:|---|---:|---:|---|
| tensor-core | tensor-core | −4.93e-04 | [−6.15e-04, −3.66e-04] | −4.760e-03 | −4.25e-04 | [−5.50e-04, −3.04e-04] |
| tensor-core | plain | −5.14e-04 | [−6.33e-04, −3.94e-04] | −4.760e-03 | −4.47e-04 | [−5.67e-04, −3.28e-04] |
| plain | tensor-core | −9.88e-05 | [−2.25e-04, +2.55e-05] | 0 | −1.00e-04 | [−2.26e-04, +2.71e-05] |
| round once | plain | +1.47e-04 | [+3.11e-05, +2.63e-04] | +2.795e-03 | +1.05e-04 | [−1.53e-05, +2.23e-04] |

Consistent with the longer windows: round once in the [BOS] row raises the NLL at position 1 (+2.8e-03 here; +2.6e-03
and +2.7e-03 at 256 and 257), the opposite direction from the tensor-core row (−4.8e-03). On positions 2-64 its
difference from the plain kernel has an interval that includes zero. No FP32 run exists for these windows, so
nearness to FP32 is not judged here.
