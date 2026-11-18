# prefill-arithmetic-v1

Why the plain prefill kernel fails the perplexity bar at 257 tokens
([text-correct-v1](../text-correct-v1/)): the nature of the difference between this engine and the Thor engine and
Hugging Face, at the project owner's request (2026-11-18).

## What the per-token files show (CPU analysis, done)

`tools/pascal/analysis/nll_anatomy.py` and `nll_tests.py` on the NLL files of the FP32 model, HF SDPA, HF eager, the
Thor engine and this engine (plain kernel); outputs in `anatomy.txt` and `tests.txt`. Confidence intervals are 95%,
bootstrap over windows.

**1. The difference is in the first 64 positions of a window, at every length, and nowhere else.**

This engine minus the Thor engine, nats per token:

| Length | Positions 1-64 | The rest |
|---|---|---|
| 256 | +5.64e-04 [+3.21e-04, +8.04e-04] | +1.08e-04 [−2.3e-05, +2.43e-04] |
| 257 | +4.87e-04 [+2.46e-04, +7.28e-04] | +2.96e-05 [−1.05e-04, +1.61e-04] |
| 4,096 | +5.70e-04 [−4.70e-04, +1.63e-03] (72 windows) | +2.08e-04 (65-1,024) and −4.95e-05 (1,025+), both spanning zero |
| 32,768 | +7.79e-04 [+4.7e-05, +1.63e-03] (positions 1-1,024; 9 windows) | −9.3e-05 (1,025-8,192, spanning zero), −1.70e-04 [−3.24e-04, −1.9e-05] (8,193+) |

Against HF SDPA the same: +6.71e-04 [+4.01e-04, +9.37e-04] on positions 1-64 at 257 and +7.9e-05 (spanning zero)
on the rest. At 256 and 257 the first 64 positions are a quarter of the tokens, which is the whole of the mean
difference; at 4,096 they are 1.6% and the means are equal.

**2. In those positions every BF16 engine scores below the FP32 model, this one least.**

Engine minus FP32, positions 1-64, length 257: SDPA −9.77e-04, Thor −7.93e-04, this engine −3.06e-04 (all three
intervals exclude zero). On the rest of the window all three are slightly above FP32 (+1.2e-04 to +2.0e-04).

**3. Per token it is not a few outliers.** This engine minus the Thor engine at 257: median 0, 49.9% of tokens higher
and 49.8% lower, rms 3.2e-02; dropping the top and bottom 1% of differences leaves +1.23e-04 of the +1.44e-04. The
two engines' deviations from FP32 have the same size (rms 0.0271 and 0.0272) and correlate at +0.31: two
realizations of the same amount of rounding noise, with a small shift between their means.

**4. The shift tracks how surprising the token is.** The slope of (engine − FP32) against the FP32 model's NLL, per
nat, at 256 and 257: SDPA −1.46e-04 and −1.47e-04, eager −1.09e-04 and −1.06e-04, Thor −1.22e-04 and −8.2e-05
(every interval excludes zero); this engine +1e-06 and −1.0e-05 (both span zero). This engine minus Thor:
+1.23e-04 [+8.2e-05, +1.65e-04] and +7.2e-05 [+2.9e-05, +1.14e-04]. At 4,096 and 32,768 that difference is
−2.8e-05 and −2.2e-05, spanning zero. So at short context SDPA, eager and the Thor engine give the surprising tokens
a little more probability than the FP32 model does, and this engine does not; with the 1% most surprising tokens
left out, their slopes vanish and this engine's is positive.

**5. What the three engines that agree have in common** is tensor cores: HF runs its BF16 GEMMs on them through
cuBLAS, the Thor engine through `mma.sync`. This engine's prefill GEMM is fused multiply-adds rounded to nearest.
Its decode path is bit-identical to Thor's, and with the tensor-core arithmetic in its prefill GEMM it reproduces the
Thor engine's pinned traces exactly ([tensor-core-model-v1](../tensor-core-model-v1/)).

## What is not known

Why. The tensor core truncates toward zero (its sums are biased low in magnitude, by about one part in 10^5 over a
2,048-term sum by arithmetic); nearest rounding is unbiased. Whether that bias is the cause, at which GEMMs it
matters, and why only the first positions show it, are open. Two readings that the data above do not separate:

- **Bias.** The truncation's systematic shrink of GEMM outputs acts on the network like a slightly lower gain, which
  shows where the model is least certain.
- **Something structural in the first positions** (very few attention keys, the first token's large activations)
  that reacts to how the low-order bits of a sum fall, whatever their direction.

## Experiment (rules written before any of it has run)

Windows of 65 tokens from the same corpus: [BOS] + 64 tokens, 4,622 windows (`windows-65.i32`,
sha256 `70b8b4fe…`), so that every scored token is in positions 1-64. The scorer of this engine, with the prefill
GEMM's arithmetic varied (`tools/pascal/arith_experiment.sh`):

| Variant | Prefill GEMM arithmetic |
|---|---|
| `plain` | FMA, nearest (the engine) |
| `exact` | the tensor-core model at every site (the Thor engine's arithmetic) |
| `rz` | FMA, every multiply-add rounded toward zero: the same direction of bias as the tensor core, about 16 times as much (2,048 truncations against 128) |
| `away` | FMA, every multiply-add rounded away from zero: the opposite bias |
| `head` / `nohead` | the tensor-core model in the output head only / everywhere but the head |
| `mlp`, `conv`, `attnproj`, `attn` | the tensor-core model in one kind of site only: the MLP's three projections; the convolution's two; attention's q, k, v, o and gate; attention's Q·Kᵀ and P·V |

Reported for each variant: mean NLL minus `plain`'s, paired, with a 95% interval.

Predictions, fixed now:

1. **Consistency.** `exact` − `plain` lies in [−8.0e-04, −2.5e-04] (the Thor engine minus this engine on positions
   1-64 was −4.9e-04 and −5.6e-04). If it does not, the positions-1-64 result above does not carry over to 65-token
   windows and the rest of this experiment is uninformative.
2. **If bias is the cause:** `rz` − `plain` is negative and larger in size than `exact` − `plain`; `away` − `plain`
   is positive. If `rz` and `away` have the same sign, or neither differs from `plain`, bias is not the cause.
3. **Where:** the single-site variants' differences from `plain` say which GEMMs carry the effect; whether they add
   up to `exact`'s is reported, not assumed.

Also run, as a separate identity check: the `exact` scorer on the 256- and 257-token windows; its NLL files should
equal the Thor engine's byte for byte (`nll-native-gemm-256.f32` `692a3f9f…`, and 257 by its recorded hash).

## Results

### The first four variants (2026-11-18)

4,622 windows of 65 tokens; `arith-first4.txt`. `plain`: mean NLL 4.293430001.

| Variant | Minus `plain`, all 64 positions | 95% CI | Position 1 alone | Positions 2-64 | 95% CI |
|---|---:|---|---:|---:|---|
| `exact` | −4.93e-04 | [−6.15e-04, −3.74e-04] | −4.76e-03 | −4.25e-04 | [−5.50e-04, −3.04e-04] |
| `rz` | −1.10e-03 | [−1.22e-03, −9.70e-04] | −4.41e-02 | −4.16e-04 | [−5.44e-04, −2.86e-04] |
| `away` | −6.91e-04 | [−8.16e-04, −5.70e-04] | −3.45e-02 | −1.55e-04 | [−2.82e-04, −3.2e-05] |

- **Prediction 1 (consistency): holds.** `exact` − `plain` is −4.93e-04, inside [−8.0e-04, −2.5e-04], and equal to
  what the Thor engine minus this engine showed on positions 1-64 of the longer windows (−4.9e-04, −5.6e-04).
- **Prediction 2 (bias): fails by the rule set for it.** Rounding away from zero lowers the NLL too. `rz` and `away`
  have the same sign, so the direction of the rounding bias is not the cause.

### What the first four showed that was not looked for

Position 1 is predicted from [BOS] alone, and [BOS] is the first token of every window. In `plain`, the NLL at
position 1 is identical for every window whose first token is the same (1,950 distinct tokens over 4,622 windows):
it is one fixed distribution, evaluated 4,622 times. A change of arithmetic moves that one distribution once, and
the move is counted in every window: `rz` lowers the NLL of position 1 by 4.4e-02 on average, nearly uniformly over
the first tokens (per distinct token: mean −4.6e-02, sd 2.0e-02). Most of `rz`'s and `away`'s difference from
`plain` is that one position.

In the longer windows the same holds across engines. Position 1, engine minus the FP32 model, length 257: SDPA
−3.17e-02, eager −3.17e-02, Thor −2.61e-02, this engine −2.14e-02; and Thor minus this engine there, −4.7e-03, is
`exact` minus `plain` at position 1 here (−4.76e-03).

A window bootstrap treats windows as independent. For position 1 they are not: it is a single draw.

### Reading under test: the first row is shared by every window

Row 0 of a prefill, the [BOS] row, does not depend on the rest of the window, so its activations, and the keys and
values every later position attends to, are the same in every window. Any arithmetic computes that one row once,
with its own rounding, and every window inherits the result. A difference between two arithmetics in that row is
then a fixed perturbation applied to all 295,808 tokens, not noise that averages out over them; its sign is whatever
it happens to be. Attention to the first token is strongest at short context, which would put the effect in the
first positions of a window and nowhere else.

That would make the 257-token failure one draw of a quantity that has no reason to favour either arithmetic, which
the bar's statistics count as 1,155 independent observations.

**Test, fixed before it has run.** The validation build can give the tensor-core arithmetic to chosen prompt
positions only (`LFM_TCM_ROWS`).

| Variant | Arithmetic |
|---|---|
| `bosexact` | tensor-core model for row 0 only, at every site; plain for every other row |
| `bosplain` | plain for row 0 only; tensor-core model for every other row |

Predictions if the reading is right:

1. `bosexact` − `plain` on positions 2-64 is negative with its interval excluding zero, and overlaps `exact`'s
   [−5.50e-04, −3.04e-04]. At position 1 `bosexact` equals `exact` exactly, value for value.
2. `bosplain` − `plain` on positions 2-64 has an interval that includes zero. At position 1 it equals `plain`
   exactly.

If instead `bosexact` shows nothing and `bosplain` carries `exact`'s difference, the reading is wrong: the effect
comes from the arithmetic of the other 64 rows.

**Result for `bosexact` (2026-11-18; `ar-exact` scorer `454dcc1b…`, `LFM_TCM_ROWS=1`):**

| Variant | Positions 2-64, minus `plain` | 95% CI | Position 1 |
|---|---:|---|---|
| `exact` (all 65 rows) | −4.25e-04 | [−5.50e-04, −3.04e-04] | −4.760e-03 |
| `bosexact` (row 0 only) | −4.47e-04 | [−5.64e-04, −3.23e-04] | −4.760e-03, equal to `exact` value for value |

`bosexact` − `exact` on positions 2-64: −2.2e-05 [−1.48e-04, +1.04e-04].

Prediction 1 holds. Giving the tensor-core arithmetic to the [BOS] row alone, one row in 65, reproduces the whole
difference between the two arithmetics; the arithmetic of the other 64 rows adds nothing detectable.

**Result for `bosplain` (`LFM_TCM_ROWS=-1`):** minus `plain` on positions 2-64, −1.00e-04 [−2.27e-04, +2.6e-05];
at position 1 equal to `plain` value for value. Prediction 2 holds: the interval includes zero. Against `exact` it
differs by +3.25e-04 [+2.05e-04, +4.43e-04].

The four cells, positions 2-64, minus `plain`:

| | Other 64 rows plain | Other 64 rows tensor-core |
|---|---:|---:|
| **[BOS] row plain** | 0 | −1.00e-04 (interval includes zero) |
| **[BOS] row tensor-core** | −4.47e-04 | −4.25e-04 |

Changing the arithmetic of 64 rows changes 98.3% of the tokens' NLL values (rms 0.033 per token) and leaves the
mean where it was. Changing it in the one shared row moves the mean by −4e-04.

## Conclusion

The difference that fails the perplexity bar at 257 tokens is made in the [BOS] row. That row is the same in every
window; each arithmetic rounds it once, differently; every later position attends to it. The result is one fixed
perturbation carried by all windows and all tokens, largest where attention to the first token is largest: the first
positions of a window. It is not a property of how the two arithmetics treat prompts in general: with the [BOS] row
held fixed, the arithmetic of the other rows makes no detectable difference to the mean.

The bar's interval is a bootstrap over windows, which takes windows as independent. For this effect there is one
observation, not 1,155. The interval's exclusion of zero says the two [BOS] rows differ, which they do; it does not
say one arithmetic gives better perplexity than the other.

Which way the draw falls relative to the FP32 model: in positions 1-64 every BF16 engine is below FP32, and this
engine's plain kernel is the nearest to it.

Not established: why the particular rounding of that row moves the mean the way it does (which elements, at which
layer); and whether the same shared-row effect explains the differences *between* the HF configurations and the
FP32 model (SDPA's own shift at 257 relative to 256), which the data here are consistent with but do not test.

## The hybrid against the bar (2026-11-18)

The validation build with `LFM_TCM_ROWS=1`: the tensor-core arithmetic for the [BOS] row, the plain kernel for every
other row (scorer `454dcc1b…`), on the full 256- and 257-token windows; `tools/ppl_compare.py` unchanged, against HF
SDPA (`records/compare-hybrid.json`; the 4,096 and 32,768 rows of that report are the plain kernel's files, the
hybrid was not run at those lengths).

| Length | PPL SDPA | PPL hybrid | dNLL | 95% CI | Verdict | Plain kernel, for comparison |
|---|---:|---:|---:|---|---|---|
| 256 | 26.5821 | 26.5798 | −8.68e-05 | [−2.14e-04, +4.17e-05] | equal: PASS | +6.89e-05, equal |
| 257 | 26.5045 | 26.5055 | +3.87e-05 | [−9.09e-05, +1.67e-04] | equal: PASS | +2.27e-04, **worse** |

Against the Thor engine the hybrid's mean differs by −4.5e-05 at 257 and +6.7e-05 at 256, both intervals spanning
zero, and its NLL at position 1 equals the Thor engine's value for value in all 1,155 and 1,160 windows.

Changing one row of the prefill moves the 257-token verdict from "worse" to "equal". Time: 1,190 s and 931 s for the
two lengths (plain: 1,005 s and 737 s), in a build whose plain kernel is not the tuned one's only cost: every GEMM
is launched twice.
