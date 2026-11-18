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

(none yet)
