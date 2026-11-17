# tensor-core-model-v1

An experiment pulled forward from the parking list by the project owner (2026-11-17): can Blackwell's tensor-core
arithmetic be modelled exactly without the hardware? Done on the workstation's CPU while the GTX 1080 was busy with
[text-correct-v1](../text-correct-v1/). No rules were set in advance; this is exploration, and what it claims is
limited to what is listed under "What is and is not established".

## Result

One model reproduces the Thor engine's first prefill GEMM bit for bit on all three multi-row fixtures.

For one `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` step, each output is

    D = mma(C, a[0..16), b[0..16))

1. The 17 terms are the accumulator `C` and the 16 products `a[i] * b[i]`, each product exact (two 8-bit
   significands).
2. Let `E` be the exponent of the term largest in magnitude. Every term is truncated toward zero to a multiple of
   `2^(E - 25)`: 24 bits plus 2 guard bits below the largest term's leading bit.
3. The truncated terms are added exactly.
4. The sum is truncated toward zero to FP32.

The Thor kernel (`lfm::gemm` `wide`, as imported at `fc130a3`) chains these steps over K in ascending blocks of 16,
then rounds the accumulator to BF16, nearest even.

## Method

The oracle is the Thor engine's pinned trace hashes (`validation/g4-gates-v2/pins/forward.json`, imported from
liquid-CUDA unchanged; Thor binary `a75e26b1…`). The trace payloads are not on the workstation, so a candidate
either reproduces a whole tensor's SHA-256 or it does not; there is no partial credit.

Scripts: `tools/pascal/tensor_core/`. Inputs: the four forward fixtures' prompts (`input_ids.json`, as the Pascal
engine tokenized them) and the checkpoint.

| Step | Script | Result |
|---|---|---|
| Embedding rows | `tcm.py` | hash equal to the pin on all 4 fixtures |
| Layer 0 `operator_norm`, in the norm kernel's own reduction order | `tcm.py` | equal on all 4, with `rsqrtf` taken as either the correctly rounded value or `1/sqrtf` |
| Control: `conv.in_proj` of the 1-row fixture (the `thin` kernel: plain `fmaf`, no tensor core) | `gemm_models.py` | equal: the harness and the BF16 rounding are right, and plain FMA agrees between Blackwell and the CPU |
| `conv.in_proj`, 31 rows: exact sums of blocks of 1, 2, 4, 8 or 16 products, rounded to nearest or toward zero (10 models; block 1 nearest is the Pascal kernel's arithmetic) | `gemm_models.py` | none equal; they differ from one another in 18 to 395 of 190,464 outputs |
| `conv.in_proj`, 31 rows: aligned truncation, block 16, 8 or 4 × guard bits 0, 1, 2, 3, 4, 5, 6, 8 × term truncation toward zero or floor × final toward zero, nearest or floor (144 models) | `aligned_models.py`, `aligned-arithmetic.log` | exactly one equal: block 16, 2 guard bits, toward zero, toward zero |
| That model on the other fixtures | `confirm.py` | equal: 31 × 6,144, 34 × 6,144 and 257 × 6,144 (1,978,368 values; the three fixtures run the kernel's three tile shapes) |

## What is and is not established

Established:

- For this GEMM (K = 2,048: 128 chained steps per output; this checkpoint's weights; normalized activations), the
  model gives Thor's BF16 outputs exactly, 1,978,368 of 1,978,368.
- Its neighbours do not: 1 or 3 guard bits, floor instead of toward zero, a final round to nearest, and blocks of 8
  or 4 each fail on the 31-row fixture. The matching model differs from an exact 16-product sum rounded to nearest in
  197 of 190,464 outputs, so the hash separates them.

Not established:

- Anything the data do not exercise: infinities, NaN, denormals, overflow, and how `E` is taken when terms cancel in
  ways these inputs never produce.
- Other `mma` shapes and types (FP16, TF32, INT8), other GPUs, `tcgen05`.
- That the hardware works this way internally; only that this arithmetic gives the same results here.
- Later tensors. Everything after `conv.in_proj` in the trace depends on further kernels (convolution, attention
  with FP32 scores, softmax, `expf` in SwiGLU) whose agreement between Blackwell, Pascal and the CPU is not yet
  tested, except `norm` and plain FMA, which agree.
- A GPU implementation. The model has only run in NumPy.

## Prior work (looked up after the result, 2026-11-17)

This model was found here without reference to the literature, and it is not new. From search results (abstracts
and summaries only; the papers have not been read):

- FTTN (Li et al., CCGrid 2024, arXiv:2403.00232): tests that reveal tensor cores' rounding mode (truncation or
  nearest), extra internal bits and block-FMA width; reports that these differ between GPU generations.
- MMA-Sim (Xie et al., arXiv:2511.10909): bit-accurate reference models of tensor cores derived by black-box
  testing, eight NVIDIA and two AMD architectures.
- "Bit-Exact AI Inference Verification Without Performance Tradeoffs" (arXiv:2606.00279): a software emulator with
  per-GPU profiles including Blackwell, described as aligning products to a maximum exponent within a fixed-point
  window, truncating what is shifted out, summing as integers and truncating to FP32. That is the form found here.
- Hawkeye (arXiv:2603.20421) and a MATLAB model set (arXiv:2512.07004): bit-exact CPU replication of tensor-core
  tile updates.

### The published model, read and tested (2026-11-17)

MMA-Sim's algorithm for NVIDIA's mixed-precision tensor cores (its "T-FDPA", as read from the paper's HTML by a
helper; the per-architecture table came through a summary and should be checked against the paper): products
exact; each term's exponent is the *sum of its operands' exponents*, the accumulator's its own; all terms are
truncated toward zero to F fractional bits below the largest exponent, summed exactly, and rounded; for Blackwell
16 products per step, F = 25 and the final rounding toward zero. MMA-Sim's hardware includes B200 and RTX PRO 6000
Blackwell, not Jetson Thor. Code: github.com/microsoft/MMA-Sim.

That is this record's model with one difference: where the window sits when a product is the largest term. This
record used the product's leading bit, which is one higher than the exponent sum when the significands' product is
2 or more.

`tools/pascal/tensor_core/tfdpa.py` runs both forms:

| Fixture | T-FDPA (L 16, F 25, RZ) | Leading-bit form | Outputs differing between them | Steps where their windows differ |
|---|---|---|---:|---:|
| `arithmetic` | equal to the Thor pin | equal | 0 of 190,464 | 3.47% |
| `baseline` | equal | equal | 0 of 208,896 | 3.51% |
| `window257` | equal | equal | 0 of 1,579,008 | 3.54% |

So these data cannot tell the two forms apart: they place the window differently on 3.5% of steps and still give
the same BF16 outputs. The published form was derived from targeted tests on hardware, which can tell them apart,
and it is the one a GPU kernel here should implement. It also confirms on Jetson Thor (through the Thor engine's
pins) the Blackwell parameters MMA-Sim reports for other Blackwell parts.

The cases listed above as not established (zeros, subnormals, infinities, NaN, cancellation) are only partly
covered in what was read of the paper; its code is the place to look.

## What it is for

If the remaining kernels also agree, a build of the Pascal engine whose prefill GEMM uses this arithmetic would have
to reproduce all of the Thor pins (4,816 trace files on the `baseline` fixture alone): exact gates in place of
statistical ones. The terms fit integer arithmetic (17 terms of at most 26 bits sum within 32 bits), so it can be
written for `sm_61`; it will be slower than the plain FMA kernel and is a validation build, not the shipping one.
