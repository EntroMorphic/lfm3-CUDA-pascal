# Experimental native LFM3 text forward pass

The separate `liquid-lfm3` executable loads real checkpoint weights and implements
batch-one BF16 text prefill/decode, final normalization and the tied output head.
The original `liquid-native` LFM2 path and its LFM3 execution guard are unchanged.
Use of the new runner requires `--experimental`.

## Contracts

- Geometry currently restricted to hidden=2048, head dimension=128, convolution
  width=7, tied embeddings and sandwich normalization. Tensor dtypes and all
  selected weight shapes are checked against the parsed configuration.
- Inputs are explicit token IDs, not native chat-tokenizer output. Validation uses
  the pinned official processor to prepare those IDs; native chat-template parity
  is not claimed.
- One unpadded, unpacked prompt per request; contiguous capacity 1..4096. Only
  empty-cache prefill followed by one-token decode is supported. Over-capacity
  requests are rejected before checkpoint upload. No ring buffer or chunked prefill.
- Text-only Omni decoding suppresses audio/image/video placeholder IDs and
  audio/mixed-mode starts. It stops on text EOS **or `text_end_token_id`**, plus any
  caller-provided stop tokens. Audio/vision encoders and audio generation are not
  implemented. Standalone text checkpoints use configured EOS tokens.
- Greedy decoding only. Host-side logits copy/argmax is included in inference
  timings. There is no sampling, batching, CUDA graph capture or production API.
- Prefill convolution uses sequential reduction; decode uses the pinned width-7
  Triton reduction. RMS normalization follows ATen's vector-4, row-count-dependent
  reduction, with BF16 cast before weight multiplication and residual addition.
- Attention uses BF16 cuBLAS QK/PV products and a native causal/windowed softmax.
  This is an eager-style reference path, not an SDPA or FlashAttention kernel.
  Persistent reduction is used through 2048 columns; the ATen register/block
  reduction is used above that. The expanded nine-case suite verifies bitwise
  vocabulary logits through the 4096-slot limit. This does not imply universal
  exactness for every prompt or SDPA parity.
- Loader still uploads all checkpoint payloads, including unused non-text weights.

## Validation and remediations

Initial native integration failed its activation acceptance gate despite matching
the baseline's first 16 next-token choices. Those reports are retained separately.
Findings and corrections:

1. A generic RMS reduction is not numerically interchangeable with ATen's schedule.
   Match vector-4 accumulation and block width selected from the logical row count.
2. Scalar attention reductions and reference cuBLAS products differ near BF16
   rounding boundaries. Use BF16 batched products and a lane-strided XOR softmax
   reduction instead; retain causal/sliding masks and grouped KV addressing.
3. Match default PyTorch BF16 cuBLAS policy and its workspace budget. Runtime
   validation uses the same container cuBLAS libraries as the oracle.
4. Text-only Omni stopping must include `text_end_token_id`, even when it is absent
   from the nested text configuration's EOS list.

Acceptance scripts do **not** automatically increase tolerances. Every captured
activation must have relative RMS error <=0.02; logits additionally require maximum
absolute error <=0.5. All teacher-forced and free-running greedy choices must match.
These finite-fixture gates do not establish universal exact/BF16 or SDPA parity.

`check_lfm3_forward.py` compares embeddings, both pre-norms, convolution projections
and products, operator outputs, FFN projections/products/outputs, layer outputs,
final normalization, and last-token vocabulary logits. It also checks same-process
reset against **every logit exactly**, stopping, suppression and invalid requests.

The source-frozen binary passed four checkpoint cases: baseline (16 tokens),
arithmetic (2 tokens, EOS), a 257-token prompt (4 tokens), and a single-token
prompt (4 tokens). **All 5,746 captured BF16 tensors are bitwise identical** to
the Transformers eager oracle, including vocabulary logits. Teacher-forced and
free-running choices match; exact-logit reset, stopping, suppression and eight
invalid-request checks pass. Evidence: `validation/forward-v3/`. Comparisons were
also rerun with the stronger `--require-exact` gate and per-trace SHA-256 records.

The combined host suite now passes 327 tests. Final-slot cache filling is verified
in the later forward-v5 and context-v3 evidence, not retroactively credited to v3.
The 86-token warmed benchmark passed exact greedy comparisons for the `af0a0db7...`
candidate; the later full-context runtime requires a fresh hash-matched measurement.
Integrated initcheck is blocked by a separately reproduced cuBLAS output tracking
limitation. See `audit/ACCEPTANCE.md`; no complete sanitizer pass is claimed.
Benchmark acceptance requires the same binary hash as the correctness suite and
exact generated IDs. Earlier failed candidate reports remain on Thor in
`build/forward-baseline-pedantic`, `build/forward-gates-v1`, `build/forward-gates-v2`,
and the preserved pre-normalization comparison reports.

The final frozen runtime (`cabeb776...`) passes **6,890 bitwise activation/logit
checks**, all nine context cases through 4096, and a hash-matched warmed 86-token
benchmark. Integrated 4096-token memcheck/racecheck/synccheck pass within the
documented NVJET exclusion scope; initcheck remains explicitly blocked. Full
results and the final disposition are in `audit/ACCEPTANCE.md`.

## Reproduction on Thor

```bash
bash tools/thor/legacy/build-forward-thor.sh
docker exec \
  -e TRITON_PTXAS_PATH=/workspace/tools/ptxas \
  -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \
  lfm3-omni-dev python /workspace/liquid-CUDA/tools/check_lfm3_forward.py \
  --checkpoint /workspace/models/LFM3-Omni-2B \
  --binary /workspace/liquid-CUDA/build/cmake/liquid-lfm3 \
  --output /workspace/liquid-CUDA/build/forward-gates-new
```

Choose a fresh output directory; the suite refuses to replace existing evidence.
`tools/thor/legacy/run-forward-sanitizers-thor.sh` runs the integrated path. Its synccheck run
explicitly excludes NVIDIA NVJET kernels because of the documented instrumentation
failure; custom kernels remain instrumented. Other tool runs do not use exclusions.
This limitation must accompany any sanitizer claim for the integrated executable.

`benchmark_lfm3.py` runs synchronized, manual batch-one greedy inference without
trace copies. It reports separate prefill time and decode forward calls/second,
three measured samples, and warmup. Model loading and text decoding are excluded.
The native and Transformers caches are different implementations; this is not a
claim of equal memory usage or a general model-serving benchmark.
