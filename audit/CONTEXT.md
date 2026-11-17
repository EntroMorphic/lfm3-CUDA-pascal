# Context/capacity boundary gate

## Status and scope

`tools/check_lfm3_context.py` is a new integration gate, **not GPU evidence yet**.
Only `tests/test_lfm3_context.py` was executed here: 33 CPU tests passed. No SSH,
CUDA, GPU workloads, runner edits, shared-script edits, or commits were performed.

The owning session reports that its initial GPU run passed capacity1, decode2,
conv7 and window255; window256 prefill was exact but first decode differed. It
has implemented physical sliding-decode key cropping to 256 to match the
official DynamicSlidingWindowCache; verification of that remedy belongs to the
owning session. These reported results are not additional runs performed here.

The reference loads the official checkpoint once with eager attention and BF16,
calls its text backbone with a fresh dynamic cache for each case, and applies the
official tied output head only to the last hidden token. All later forwards are
single-token teacher forcing. The oracle and its caches are freed before native
processes start. The script checks every vocabulary logit at every forward, exact
teacher argmax IDs, selected actual greedy generations, and the native
same-process exact-logit reset replay. Teacher forcing continues even after EOS;
free generation follows stop semantics and reports whether it actually reached
the final slot. Every teacher case consumes its last legal cache slot.

Default acceptance is **bitwise BF16**, including signed zero, plus finite values,
correct tensor lengths, and exact token IDs. `--atol`/`--rtol` are explicit opt-ins
to `abs(native-reference) <= atol + rtol*abs(reference)`; no automatic fallback,
threshold adjustment, or token-only success is allowed. Exactness is still
reported if tolerances are requested. Use defaults for the initial run.

## Invocation (main session should serialize GPU execution)

Inside the same pinned reference container used for `check_lfm3_forward.py`:

```bash
python tools/check_lfm3_context.py \
  --checkpoint /path/to/LFM3-Omni-2B \
  --binary /path/to/liquid-lfm3 \
  --output /fresh/path/context-smoke --suite smoke

python tools/check_lfm3_context.py \
  --checkpoint /path/to/LFM3-Omni-2B \
  --binary /path/to/liquid-lfm3 \
  --output /fresh/path/context-full --suite full
```

The output path must not exist. The CUDA library path is set identically to the
existing forward gate. Each native subprocess has a configurable 1800-second
timeout. `summary.json` is updated atomically before each case/step; failure is
nonzero and preserves `status: failed`, completed comparisons, case, step, phase,
and error message. Only `status: passed` is passing evidence.

### Reuse an expensive oracle after rebuilding native

```bash
python tools/check_lfm3_context.py \
  --checkpoint /path/to/LFM3-Omni-2B --binary /path/to/rebuilt-liquid-lfm3 \
  --output /fresh/path/context-rerun --suite full \
  --reuse-reference /previous/path/context-full
```

The new `reference-manifest.json` is persisted **before native execution**. Reuse
loads no model, torch, or tokenizer, verifies every cached vector's SHA-256,
length and finiteness, verifies oracle/request consistency and source SHA-256,
then copies only oracle artifacts to the fresh output. All native checks rerun
against the new binary. Weight identity uses absolute path, size, and mtime_ns
(explicit local identity, not a cryptographic weight digest), so modifying or
moving weights invalidates normal reuse.

For old failed runs without a manifest, optional `--allow-legacy-reference`
explicitly trusts their original checkpoint provenance. It loads **only the CPU
processor/tokenizer**, reconstructs requests from the same deterministic content,
checks any saved teacher requests and saved greedy oracle, and imports all saved
reference vectors. Metadata is visibly marked `legacy-adopted`; original source
or weight identity cannot be proven retrospectively. Without this opt-in, a
manifest-less source is rejected. Neither mode reruns expensive oracle GPUs.

| Suite | Prefill + decode cases | Official model loads | Native model loads |
|---|---|---:|---:|
| smoke | 1+0, 1+1, 6+2, 255+2, 256+1, 257+1 | 1 | 8 |
| full | smoke, 1024+2, 4095+1, 4096+0 | 1 | 12 |

Smoke: 776 reference prefill tokens + 7 decode calls; native at most 2064
prefill tokens + 18 decode calls including reset and greedy checks. Full: 9991
reference prefill tokens + 10 decode calls; native at most 28684 prefill tokens
+ 26 decode calls. Early greedy stopping can reduce native counts. This is a
workload estimate, not an elapsed-time estimate. Native subprocess model loading
is repeated by the runner's existing one-request interface.

At 128000 vocabulary, full-suite raw retained reference + native logits occupy
about 9.7 MB (38 vectors of 256000 bytes). Teacher requests set the owner's new
`trace_logits_only: true`; no `/dev/null` symlink workaround remains. Missing
logits, symlinks, or any unexpected activation filenames fail the gate. Native
must include the new option to avoid intermediate activation host copies.
These runs are correctness checks, not throughput benchmarks.

## Rejections before model loading

Twelve invalid requests run against a deliberately **config-only directory with
no model.safetensors**. Each must return code 1, emit the relevant request/capacity
diagnostic (not a missing-weights error), and leave no output JSON:

- capacities 0 and 4097;
- prefill overflow, teacher decode overflow, generation overflow at capacity 1;
- 4097-token prefill at capacity 4096;
- decode or generation continuation after a 4096-token prefill.
- `warmup_runs=true` (must be a number) and `warmup_runs=11` (at most 10);
- `check_reset=1` and `trace_logits_only=1` (must be boolean).

Unlike running against a real checkpoint and checking only exit code, this
distinguishes request rejection from failures after attempting weight loading.
Configuration parsing before request validation is expected and permitted.

## Runtime review / remedies for the owning session

Reviewed baseline `2bd64e3` `native/src/lfm3_main.cu`; no runtime source was edited.

1. **Confirmed baseline validation-order gap:** `warmup_runs` and `check_reset`
   were parsed after `Runner` construction (baseline lines 360, 367–368, 381).
   Invalid values therefore caused weight/GPU allocation before rejection.
   Remedy: validate both before constructing `Runner`, as for capacity/count.
    The owner moved those checks before construction. All twelve config-only
    controls passed in `context-gates-v3`, including these type/order regressions.
2. **Tracing API limitation remedied by owner:** baseline `Runner::dump` lines
   202–208 had no logits-only mode. The owner now reports adding validated
   `trace_logits_only` and returning before host allocation/copy for non-logits.
    This gate requests that mode, rather than retaining its original disk sinks.
    The full GPU gate verified the logits-only output filenames.
3. **Numerical red-team hypothesis, not a confirmed bug:** the softmax kernel
   hardcodes a one-warp lane-strided reduction through length 4096. PyTorch's
   eager softmax may select a different reduction schedule at larger row widths;
   sliding dynamic caches can also expose a different logical key width than
   the native full-capacity cache. Existing length-257 evidence does not establish
   exactness at 1024/1025 or 4096. Source research below establishes that the
   CUDA forward reduction schedule changes at **2048/2049**, not 1024/1025.
   The full suite covers both persistent and register-kernel ranges.
    This investigation found a real sliding decode product-width discrepancy and
    CPU-vs-GPU RoPE frequency construction difference (observable at position850).
    After those fixes and the >2048 reduction path, `context-gates-v3` passed all
    nine cases with **bitwise BF16 logits**, including 4095+1 and 4096 prefill.
    The original failures remain preserved; no tolerance was increased.
4. **Capacity guard review:** `forward` checks `rows <= context-cursor` before
   embedding/scatter; preflight uses `input + max(count-1, teacher_count) <=
   capacity`, correctly treating the final generated ID as unconsumed. The
   `probabilities` local `elements[128]` fits exactly 4096/32 iterations at the
   advertised maximum. No off-by-one/OOB defect was confirmed by source review.
5. **Memory scope:** score and probability buffers alone occupy 1 GiB combined
   at 16 query heads and capacity 4096 (two BF16 16×4096×4096 arrays), independent
   of sliding attention. This is expected prototype quadratic storage, not a
   ring/sliding-capacity implementation. The suite exercises logical sliding
   masks, not eviction/ring-buffer behavior or chunked multi-token continuation
   (which the runner intentionally rejects).

## Host verification

```bash
python -m pytest -q tests/test_lfm3_context.py
```

Tests cover strict/tolerant BF16 comparison, nonfinite/missing/shape corruption,
signed-zero exactness, config-only rejection evidence, logits-only requests,
reset evidence, dependency-free help, failure case/step persistence, verified
reference reuse/corruption/provenance rejection, explicit legacy adoption, and
mocked end-to-end success/failure. Mocked results are not GPU numerical validation.

## PyTorch v2.10.0 CUDA softmax source derivation

Fetched upstream **v2.10.0**, not current main. This is source-derived guidance,
not GPU verification of a replacement kernel:

- [SoftMax.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/SoftMax.cpp)
  `softmax`: fused `half_to_float=true` is special-cased **only for FP16**, not
  BF16. `softmax(bf16_scores, dtype=float32)` materializes FP32 input and calls
  `_softmax(converted, dim, false)`.
- [SoftMax.cu](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu)
  `softmax_cuda_out`: CUDA calls `host_softmax<..., false, false>`; the
  `use_fast_softmax=true` variant is ROCm-only here. `inner_size==1` enters the
  regular row path. In the **!half_to_float** branch, persistent forward is
  selected when `dim_size <= 2048 && dim_size*sizeof(float) <= 8192`.
- [PersistentSoftmax.cuh](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/PersistentSoftmax.cuh)
  `dispatch_softmax_forward` explicitly supports 2048, despite a stale nearby
  comment saying 1024. Do not use the 1024 threshold in the half-to-float or
  backward branches for this BF16→FP32 forward operation.
- [block_reduce.cuh](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/block_reduce.cuh)
  `BlockReduce`, `WarpReduce` establish the two-stage shuffle-down tree below.

### Width 1025–2048: persistent path

`next_power_of_two=2048`, `WARP_SIZE=32`, `WARP_ITERATIONS=64`, `WARP_BATCH=1`.
Launch shape is `(32,4,1)` (128 threads, four independent rows per block), not
1024 threads. Lane `l` handles `l+32*i` for `i=0..63`; indices outside logical
width load `-infinity`. Each lane scans its entries in increasing `i`, then max
uses XOR shuffle offsets **16,8,4,2,1**. For sum, start at FP32 zero, compute
`std::exp(element-max)` and add sequentially in increasing `i`, then use the same
XOR tree. Output is the stored exponential **divided by the sum**. The native
one-warp-per-row arrangement can reproduce the per-row arithmetic; four-row
packing is not necessary for numerical equivalence.

### Width 2049–4096: register path, no float4 vectorization

`SoftMaxForward_getBlockSize(dim_size)=1024`. One block handles one row;
`potential_register_count=ceil(dim_size/1024)`, so dispatch is:

| Logical row width | Kernel specialization | Per-thread scalar indices |
|---|---|---|
| 2049–3072 | `cunn_SoftMaxForwardReg<..., reg_cnt=3>` | `t`, `t+1024`, `t+2048` if valid |
| 3073–4096 | `cunn_SoftMaxForwardReg<..., reg_cnt=4>` | previous plus `t+3072` if valid |

`potential_reg_cnt < 10` takes precedence over `can_use_smem`; **neither alignment
nor width divisible by four selects the ILP/float4 path in this range**. Register
loads are scalar and block-strided, not adjacent groups of four per thread.
Dynamic shared memory for the reduction is `1024/32*sizeof(float) = 128` bytes.

For both max and sum, reduce exactly as follows:

1. Each thread starts max at `-FLT_MAX`, and scans valid scalar registers in
   increasing register index. Invalid indices do not contribute.
2. Within each warp, perform `op(value, shfl_down(value, offset))` for offsets
   **16,8,4,2,1**, in that order. This is **not XOR**, and is not a shared-memory
   binary halving of 1024 thread values.
3. Synchronize; each lane 0 stores its warp aggregate to `shared[warp_id]`.
   Synchronize; threads 0..31 load the 32 shared warp aggregates in increasing
   warp-ID order. Warp 0 applies the same shuffle-down tree. Thread 0 stores
   result in `shared[0]`; synchronize and broadcast `shared[0]` to the block.
4. After finding max, each thread starts FP32 sum at zero and accumulates
   `sum = sum + std::exp(reg[i]-max)` sequentially for valid `i=0,1,2[,3]`.
   Apply the same two-stage shuffle-down reduction to these partial sums.
5. For each valid register, recompute `std::exp(reg[i]-max) / sumAll` as FP32,
   materialize FP32 probability, then cast to BF16 for the attention PV GEMM.

Use ordinary `expf`/CUDA's float `std::exp` behavior, **not `__expf`**, and ordinary
division as the source does, **not reciprocal times exponential**. Do not enable
`--use_fast_math` or reassociate the sums when reproducing this schedule.

Suggested owner's primitive GPU checks: 1024, 1025, 2048, 2049, 3072, 3073,
4095, 4096, with BF16-rounded scores promoted to FP32, realistic causal/window
masks and cancellation-sensitive exponent sums. Full-attention width is the
whole logical key row; a physically cropped sliding decode row remains width
256 and must stay on its persistent path. The reported sliding-cache fix should
therefore be kept separate from the >2048 register schedule change. No tolerance
relaxation is justified by either mismatch.
