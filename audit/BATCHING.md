# Opt-in two-lane decode prototype — standalone only

`native/src/lfm3_decode_batch.cuh` must be included after `lfm3_main.cu` with
`LFM3_RUNTIME_ONLY`. It supplies `lfm3_decode_batch::forward(vector<Runner*>,
vector<int>)`, returning one full float-logit vector per lane. Exactly one token
per owner, one or two owners, is supported. One lane explicitly falls back to
`Runner::forward`; two lanes execute actual two-row dense GEMMs, not two serial
forwards. This is a standalone experiment, **not worker batching acceptance**.

## Execution and isolation contract

The caller externally serializes all operations with one CUDA owner/default
stream. Distinct runners must share the identical Safetensors object and model
geometry/layer specifications; capacities and positive cursors may differ.
Tracing is rejected to avoid producing misleading serial golden files. All
metadata validation precedes device mutation. Neither lane may already be at
capacity. Lane A must have room for two activation rows (`context>=2`).

A supplies two-row activation scratch and its cuBLAS handle/pointer arrays.
Convolution in/out projections, Q/K/V/gate/output projections, MLP projections,
and tied output logits use actual rows=2 GEMMs sharing weights. Embedding,
normalization, rotary, head gating, SwiGLU, and residual work cover two rows.
Hidden normalization explicitly uses the original one-row reduction schedule;
Q/K head normalization uses the original per-lane head-count schedule rather
than deriving threads from twice the head count. Rotary uses each owner's cursor.

Convolution dispatch remains one kernel per lane with that owner's seven-value
state. KV scatter uses each owner's context stride and cursor. Attention remains
one head-pointer GEMM batch per lane, using its exact serial end/key_start/window
with no padding. Dedicated score/probability buffers each hold
`max(capacityA,capacityB)*heads` elements; neither borrows smaller lane A's serial
attention scratch. Dedicated logits hold `2*vocabulary` elements, and both output
rows are copied, checked finite, and returned. This is **partial operator
batching**, not fused ragged attention or a serving scheduler. Scratch allocation
on every call is intentionally unoptimized.

Cursors/steps publish only after successful logits device-to-host copy and
finite-value validation. An execution error can leave physical state partially
modified: the caller must discard **both** participating caches/runners, even if
their host cursors did not advance. There is no rollback or GPU fault-isolation
claim. Metadata rejection itself performs no device work; it does not require
discard. Do not call concurrently with serial forwards, resets, or other batches.

## Standalone GPU probe

`native/probes/lfm3_decode_batch_probe.cu` takes:

```text
decode-batch-probe --experimental MODEL_DIRECTORY OUTPUT_DIR [scratch|sliding|softmax|tiled|upper] [normal|compact]
```

The optional CMake target `lfm3-decode-batch-probe` is excluded from the default
build. Enable `LIQUID_LFM3_OPS_TESTS` and explicitly build this target; it uses the
same cuBLAS/CUDA and `--fmad=false` policy as the single runner. Worker and client
integration are intentionally absent.

The output directory must not exist. Failures retain `checks.jsonl` and
`summary.json`; no overwrite of an existing evidence directory is permitted.
By default, four runners share one weight upload **in this test only**: two
batched owners and two independent serial reference owners. Omitted fixture means
`scratch`; omitted reference strategy means `normal`. To select compact mode,
provide both final arguments, for example `upper compact`. `reference_mode`
(`normal` or `compact`) and GPU runner-owner count are explicitly recorded in evidence. Fixture schedules,
counts and exact assertions below are identical for both strategies.

| Fixture | Capacities A/B | Prompt lengths A/B | Expected comparisons / paired calls / retired checks |
| --- | --- | --- | --- |
| scratch (default) | 12 / 300 | 3 / 257 | 59 / 10 / 35 |
| sliding | 270 / 280 | 254 / 256 | 47 / 17 / 9 |
| softmax | 2062 / 2074 | 2046 / 2048 | 49 / 17 / 11 |
| tiled | 4112 / 4124 | 4096 / 4098 | 49 / 17 / 11 |
| upper | 2 / 32766 | 1 / 32764 | 5 / 1 / 1 |

All fixtures require11 metadata rejections; non-upper fixtures also require2
reset cache checks. Configured and observed counts are recorded and must match.
Default B's attention length exceeds A's serial scratch144 elements per head;
all fixtures require nonzero sliding key_start. Softmax explicitly requires both
<=2048 and >2048 attention products; tiled requires both prefills exceed their
query tiles. Upper keeps the two participating capacities' sum exactly32768
(the test-only serial reference allocations are additional).

Each reference independently chooses raw argmax inputs, which are fed identically
to the matching batch lane; EOS does not terminate this low-level decode probe.
Every initial prefill and decode compares the complete vocabulary float vector
**bit-for-bit** (per-float `memcmp`, distinguishing positive/negative zero),
argmax ID, cursor, step, and **every byte of the full physical KV/convolution
cache**, including future/unused KV slots, against its serial reference. There
is no tolerance fallback. Each single-lane call snapshots the retired owner
before/after, checks byte equality against its serial reference, and checks its
counters. Non-upper fixtures first retire B during an A-only call, resume B with
reversed batch ordering, then reset and replay both prefills. This independent
phase avoids overfilling default A's nine remaining positions.

The standard phase executes six paired steps, retires A for two B-only calls,
then resumes both for three paired steps. Further pairs fill A to capacity;
reject A-at-capacity, fill B via single-lane fallback, reject B-at-capacity.
Upper explicitly substitutes one paired call, A-capacity rejection, one B-only
call and B-capacity rejection, rather than claiming six-step or reversed-lane
coverage it cannot execute. Successful/failed case progress is flushed to JSONL.

Invalid duplicate/null owners, missing weights, distinct non-null weight identity,
geometry, tracing, token, zero cursor, and capacity requests must reject without
metadata advancement or any bit change in either owner's complete cache. The
identity test uses a temporary one-past aliasing shared_ptr, retaining the single
checkpoint upload; its pointer must be rejected before any dereference.

Changing dense GEMM rows from1 to2 may select a different cuBLAS algorithm and
change BF16 outputs. The probe must fail if that happens: retain evidence and
investigate the algorithm/schedule, **do not weaken exact assertions**. This
initial tiny fixture passed exact logit/ID/cursor comparisons in
`validation/decode-batch-v1`, but did not establish full physical-cache parity or
the larger scratch/window boundaries. The strengthened probe subsequently **passed
all five fixtures** on Thor: `validation/decode-batch-v2/`. It requires209 exact
full-vocabulary/full-cache comparisons,62 actual paired GPU calls,55 pre-mutation
rejections and67 retired-lane checks. Binary `465d8501...` and source hashes are
recorded in `verification.json`. Those earlier logit comparisons used numeric
float equality and therefore do **not** establish signed-zero bitwise parity.
The current per-float byte comparison strengthens that gate. Both normal and
compact modes now pass new all-five-fixture GPU runs; see the final evidence below.
No worker integration, serving-concurrency,
cancellation, throughput, sanitizer, or soak acceptance is inferred. No host fake
is presented as GPU proof.

Compile-only validation succeeded using `/usr/local/cuda/bin/nvcc`, `g++-12`,
CUDA C++17, native includes, and `-arch=sm_80`. Only compile checks were performed
in that initial revision. The final Thor build targets `sm_110`; numerical and
full-cache GPU parity above are separate actual execution evidence. Standalone
memcheck coverage is being checked separately rather than inferred from parity.

## Optional compact reference strategy — test-only

`compact` allocates only the two actual GPU runners (and the same single shared
weight upload). Each serial oracle's complete physical KV/convolution cache,
cursor and step live in CPU snapshots. Before each actual batch/fallback call:

1. Capture the actual owner's entire cache and counters, including unused/future
   KV slots. Restore that lane's previous CPU oracle state into the same runner.
2. Execute the **unmodified** `Runner::forward` with the serial oracle's input;
   capture its full post-cache/counters and returned logits as reference results.
3. Restore the saved actual cache/counters, synchronize, and verify every cache
   byte and both counters match the saved actual state. Only then publish the new
   CPU oracle state. Repeat independently for the other lane where applicable.
4. Run the actual batch/fallback and compare full logits, IDs, counters and every
   physical cache byte against the oracle post-state **before any next step**.
   Retired owners still have before/after and oracle-reference cache checks.

Prefill uses the same save/run/capture/restore flow from the initial zero-cache
state. Reset is explicit in both modes: compact temporarily restores the oracle
state, calls the existing `Runner::reset`, captures the reset oracle state, then
restores the saved actual state. The normal path retains independent runners and
its original operation order. Metadata-rejection cases never call an oracle or
forward; they still snapshot both complete actual caches before/after rejection.
All eleven rejection, retirement, reversal, reset, capacity and count assertions
remain enabled; there is no tolerance change or output/GEMM zero-fill workaround.

Only legitimate KV/convolution cache and cursor/step are restored. Activation,
GEMM-output, pointer-array and cuBLAS workspace contents are deliberately **not**
initialized or restored. Compact oracles consequently share scratch/handles with
actual execution and are not as independent as the default reference strategy;
normal remains the primary independent-reference gate. Any failed oracle forward,
snapshot or restoration aborts the probe and preserves failure evidence; no
exception-time rollback or continued GPU/cache usability is promised.

Compact uses extra CPU snapshot storage and full-cache host/device transfers,
including transient actual/post/reference copies. Host allocations can still
pressure shared/UMA memory, and instrumentation can be slower. It does **not**
establish the root cause of the previous upper memcheck600-second timeout, nor
guarantee avoiding it. Scratch memcheck success does not imply upper acceptance;
the timed-out instrumented run and overlapped SDPA follow-up are not acceptance
evidence. Process-group/descendant cleanup is the main owner's wrapper concern,
not implemented by this probe.

## Final bitwise and instrumentation evidence

Both modes pass all five fixtures on binary `db30e878...`:
`validation/decode-batch-v3-normal/` and `validation/decode-batch-v4-compact/`.
Each mode retains209 full-logit-bit/full-cache comparisons,62 true paired calls,
55 rejections and67 retired checks. Normal uses four GPU runners; compact uses
two. Their actual records are postvalidated against strict mode, owner count,
geometry, coverage, bit-comparison and non-integration metadata. This establishes
the specified finite standalone arithmetic/cache behavior, not serving batching.

The first four-runner upper memcheck timed out600 seconds; a separate compact
instrumented diagnostic timed out120 seconds, both **not passes**. The latter
printed100 launches, ending with a cuBLAS kernel in serial oracle prefill. This
does not locate the actual timeout: the default print limit may truncate the
launch ledger. The runner now requests unlimited printing explicitly. Peak
aggregate RSS was about6.05GiB with minimum observed
host MemAvailable92.73GiB; this does not establish a GPU-memory root cause. See
`validation/decode-batch-memcheck-v2-compact/`. All same-group children were
stopped before proceeding. Scratch memcheck on the earlier binary was clean,
but no final upper-bound memcheck certificate is claimed.

The final hardened runner's scratch numerical fixture also passes in
`validation/decode-batch-final-runner-v5/` in5.14 seconds with complete group
cleanup. This is numerical/lifecycle verification, not a memcheck run.

The fail-fast fixture runner is `tools/check_lfm3_decode_batch.py`; process,
signal, timeout and evidence-validator remediation is documented in
`OWNED_GPU_JOBS.md`. Production worker/service batch integration, per-lane batch
cancellation, batched throughput/resource/soak and complete sanitizer gates remain
open. Merely two sequential worker requests is never represented as GPU batching.
