# Native grouped worker decode

## Scope and protocol

`native/src/lfm3_worker.cu` supports protocol-2 `generate_batch` with exactly
two distinct existing sessions. The outer object contains only `op`, `rpc_id`,
and `lanes`; each lane contains the existing generation metadata, but no
`op`/`rpc_id`. Both lanes' schema, session existence, budgets, sampling options,
and trace options are checked before any lease/cache/weight mutation.

Ready advertises `decode_batching:true` separately from idle
`gpu_batching:false`. Existing single-generation execution and integer trace
references remain unchanged. This is an explicit group, not opportunistic
request scheduling or independent full-serving acceptance.

Prefills and prompt reuse continuations execute serially. Generation advances
through the real `lfm3_decode_batch::forward` only when both lanes have a pending
nonfinal token; otherwise it uses the remaining `Runner::forward`. Final output
tokens, including EOS, are never forwarded. Each lane owns its sampler/RNG,
lease, physical cache, consumed history, and stopping decision. Shared model
weights and the existing session ledger are reused. Aggregate admission stays
32768 tokens/two sessions.

The reply includes ordered `results`, outer/lane `paired_decode_calls`, and
`gpu_batching` based on actual participation, not merely requested grouping.
Lane results include cancellation/cache validity, session ID, generated IDs,
reuse/consumed counts, sampling settings and optional last logits. A successful
lane commits once and is excluded from all subsequent cancellation/discard
operations, including peer execution failures.

## Cancellation and failure boundaries

One borrowed nonblocking pipe has one `CancellationReader`, one bounded drain,
and one partial-frame buffer across both single and grouped RPCs. Group frames
are little-endian u64 `((lane + 1) << 32) | outer_rpc_id`, tags 1/2, outer IDs
1..INT_MAX. Valid foreign tagged frames are ignored by single polling; invalid
tags, zero/out-of-range outer IDs, EOF, descriptor changes and read errors fail
closed. Poll generations cannot decrease or change single/pair mode in place.
Matching lane latches persist independently; complete and partially received
future frames are not queued to cancel a subsequent request. Drain byte and
attempt budgets are retained, including interrupted and partial reads.

Host polls occur before and after each serial prefill/continuation, after
initialization/reset and before prefill, around generation decode, and before
each commit. Cancellation can discard one lane while the peer continues.
Discard destroys only that lane's Runner, clears physical consumed/logit state,
and aborts its lease, preserving prior committed transcript/consumed history as
invalid-cache history. A completion/cancellation race may complete normally
after the final boundary. CUDA work already launched is not interrupted.

Native execution errors discard **all still-inflight lanes**. Already committed
lanes are not changed. This is not a CUDA context-fault isolation guarantee.

## Trace and resource accounting

Grouped sampling traces live under the startup audit root at
`<rpc_id>/lane0|lane1/<step>.logits.bf16`. Replies return only safe generated
relative strings (`"<rpc_id>/lane0"`), never request-provided filesystem paths.
Single traces keep their previous integer RPC reference.

`stats` calls `cudaDeviceSynchronize` and `cudaMemGetInfo` only after weights
have been loaded. It then emits unsigned byte counts `cuda_free_bytes`,
`cuda_total_bytes`, and `device_allocated_bytes` (= total - free). These are
**device-wide CUDA/unified-memory accounting**, not per-worker allocation,
resident physical memory, or host RSS. No memory threshold/policy is asserted.
Unloaded stats do not initialize a CUDA context and omit these byte fields.

## Local evidence (no GPU execution)

Executed:

```text
python -m pytest -q tests/test_text_batch_cancellation.py tests/test_text_cancellation.py tests/test_text_sessions.py tests/test_text_serving_limits.py
71 passed

/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_80 -ccbin g++-12 -I native/include -c native/src/lfm3_worker.cu -o /tmp/user/1000/opencode/lfm3-worker-batch.o
compile-only succeeded
```

New tests compile the pipe reader and the **actual worker scheduler source**
with fake host Runners, in strict C++17 and UBSan configurations. They cover
both tag orders, duplicate/stale/future/partial frames, malformed frames,
disabled reader, bounded drains and backpressure; paired/ragged decode, EOS,
reuse, trace layout, cancellation before prefill/while a peer initializes,
preserved committed history, both-lane cancellation, fail-closed execution,
and immutability of a completed lane after peer failure. Metadata/schema,
duplicate-session and capacity counterexamples assert zero forwards, no weight
load and unchanged session/lease state. Fake logits/trace storage are not
numerical CUDA evidence.

The system `/usr/bin/nvcc` first failed with default architecture BF16 support,
then with its GCC 11 `std::function` compatibility issue; the explicit CUDA
13/GCC 12 compile above succeeds. No GPU, SSH, commits or subagents were used.

Pending parent-session GPU verification: grouped-vs-serial exact logits and
sampling traces, ragged/EOS/reuse/cancellation integration, extreme capacities
3/32765, sanitizer checks, and device-memory/resource acceptance. Existing
standalone batch-kernel evidence and single-path evidence are not replaced or
upgraded by this host-only scheduler validation.
