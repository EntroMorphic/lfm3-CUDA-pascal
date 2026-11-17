# Session ledger and experimental resident worker

Current internal protocol is **2**: positive monotonically increasing `rpc_id`
is mandatory on every request and echoed on every response; clients fail closed
on mismatches and surplus frames. Rebuild clients and workers together. The
strengthened sampling/32K gate is `validation/worker-redteam-v4/` (worker
`1f3401db...`), including independent per-step sampled-logit replay with a verified
non-greedy choice. Historical evidence below retains its original binary scope.

`native/include/lfm/text_sessions.hpp` is header-only C++17 bookkeeping with
no CUDA dependency. It neither allocates GPU caches nor changes any runner,
protocol, frontend, or model execution. Persistent resident sessions are **not
implemented** by the header alone. The worker below uses the ledger and has
passed its finite GPU cache-isolation gates; full-serving acceptance remains open.

## Integration contract

1. Construct `lfm::text::SessionLedger(max_sessions, token_budget)`. Create each
   opaque session with a fixed token capacity in `[1, 32768]`. IDs are case-sensitive
   ASCII `[A-Za-z0-9_-]`, 1–64 bytes, never paths. Creation reserves full capacity;
   rejection never evicts another session. Reset retains the reservation; close
   releases it. Convert actual per-token KV bytes and other GPU overhead into a
   suitable token budget externally; this is not a byte-accurate GPU allocator.
2. `begin(id, full_prompt_tokens, requested_output)` validates the **sum**, without
   subtracting the final predicted token. Empty prompts and negative token IDs are
   rejected; vocabulary bounds belong to the caller. Only one lease per session
   may be active. Independent sessions may have interleaved leases.
3. Read `lease.plan()`. A full-prefill plan requires resetting that session's device
   cache before forwarding the complete prompt. Otherwise retain exactly
   `reuse_tokens` and forward `prompt[reuse_tokens:]`. Reuse requires the entire
   committed consumed prefix to match, with at least one new prompt token. Equal
   or shorter prompts and mismatches fall back to full prefill: there is no cache
   truncation or saved-logits assumption. Plans never imply cross-session reuse.
4. After successful model execution, `commit(generated, consumed)` publishes
   prompt + generated as the transcript. For nonempty generated output, consumed
   must equal that transcript **minus its final predicted token**, including when
   the final prediction is EOS. Earlier generated tokens, including any EOS the
   caller elects to continue past, must be consumed. Early EOS may produce fewer
   tokens than requested. A nonzero request cannot commit empty output; cancel it.
   A zero-output request commits an empty generated vector and the entire prompt
   as consumed. The caller must honestly report device work: this host ledger
   cannot verify GPU state, tokenizer/model identity, or EOS stop policy.
5. Failed admission/validation and invalid commits preserve committed history.
   Invalid commits leave the lease active for correction or abort. `abort()` or
   lease destruction retains the previous transcript/consumed history but marks
   its cache invalid, even if execution had not started. To restore, begin with
   the saved transcript and recompute; do not reuse invalid device state.

Reset/close reject inflight sessions. Leases are move-only and cancel on scope
exit or overwrite; replay after commit/abort/move fails. Internal shared ownership
keeps leases safe even if the ledger facade is destroyed. Snapshots are copies;
the plan reference is valid only for its lease's lifetime and until it is moved.
No raw session references escape. Tickets do not wrap; exhaustion rejects begin.

**External serialization is required**, including lease destruction. There are no
mutexes and no thread-safety claim. The eventual worker must isolate actual KV
allocations by session, obey plans and cancellation, and invalidate/reset cache
after device failures. Any model/tokenizer/cache-layout change requires resetting
affected sessions. These integration responsibilities are not tested GPU features.

## Host validation

`python -m pytest -q tests/test_text_sessions.py` compiles with C++17 and strict
warnings, then covers A/B interleaving, prefix fallbacks, EOS consumption,
capacity/quota rejection, ID validation, 32K boundaries, invalid/replayed commits,
RAII cancellation, inflight reset/close rejection, and facade/lease lifetimes.

## Resident worker protocol increment

`liquid-lfm3-worker --experimental MODEL_DIRECTORY` is a bounded JSON-lines
protocol on stdin/stdout, not a network listener. It is single-threaded and does
not implement GPU batching or concurrent cancellation. Commands:

- `create`: `session_id`, `capacity`; at most2 sessions, aggregate reserved token
  capacity32768. Creation is host-only; weights/caches are loaded lazily.
- `generate`: full `input_ids`, positive `max_new_tokens`, optional validated
  sampling parameters. Reuse is planned from that session's consumed token prefix;
  prefix mismatch/equality falls back to fresh prefill. Last predictions are not
  incorrectly treated as already forwarded. Debug `return_logits` exposes final
  raw vocabulary logits for correctness checks only.
- `inspect`, `reset`, `close`: explicit per-session lifecycle; `stats` reports
  admission and whether weights have been loaded. `shutdown` exits cleanly.

The runtime shares one read-only checkpoint upload, but every session owns its
activation, convolution and KV allocations. Admission is bounded by configured
capacity, not an asserted exact GPU-byte quota. Sampling RNG belongs to each
generation request. Metadata is validated before acquiring a mutation lease;
execution failure destroys that session's runtime and invalidates its ledger
cache while preserving prior committed transcript bookkeeping. CUDA-context
poisoning/restart semantics still require process-level serving acceptance.

Frames are limited to4MiB, IDs to safe ASCII64 bytes, token/context fields have
strict bounds, unknown request fields and modality placeholder inputs are rejected.
This is an internal experimental worker; external authentication, socket protocol,
admission queues, asynchronous cancellation, actual GPU batching and production
fault recovery remain open gates.

The worker reuses the single runtime translation unit under `LFM3_RUNTIME_ONLY`,
not a duplicated kernel implementation. Shared-weight ownership was added to the
runtime constructor; the rebuilt single-request executable passes its6890 exact
short-context regressions. Long-context hash-matched revalidation is recorded
separately rather than inferred from earlier candidate results.

`tools/check_lfm3_worker.py --preflight-only` passed on Thor with CUDA disabled:
quota, duplicate/invalid IDs, malformed sampling, overflow, unknown fields and
placeholder requests reject before loading weights. Host control-pipe tests
cover malformed/oversized/timed-out handshakes, write deadlines and process cleanup.

The strengthened full GPU gate subsequently **passed** against independent
single-runner oracles: raw float logits, generated IDs, consumed/transcript counts,
and final-prediction accounting match across A/B interleaving, multi-token
generation followed by prefix continuation, B resumption after A advances/resets,
invalid metadata, prefix mismatch, reset, early EOS and close. The worker binary
is `3bc90ece...`; the single-runner oracle is `faf3d2df...`. Their hashes were
captured before execution and rechecked afterward. Evidence:
`validation/worker-gates-v2/`. All6890 exact short-context checks also pass for
the shared-weight single runner in `validation/forward-resident-shared-v1/`.

### Independent review remediations

1. An earlier one-token-only gate could pass wrong/empty generated IDs and never
   exercise the generated-token forward loop. It now checks exact emitted IDs,
   ledger lengths, multi-token continuation, early stopping and B resumption.
   Seven host counterexample tests reject each broken behavior and wrong-size
   oracle logits, preserving failed reports rather than losing partial evidence.
2. A response timeout did not cover blocking pipe writes. Reads and nonblocking
   writes now share one deadline; timeout kills/reaps the process. A fake worker
   that handshakes but never reads a1MiB request is rejected promptly, and cleanup
   is idempotent. These are validator/client-transport tests, not network serving.
3. Failed `Runner` construction after cuBLAS handle acquisition leaked that
   handle because its destructor never ran. `finish_handle_setup` now destroys
   and clears it on workspace/math/reset setup failure. A host UBSan test injects
   failure at all three setup stages and verifies exactly one cleanup, success
   destruction, original exception preservation and handle clearing. Real CUDA
   setup-failure injection is not claimed by this host test.

The GPU gate establishes these finite resident-cache cases, not production
concurrency, kernel batching, asynchronous cancellation, fault/soak acceptance or
complete sanitizer coverage. Those remain explicit release gates.

The subsequent `validation/worker-gates-v3-32k/` gate also passes resident32767
prompt +1 prediction, exact single-runner logits/ID/counts, and non-mutating
overflow rejection when another output would exceed the combined32768 budget.
Worker/single hashes remain `3bc90ece...` / `faf3d2df...`. The latter also repeats
the independent SDPA numerical gate in `validation/context32k-resident-shared-v1/`.
