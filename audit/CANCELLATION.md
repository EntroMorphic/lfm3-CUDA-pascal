# Cooperative token-boundary cancellation

`native/include/lfm/text_cancellation.hpp` is Linux host-only C++17 bookkeeping
and nonblocking pipe I/O. **The header alone does not implement cancellation in
the worker, client, HTTP server, or CUDA runtime.** No GPU preemption, asynchronous
execution, cancellation latency, or cross-session GPU fault isolation is claimed
by the header. The integrated worker/client/HTTP increment is verified below.

## Interface and framing

`lfm::text::CancellationReader reader(fd)` borrows a readable, nonblocking pipe
descriptor >=3. It validates `F_GETFL`, access mode, `O_NONBLOCK`, and FIFO/pipe
type. `-1` disables reading. Other negative descriptors, standard descriptors,
blocking/write-only descriptors, non-pipes, and closed descriptors reject.
The reader never closes the descriptor or changes its flags; its lifetime and
all calls require one externally serialized owner. Do not close, replace, or
alter a borrowed descriptor concurrently. Copying/moving the reader is disabled.

The parent writes **exactly 8 little-endian bytes per frame**, encoding a
positive protocol-2 RPC ID <=`INT_MAX`. It must set its write end nonblocking and
write each complete frame in one syscall; 8 bytes are <=`PIPE_BUF`. On `EAGAIN`,
retain/retry the whole frame within a bounded parent-side policy, or report that
cancellation was not delivered. Never claim cancellation merely because a write
was attempted. Parent shutdown/broken-pipe handling is not supplied here.

`bool requested(uint64_t rpc_id)` accepts positive, nondecreasing RPC IDs, never
reused for different work. Disabled readers return false for valid IDs. A complete
matching frame latches true for repeated polls of that RPC. Advancing to another
RPC clears the latch. Complete stale **and future** targets are discarded, never
saved for later execution. Partial frames survive polls and RPC transitions;
their decoded target is compared with the RPC being polled when completed.
No flag from another RPC can cancel the current RPC. Do not poll an already
completed RPC again: the owner must advance its ID with the next operation.

Each poll reads at most8192 bytes and attempts at most1024 reads, including
interruptions. It returns at `EAGAIN`/`EWOULDBLOCK`, preserves partial frames, and
retries `EINTR` only within the attempt budget. A continuously writing parent
cannot make one poll drain indefinitely. A match behind the budget may require
later polls; returning false does not prove there is no matching buffered frame.

Zero/out-of-range decoded IDs throw `std::system_error(EPROTO)`, even for foreign
targets. EOF throws `std::system_error(EPIPE)`, whether clean, mid-frame, or after
a matching frame. Other I/O errors throw their errno. These failures are sticky:
subsequent polls throw rather than silently resuming with untrusted control state.
Descriptor/flag validation repeats before each poll; a changed blocking flag is
an `EINVAL` channel failure. Invalid/decreasing poll arguments instead throw
`std::invalid_argument`. Constructor errors reject setup before any device work.

## Integrated ownership contract

1. Parent creates a dedicated cancellation pipe, marks both ends nonblocking,
   explicitly passes only the read descriptor to the worker, and closes unused
   ends. Keep generation protocol IDs unique/monotonic and reject ID exhaustion.
   JSON RPC framing/response correlation remains separate from this byte pipe.
2. The existing single CUDA owner polls before prefill/execution, after every
   completed `forward()`, and before further sampling/forwarding or final commit.
   No reader thread mutates GPU buffers, session maps, or ledger leases.
3. On a matching cancellation, owner aborts the active lease and discards that
   session's physical runner/cache. Preserve its last committed transcript and
   consumed history, invalidate reuse, and leave other session caches untouched.
   The helper performs none of these actions. Control-channel failure must enter
   explicit fail-closed worker/service handling, not become a false poll result.
4. Distinguish parent cancellation delivery/request acknowledgement from the
   owner's terminal cancelled result. Define completion-versus-cancel races and
   make the owner the sole authority publishing either terminal outcome. A late
   frame for a completed RPC must not affect the next request.
5. Current synchronous CUDA `forward()` cannot observe cancellation during its
   kernels or full prefill. Checks occur at host/token boundaries; no bounded
   cancellation latency or kernel interruption follows from this design. Chunked
   prefill would be a separate runtime/numerical-validation change.

## Host validation

`python -m pytest -q tests/test_text_cancellation.py` compiles strict C++17 and
UBSan variants when g++ is available. Tests cover disabled/invalid descriptors,
borrowed lifetime, changed/closed descriptors, little-endian and all partial-frame
splits, stale/future/current/duplicate IDs, sticky malformed/EOF failures, pipe
backpressure and exact drain-budget accounting.
Deterministic wrapped-read cases also exercise an endless byte source, endless
`EINTR`, and one-byte reads without relying on the host pipe capacity.
The simulated owner uses the actual
session ledger to verify cancellation before execution, preservation of committed
history and B's cache metadata, and rejection of late cancellation against the
next RPC. This is not a worker integration, real cache-discard, or GPU test.

The initial host run passed18 cancellation cases (9 each in strict and UBSan
builds), plus the6 existing session-ledger cases. No GPU validation was run.

## Worker, transport and HTTP integration

`WorkerClient(..., cancellation=True)` creates the dedicated pipe, passes its read
FD explicitly through the signal-mask-restoring exec launcher, and validates the
worker capability. The CLI enables this; other callers may leave it disabled.
Only the CUDA owner touches runners and leases. It polls before initialization,
again after initialization/reset immediately before prefill, between complete
forwards, before sampling, and before committing the result. A match discards only
that runner and aborts its mutation lease. `inspect.consumed_tokens` now measures
the physical cache; `committed_consumed_tokens` and `transcript_tokens` retain
the last committed history even after physical cache destruction.

Authenticated `POST /sessions/ID/cancel {}` returns202 **request acknowledgment**
for a leased generation. Idle/lifecycle work returns409; a client without the
capability returns501. Queue/pre-dispatch cancellation is handled on the owner by
resetting that session. A running RPC receives its targeted ID through the pipe.
The original generate response eventually returns `cancelled:true`,
`finish_reason:"cancelled"`, exact partial IDs/text and `cache_valid:false`, or
normal completion if completion won the race. Control-pipe backpressure/failure
returns503 rather than falsely acknowledging delivery. Cancellation is not a
generic HTTP timeout, kernel preemption, initialization interruption or bounded
latency promise. A slow full prefill still has to finish before the next poll.

### Adversarial remediation

Independent review found a SIGINT window after closing an inherited read FD but
before retiring ownership. Cleanup could double-close it and skip child teardown.
Descriptor retirement and complete cleanup now mask interruption, serialize
concurrent cleanup and always terminate the child despite FD-close errors.
A real SIGINT injection and live-pipe backpressure counterexample originally
failed: `validation/redteam-cancellation-before.log`. Both now pass. A UBSan fake
Runner test injects cancellation during construction and requires zero forwards
after the final prefill-boundary check.

The first native gate failed because inspect conflated physical and retained
committed counts. That failure is preserved in `validation/native-cancellation-v1/`.
The corrected gate requires **both** zero physical count and unchanged committed
history; it was not relaxed to discard historical accounting.

### Evidence

- Full host suite:667 passes, `validation/host-cancellation-remediated-v1.log`.
- Native gate: `validation/native-cancellation-v3/` observes a live completed decode
  trace before cancellation, requires exact partial IDs, checks abort/history/B
  preservation, B's exact continuation, fresh A recovery, pre-dispatch cancellation
  and stale-ID rejection.
- Actual authenticated HTTP gate: `validation/http-cancellation-v1/` verifies
  request202 versus terminal200, auth rejection, physical/history metadata,
  peer preservation, idle409 and exact fresh recovery. Partial text is copied to
  `../tmp/http-cancellation-v1/partial-response.txt`.
- Native/HTTP gates use worker `842a46f1...`. The HTTP/admission regression also
  passes in `validation/http-cancellation-regression-v5/`. All6890 batch-one exact
  comparisons remain intact for single runner `ad757f34...` in
  `validation/forward-sampler-fastpath-v2/`.
- Final resident32K and independent stochastic replay also pass for the same
  worker in `validation/worker-cancellation-fastpath-v6/`.

These establish finite token-boundary cancellation behavior, not universal fault
isolation or preemption. True paired decode remains a separate unintegrated GPU
prototype; GPU-memory/resource/long-soak acceptance and sanitizer gaps remain open.
