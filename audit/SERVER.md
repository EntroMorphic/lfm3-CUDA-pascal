# Local text server (experimental)

Latest cancellation acceptance:667 host tests plus native and authenticated HTTP
token-boundary cancellation gates pass for worker `842a46f1...`; see
`CANCELLATION.md` and `validation/cancellation-verification.json`. Physical cache
discard and retained committed history are reported separately. Earlier scoped
verification below is historical, not a claim of kernel/prefill preemption.

## Additional signal-mask remediation

Independent verification of the startup fix found that blocking SIGTERM/SIGINT
around process creation also blocked them in the exec'd child (observed child
mask `0x4002`). `restore_signals=True` does not restore that mask. The transport
now uses a small Python exec-only launcher to restore the original mask before
executing the worker, without a thread-unsafe `preexec_fn`. A host regression
checks exact child/parent masks and requires ordinary SIGTERM exit within one
second, independently of forced-cleanup fallback. This closes the signal-mask
defect; native HTTP evidence is recorded separately below when available.

`tools/lfm3_server.py` adds an authenticated, IPv4-loopback-only HTTP wrapper
around `liquid-lfm3-worker`. It supports **explicit fixed two-lane decode groups**,
not automatic/dynamic microbatching, a public-facing server,
or evidence that the native GPU correctness gates passed. Every inference and
lifecycle RPC runs through one serialized owner thread. Native weights are shared;
each of at most two sessions has its own native KV/cache allocation. Aggregate
reserved capacity is at most 32768 tokens, not 32768 per concurrent session.

### Explicit-group host verification

**228 host tests passed** across `test_lfm3_service_batching.py`,
`test_lfm3_service_batch_redteam.py`,
`test_lfm3_worker_batch_client.py`, `test_lfm3_server.py`,
`test_lfm3_server_cancellation.py`, `test_lfm3_worker_client.py`,
`test_lfm3_worker_cancellation.py`, `test_lfm3_server_sockets.py`,
`test_lfm3_server_redteam.py`, and `test_lfm3_server_gate.py` (all in `tests/`).
These use fake workers/frontends and local pipes/HTTP harnesses: they verify one
group RPC, independent policies/cancellation/leases, strict accounting and
malformed-second-lane rejection, deadline/backpressure bounds, authentication,
framing and legacy serving compatibility. They do **not** establish actual GPU
batching, native two-row GEMM correctness, throughput, or cancellation latency.

## Explicit startup (not run by the implementation/tests)

Supply `--worker /absolute/path/liquid-lfm3-worker --checkpoint /path/to/checkpoint`
to `python tools/lfm3_server.py`, plus `--token-file /path/to/token` (owned regular
file, exact mode 0600, no symlink). Alternatively configure `LFM3_SERVER_TOKEN`.
The token must contain 32–256 URL-safe characters; generate a cryptographically
random value, not a human password. Tokens, prompts and session IDs are not logged.
Worker stderr is discarded rather than captured in an unbounded pipe/log. The
process environment supplies CUDA library discovery; the server does not change
the checkpoint, model configuration, or library paths. Port defaults to 8080;
the only accepted bind address is the literal `127.0.0.1`.

The official local tokenizer frontend is used without a Transformers model,
processor, audio codec, remote download, or checkpoint-code execution. Its existing
tokenizer dependency must already be installed; HTTP/transport need only stdlib.

## API

All POSTs require `Authorization: Bearer TOKEN`, `Content-Type: application/json`
and exactly one decimal `Content-Length`. Transfer encoding is rejected. Objects
reject duplicate keys, nonfinite numbers and nesting deeper than 64. UTF-8 is
strict; the frontend also rejects escaped invalid Unicode and reserved token
placeholders. Unknown request/message/sampling fields are rejected. No tools are
executed; plain tool-response text is supported by the existing frontend, but
tool definitions/calls and multimodal inputs are not.

* `POST /sessions` with `{"capacity":32768}` returns a random 192-bit
  `session_id`, capacity, and `gpu_batching:false`. For two sessions choose smaller
  allocations whose sum fits 32768. Configuration may lower either quota.
* `POST /generate` with `{"session_id":"ID","messages":[{"role":"user",
  "content":"Hello"}],"max_new_tokens":32,"sampling":{"do_sample":false}}`.
  Sampling is optional and follows `lfm3_chat.sampling_options`, including explicit
  seed validation. Return fields: `session_id`, decoded `text`, `generated_ids`
  (including a final stop token if present), `reused_tokens`, `consumed_tokens`,
  `do_sample`, `seed`, `gpu_batching:false`, `cancelled`, `finish_reason` and
  `cache_valid`. Normal completion reports `cancelled:false`, `cache_valid:true`,
  and `finish_reason:"stop"` or `"length"`.
* `POST /generate_batch` with `{"requests":[REQUEST_A,REQUEST_B]}`, where each
  request is a standard `/generate` body for a distinct existing session. Both
  bodies' metadata are validated before any native work, both leases are acquired
  atomically before encoding, and encoding completes before one owner-queue job
  submits one native `generate_batch` RPC (never two serial generation RPCs).
  Returns `{"results":[REPLY_A,REPLY_B],"gpu_batching":BOOLEAN,
  "paired_decode_calls":INTEGER}` in request order, without an outer `ok` field.
  Each result has standard generation fields plus the same actual paired-call
  count and batching flag. `gpu_batching` is true **iff** actual paired decode
  calls are positive; capability alone is not evidence of paired execution.
  Counts cannot exceed either lane's requested `max_new_tokens - 1`, either normal
  lane's `len(generated_ids) - 1`, or a cancelled lane's `len(generated_ids)`.
  When neither lane is cancelled the count must equal
  `min(len(REPLY_A.generated_ids), len(REPLY_B.generated_ids)) - 1` exactly;
  an EOS sampled during prefill therefore correctly permits zero paired calls.
  Prefill remains serial. Decode can use the native two-row GEMM while both lanes
  are active, then retire EOS/length/cancelled lanes independently and continue
  the peer. Each lane retains its own sampler, seed, stop policy, cache and
  cancellation event. No new sessions or extra capacity are allocated by batching.
* `POST /sessions/ID/reset` with `{}` clears that cache but retains its allocation.
* `POST /sessions/ID/close` with `{}` releases that session/allocation.
  Reset/close acknowledgments (including idle/discard allocation closes) must
  match the requested session whenever native includes a `session_id` field;
  a mismatch fails the entire service closed. Omitted identity remains supported
  for legacy protocol-2 replies, not treated as proof of a different identity.
* `POST /sessions/ID/cancel` with `{}` requests cancellation of that session's
  currently leased generation; see the cooperative cancellation contract below.
* `GET /health` is unauthenticated and returns only `ready` and
  `gpu_batching:false` (HTTP 503 after worker failure).

There is no automatic coalescing of separate `/generate` arrivals, streaming,
automatic session history, or silent
history truncation. Send the full conversation each time. The native worker checks
the cached prefix and re-prefills on mismatch. The full rendered prompt plus
**all requested** output tokens must fit the session capacity, capped at 32768.
An individual 32K allocation uses the entire aggregate budget. A session ID is
not a separate authorization principal: all users of the global bearer are
trusted local peers with access to all sessions if they know their IDs.
Sessions expire after 300 seconds idle by default (`--session-idle-timeout`,
positive finite seconds, at most 86400). Creation and completion of generation
or reset restart that interval. A queued/executing session lease never expires.
Expiration closes the native allocation and removes its session ID; subsequent
use returns 404. Clients must create a new session after expiration.

## Bounds, failure and operational limits

### Cooperative cancellation (experimental)

The cancel endpoint requires the same bearer authentication and strict JSON/framing
as other POSTs. It returns HTTP **202** with `cancel_requested:true`, `session_id`,
and `gpu_batching:false` only while a generation lease exists (encoding, queued,
native execution, or decoding). This acknowledges intent, **not terminal success**:
completion may win the race. Repeated requests while the same generation is leased
are idempotent requests. Idle/finished sessions and create/reset/close leases return
409; unknown IDs return 404. Workers without `supports_cancellation:true` return
501 `cancellation_unavailable` before any cancellation event is set. The production
CLI enables the transport capability with `cancellation=True`; normal calls remain
compatible with legacy/fake workers without the cancellable-call interface.
The CLI also requires `batching=True`: protocol-2 ready must advertise strict
boolean `decode_batching:true`, while idle `gpu_batching` remains false. The
transport constructor defaults to `batching=False` for legacy callers/handshakes;
`supports_batching` reflects the actual ready capability, not the constructor
request. A batch endpoint call with no capability returns 501.
A transport notification failure returns 503 `cancellation_transport_failed`, not
a successful acknowledgment; its already-set event may still be observed by the
owner/transport. It does not authorize handler-thread cache mutation.

A per-generation event and operation marker are installed atomically with the
lease. The handler sets that event and makes only a nonblocking targeted transport
notification; it never resets a runner/cache or performs inference. Holding the
lease identity stable through notification prevents an old request from targeting
the next generation. The cancellable transport receives that same event so a
request arriving before native dispatch/active registration is not lost.
For explicit groups, both session identities are installed under the transport's
active lock **before** the single group write; pre-set cancellation events are
bridged independently. Control frames remain little-endian u64: single requests
use their RPC ID; group lane index `i` (zero-based) uses
`((i + 1) << 32) | rpc_id`. Repeated cancellation sends at most one frame per lane.
Cancellation of an encoding/queued batch is passed to native as a pre-set lane
event, allowing owner-side reset/abort without cancelling its peer. A 202 still
acknowledges only intent, and ordinary completion may win.

Before native generation, the owner observes cancellation and performs a validated
native reset for that session only. Its terminal generation response is HTTP 200,
`cancelled:true`, `finish_reason:"cancelled"`, `text:""`, `generated_ids:[]`,
`consumed_tokens:0`, `reused_tokens:0`, and `cache_valid:false`, with the requested
`do_sample`/`seed` and `gpu_batching:false`. Running native cancellation may return
a partial (possibly empty) transcript, decoded under the same absolute request
deadline, with those same cancellation/count/cache flags. Partial IDs still obey
vocabulary, modality-suppression, count, and stop-token checks; only cancelled
responses may be empty or short without EOS. Unsolicited native cancellation or
invalid cancellation metadata fails the entire service closed, rather than
inventing a valid cache. A valid cancellation invalidates only the target cache;
other session allocations remain intact. Keep sending full conversation history
on subsequent generation; the aborted transcript is not cached history.

The lease is not released until the owner finishes native generation/reset and
validation (or ordinary admission/deadline cleanup). Late CPU-only completion
cannot release a newer generation's lease. A requested cancellation does not
extend the deadline or guarantee a terminal cancelled response: admission errors,
request expiry, disconnects, or normal completion may intervene. Slow in-process
encoding/decoding is not preempted; the existing daemon/deadline bounds still apply.
Cancellation during native prefill waits for the full current forward operation;
this is cooperative cancellation, **not GPU preemption**, a cancellation latency
SLA, or native acceptance evidence. Service/HTTP host tests are separate from
native-worker/client cancellation verification and GPU correctness gates.

Focused local verification: **101 host tests passed** with
`python -m pytest -q tests/test_lfm3_server_cancellation.py tests/test_lfm3_server.py tests/test_lfm3_server_sockets.py tests/test_lfm3_server_redteam.py::test_request_deadline_includes_tokenization_before_native_admission`.
The cancellation suite exercises authenticated admission, unsupported workers,
queued/encoding/running requests, the pre-dispatch event bridge, duplicates,
completion races, deadline lease retention, invalid partial/unsolicited metadata,
and peer-cache isolation using fake workers and in-memory HTTP handlers only.
No native/GPU cancellation acceptance or remote soak-source update is implied.

### Resource and deadline bounds

Defaults: eight simultaneous HTTP handlers, four pending owner-queue jobs (plus
one executing), 4 MiB request/response bodies, 4096 messages and 4 MiB combined
conversation text under the existing frontend. Admission is nonblocking: session
quota, queue saturation or tokenizer contention returns 429; an already leased
session returns 409. Tokenization and metadata/budget checks finish **before GPU
queue admission**. A session lease includes validation, queue wait and execution;
reset/close cannot race generation. No stdin writes from HTTP threads occur.
The listener backlog is the stdlib bounded socket backlog; connections exceeding
handler capacity are closed without spawning a thread, not promised a JSON 429.
Frontend execution uses one daemon thread and a queue bounded to one job, with a
nonblocking admission lock held until that CPU job finishes. Encode and decode
share this serialization; contention rejects new encoding with 429 rather than
spawning threads. The generation deadline starts before frontend processing and
the same absolute deadline covers encoding, owner-queue wait, and decoding.
An explicit batch uses one shared absolute request deadline for both lanes (each
therefore retains the same absolute encode/admission/decode bound). Both sequential
encodes run in one serialized frontend job; both decodes likewise run in one job,
checking expiry before proceeding to the next lane. Both native replies are fully
validated before any decoding or response publication. A running native batch
holds both leases through native completion even after its caller abandons the
request; only the owner releases them. Frontend timeout never permits late GPU
admission. Batch admission consumes one bounded queue slot, not one per lane.
Expired encoding is never admitted to native execution. A slow encoder or decoder
returns 504 to its waiting caller within the remaining budget; slow decoding also
releases the native owner to handle other jobs at that deadline, without invalidating
otherwise valid caches. Once native work has finished (or before it ever starts),
the lease can safely be released even if CPU-only frontend work is still running.
Late frontend completion never changes a session lease or submits GPU work.

An in-process frontend call is **not preemptible**: after caller timeout its CPU
work may continue in the sole daemon, retaining frontend admission until completion;
its late result is discarded. Shutdown does not join that daemon or register an
unbounded executor exit join. This bounds thread population and caller waits, not
CPU execution time, memory consumption, or interpreter/GIL starvation. Process CPU
isolation and hard CPU cancellation are not claimed.

Header and body deadlines default to ten seconds each (header trickling has an
absolute deadline). Socket writes also time out. Each worker RPC uses one shared
absolute write/read deadline, default 300 seconds, and 4 MiB bounded framing.
The generation request budget defaults to 330 seconds; expired queued jobs do not
execute. HTTP 504 while native execution is running is **outcome unknown**, not
cancellation: its lease remains held until native execution completes/fails. A
frontend timeout does not cancel its CPU job, but cannot trigger late native work.
A disconnected caller similarly
does not cancel execution. A create that exceeds its request deadline is special:
the owner closes its newly created allocation after completion, before releasing
the reserved quota, instead of retaining an unreachable ID. Completion publication
and abandonment are synchronized so a timeout racing completion cannot strand it.
A lost successful HTTP create response is recovered by idle expiration, not by
idempotent retry: retrying can return 429 until that allocation expires. The owner
checks expiration between every job and at least every 50 ms while idle. Native
close RPCs are serialized with inference, so expiry can be delayed by the bounded
current RPC and pending expiration closes (at most two sessions); it never runs
native work on a handler thread. A failed expiration close fails the entire service
closed. There is no request-rate limiter.

Any native error, pipe EOF/deadline, malformed response or inconsistent generation
metadata fails the service closed, clears **all** server session records, terminates
and reaps the process group, and makes subsequent admitted work return 503. There
is no automatic restart: stop/restart the server, then create fresh session IDs.
Native error text is not exposed to clients. Worker cleanup uses bounded TERM/KILL
waits and closes pipes. The bearer environment variable is removed from the
native child's environment. Service shutdown rejects new work, invalidates sessions,
drains rejected queued jobs, waits for the bounded running operation, then closes
the worker. The CLI installs SIGTERM handling before worker construction, retains
ownership throughout partial initialization, and handles SIGINT and SIGTERM through
this cleanup path, including stalled handshakes. Spawn briefly masks these signals
until the child handle is recorded; cleanup retains bounded KILL escalation even
when a child or descendant ignores/blocks TERM. A
supervisor should still terminate the whole service group if force-killing the
Python process, since SIGKILL cannot run cleanup.

This focused stdlib wrapper has no TLS, per-client ACLs, durable sessions,
multi-process serving, true GPU batching, GPU-preemptive cancellation, public
deployment hardening, throughput SLA, or demonstrated multi-user load stability.
Do not expose it via a public proxy or reinterpret loopback plus a global bearer
as tenant isolation. Health readiness is process/service state, not a GPU probe.

## Host verification

`python -m pytest -q tests/test_lfm3_worker_client.py tests/test_lfm3_server.py`
uses fake subprocesses, fake workers/frontends and in-memory HTTP handlers. It
does not start any network listener, load a model, use SSH, or access a GPU.
Coverage includes strict JSON, frame limits, EOF/timeout/write-backpressure and
cleanup, handshake rejection, quota/budget validation, lifecycle isolation,
per-session leases, serialized ownership, queue rejection, timeout lease retention,
bounded slow encode/decode waits, single-deadline propagation, late frontend result
discard without releasing a newer lease, frontend completion/timeout races,
shutdown without joining a stalled CPU frontend,
abandoned-create reclamation, idle allocation expiration, startup SIGTERM descendant
cleanup, fail-closed invalidation (including ambiguous nested sampling metadata),
token-file permissions and mocked-handler HTTP validation.
Real native GPU gates and end-to-end HTTP/load tests remain separate requirements.

Deadline remediation focused verification: **67 host tests passed** with
`python -m pytest -q tests/test_lfm3_server.py tests/test_lfm3_server_sockets.py tests/test_lfm3_server_redteam.py::test_request_deadline_includes_tokenization_before_native_admission`.
This selection excludes the separately owned worker-protocol duplicate-reply
regression and does not claim native/GPU acceptance of the deadline changes.

Host contract/regression evidence is not GPU acceptance or serving load evidence.
Separate socketpair framing tests are maintained in `tests/test_lfm3_server_sockets.py`.

The complete stabilized host suite passes **552 tests** (including 15 actual
socketpair parser/framing/deadline/admission tests and seven HTTP gate
counterexamples): `validation/host-http-remediated-v1.log`. Socket tests also
verify that a non-reading peer cannot retain a response writer or provoke a
second reply after a partial timed-out response. Counterexamples reject wrong
control IDs, lost cache reuse, failed recall/reset/invalidation and binary changes.
`tools/check_lfm3_server.py` separately starts an explicit temporary authenticated
loopback listener for native acceptance; these host counts do not imply it passed.

## Native HTTP verification

The corrected `http-native-remediated-v2` run **passed** against worker
`3bc90ece...`. Evidence: `validation/http-native-remediated-v2/`; actual response:
`../tmp/http-native-remediated-v2/control-response.txt`. Local and remote server,
transport and gate source hashes match and are recorded in `verification.json`.

The finite gate checks exact control IDs, authenticated create/quota/rejections,
split-delimiter rejection, A/B history/cache reuse, sampled code-word recall,
reset parity, close, normal SIGTERM fault injection, health failure and all-session
invalidation. Three eight-client rounds yield six successful exact-output requests
and eighteen busy-session409 rejections, followed each round by exact output from
both sessions. No worker/test process remains after cleanup. This is evidence of
bounded admission and serialized execution, not GPU batching or a throughput SLA.

The failed first run and traceback remain in `validation/http-native-remediated-v1/`.
Its assertions were not weakened. Native cancellation, sustained soak/resource/
performance acceptance and the integrated sanitizer gaps remain open.

### Final-source confirmation

Two further cleanup regressions ensure final reap timeout still closes the
parent's pipes and cleanup failure cannot kill the owner thread before callers
receive503. The service remains failed with `cleanup_failed=true`; a reap timeout
is not a claim that an uninterruptible child has exited. The full suite now
passes **554 tests** (`validation/host-http-remediated-v2.log`).

The final-source native rerun **passed** in `validation/http-native-remediated-v3/`:
six exact-output successes, sixteen409 busy and two429 contention rejections over
three eight-client rounds; both sessions remain exact after every round. Ordinary
SIGTERM fault injection and all-session invalidation pass, and no worker/test
process remains. Remote/local source hashes match `verification.json`. Actual
control text: `../tmp/http-native-remediated-v3/control-response.txt`.
