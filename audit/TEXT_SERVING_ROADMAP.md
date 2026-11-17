# Full text-serving acceptance target

**Consolidated current release gates:** `ACCEPTANCE.md` records1128 host passes,
exact full eager-context4096, independent32K SDPA, actual grouped-worker memcheck,
final561-cycle/600-second grouped resource soak and balanced20-per-mode AB/BA
benchmark. Production initcheck, unexcluded synccheck and complete32K racecheck
remain unresolved. Lower-count/standalone-only notes below are historical.

Current increment: `GROUPED_SERVING.md` records real fixed-two-lane native/HTTP
group dispatch,905 host passes and actual grouped parity/cancellation/aggregate32K
plus authenticated HTTP smoke. It closes the prior standalone-only limitation
for that explicit interface, not automatic microbatching. Extended grouped resource
soak and complete sanitizer acceptance remain separate. Earlier increment notes
below preserve their historical scope and binaries.

Latest red-team increment: `REDTEAM_ROUND2.md`. RPC reply correlation and frontend
absolute deadlines were remediated; HTTP negative fixtures and resident stochastic
replay were strengthened.575 increment-scoped host tests and native resident/HTTP
gates pass for protocol2 worker `1f3401db...`. Soak/cancellation/batching and
integrated sanitizer acceptance remain separate gates.

Subsequent verified increments: `SOAK.md` records a finite30-cycle host/worker
RSS/descriptor gate and end-to-end latency (not a GPU-memory or SLA certificate).
`CANCELLATION.md` records667 host tests and native/HTTP token-boundary cancellation
with physical-cache discard, committed-history preservation, peer isolation and
exact recovery. The final worker `842a46f1...` also passes resident32K budget and
independent stochastic replay. True paired decode is still a standalone prototype,
not worker/service batching; long soak/GPU-resource/performance and integrated
sanitizer gates remain open.

Paired-decode progress: the standalone normal/compact five-fixture gates now pass
bitwise logits, complete physical caches, lane retirement/reversal and32K capacity
boundaries. See `BATCHING.md`. Hardened fixture controllers cover termination,
startup/setup failures and group cleanup; failed/overlapped evidence is preserved
in `OWNED_GPU_JOBS.md`. Next implementation work remains explicit grouped worker
dispatch and per-lane cancellation/sampler/lease ownership; the current RPC client
and HTTP owner are serialized and must not be labeled batched serving.

Requested scope: text preprocessing/decoding, native generation with sampling,
multi-turn sessions, batching/concurrency, and the model-card context maximum.
Existing bounded experimental acceptance is preserved, not generalized silently.

## Context source of truth

`LiquidAI/LFM3-Omni-2B` model card, both current and pinned revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`, states **32K tokens, input + output**.
Source: https://huggingface.co/LiquidAI/LFM3-Omni-2B#model-details
The local `text_config.max_position_embeddings=128000` is not the advertised
context promise. No 256K context claim is established for this checkpoint. If
the target checkpoint/model-card changes, re-evaluate explicitly.

## Work gates

1. **Bounded-memory 32768-token runtime.** Tile attention query rows, preserve
   the previously validated <=4096 schedule, enforce a combined input/output
   serving budget, validate long-position RoPE and long-row softmax. Prefill,
   final-slot decode, overflow, reset, and long prompt logits require evidence.
   Bring-up requests above4096 require `experimental_long_context=true`; the
   public launcher is not yet raised to32K. This is not accepted32K serving yet.
2. **Tokenizer-only frontend.** Avoid importing the Omni processor/audio codec
   for text requests. Six official processor parity cases passed on Thor with
   exact UTF-8 rendering and token IDs. Integration into the entry point pending.
3. **Sampling.** Host-only sampler supports validated temperature/top-k/top-p/
   min-p/repetition penalty and seeded RNG. Host adversarial/statistical tests
   pass; native request/response and CLI integration, reference distribution
   comparisons and end-to-end reproducibility remain pending. No torch RNG parity
   claim is implied.
4. **Persistent multi-turn sessions.** Explicit cache ownership, isolation, reset,
   capacity accounting, cancellation, restart/error semantics and transcript
   parity. No reuse of unconsumed final predictions without forwarding them.
5. **Batching/concurrency and serving.** Independent sequence positions and
   convolution/KV states, bounded admissions/memory/queue, stop/cancellation,
   protocol limits and deterministic failure/recovery. A serialized subprocess
   wrapper must not be represented as GPU batching.
6. **Release acceptance.** Adversarial/unit/integration coverage, native/reference
   logits and greedy/distribution checks, request isolation, soak and resource
   measurements, hash-matched benchmark, and explicit sanitizer disposition.
   The reproduced cuBLAS initcheck tracking blocker is still open; no zero-fill,
   custom-kernel suppression or weakened numerical gate closes it.

## First implementation increment

`text_serving_limits.hpp` specifies32768 and128-query tiling. Attention score and
probability scratch at maximum capacity is256MiB combined, rather than64GiB for
two quadratic BF16 matrices. KV and other activation allocations are additional.
The new long-row softmax has32 scalar registers/thread to cover32768 columns;
its arithmetic above4096 is bring-up, not an exact ATen equivalence certificate.

The first bring-up binary completed8192- and32768-token prefills with finite
logits and preserved all6890 bitwise short-context activation checks. The combined
host suite passes385 tests. Evidence: `validation/32k-foundation/`. This is **not**
accepted full32K text serving: independent long-context numerical comparison,
decode/reset boundaries, new-kernel sanitizer coverage, frontend/sampling runtime
integration and the later session/concurrency gates remain required.

`check_lfm3_32k.py` runs a separate SDPA oracle at8192,32767+1, and32768. It uses
predeclared maximum logit error0.5 and relative RMS0.02 plus exact argmax/reset;
this is explicitly different from the older eager bitwise gate. It preserves
failure evidence and never raises tolerances to close a failed test. That initial
comparison is pending; SDPA is chosen to avoid a quadratic eager oracle's memory
peak, not to label a different backend as exact eager parity.

The first SDPA comparison **passed** for binary `ffbab18f...`:8192 prefill,
32767+1 cached decode, and32768 prefill. Maximum observed logit absolute error
was0.1875 and relative RMS0.003131; every argmax matched and native reset logits
were exact. Evidence: `validation/context32k-sdpa-v1/`. This closes those finite
fixtures, not all long-position/backend behavior or full text-serving acceptance.
Later runtime changes require hash-matched revalidation.

## Sampling/frontend integration increment

The native runtime now accepts sampling fields with strict validation before GPU
upload. A reviewed false-pass in the first integration harness was remediated:
reproducibility alone cannot prove that sampling is wired in. The new gate
independently replays logits in `text-sampling-oracle` and requires a verified
non-greedy choice. It passes seed reproducibility, warmup RNG isolation, top-k1
equivalence, exact-logit reset, suppression/stopping and14 config-only rejections.

`lfm3_chat.py` now uses only the tokenizer/template frontend for normal requests,
exposes the sampling fields and `--messages` textual history, and enforces the
strict combined32K serving budget. The control prompt reproduces the previously
validated response without loading the codec. A sampled four-message history
correctly recalled the requested code word `cobalt`. This is history replay through
prefill, **not persistent KV sessions** or GPU batching.

The same integrated binary (`85028f2a...`) passes6890 short-context bitwise checks
and repeats the8192/32767+1/32768 SDPA gate with the same bounded errors. The host
suite has422 passing tests. User-facing actual outputs are copied locally to
`../tmp/control-codecfree-001/` and `../tmp/multiturn-sampled-001/`.

The next resident-worker increment now has finite GPU cache-isolation evidence;
see `SESSIONS.md` and `validation/worker-gates-v3-32k/`. Actual GPU batching, bounded
external request serving/concurrent cancellation, soak and performance/resource
acceptance remain open. The32K integrated sanitizer run passed memcheck and
NVJET-excluded synccheck, but racecheck timed out after1200 seconds and is **not a
pass**; the cuBLAS initcheck blocker also remains open. Evidence:
`validation/sanitizer32k-sampling-v1/`. Small maximum-dimension shared-memory
probes passed all four tools without exclusions (60 fixtures, including softmax
width32768): `validation/serving-ops-sanitizers-v1/`. They do not substitute for
integrated coverage. The shared-weight runner repeats the32K SDPA checks:
`validation/context32k-resident-shared-v1/`. Existing/resident host suites pass470
tests, excluding the separately developing HTTP/client suites.
Do not represent successful components as complete full-serving or100/100 acceptance.

## Authenticated local HTTP increment

The loopback-only bearer-authenticated service passes554 host tests plus the
finite native HTTP gate (`validation/http-native-remediated-v3/`). Verification
includes control/history/cache/reset parity, auth and malformed inputs, process
failure invalidation, and three eight-client admission rounds. Each round accepts
two serialized session requests and rejects six busy/contending peers, then confirms exact
output from both sessions. This is **not GPU batching** or sustained-load acceptance.

Independent review remediated startup cleanup, unreachable timed-out creation
quota, idle allocation expiration, contradictory sampling metadata and inherited
blocked signals; the failed native first run remains preserved. See `SERVER.md`.
True GPU batching, native cancellation, soak/resource/performance gates and the
integrated racecheck timeout/initcheck blocker still prevent full acceptance.

`text_sampling.hpp` and `lfm3_text_frontend.py` are independently tested components;
neither is silently advertised as fully integrated native serving. Their detailed
contracts and remaining validation are in `SAMPLING.md` and `TEXT_FRONTEND.md`.
