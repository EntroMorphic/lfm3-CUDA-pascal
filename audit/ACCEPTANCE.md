# Text-inference acceptance iteration

This ledger separates verified native behavior from tool limitations. An open
external validation limitation is **not** a clean full-sanitizer certificate.

## Current text-serving status (later increments)

**Full acceptance remains false.** Current default-math single `e436191d...`,
worker `1aa1e0c2...` and paired probe `345f1ded...` have the following verified gates.
All historical failures below retain their original artifact/binary scope.

| Gate | Current evidence |
| --- | --- |
| Host suite |1245 passed, `validation/host-boundary-final-v3.log`|
| Original eager activations |6890 exact checks, `forward-bounded-prefill-v1`|
| Full eager context through4096 |9 fixtures, zero tolerances,19 full-vocabulary logit steps, exact reset and12 pre-weight rejections, `context-full-bounded-release-v2`|
| Independent32K SDPA |Unchanged0.5 max-error/0.02 relative-RMS limits, exact argmax/reset, `context32k-bounded-prefill-v1`|
| Integrated grouped execution |73 exact logit steps, independent non-greedy replay, retirement/reuse/EOS/cancellation/recovery, nonempty history and aggregate32K, `grouped-worker-bounded-prefill-v2`|
| Standalone paired matrices |Five fixtures and209 comparisons in each normal/compact mode|
| Memcheck |Standalone scratch and upper (177.11s), actual grouped worker, and integrated full32K all clean|
| Serving regressions |Resident32K/sampling, scalar cancellation and three eight-client HTTP admission rounds pass|
| Final grouped soak |561 exact groups over600.12s, fixed RSS/FD/device-wide memory and recovery limits, `http-bounded-grouped-soak-v4`|
| Balanced benchmark |20 warmed AB/BA samples per mode, numerical/source hashes bound; median paired wall ratio1.62445, `http-group-benchmark-v1`|

The first full-context attempt stopped on a stale capacity-zero diagnostic string;
its failed evidence is preserved in `context-full-bounded-release-v1`. Only that
message expectation was updated for the32768 limit. Nonzero exit/no output/pre-weight
rejection assertions and all zero numerical tolerances remain unchanged.

**Open required gates:** production initcheck, unexcluded synccheck and full32K
racecheck. Synccheck reports internal untracked NVJET launches despite correct
application output. Default initcheck remains blocked with old/new official cuBLAS;
pedantic math was rejected by4205 original baseline bitwise failures, not adopted.
The unbounded racecheck attempt was kernel-correlated global-OOM, not a zero-hazard
pass. Its memory-capped, single-analysis-worker follow-up also remains non-accepting:
900.04s timeout without application JSON, no OOM/guard trip, stable hashes and
verified descendant cleanup. Both temporary containers are stopped. See
`RACECHECK_MEMORY.md`; no full32K racecheck pass or completed coverage is established.

Explicit `/generate_batch` is real fixed-two-lane GPU batching; automatic
coalescing/streaming and kernel/prefill preemption are not claimed. See
`GROUPED_SERVING.md`, `GROUP_BENCHMARK.md`, `LONG_PREFILL_WORK.md` and
`INTEGRATED_SANITIZER_RUNNER.md`. Checkpoint-free vendor evidence is prepared
locally; no external publication, driver/BSP replacement or waiver is authorized.

### Measured initcheck boundary — diagnostic only

`BEDROCK_MEASUREMENTS.md` records whole-output numerical checks, an eight-setting
factorial experiment, actual launch/allocation traces and scratch before/after
snapshots. The one-row default path selects NVJET split-K and reports 12,288 4-byte
reads from workspace offsets0..49,148; all final outputs match. Changing compute
type alone changes the observed path/result in that fixture. A read-only snapshot
shows 23,222 scratch bytes physically change during GEMM, while a post-completion
host read is still reported uninitialized. Two scratch-seeded diagnostics select
the same kernels and produce the same bytes without those reports; they are
**not acceptance fixes**. Exact instruction-level tracking and internal-read
ordering remain unresolved in the original execution. A calibrated SDK observer
finds all expected scratch partials **before reduction** in three explicitly
synchronized diagnostics. A no-cuBLAS4-D TMA store reproduces physically changed,
correct output reported uninitialized; ordinary stores are clean, and the same
minimal binary completes memcheck/synccheck cleanly. This is consistent with
NVIDIA's documented unsupported tensormap/tcgen05 coverage, not vendor-confirmed
attribution or a production waiver. Full-runtime synccheck and32K racecheck gates
are unchanged. See `BOUNDARY_OBSERVER.md` and `TMA_STORE_FINDINGS.md`.

### Thin-hybrid decode remediation candidate (not promoted)

Candidate `thin-hybrid-runtime-candidate-v4` keeps v4's arithmetic and replaces only its decode
(rows ≤ 8) producer with v8 (coalesced 16-byte loads, quad XOR-butterfly transpose in registers,
no shared memory) for m ≥ 2048 and v5 U16 below that. Single `327a9096...`, worker `3442a068...`.
Its producers are bitwise-identical to v4's on 203 random-input cases, the real layer-0 fixture
and 192 tail cases under each sanitizer.

It independently passes:
- all 6890 eager checks
- scoped integrated initcheck, unexcluded synccheck, memcheck and racecheck (29.3 s), with zero
  findings
- context through 4096, both paired matrices, the grouped worker and 32K SDPA

Balanced warmed AB/BA 20-sample decode: median paired ratio **0.8450810816** against production
(v4: 3.445809). Prefill is unchanged from v4 (0.1099 s against 0.0320 s). The full 32768-prefill
guarded racecheck timed out at 900.6 s, with no OOM and clean cleanup. That fixture runs only
v4's unchanged prefill `wmma_partials`, and its timeout is not credited or waived. HTTP,
cancellation, soak and grouped-throughput gates are not repeated. Production is unchanged; full
acceptance remains false. See `THIN_HYBRID.md`.

### Native ordinary-store remediation candidate (not promoted)

Latest v4 candidate `81df4dea...` independently passes6890 eager checks, nine
exact context fixtures, both209-comparison paired matrices,73 grouped-worker
steps, three32K SDPA cases and all four scoped sanitizers without exclusions.
Its four complete32K logit outputs match production bitwise. However, balanced
warmed20-sample AB/BA measurement shows **3.445809175593137x decode slowdown**.
Full32768-prefill guarded racecheck timed out at900.605s without application
JSON or a canonical footer; upper initcheck/synccheck/memcheck were not launched.
Production/library hashes are unchanged and owned processes/container are cleaned
up (`wmma-v4-final-cleanup.json`). HTTP/cancellation,
grouped performance, soak and deployment gates have not been repeated for v4.
Production is unchanged; full acceptance remains false. See `WMMA_REMEDIATION.md`
for exact hashes, frozen sources, scope and preserved upper failure evidence.

`WMMA_REMEDIATION.md` records an isolated native WMMA linear path that preserves
the measured split-K arithmetic while avoiding the vendor TMA stores. Candidate
single `d6ed2fef...` passes all6890 unchanged eager checks and all nine exact
context fixtures through4096. All6890 regenerated reference hashes also match the
original reference files. Its original four-token fixture passes integrated
initcheck, unexcluded synccheck and memcheck with exact original application output.
This is a real backend candidate, not scratch seeding or checker suppression.
Production still uses the original hashes above. Upper/context/worker, performance,
resource and full32K racecheck acceptance must not be inferred from this increment.

The same v1 candidate also passes the unchanged three32K SDPA cases; all four
complete native logit files additionally match the original binary bitwise. Both
five-fixture paired matrices pass209 comparisons per mode. Its grouped worker
`d527aee0...` passes73 exact full-logit steps, independent non-greedy replay and
aggregate32K/cancellation/recovery checks. The first worker attempt used an absent
oracle path and never launched; the corrected-path retry preserves the original
oracle hash. Production promotion and performance/resource/upper-sanitizer/full
racecheck acceptance remain separate. A newer input-load optimization must repeat
its own gates rather than inheriting v1 results.

### Prior standalone and single-serving increments (historical)

The prior standalone increment: paired decode passes five fixtures in both normal
and compact reference modes,209 bitwise-logit/full-cache comparisons per mode.
See `BATCHING.md` and `OWNED_GPU_JOBS.md`. This is **not integrated service
batching**. The complete host suite at this increment passes722 tests, with
follow-up diagnostic regressions bringing the final suite to724 passing tests.
Clean serialized32K SDPA
evidence is `validation/context32k-fastpath-v4/`; the overlapped v3 is explicitly
non-accepting. Initcheck, upper-bound paired memcheck and integrated racecheck
remain unresolved; the latest300-second timeout is not a pass.

The historical batch-one milestones below are preserved. The codec-free frontend,
seeded sampling, fixed32K numerical checks, resident cache isolation and local
authenticated HTTP service now have separate evidence. The current HTTP increment
passes667 host tests and finite native control/history/reset/fault, eight-client
admission, host-resource soak, and cooperative token-boundary cancellation gates.
Latest evidence: `validation/cancellation-verification.json`, `CANCELLATION.md`
and `SOAK.md`. This is serialized serving, **not worker/service GPU batching**.
Long-soak/GPU-resource/performance acceptance, integrated racecheck
(1200-second timeout) and blocked
cuBLAS initcheck remain open. See `TEXT_SERVING_ROADMAP.md`, `SESSIONS.md` and
`SERVER.md`; no full-serving or100/100 acceptance is implied.

## Priority 1 — typed prompt to native text

Closed for the supported single-turn, batch-one, greedy interface. Commit
`d86d657` adds `tools/lfm3_chat.py`, 68 host regressions, and an actual Thor smoke
result. Local user-facing artifacts: `../tmp/chat-smoke-001/`.

Red-team cases cover malformed/mismatched native responses, modality leakage,
stopping, bounds, process failure/timeout, JSON ambiguity and output overwrite.
Language-model inference is entirely native; the official Python processor does
formatting/tokenization/decoding and loads its auxiliary audio codec. This is not
native-tokenizer parity, batched serving or multimodal inference.

## Priority 2 — acceptance and throughput

Remediated benchmark issues: the native warmup originally exercised just one
decode while the reference warmed a full greedy sequence; it now warms the full
requested sequence. Gate validation cannot pass vacuously on empty cases or
nonboolean flags, cannot be disabled by Python `-O`, and requires exact numerical
evidence for the measured binary. Timing/call counts, stop/suppression policy,
every measured sequence, length mismatches, finite values and unchanged binary
identity are checked. The benchmark harness has 102 host regressions.

The `af0a0db7...` candidate passed a three-sample, warmed, 86-token comparison with
all generated IDs equal to Transformers eager: median native decode 39.45 tok/s,
Transformers eager 33.05 tok/s. Raw evidence: `validation/benchmark-eager-v2/`.
These are synchronized backend-specific batch-one loops, not isolated kernels,
best-SDPA comparisons or universal speedup claims. Later runtime revisions require
their own hash-matched correctness/benchmark evidence.

### Open external limitation: initcheck output tracking

The integrated CUDA 13.0 sanitizer run passed memcheck and racecheck; synccheck
passed with NVIDIA NVJET excluded because of its previously reproduced internal
instrumentation failure. Initcheck reported uninitialized convolution inputs and
was stopped rather than credited as passed.

`lfm3-projection-probe` isolates the producer/consumer boundary with no checkpoint:

- A CUDA-fill producer plus the same CUDA consumer passes initcheck.
- An intentionally missing producer is detected, validating the negative control.
- A cuBLAS BF16 GEMM produces the correct 128 sampled values, but initcheck labels
  those values uninitialized at the consumer. No output initialization, host
  roundtrip, custom-kernel exclusion or suppression is used to conceal this.

The reproduction was red-teamed again: a uniform GEMM output might accidentally
match stale memory from the fill control. The final probe instead uses a
nonuniform per-channel GEMM pattern, distinct from the constant fill control, and
checks every sampled result against its exact analytic value. The same tracking
failure persists. Evidence: `validation/projection-nonuniform-final/`.

The reproduction persists with Compute Sanitizer 2025.3.1, 2025.4.1, and 2026.2.0;
enabling tensor-op checks on the latter did not resolve it. Tested cuBLAS reports
version 130100 on Thor. Isolated sanitizer downloads were SHA-256 verified; no
system CUDA/driver installation was changed.

`check_lfm3_sanitizer.py` now runs positive/negative producer controls before a
potentially very slow integrated initcheck. It records `passed: false` and
`blocked-cuBLAS-output-tracking` on the reproduced condition, exits 86, and
preserves the logs. It cannot credit missing logs, internal errors, failed
controls or incorrect producer values as a clean result. Six host regressions
verify this fail-closed behavior. Evidence: `validation/forward-sanitizer-v2/`.

The frozen `cabeb776...` runtime also passes integrated **memcheck, racecheck and
synccheck at 4096-token prefill** with no timeout. Only synccheck excludes NVJET;
custom kernels, including the new large softmax and RoPE path, remain instrumented.
The final report still has `passed: false` because initcheck is blocked. Evidence:
`validation/forward-sanitizer-final4096/`.

**Disposition:** bounded numerical/native inference acceptance can be measured,
but complete integrated global-memory initialization coverage remains blocked by
this reproduced library/tool interaction. Reporting this as “pristine” would be
incorrect.

## Priority 3 — cache/context red-team

The stricter context suite found and retained these failures:

1. A 256-token prefill matched exactly but the next sliding decode did not.
   Native masked a larger attention product while Transformers cropped to the
   last 256 keys/values. Native now matches the physical decode product width.
2. A 1024-token prefill exposed RoPE frequency construction differences. The
   checkpoint constructs inverse frequencies on CPU; native recomputed with GPU
   `powf`. At frequency 18, a one-ULP difference changes BF16 cosine at position
   850. Native now constructs FP32 frequencies on CPU before upload.
3. Source review established a separate ATen FP32 softmax register/block reduction
   above 2048 columns. A 1024-thread, two-stage shuffle-down path now covers that
   range; shorter rows retain the persistent XOR reduction.
4. `warmup_runs`, `check_reset`, and logits-only tracing flags must be validated
   before checkpoint upload. Config-only negative requests test that order.

`check_lfm3_context.py` tests capacities 1/2, convolution handoff, 255/256/257
windows, 1024+2, 4095+1, and 4096 prefill with exact-logit reset and final-slot
checks. Twelve overflow/type controls use a configuration with **no weights** to
prove rejection before loading. Strict bitwise comparisons are never silently
relaxed. Reusable oracles are hashed before native execution; failures preserve
case, step and partial evidence. Logits-only tracing avoids multi-GB activation
copies and unsafe sink workarounds.

The source-frozen `cabeb776...` candidate passed all nine context cases:
**19 complete vocabulary-logit vectors (2,432,000 BF16 values) bitwise equal** to
the pinned eager oracle, with exact-logit reset and all twelve pre-upload controls.
Saved evidence: `validation/context-v3/`; the failed predecessor is preserved in
`validation/context-failed-v2/`. Raw oracle vectors remain in the Thor build tree;
the checked-in manifest records their SHA-256 values and explicit weight identity.

## Frozen-candidate final verification

Runtime SHA-256:
`cabeb776f6221895810ad226c673ff12fa911d13a5bca19d96f94aea53f1c27c`.
The final forward, context, benchmark and 4096-token sanitizer records all identify
this exact runtime binary.

- **6,890 bitwise activation/logit comparisons** pass, now also capturing actual
  eager rotary inputs, attended outputs and probabilities. Teacher-forced and
  free-running generation, exact reset, stopping and suppression pass unchanged.
- All nine context cases and twelve config-only controls pass, as above.
- Three measured runs per backend generate identical **86-token** responses.
  Median warmed decode is **39.83 tok/s native**, **33.37 tok/s Transformers eager**;
  median prefill is 31.06 ms and 35.84 ms respectively. See benchmark scope and
  limitations above; timings exclude model loading and text decoding.
- Integrated memcheck/racecheck/synccheck at 4096-token prefill pass within scope;
  complete initcheck is still **blocked**, with the strengthened nonuniform
  reproduction preserved. The final combined sanitizer report is not passing.
- **327 host tests pass** (`validation/host-acceptance-final.log`). The original
  repository's recorded source hashes remain unchanged.
- Actual typed-prompt → native generation → decoded text succeeds for the frozen
  runtime; local artifacts are in `../tmp/lfm3-native-final/`.

Final evidence: `validation/forward-final/`, `validation/context-v3/`,
`validation/benchmark-eager-final/`, `validation/forward-sanitizer-final4096/`, and
`validation/acceptance-final.json`. Earlier candidate evidence remains historical.

**Disposition:** owned findings in priorities 1 and 3, and the benchmark/harness
findings in priority 2, are remediated and verified for the stated coverage.
“Pristine” overall acceptance is **not achieved**: initialization coverage requires
a compatible NVIDIA tooling/library/driver stack or vendor remediation. No system
driver upgrade, suppression of our kernels, or numerical-tolerance relaxation was
used to manufacture closure.
