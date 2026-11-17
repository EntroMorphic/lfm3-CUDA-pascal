# Native ordinary-store GEMM remediation candidate

## Current disposition

The first isolated runtime candidate (v1) replaces the measured NVJET linear path with
native BF16 WMMA and ordinary global stores. It passes all **6,890 unchanged eager
checks**, all **nine exact context fixtures through4096**, and integrated initcheck,
unexcluded synccheck and memcheck on the original four-token single fixture.
It also passes the three32K SDPA cases, both five-fixture paired matrices and
the integrated grouped-worker gate. **Production is not switched and full
acceptance remains false.** Resource, balanced performance, upper sanitizer and
full32K racecheck acceptance are separate. An input-load-only v2 optimization is
recorded below; further revisions must repeat their own gates rather than inherit
v1 evidence.

Candidate single SHA256:
`d6ed2fefc5e8691cb574b4468e91e9e1cd5361c605f66f5cdac50fb8450d288d`.
The original single/worker/paired source paths and binaries remain unchanged.

## Failed alternatives retained

1. Explicit workspace budgets0 and16KiB preserve the small fixture's numbers but
   still report6,528 initcheck errors and internal synccheck failures.
2. Disallowing reduced-precision reductions preserves the small fixture's numbers
   but still reports12,288 initcheck errors and internal synccheck failures.
3. The first real-weight Lt screen correctly rejected an ambiguous tensor suffix:
   audio, language and vision each have a layer0 convolution projection. The
   corrected extractor selects the complete language-model key and verifies shape.
4. Initial Lt enumeration executes nine configurations per real fixture. Only two
   match all6,144 decode outputs; both report12,288 initcheck errors and internal
   synccheck failures. The other tested decode defaults differ by2–3 BF16 values.
   The first34-row instrumented prefill candidate times out at60s; no pass is
   credited. Expanded split-K enumeration executes107 decode configurations
   (24 exact) and80 prefill configurations (7 exact). Numerically exact decode
   reduction-scheme variants1/4 still reproduce both checker failures.

Evidence: `blas-policy-screen-v1`, `lt-policy-screen-v1`, `lt-policy-screen-v2`,
`lt-policy-split-scan-v3`, `lt-policy-reductions-v4` under `validation/`.
Enumeration is a bounded candidate search, not a complete proof about every
possible library algorithm or configuration.

## Native arithmetic and stores

`native/probes/lfm3_wmma_policy_probe.cu` uses one warp per16x16 output tile, BF16
WMMA in16-element K steps, FP32 partial accumulators, an ordinary-store FP32
scratch matrix, and a native reduction/materialization kernel. Shared-memory
zero padding covers only input tails; scratch/output is never preinitialized.
Producer and consumer outputs are checked in full against real checkpoint traces.

For the actual layer0 projection:

| Fixture | One partition | Two partitions | Four partitions |
| --- | --- | --- | --- |
| Decode,6,144 values |3 different |**0 different** |2 different |
| Prefill,208,896 values |**0 different** |192 different |205 different |

Both exact choices independently pass initcheck, memcheck and synccheck without
exclusions. This demonstrates why blindly changing reduction order is not a fix.
See `wmma-policy-screen-v1`. All12 standalone observations, including rejected
partition choices, are independently replayed from complete producer/consumer
bytes and original reference outputs in `validation/wmma-policy-replay-v1.json`.
Checkpoint weight bytes remain remote and are not copied into this repository.

`tools/prepare_lfm3_wmma_runtime.py` creates an isolated source mirror and build.
Only `Runner::linear` dispatch changes. Its generated header incorporates the
exact standalone WMMA kernel source, checked by host tests and recorded hashes.
It queries cuBLASLt's first heuristic with the original8,519,680-byte budget to
obtain the partition count. In this pinned experiment, algorithm ID66 selects
the native path; other IDs retain the original cuBLAS call. Unsupported partition
geometry throws rather than silently changing arithmetic. Both attention batched
GEMMs, all normalization/sampling/cache code and all numerical gates are unchanged.

This ID-based heuristic mapping is **candidate policy for the pinned environment**,
not a portable vendor guarantee. It must not be generalized across library/GPU
versions or presented as an accepted deployment policy without further checks.
The native scratch is distinct, owned and fully written by each producer; it is
not an initialization workaround for the original vendor scratch.

To regenerate a fresh candidate (not a production switch):

```sh
python3 tools/prepare_lfm3_wmma_runtime.py --output build/wmma-runtime-candidate-NEW
cmake -S build/wmma-runtime-candidate-NEW -B build/wmma-runtime-candidate-NEW/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/wmma-runtime-candidate-NEW/build --target liquid-lfm3 liquid-lfm3-worker lfm3-decode-batch-probe -j4
```

Build on Thor in compiler image
`sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f`;
execute in the original pinned runtime/library environment, not compiler-image
cuBLAS. `validation/wmma-runtime-source-v1/` preserves the complete emitted source
mirror and binary identities. The candidate single's static SASS contains HMMA
and no `UTMASTG`; disassembly metadata binds the actual binary in
`validation/wmma-runtime-disassembly-v1/`. This is not a dynamic instruction count
or a claim about external library modules.

## Numerical and integrated checks

- Original `tools/check_lfm3_forward.py`: all6,890 exact activation/logit checks,
  greedy/teacher IDs, suppression/stopping and reset pass. The regenerated
  reference files also match **all6,890 original reference hashes**, independently
  replayed; no goldens or tolerances changed.
- Original full `tools/check_lfm3_context.py`, reusing its pinned original
  references: nine fixtures,19 complete vocabulary logit steps, zero tolerances,
  reset and12 pre-weight rejections pass.
- The original single-token-prompt/four-generated-token request, with reset and
  expected original application output, passes integrated initcheck (16.205s),
  unexcluded synccheck (2.727s), and memcheck (18.553s). These are scoped elapsed
  tool observations, not serving speed benchmarks.
- Original32K SDPA gate:8192,32767+1 and32768 cases pass unchanged0.5 maximum-error
  and0.02 relative-RMS bounds, exact argmax and reset. Observed maxima are0.15625
  and0.0031099559273570776 respectively. All four complete native logit files
  additionally match the original production binary's saved outputs **bitwise**.
- Both normal and compact paired matrices pass all five fixtures and209 exact
  comparisons per mode, including the upper capacity split.
- Integrated grouped worker passes73 complete-vocabulary bitwise steps,
  independent non-greedy logit replay, retirement/reuse/EOS, targeted cancellation,
  recovery, stale tags and aggregate32K checks. The first worker attempt failed
  before launching the worker because the oracle path was wrong. Its original
  failure is kept; the retry uses `build/cmake/text-sampling-oracle` with the same
  previously accepted SHA256 `8ff4ee7c563beeacd09ba192245b2a40d1cca671016ce711930c74373dc9f0f0`.

Evidence: `wmma-runtime-forward-v2`, `wmma-runtime-context-v1`, and
`wmma-runtime-sanitizers-v1`. The first forward attempt failed during reference
Triton assembly because it omitted the documented Thor assembler variables; it
never reached native comparison. That failure is retained as
`wmma-runtime-forward-v1`. The successful retry uses the existing documented
`/workspace/tools/ptxas` path for both Triton assembler variables, with its hash
pinned; no system tool replacement occurred.

Additional evidence: `wmma-runtime-32k-v1` (356.840s),
`wmma-runtime-paired-normal-v1` (296.133s),
`wmma-runtime-paired-compact-v1` (303.535s), and
`wmma-runtime-grouped-worker-v2` (185.834s). These are gate wall times, not throughput
comparisons. Failure evidence is `wmma-runtime-grouped-worker-v1.stdout.log`.
`wmma-runtime-32k-v1/original-native-replay.json` binds all four original/candidate
complete logit hashes.

Actual generation responses are copied to project-root
`../tmp/wmma-runtime-candidate-v1/`, including decoded text linked to raw response
hashes. The current host suite passes1287 tests
(`validation/host-wmma-candidate-v4.log`), including fail-closed host models for
missing/duplicate checker footers, internal checker errors, incorrect application
IDs, missing output and timeout.

## Subsequent revisions: measured, not silently promoted

- **v2 — direct complete-tile input loads:** all6890 original checks and the
  three scoped initcheck/synccheck/memcheck checks pass. Its scoped racecheck
  times out at300.002s without application output or a canonical footer. The
  timeout remains failed evidence in `wmma-runtime-sanitizers-v2`.
- **v3 — register-only thin GEMM:** for rows1..8, use PTX
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, four independent warps
  per block where geometry permits, and ordinary FP32 stores. No shared memory
  is used by this thin producer. All6890 original eager checks pass again.
  Scoped initcheck, unexcluded synccheck and memcheck pass; scoped racecheck
  completes in28.323s with zero hazards/errors/warnings and exact application
  output. This is **not** full32K racecheck acceptance.
- **v3 cost:** a balanced warmed AB/BA experiment with20 samples per binary,
  all original token IDs exact, measures median decode0.3796655s for the original
  and1.341545s for v3. Median paired candidate/original decode ratio is
  **3.530298008783833**. Median prefill is0.02873435s versus0.11115s. Model loading
  and warmup are outside those native inference counters. This measured regression
  prevents a silent production switch; correctness alone is not enough.
- **v4 — aligned operand loads, under validation:** disassembly of the actual v3
  image finds120 static `LDG.E.U8` sites in the thin producer, caused by the
  four-byte `memcpy` bit loader. Static sites are not dynamic execution counts.
  Replace that loader with `ld.global.u32`, preserving operand bits. CUDA
  allocations and the validated K/partition multiples of16 plus even BF16
  offsets guarantee the required four-byte alignment. Static inspection verifies
  four-byte-aligned payload offsets and input widths divisible by16 for all154
  text matrix tensors; see `wmma-operand-alignment-v1.json`. The indexed loader
  independently places tensors on256-byte boundaries. New tail fixtures cover
  m1/17/63/65, n1/2/7/8/9/17, K16/32/48/64, partition counts1/2/3/4, matrix tails,
  whole-warp retirement, partially populated last blocks and both paths. Completed
  v4 results and their remaining limitations are recorded below.

Evidence: `wmma-runtime-forward-v3`, `wmma-runtime-forward-v4`,
`wmma-runtime-sanitizers-v2`, `wmma-runtime-sanitizers-v3`,
`wmma-runtime-racecheck-v3`, `wmma-runtime-benchmark-v1`, and
`wmma-runtime-disassembly-v3`. The benchmark requires hash-matched complete6890
eager evidence before either binary runs. Racecheck parsing requires its own
canonical hazard footer and treats warnings as non-clean; missing or contradictory
footers cannot become zero-hazard results.

## v4 completed matrix; production unchanged

The aligned-load candidate single is
`81df4deae203b3cd646bfa1827317389e900a981e8d9db72dd26725a413799d0`;
worker is `1bfe0929b505fbca029b0581b4b173dc499119a70c6049e82434796456342902`.
`validation/wmma-runtime-source-v4/` freezes the complete generated mirror,
binary identities, original kernel/helper and execution runners. The generator
ran locally; its local source is preserved with that mirror. The completed parent
`wmma-v4-gates.json` records stable start/end binary and runner hashes.

Verified on v4, without transferring earlier binaries' credit:

- Six synthetic tail fixtures under all four tools:24 exact observations.
- All6890 unchanged eager checks, nine exact context cases through4096.
- Normal and compact paired matrices:209 comparisons each, independently replayed
  with the original assertions (`wmma-v4-local-replay.json`).
- Grouped worker:73 bitwise full-vocabulary steps and independent sampling replay.
- All three32K SDPA cases with unchanged limits, argmax and reset assertions.
  All four complete saved native logit files also match production bitwise;
  see `wmma-runtime-32k-v4/original-native-replay.json`.
- Four-token integrated initcheck5.893s, unexcluded synccheck2.325s,
  memcheck6.745s and racecheck28.229s: exact application output and zero errors;
  racecheck also has zero hazards/warnings. These are scoped checks only.
- Host suite:1287 passed (`host-wmma-candidate-v4.log`).

Balanced warmed AB/BA20 samples per binary remain a performance failure for
promotion: original median prefill0.02844955s/decode0.378834s; v4
prefill0.111122s/decode1.305875s. Median paired decode slowdown is
**3.445809175593137**. All measured responses retain the expected IDs.
Raw/decoded actual responses are in project-root
`../tmp/wmma-runtime-candidate-v4/`; original responses remain intact.

Full32768-prefill all-launch racecheck **timed out after900.605s** with one
analysis worker,32GiB owned-RSS budget and16GiB minimum host availability.
It produced neither application JSON nor a canonical zero-hazard footer. The
resource guard did not trigger; cgroup OOM/kill counters remained zero. Artifact
hashes stayed stable and owned-group cleanup completed. The parent returned86
and correctly did **not launch** upper initcheck/synccheck/memcheck. These tools
remain untested at this upper fixture, not failed or clean by inference.
Complete failure evidence and commands are preserved in
`validation/wmma-runtime-upper-racecheck-v4/` and `wmma-v4-upper-gates.json`.

An independent host process scan found no live inference or sanitizer processes.
All three production binaries and all three pinned CUDA libraries retain their
original SHA256 values. The owned `lfm3-policy-v1` container was stopped and
verified exited (exit137 after Docker stop; `OOMKilled:false`). See
`validation/wmma-v4-final-cleanup.json`. Production remains unchanged and full
acceptance remains false; no timeout or incomplete checker result is waived.

## Remaining release gates

No production promotion, upper-context sanitizer certificate or full racecheck
completion is implied. Legacy serving/cancellation, balanced performance,
resource soak and upper sanitizer/racecheck gates must retain their original
assertions and complete on the selected final binary. There are no suppressions,
checker kernel exclusions, scratch seeds, pedantic math, system library changes,
vendor submissions or waivers in this remediation.
