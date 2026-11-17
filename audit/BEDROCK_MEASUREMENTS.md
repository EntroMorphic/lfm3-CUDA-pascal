# Bedrock measurements — diagnostic, not acceptance

Production runtime and original probes remain unchanged. New code is confined to
an EXCLUDE_FROM_ALL standalone target and its measurement runner/tests. No driver,
BSP, global library or serving math change is made.

## Subsequent boundary and instruction-isolation evidence

`BOUNDARY_OBSERVER.md` adds calibrated GPU observations with explicit boundary
synchronization: all12,288 expected scratch words are correct before reduction
in three runs. Captured producer SASS contains `UTMASTG.4D`; the reported reducer
offset is a global load. Static instructions are not dynamic execution counts.
`TMA_STORE_FINDINGS.md` adds a no-cuBLAS4-D TMA-store reproduction, distinct values,
physical pre/post changes, and clean minimal memcheck/synccheck observations on
the same binary. This is consistent with NVIDIA's documented unsupported
tensormap/tcgen05 initcheck coverage, not exact vendor attribution or proof of
original unsynchronized ordering. Production gates remain unchanged. Historical
statements below retain the evidence scope/time at which they were recorded.

## Questions

1. Do producer and consumer outputs match an independent CPU oracle over every
   output element, rather than only128 sampled values?
2. Does the consumer retain initialization-read reports when every output matches?
3. Which individual API setting, or interaction, changes the measured result?

No instruction/backend root cause is inferred from enum names. Actual launch
selection has not yet been measured by this runner. Initcheck, synccheck and
racecheck are distinct unresolved gates, not assumed manifestations of one cause.

## Oracle and input scope

The C++ probe does not generate an expected result. It uploads binary BF16 inputs
provided by the Python host, executes the selected producer, consumes **all**
outputs into a separate allocation, downloads producer and consumer independently,
and writes both raw arrays. Neither output allocation is initialized before GEMM.
`beta=0` and the baseline default-stream submission order are preserved.

The CPU oracle reads the actual host input bytes. Each vector must be exactly
four-element periodic across its full length. It converts those measured BF16
values to exact integer units of1/32, computes integer dot products using the
verified repetition factor, and independently rounds the result to BF16 using
round-to-nearest/ties-to-even. Truncated or nonperiodic inputs are rejected.
CPU tests compare this oracle against literal dot products over small matrices,
exercise BF16 ties, last-element corruption, and cross-column indexing.

Legacy inputs reproduce constant0.5 input rows and channel-dependent weights.
Asymmetric inputs vary within each vector, across output channels, and across
all257 distinct input rows. Tests establish distinguishability on the small
oracle fixture. The exact whole-output oracle is not substituted for any original
production numerical reference. A difference from the mathematical BF16 oracle
would be a measured discrepancy requiring investigation of numerical semantics;
it would not by itself prove a cuBLAS or application bug.

## API controls

The three independent knobs are compute type (default/pedantic), algorithm enum
(tensor/default), and handle math (default/pedantic). The runner records requested
settings and actual enum values, queried handle math, default-stream identity,
cuBLAS/runtime/driver versions, GPU compute capability and workspace bytes.
Unsupported configurations are recorded as failures, never as clean results.
The isolation phase enumerates all8 combinations; it does not call every clean
configuration a valid production remedy.

Host monotonic stage markers distinguish API submission returns from the later
blocking downloads. They are **not GPU kernel timings**. An explicit producer
synchronization option exists in the standalone probe for a future controlled
experiment; both current runner phases select `sync=none`.

## Execution and evidence

The baseline phase has2 controls plus12 GEMM observations: both patterns, both
shapes (1/257 rows, width2048), three repeats each. The isolation phase has2 small
controls plus24 GEMM observations: asymmetric257 rows, all8 configurations, three
repeats. Every case has a90-second bound by default, owned-group cleanup, and a
dedicated output directory. Timeouts/failed controls stop the phase. Inputs,
expected arrays, outputs, raw logs and start/end hashes are retained remotely.
The temporary GPU container has48GiB hard memory and memory-plus-swap limits and
uses exact read-only copies of serving CUDA/cuBLAS libraries.

`measurement-completed` means the planned observations were collected. It does
**not** mean numerical or sanitizer acceptance; discrepancies and reported error
counts remain explicit per case, and `full_acceptance` is always false.

The first baseline attempt stopped at dependency preflight because `OwnedJob`'s
exec launcher requires an absolute executable path. Its failed evidence is kept
in `validation/projection-bedrock-baseline-v1/`. The runner now resolves `ldd`
explicitly; tests require that absolute path. No GPU measurement occurred in v1.

## Completed observations

All statements below are scoped to these exact fixtures and pinned artifacts,
not a vendor-confirmed mechanism or general GEMM correctness certificate.

### Whole-output numerical check

`projection-bedrock-numerical-v3` completes all12 GEMM observations without
instrumentation: legacy/asymmetric,1/257 rows, three repetitions each. Every
producer and consumer element matches the CPU oracle, including all1,579,008
elements in each257-row observation. No sanitizer coverage is credited to this
uninstrumented phase.

The initial full-consumer initcheck phase (`projection-bedrock-baseline-v2`)
completes three legacy1-row observations with all6,144 outputs correct but12,288
internal split-K errors each. Its first257-row observation times out at90s after
consumer submission, before the blocking download returns. Submission markers
do not prove producer completion or identify which GPU/tool operation takes the
time. That phase remains failed, with no257-row numerical conclusion.

### Independent API-setting experiment

`projection-bedrock-isolate-row1-v4` completes all8 configurations three times
each on asymmetric1-row inputs. All24 observations match every output. Each of
the12 default-compute observations has12,288 internal split-K errors, regardless
of algorithm enum or handle math. All12 pedantic-compute observations have zero
errors. Thus compute type changes the measured outcome in this fixture; neither
other knob alone does. This is not yet instruction-level causality.

`projection-bedrock-compute-only-row257-v7` also completes three asymmetric257-row
observations with only compute type changed to pedantic: all1,579,008 producer
and consumer elements match and initcheck reports zero errors. Production math
remains unchanged; its original bitwise contract still rules out adopting the
previous pedantic runtime experiment.

### Actual launches and allocation ranges

The mapped probe prints allocation bases/sizes. The unlimited diagnostic trace
`projection-bedrock-launches-row1-v6` records this default-compute sequence:

1. `nvjet_sm110_tst_128x8_64x12_2x1_v_bz_splitK_TNT`
2. `cublasLt::splitKreduce_kernel` (mangled full name retained in raw log)
3. `consume_all`

All12,288 displayed uninitialized4-byte reads in the reduction map to the user
workspace, at distinct offsets0,4,...,49,148. The complete reported read range is
therefore49,152 bytes. It is not the output allocation. Changing only compute
type records a `gemv2T_kernel_val` followed by `consume_all`, with no displayed
uninitialized-read reports. The earlier v5 trace used print-limit4 and was partial;
it is retained, not called a complete launch trace.

Launch dumping itself contributes to this tool's reported error total. V6 totals
are12,291 for default (12,288 reads plus3 printed launches) and2 for pedantic
(2 printed launches). These diagnostic totals are never credited as sanitizer
acceptance. Strict no-dump results above provide the initialization comparisons.

### Scratch overwrite experiment

`projection-bedrock-workspace-{none,a5,5a}-v9` completes three observations per
variant. Only workspace is seeded, never GEMM output. Across all nine observations:

- Every final producer and consumer output matches the independent CPU oracle.
- All49,152 post-completion scratch bytes are identical across variants/repeats.
- Interpreted as12,288 FP32 values, they exactly match two contiguous copies of
  half the independent integer-dot-product result: zero mismatching values.
- In the seeded variants, zero snapshot bytes equal the respective seed byte.
- Untouched-workspace runs report13,824 errors with the added scratch download;
  seeded runs report zero. No seeded run is treated as an acceptance workaround.

These observations establish post-completion contents and a diagnostic dependency
on scratch initialization. They do **not** observe the workspace at the instant
of each internal reduction read, prove a specific instruction is untracked, or
exclude an ordering defect. Kernel-path comparison under seeding is separately
recorded; it must not be assumed from unchanged API arguments.

The completed unlimited traces `projection-bedrock-workspace-trace-{a5,5a}-v10`
and `projection-bedrock-workspace-trace-none-v11` record the same three kernel
names (NVJET split-K, split-K reduction, consumer), all on stream0x0. Seeding
therefore did not change the observed selected kernel sequence in these runs.
The untouched trace has12,288 device4-byte read reports and1,536 host-API source
access reports on the scratch download. Neither seeded trace has either kind.
Dump totals additionally include3 printed launch records; those are not semantic
error counts and are not accepted sanitizer results.

The first scratch attempt v8 mistakenly downloaded unused, uninitialized workspace
in the fill control. The raw log reports1,536 host-API source-access errors. The
phase failed before any GEMM. The runner was corrected to apply scratch interventions
only to GEMMs; controls and their original strict assertions were retained. That
failed evidence and its frozen source are preserved, not reinterpreted as a cuBLAS
failure. The final probe/runner has74 host tests plus7 offline-analysis tests;
the combined suite passes1209 tests (`validation/host-bedrock-final-v3.log`).

The next standalone probe revision adds a read-only workspace snapshot before
the GEMM, without initialization. This separately labeled diagnostic tests whether
the physical bytes changed during the call, rather than assuming fresh allocation
contents or treating a post-completion snapshot as evidence of write timing.
It adds a blocking read before GEMM and cannot be conflated with the original
baseline.

`projection-bedrock-workspace-preimage-v12` completes that measurement. The
preimage is49,152 zero bytes; this is an observed value, not a guarantee that
`cudaMalloc` initializes storage. After GEMM,23,222 bytes and12,152 FP32 words
change. The whole snapshot matches the independently computed partial sums.
Initcheck reports1,536 host-API source accesses on the preimage download and
another1,536 after the three kernel launches. It also reports12,288 reduction
reads. The physical region changed during the call but is still reported as
uninitialized on the post-completion read. This establishes a physical-write/
initialization-report discrepancy, not its precise instruction-level cause or
proof that every earlier internal-read diagnostic is false.

The original kernels and stream0x0 sequence remain visible in that trace. Raw
before/after snapshots, complete logs, tool totals and frozen source versions are
retained. Large logs are losslessly gzip-compressed locally with original and
compressed SHA256 identities; originals remain on Thor.

## Replay and disposition

`tools/analyze_lfm3_projection_bedrock.py` replays the stored observations without
a GPU. It checks complete factorial coverage, full-output comparison bindings,
raw log hashes, allocation ranges, kernel/stream comparisons, scratch bytes and
before/after changes. Output is `validation/bedrock-consolidated-v1.json`.
This is trusted-artifact replay, not attestation that a third party ran the GPU.
Full-size raw matrices remain on Thor with file manifests and independent CPU
rechecks; local scratch snapshots are also retained directly.

All GPU jobs completed or stopped with verified process-group cleanup. The
temporary48GiB container has been stopped. Its measured cgroup peak was
1,257,009,152 bytes with zero OOM events. Production single/worker/paired binaries
remain hash-identical to the accepted default-math binaries, recorded in
`validation/bedrock-environment-v1.json`. No production seed/zero-fill, math
change, driver/BSP change, external publication or acceptance waiver occurred.

## Remaining distinction

Correct post-completion scratch/output data plus initialization reports do not
alone distinguish an unwitnessed write from a genuine read-before-write followed
by later writes. The next mechanistic evidence must observe producer/reducer
boundaries or otherwise establish which writes precede the reported reads.
Likewise, none of these measurements resolves the separate synccheck or full32K
racecheck gates. All new work is diagnostic; production acceptance stays false.
