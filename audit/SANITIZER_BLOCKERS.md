# Sanitizer blocker follow-up

Neither a timeout nor correct sampled output is a full sanitizer pass. No output
preinitialization, instruction suppression, selected-block coverage reduction or
disabled checking was used.

## Latest measurement-only localization

`BEDROCK_MEASUREMENTS.md` and `validation/bedrock-consolidated-v1.json` narrow
the one-row initcheck symptom to the NVJET split-K/reduction workspace path.
Whole-output checks, isolated API settings, complete diagnostic launch traces,
allocation mapping and workspace before/after snapshots are measured separately.
They establish a physical-write/initialization-report discrepancy after GEMM,
but not the precise untracked instruction or freedom from an internal ordering
defect. Scratch seeding occurs only in labeled diagnostics, never production or
acceptance runs. The original full-consumer default 257-row instrumented case
timed out and remains incomplete. Synccheck and racecheck are not resolved by
these findings; no vendor-confirmed root cause or workaround is claimed.

### Subsequent instruction-isolating measurements

`BOUNDARY_OBSERVER.md` records calibrated, synchronized pre-reduction snapshots:
all12,288 expected workspace words are correct before reduction in three runs.
`TMA_STORE_FINDINGS.md` records a no-cuBLAS4-D tensor-store reproduction with
distinct outputs and read-only preimages: all12,288 words physically change to
correct values yet initcheck reports downstream reads uninitialized. Ordinary
stores do not. The same binary's six memcheck and six synccheck cases are clean,
without exclusions. This does not resolve full-runtime internal synccheck failure,
prove original internal-read ordering, or complete32K racecheck. The documented
unsupported tensormap/tcgen05 limitation is consistent with these results but is
not vendor confirmation of this instruction/configuration.

## Native remediation candidate

Latest v4 passes all four scoped integrated tools (including unexcluded
synccheck/racecheck),6890 unchanged eager checks and numerical32K coverage.
This does not resolve production's historical checker failures or establish
full32768 sanitizer coverage. The all-launch guarded upper racecheck timed out
at900.605s without application JSON or a canonical zero-hazard footer. No OOM or
resource-guard stop occurred; upper initcheck/synccheck/memcheck were not launched.
Owned descendants were independently checked absent and the container stopped;
production binaries and pinned libraries retain their original hashes.
the candidate also measures3.445809175593137x slower decode in balanced warmed
testing and is not promoted. Frozen v4 sources/results and limitations are in
`WMMA_REMEDIATION.md`.

An isolated WMMA linear backend now passes6890 original exact eager checks, nine
exact context fixtures through4096, and the original four-token request under
initcheck, unexcluded synccheck and memcheck. It reproduces the measured split-K
arithmetic using ordinary stores instead of preinitializing vendor scratch. See
`WMMA_REMEDIATION.md`. Production is not promoted; original production blockers
and full32K racecheck remain open. The historical vendor probe matrix below still
fails and is not relabeled clean because a different backend works.

## Initcheck sanitizer-only version/shape matrix

`tools/check_lfm3_initcheck_matrix.py` holds the checkpoint-free probe binary,
resolved cuBLAS/runtime libraries and driver fixed, changing only sanitizer and
shape. It tests257x2048,1x128,257x128 and1x2048 with CUDA-fill, intentionally
missing-write, and GEMM producers. The unchanged probe checks **128 sampled
values**, not every output. Nine host classification counterexamples prevent
numeric failure, timeout or an unrelated library error from being mislabeled as
the established downstream tracking reproduction.

Official standalone packages were verified against redistribution manifests:

-13.4.92 from CUDA13.4.2 (2026-09-16), product Compute Sanitizer2026.3.0.0.
-13.5.25 from CUDA13.5.0 (2026-11-04), product Compute Sanitizer2026.4.0.0.

Only standalone tools were extracted under `build/`; no driver, toolkit or
serving-library replacement was made. Archive and manifest hashes are preserved
in `validation/initcheck-version-shapes-v2/`. Official manifests:
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.2.json
and https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json.

Measured versions2026.2/2026.3/2026.4 yield:

-12/12 CUDA-fill controls clean.
-12/12 intentional missing-write controls detected.
-9/12 GEMM cases reproduce uninitialized reads in downstream `read_output` despite
  correct sampled values (three shapes per tool).
-3/12 GEMM cases (1x2048, all tools) report an uninitialized read inside
  `cublasLt::splitKreduce_kernel`. This is a **distinct unresolved diagnostic**, not
  automatically the same tracking bug, and the strict matrix remains failed.

cuBLAS reports130100. Resolved serving libraries are pinned by SHA256 in the
matrix summary: cudart `7bdba2b5...`, cuBLAS `7761d260...`, cuBLASLt `3839e12c...`.
All GEMM probes printed correct128-sample results; that does not waive either
diagnostic. The raw failed summary is preserved, not changed to a pass.

NVIDIA documents some unsupported `tensormap`/`tcgen05` initcheck accesses, making
an instrumentation gap plausible but **not a confirmed diagnosis here**:
https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations.
The separately documented cuBLAS issue6196059 concerns other async-copy/OOB
behavior; do not equate it to these uninitialized-read reports without confirmation.

Next vendor reproduction should contain the existing checkpoint-free projection
probe, build command, three controls, this shape/version matrix, exact libraries,
Thor/BSP/R580 details and128-sample scope. Ask for a supported Thor tooling pair
and tracking ID, not a suppression. No external report has been published.

## Integrated32K racecheck

The original1200-second monolithic timeout remains open. Its log contains only
the tool banner, so there was no evidence locating the slow launch. The60 small
operator fixtures passing four tools do not replace integrated coverage.

Compute Sanitizer2026.4 `--help` confirms `--dump-kernel-launches`,
`--print-session-details`, `--racecheck-num-workers`, `--launch-skip` and
`--launch-count`. Next diagnostic run must keep the full accepted32K request and
binary, enable launch/session detail, and retain elapsed/CPU/RSS observations.
No launch filtering or force-blocking mode is permitted in the full-run gate.
If partitioning is later required, its launch ledger must cover the union of
**every** original launch and be reported as partitioned coverage, not a
monolithic pass. A single slow launch still requires direct investigation.

The all-launch2026.4 diagnostic in `validation/racecheck32k-diagnostic-v3/`
timed out after300.02 seconds with stable binary/tool/request/reference hashes.
No matching live GPU descendants remained after cleanup. It printed100 launches,
ending at `probabilities_large`, but this is not reliable bottleneck localization:
the default print limit was100. Its150 resource samples contain only the launcher,
so their7.6MB peak cannot be represented as aggregate sanitizer/application RSS.
Process-tree traversal did not observe the reparented children. No CPU/GPU-memory
cause is established. The failed evidence is retained unchanged.

The diagnostic now explicitly uses `--print-limit 0` (reporting only, no reduction
in checking) and the same process-group resource sampler as the batch runner,
including reparented live group members and host MemAvailable/SwapFree. A host
regression verifies observation after the direct parent exits. A short bounded
follow-up is diagnostic only; none of these changes closes racecheck acceptance.

That corrected follow-up is retained in
`validation/racecheck32k-diagnostic-v4/`:60.02-second timeout,29 printed launches,
three observed group members, peak aggregate RSS7.395GiB and minimum host
MemAvailable91.551GiB. The final printed launch is a normalization kernel; the
CUDA target accumulated9652 user and247 system clock ticks by the last58-second
sample. This indicates substantial host CPU work but does not isolate a single
slow launch, GPU-memory cause or sanitizer defect. Hashes stayed stable and group
cleanup completed; independent post-run inspection found no matching live GPU
descendants. Neither the short diagnostic nor the prior300/1200-second timeout
is an acceptance pass. Further tuning should measure analysis-worker counts or
complete-launch partitions with a full coverage ledger, not repeat blind waits.

## Final paired-binary memcheck follow-up

`validation/decode-batch-final-memcheck-v4/` uses final probe `db30e878...` and
strict normal reporting (no launch dump). Scratch **passes zero errors** in15.48
seconds. The unchanged upper-bound normal fixture times out after900.00 seconds;
that case remains open. Group cleanup and independent process inspection confirm
no live matching descendants before subsequent GPU work. Peak observed aggregate
upper RSS is5,105,082,368 bytes and minimum host MemAvailable100,019,904,512 bytes;
neither value establishes a GPU-memory cause.

The earlier final scratch run with unlimited launch dumps completed the numerical
fixture but returned86 and printed49,754 launch records and49,754 summary errors.
No invalid-access report was established from that log. This appears to be a
reporting/counting interaction, not clean instrumentation evidence. Its strict
failed status remains preserved in
`validation/decode-batch-final-memcheck-scratch-v3/`; the76MB raw log is retained
losslessly as gzip with original/compressed hashes. The subsequent no-dump scratch
zero-error run—not reinterpreting that failed summary—establishes scratch memcheck.

The later long-prefill runtime change has independent unchanged SDPA/eager/grouped
numerical evidence (`LONG_PREFILL_WORK.md`). New probe `345f1ded...` then passes
strict scratch **and upper** memcheck in
`validation/decode-batch-bounded-memcheck-v1/`, upper177.11s with zero errors.
This resolves the standalone upper memcheck gate for the changed program, not by
waiving the original timeout. Full original fixtures and normal reference/cache
checks remain. Integrated32K tools, worker-dispatch instrumentation and initcheck
remain separately required.

## Memory-capped racecheck follow-up

The optimized full32K unbounded attempt was kernel-correlated global-OOM, with
missing application output despite a zero-hazard footer. The fresh hard-capped
temporary-container run in `validation/racecheck32k-guarded-v2/` avoids OOM but
times out at900.04s with only a sanitizer banner and no application JSON. One
analysis worker does not reduce launch coverage, but incomplete execution cannot
establish coverage or race freedom. Hashes stay stable, cleanup is independently
verified, and both temporary containers are stopped. Full acceptance remains
blocked by racecheck, production initcheck and unexcluded synccheck. Further
vendor submission or system/BSP maintenance requires explicit authorization;
neither has been performed. See `RACECHECK_MEMORY.md` for exact resource evidence.
