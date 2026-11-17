# Checkpoint-free cuBLAS/initcheck reproduction packet

## Status and scope

**Unresolved; not vendor-confirmed, not acceptance, no waiver.** The preserved
36-case summary is `failed-controls-or-tool`, with integrated initcheck acceptance
false. Compute Sanitizer 2026.2/2026.3/2026.4 each ran the same four shapes and
three producers: CUDA fill, intentionally missing write, and cuBLAS GEMM.
There are 12 clean fill controls, 12 detected missing-write controls, nine GEMM
downstream `read_output` reports and three distinct internal
`cublasLt::splitKreduce_kernel` reports (GEMM 1x2048). Do not conflate the latter
with the downstream tracking hypothesis or classify them as passing controls.

The unchanged source checks exactly the first **128 BF16 output elements**,
not the entire output of size rows * 3 * width. GEMM uses nonuniform weights,
constant 0.5 inputs, beta=0 and analytically expected values
`width * ((i % 13) - 6) / 16`. The fill control expects `width`; the missing
control deliberately has no producer or numerical pass. No output prezero,
suppression, changed checking or model/checkpoint is involved.

## Build and verify a packet (host only)

From the repository root, using an existing destination parent:

```sh
python tools/check_lfm3_sanitizer_repro.py --output /tmp/user/1000/opencode/vendor-initcheck-packet
python tools/check_lfm3_sanitizer_repro.py --verify /tmp/user/1000/opencode/vendor-initcheck-packet
```

Output must not exist. The builder does not execute CUDA, SSH, upload, download,
read environment secrets, follow recorded library paths or copy arbitrary files.
It copies exactly 80 pinned source/evidence files plus this note, the builder and
a minimal standalone CMake recipe. Existing package download metadata carries
redistribution manifest/archive hashes; library, original probe binary and tool
hashes remain in the raw summary. Binaries/libraries/archives are not bundled.
Absolute paths inside raw logs/commands are inert historical provenance only.

`packet-manifest.json` hashes every copied/generated file. Verification rejects
modified or absent controls, changed summary/status, extra files, traversal and
symlinks. A hard-coded aggregate anchor also rejects replacement of the pinned
evidence even if someone recomputes its packet manifest. This is a local review
anchor, not a signature or independent attestation; a party able to replace both
the verifier and anchor is outside this integrity model. Zero exit status means
**packet integrity only**. Failed sanitizer status is deliberately preserved.

## Exact reproduction commands (GPU owner only; not run by builder)

On the original Thor installation, from the packet root, with CMake, CUDA and
Python available, provision the original libraries/tool packages locally first:

```sh
cmake -S . -B build/repro
cmake --build build/repro --target lfm3-projection-probe -j2
python tools/check_lfm3_initcheck_matrix.py \
  --sanitizer /workspace/liquid-CUDA/build/sanitizer-13.3.27/cuda_sanitizer_api-linux-sbsa-13.3.27-archive/compute-sanitizer/compute-sanitizer \
  --sanitizer /workspace/liquid-CUDA/build/sanitizer-13.4.92/cuda_sanitizer_api-linux-sbsa-13.4.92-archive/compute-sanitizer/compute-sanitizer \
  --sanitizer /workspace/liquid-CUDA/build/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer \
  --probe "$PWD/build/repro/lfm3-projection-probe" \
  --output "$PWD/build/repro-matrix" --timeout 120
```

The runner uses owned process groups and unconditional bounded cleanup. Each
case has a 120-second deadline (36 cases), no launch/kernel filter, and unchanged
initcheck settings with error exit code 86. Output must be new. Each invocation
and return code is recorded. Inspect all three producers, all shapes and all tool
versions; an absent/misdetected negative control invalidates the experiment.
The historical matrix is expected to return 1, not a successful sanitizer gate.
Rebuilding does not promise a byte-identical binary: compare new compiler/build,
resolved library and tool identities before making same-environment claims.

## Available environment facts and missing vendor prerequisites

Reported environment: NVIDIA Thor, R580 driver family, CUDA 13.0 installation;
source build target sm_110. cuBLAS reports 130100. Exact resolved libcudart,
libcublas and libcublasLt paths/SHA256 and tool version/build strings/SHA256
are preserved in `validation/initcheck-version-shapes-v2/summary.json`.
2026.4 is standalone package 13.5.25, not a toolkit/driver upgrade. The copied
evidence does **not** establish the exact BSP, driver patch, OS image, compiler
version or officially supported Thor/BSP/tool pair. The GPU owner must obtain
those facts locally for vendor triage; do not invent them or attach unrestricted
environment dumps. Request vendor-supported combinations and a tracking ID.

The parent session subsequently collected narrow local facts in
`validation/thor-supported-config-facts-v1.json`: host `/etc/nv_tegra_release`
reports **R38 revision2.2**, GCID42205042, kernel6.8.12-tegra. The open-kernel
module reports a `TempVersion` release build rather than a precise numeric driver
patch; do not infer that patch from the historical R580-family description.
The build image (named r38.3) separately reports CUDA compiler **13.0.88**.
The serving container has no `/usr/local/cuda/bin/nvcc`; that failed query is
preserved. Image naming is not evidence of the running host BSP. These facts
narrow the vendor question but do not establish a supported BSP/tool pairing.

Documented unsupported tensormap/tcgen05 accesses make an instrumentation gap
plausible, not a confirmed explanation of these traces. Compatibility cannot be
inferred solely from CUDA minor compatibility. No vendor has confirmed a bug or
approved acceptance. Keep internal split-K diagnostics separately actionable.

## Next integrated diagnostics (outside this checkpoint-free packet)

The full 32768-token prefill fixture, binary, request/reference hashes and exact
result checks must remain unchanged. Do not copy checkpoint/request material into
this packet. The GPU owner can use `tools/check_lfm3_racecheck_diagnostic.py` in
the main repository with explicit sanitizer/binary/checkpoint/request/reference
paths, a fresh output directory, bounded timeout and worker count. Existing
`--print-limit 0`, launch/session logs, owned-group CPU/RSS sampling and cleanup
are already fixed: do not present them as new mitigations. Compare bounded
worker-count experiments (including default 0) with elapsed time, aggregate group
RSS, CPU ticks and host memory; neither the last printed launch nor a timeout
locates a bottleneck or constitutes a pass. No filters, force-blocking launches,
suppression, selected-block instrumentation or reduced hazard tracking.

For upper-bound memcheck diagnostics, retain the identical maximum-capacity
fixtures and exact numerical/output checks, collect launch/session logs and
group resources, and set explicit per-fixture wall-clock deadlines. Run all
required upper-bound prefill/decode/batch fixtures, not a smaller proxy; preserve
all failing/timeout logs and before/after hashes. Changing reporting verbosity
does not change coverage. Any partitioned racecheck follow-up needs an explicit
ledger whose union covers every original launch, no gaps, with matching fixture
and binary/tool hashes. Label it partitioned evidence, never a monolithic pass.
These are diagnostic plans, not performed work or closure of acceptance blockers.
