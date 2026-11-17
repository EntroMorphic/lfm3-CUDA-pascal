# Isolated PEDANTIC runtime diagnostic (not accepted)

The reported 36-case cuBLAS matrix v3 found downstream/split-K failures with
130100 default policy and downstream failures with 131000 default policy;
small PEDANTIC GEMMs were clean on both versions. This motivates an isolated
runtime experiment, **not** a production policy change or acceptance claim.

## Generate and build

From the project root, using a **new, nonexistent** output directory:

```sh
python3 tools/prepare_lfm3_pedantic_diagnostic.py --output build/pedantic-diagnostic-v1
```

Transfer that generated directory (not the checkpoint or repository) to
`/workspace/liquid-CUDA/build/pedantic-diagnostic-v1` on Thor. Inside the chosen
CUDA build image, run:

```sh
cmake -S /workspace/liquid-CUDA/build/pedantic-diagnostic-v1 -B /workspace/liquid-CUDA/build/pedantic-diagnostic-v1/build -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/liquid-CUDA/build/pedantic-diagnostic-v1/build --parallel --target liquid-lfm3 liquid-lfm3-worker lfm3-decode-batch-probe
```

The default includes all three targets; `--without-probe` omits the paired probe.
Target names are familiar but every output is confined to this separate build.
No production CMake, sources, binaries, defaults, CPU/legacy paths, or math policy
are modified. No GPU execution is performed by the generator or its tests.

## Pinned scope and policy

The mirror contains only the four named LFM3 runtime sources (including
`safetensors.cpp`), the optional decode-batch probe, project `native/include`
`.hpp`/`.cuh` headers, MIT license, and PyTorch license. No arbitrary file copy,
model, checkpoint, environment, credentials, or data directory is included.
Symlink inputs, including symlinked header directories, are rejected.
Original names preserve the worker/probe's relative runtime includes.

The generator requires exactly three `CUBLAS_COMPUTE_32F` and three
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` tokens in main, two of each in the batch header,
and one `CUBLAS_DEFAULT_MATH` token in main. Exact-word replacements select
`CUBLAS_COMPUTE_32F_PEDANTIC`, `CUBLAS_GEMM_DEFAULT`, and `CUBLAS_PEDANTIC_MATH`.
The existing `finish_handle_setup` wrapper, noexcept destroy callback, and
exception cleanup header remain intact. All other bytes, including the current
mask planner/optimized 4096 guard, remain unchanged. Production's numerically
validated default candidate is not replaced by this experiment.

The portable CMake file uses mirror-relative paths, CUDA architecture 110,
C++20/CUDA20, and the originals' CUDA `-O3 --fmad=false` flags. It does not add
`--expt-relaxed-constexpr` or any fast-math option. Manifest build metadata
records this explicitly. The manifest pins source origins, original/copied/
modified SHA-256 values, exact replacement counts, original CMake hash, and
generated CMake hash. `copied_sha256` denotes the byte-identical input before
the in-memory diagnostic transformation; `modified_sha256` is the emitted file.
The manifest is provenance, not a validation certificate.

## Required next gate: unchanged full eager acceptance

`manifest.json` always starts with `diagnostic_only: true`,
`full_acceptance: false`, and `status: generated_not_built_or_validated`.
**All 6890 eager checks using the ORIGINAL `tools/check_lfm3_forward.py` and
unchanged exact gates are REQUIRED before considering any runtime use.**
Point `--binary` at this mirror's `build/liquid-lfm3`, use the original
checkpoint, and retain the complete output in a new evidence directory.
The parent session owns checkpoint/image selection and that execution.
Small-GEMM sanitizer cleanliness is not full-runtime or numerical acceptance.

Any bitwise mismatch, unsupported math API, build failure, or runtime failure
is a failed diagnostic: **keep the failed evidence; do not relax gates, replace
goldens, silently fall back, or adopt the diagnostic runtime.** Even successful
eager checks do not by themselves establish worker/paired/sanitizer acceptance.
Do not overwrite the initial manifest to claim acceptance; record gate results
separately with logs and hashes.

CPU-only generator regression tests:

```sh
python3 -m pytest -q tests/test_lfm3_pedantic_diagnostic.py
```

## Parent execution: rejected without adoption

The isolated single binary `a723ebd5...` built on Thor with CUDA13.0.88/GCC13.3.
The unchanged original eager gate stopped at baseline:4,205 of4,240 baseline
activation/logit comparisons failed bitwise equality, beginning at convolution
projection output. Teacher argmax happened to match, which does not satisfy the
required exact contract. The remaining fixtures were not credited as passed.

Evidence: `validation/forward-pedantic-diagnostic-v1/` and its separate mirror
manifest. `require_exact:true` and every original threshold/golden remain intact.
No retry with weaker assertions, fallback or runtime adoption occurred. Production
single remains `e436191d...` with its6890 exact checks. Clean small pedantic GEMMs
therefore do **not** resolve production initcheck under the required policy.
