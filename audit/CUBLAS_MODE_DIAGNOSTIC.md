# Supported cuBLAS mode diagnostic — not acceptance

This new checkpoint-free probe explores documented backend options without a
driver install, runtime edit, output preinitialization, suppression, preload or
unofficial flags. The golden `native/probes/lfm3_projection_probe.cu` is unchanged.
The existing default produces 128 correct nonuniform samples but has downstream
consumer initcheck reports under 2026.2/.3/.4. The 1x2048 internal split-K report
is a separate failure, not evidence of the same consumer-tracking hypothesis.
Neither is vendor-confirmed as a bug; neither is resolved by this diagnostic.

## CLI and supported semantics

`lfm3-projection-modes-probe kernel|missing|gemm ROWS WIDTH MODE`

Rows are 1..257, width is 128 or 2048; all other tokens are rejected.

| Mode | Compute type | Algorithm enum | Handle math |
| --- | --- | --- | --- |
| `default` | `CUBLAS_COMPUTE_32F` | `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | unchanged default |
| `pedantic` | `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUBLAS_GEMM_DEFAULT` | `CUBLAS_PEDANTIC_MATH` |
| `default-algo` | `CUBLAS_COMPUTE_32F` | `CUBLAS_GEMM_DEFAULT` | unchanged default |

All GEMMs retain BF16 A/B/C (`CUDA_R_16BF`), FP32 alpha=1/beta=0,
the same transpose/layout, workspace size and initialization patterns. Default
math reproduces the golden calculation, not necessarily a byte-identical binary.
The first 128 elements are compared **exactly** with
`width * ((i % 13) - 6) / 16`; this is not full-output validation. The CUDA-fill
positive control still expects `width`. The missing-write negative control still
reads uninitialized output and deliberately has no numerical pass. No GEMM or
missing-control output is prefilled, including in pedantic mode.

Official NVIDIA cuBLAS documentation (retrieved 2026-11-04, live v13.5):

- [Compute types](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t):
  `COMPUTE_32F` uses at least 32-bit compute/intermediate precision; pedantic
  prescribes 32-bit arithmetic throughout and disables optimizations such as 3M.
- [Math modes](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t):
  pedantic requests prescribed precision/standardized arithmetic for robustness,
  testing and debugging, and may reduce performance.
- [Tensor Core usage](https://docs.nvidia.com/cuda/cublas/index.html#tensor-core-usage):
  cuBLAS may automatically select Tensor Cores unless disabled by pedantic modes.
- [Algorithm enum](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmalgo-t):
  the enum has no effect on sm_80 and newer; `DEFAULT` means heuristic selection,
  **not forced SIMT**. `DEFAULT_TENSOR_OP` is deprecated but retained solely for
  the golden default comparison. No kernel selection is promised by default-algo.
- [GemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex):
  datatype/compute combinations and execution support are checked by cuBLAS.
  An unsupported combination or any API/CUDA failure is a nonzero failed case,
  never a clean result; no fallback retry is attempted.

## Build and evidence boundaries

The parent added an excluded diagnostic CMake target separately. Compile standalone with a local CUDA 13 toolkit,
without running the resulting executable:

```sh
/usr/local/cuda-13.5/bin/nvcc -std=c++17 -arch=sm_80 native/probes/lfm3_projection_modes_probe.cu -lcublas -o /tmp/user/1000/opencode/lfm3-projection-modes-probe
python -m pytest -q -p no:cacheprovider tests/test_lfm3_projection_modes_contract.py
```

The build architecture above is only a local compile check, not the Thor target.
The probe prints the exact compiler version, CUDA runtime/header build version,
cuBLAS header build components, requested mode/compute/algo, actual handle math
and `cublasGetVersion` library version. It prints configuration before allocations
and library version before math setup, so unsupported modes retain identity.
Record binary/source hashes, resolved library hashes and sanitizer versions
externally; these printed versions alone do not identify the loaded files.
Host source-contract tests neither launch CUDA nor establish native integration.
Local host validation: the CUDA 13.5 standalone compile above succeeded, and all
four source-contract tests passed. The binary was not executed.

## Bounded two-library matrix (GPU owner only; not executed here)

After notification that the parent's frozen soak and paired runtime numerical /
memcheck validation have finished, acquire exclusive GPU ownership. Use the
**same CUDA 13.0-built probe** with the current serving cudart for both libraries.
The new runner does not build, install, download or modify any library. From the
remote repository root, the exact planned command is:

```sh
python tools/check_lfm3_cublas_modes.py \
  --sanitizer /workspace/liquid-CUDA/build/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer \
  --probe /workspace/liquid-CUDA/build/cmake-bounded-prefill-v1/lfm3-projection-modes-probe \
  --library-dir /opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib \
  --library-dir /workspace/liquid-CUDA/build/cublas-13.10.0.20-diagnostic/libcublas-linux-sbsa-13.10.0.20-archive/lib \
  --expected-version 130100 \
  --expected-version 131000 \
  --output /workspace/liquid-CUDA/build/cublas-modes-13.1-vs-13.10-v3 \
  --timeout 60
```

Output must not already exist. The first explicit root is serving cuBLAS
13.1.0.3; the second is the parent's archive-hash-verified isolated 13.10.0.20.
For each root, `LD_LIBRARY_PATH` begins with that selected root, then the first
explicit serving root as fallback, then nonempty inherited entries. No automatic
venv prefix can override the selection. Preload/audit overrides are rejected.
Before any GPU case, owned, bounded `ldd` resolves **both** cuBLAS and cuBLASLt
inside each selected canonical root (including symlink checks), records every
resolved dependency path/hash and requires the same resolved cudart for both.
For versioned cuBLAS filenames, the expected `cublasGetVersion` integer is derived
using NVIDIA's header formula `major*10000 + minor*100 + patch`; an explicit
expectation must agree with that derivation. Optional `--expected-version` flags
must occur exactly twice, in library-root order, as integers in 130000..139999.
For an ABI-only real basename **`libcublas.so.13`**, an explicit expectation is
required; it is recorded as `expected-config-verified-per-case` provenance, **not
an observed API version**. Without it, ABI-only filenames still fail preflight;
arbitrary unrecognized filenames fail even with an expectation. The runner also
records the tool version under each environment.

The parent reported that v1 stopped **before GPU cases** because the serving pip
library's real basename is ABI-only. That failed evidence remains preserved at
`build/cublas-modes-13.1-vs-13.10-v1`; v2 uses a fresh output directory. The
configured expectations above come from the old actual API logs (130100) and the
official 13.10.0.20 archive version (131000), respectively. Every case must still
print the matching actual `LIBRARY cublas=...` value plus matching mode/math and
success-line version. An actual-version mismatch in a control fail-stops after
cleanup even if all resolved library hashes are unchanged. No probe rebuild or
C++ change is needed; the new archive's API version remains unobserved until run.

There are 18 cases per root: three modes × shapes **257x2048 and 1x2048** ×
three producers; 36 total. Every invocation uses `--tool initcheck
--error-exitcode 86 --print-limit 4`, without launch dumps, record-errors,
filters, suppressions or altered initialization. Positive/GEMM clean requires
exit 0, exactly one zero-error summary, matching configuration/library/math
identity and the exact 128-sample success line. Missing-write detection requires
exit 86, a nonzero summary, matching identity and intentional-control line plus
the downstream `read_output` uninitialized read. A wrong numerical result or
mode/version cannot be clean.

Timeouts or failed positive/negative controls stop further GPU cases **after
owned process-group cleanup**. GEMM downstream tracking, internal split-K,
unsupported/API errors and unexpected failures remain distinct blockers while
the remaining cases continue. Raw logs, commands, durations, failed/incomplete
records and source/runner/probe/tool/dependency hashes are retained. Final hashes
and repeated per-root `ldd` checks detect changed files or library symlink targets.
`diagnostic_completed` indicates all 36 cases ran, even if GEMM was blocked;
status `blocked` returns 86, failed controls/timeouts return 1. Only an all-clean
diagnostic returns 0. **Every status keeps `integrated_initcheck_accepted=false`.**
The old matrix runner is not used: it has a different CLI and library selection.

Host-only runner tests cover classification, wrong output/mode/version, root
alias escape, environment precedence, complete/fail-stop fake matrices and real
owned CPU-helper cleanup after exit, timeout and injected start/wait exceptions:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_lfm3_cublas_modes_gate.py tests/test_lfm3_projection_modes_contract.py
```

The host suite additionally checks ABI-only expectations, versioned-file
disagreement, omitted expectations, strict CLI counts/types/ranges and a fake
control-version mismatch with unchanged valid library hashes. No sanitizer
matrix or GPU execution was performed by this agent.
Local revised host result: **43 tests passed**.

The parent subsequently ran v2: the fill control was clean and the missing-write
control correctly produced 128 downstream uninitialized-read errors with exit
86. The old classifier incorrectly counted the informational `124 errors were
not printed` footer as another total and stopped after two cases. That original
failed-classification evidence is preserved unchanged in the v2 directory.
The revised parser accepts only canonical full lines matching
`(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$`. It still requires exactly one
canonical total; missing/truncated totals or multiple totals fail. Informational
print-limit footers neither supply a total nor alter the actual diagnostics.
Regression tests read the original v2 missing-control sanitizer/application logs
and require negative-control detection with all identity/exit/output checks
unchanged; a footer claiming zero unprinted errors alone cannot be clean.
Use the fresh **v3** output command above; no GPU rerun was performed here.
Local canonical-summary regression result: **46 host tests passed**.

Even a numerically correct, initcheck-clean mode is **not acceptance**, nor proof
that the known findings are fixed. Changing runtime math may break the 6,890 eager
bitwise comparisons and performance. No automatic runtime adoption: a separate
review and unchanged full runtime correctness/sanitizer/performance gates are
required. No GPU, SSH or native execution was performed for this host-only work.

## Parent-session completed36-case diagnostic

`validation/cublas-modes-13.1-vs-13.10-v3/` completes all36 cases with stable
source/probe/tool/dependency identities and unchanged serving cudart. Actual
cuBLAS API versions130100 and131000 match the configured expectations. The new
13.10.0.20 archive was downloaded/hash-verified and extracted only under `build/`;
no serving-library, toolkit, driver or global-environment replacement occurred.

- All12 CUDA-fill controls are clean; all12 intentional missing-write controls
  are detected.
- Four pedantic GEMM cases are clean with correct128-sample numerical results.
- Six default/default-algo GEMM cases retain downstream tracking diagnostics;
  two old-library1x2048 cases retain the distinct internal split-K diagnostic.
- The newer library does not clear default-mode initcheck. Pedantic is a supported
  diagnostic lead, not a runtime fix or a waiver of the original bitwise policy.
  An isolated generated-source runtime must pass the original6890 exact checks
  before it can be considered; production math remains unchanged.

V1's ABI-only filename preflight failure and v2's informational-footer parsing
failure are preserved unchanged. V3 fixes the runner, not the GPU findings; it
returns blocked/86 and keeps `integrated_initcheck_accepted:false`.
