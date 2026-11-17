# LFM3 operator remediation ledger

Scope: the isolated operator/configuration/checkpoint milestone. The model forward
pass remains disabled. This is not a full native inference acceptance certificate.

## Changes and staged verification

| Finding | Remediation | Verification |
| --- | --- | --- |
| R1 checkpoint validation | Strict duplicate/nonfinite/UTF-8 JSON; typed config and dimensions; complete payload range/layout validation across modalities; explicit unsupported-bias rejection | All five original failures pass. Expanded host suite: 110 passed. Real `model.safetensors` remains valid: 286 text + 379 other tensors. |
| R2 invalid positions | Python active-row range validation plus independent synchronous C-ABI validation; checked launch arithmetic | Invalid negative, capacity, oversized and INT_MAX positions rejected without output writes. |
| R3 unchecked tensors | Mandatory checked Python wrapper for shape, dtype, contiguity, device, gradients, writable overlap, scalar and launch limits; raw pointer/device guards | 48 boundary checks, including producer-stream readiness, nondefault streams, valid calls after rejection, and graph-capture rejection without invalidating the surrounding graph. |
| R4 cancellation | Explicit separate native prefill (sequential) and decode (8-wide butterfly) contracts matching the pinned checkpoint | Original 2^25/2^30 cancellation cases now exact. 336 exact comparisons across lengths 1,2,6,7,8,17,257; batch 1/3; widths 128/2048; ordinary/mixed-magnitude fixtures, cache handoff and reset. |

No assertion was relaxed to close these findings. The new contiguous output
contract caught a test-fixture allocation preserving a transposed stride; the
fixture was corrected rather than weakening the boundary.

The original numerical sweep was repeated with three seeds: 145 checks each,
435 total. Both the direct nvcc test bridge and the CMake Release bridge compile;
the CMake build also passes operator, boundary and convolution suites. These
measurements are correctness checks, not performance claims.

## Sanitizer verification

The initial remediated sweep passed memcheck and racecheck. Synccheck then failed
with **48 internal instrumentation errors in NVIDIA NVJET reference GEMMs**. Both
Compute Sanitizer 2025.3.0 and 2025.3.1 showed that failure. The failed logs are
retained; they are not credited as passes.

The sanitizer harness now uses a CPU eager attention oracle, avoiding those
third-party reference GEMM launches while leaving every custom CUDA kernel
instrumented. Ordinary GPU-eager numerical regressions remain separate. No kernel
exclusion filter or sanitizer-error suppression is used.

Final source-frozen matrix (Compute Sanitizer 2025.3.1, raw logs prefixed
`remediated-final`): **all 12 runs passed**.

| Tool | Boundary (48 checks) | Convolution (336 checks) | Operators (145 checks) |
| --- | --- | --- | --- |
| memcheck | 0 errors | 0 errors | 0 errors |
| racecheck | 0 errors/warnings | 0 errors/warnings | 0 errors/warnings |
| synccheck | 0 errors | 0 errors | 0 errors |
| initcheck | 0 errors | 0 errors | 0 errors |

All final reports identify the same direct-nvcc candidate library SHA-256:
`734ec4ba25d3d924e7fcc86d95139b65170eb517a9a32d3c76c0513f966501b3`.
`validation/remediation-summary.json` records the matrix and reviewed source
hashes. `validation/remediated-final-redteam.json` has zero remaining findings in
the original adversarial probes. Historical failure logs remain unchanged.

**Disposition: R1–R4 closed for this bounded milestone.** The execution-guard
regression still passes: the compiled executable rejects LFM3 before attempting
unimplemented forward execution. No blanket 100% correctness or full-model parity
claim follows from these results.

## Remaining model-integration gates

- Native weight mapping/forward integration and actual checkpoint activation/logit
  parity are not established. The executable still rejects LFM3 before allocation.
- SDPA parity is not established by eager-reference operator tests.
- Attention remains a bounded correctness prototype with a 4096-slot contiguous
  cache, not a full-context optimized implementation.
- Packed/padded/incremental convolution prefill is unsupported and must not be
  inferred from the empty-cache prefill API.
- Full LFM2 generation regressions, full native Omni, production concurrency and
  performance are not certified by this milestone.

Original source files remain hash-verified and unchanged. Historical failures and
pre-remediation binaries/reports are retained separately from candidate evidence.
