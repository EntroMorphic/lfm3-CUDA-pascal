# Adversarial review: LFM3 operator milestone (historical baseline)

**Baseline verdict: do not integrate the model forward pass.** This document records
the original findings, not current open-defect status. Remediation and fresh
verification are tracked in [REMEDIATION.md](REMEDIATION.md). The execution guard stays.

## Confirmed findings

### R1 — Incomplete checkpoint validation (high, before trusting the inspector)

Location: `tools/inspect_checkpoint.py`, `text_schema` and `inspect`.

- Floating-point offsets such as `0.0` pass arithmetic comparisons even though the
  safetensors format requires integer offsets.
- Boolean dimensions compare equal to integer dimensions of one in Python.
- Byte spans are checked only for expected text tensors. A non-text tensor may
  overlap the embedding payload without rejection.
- Duplicate JSON header names are silently collapsed by `json.loads`.
- A config specifying convolution biases can pass even when bias tensors are absent.

All five adversarial acceptance tests failed against the reviewed baseline, meaning invalid input was
accepted. The installed safetensors reader independently rejects the floating
offset, boolean shape, and cross-modality overlap fixtures. A valid control fixture
loads successfully in that reader. Duplicate-name rejection is a stricter integrity
policy; no claim about the reader's duplicate-name behavior is made here.

Required remediation: strict JSON decoding; exact integer/non-boolean shape and
offset types; whole-file layout validation; shared/synchronized configuration
contracts. Keep the original valid checkpoint check and malformed-file controls.

### R2 — Invalid attention positions accepted (high, before runtime use)

Location: `native/include/lfm/lfm3_ops.cuh`, `attention_decode`, and the test ABI.

Positions -1, capacity, and capacity+2 return success. Clamping `end` to capacity
does not establish a valid absolute cache index; negative positions can produce a
zero result and out-of-range positive positions silently use the wrong range.

Required remediation: enforce `0 <= position < capacity` for every active row before
launch, or expose and check a device error/status contract. Overflow-safe position
arithmetic and invalid-position regressions are required. Returning plausible
numeric output is not a substitute for rejection.

### R3 — No checked tensor boundary (high if reused beyond the test fixture)

Location: `tools/check_lfm3_ops.py::Bridge.call` and `native/tests/lfm3_ops_bridge.cu`.

The C ABI intentionally takes raw pointers; its caller must uphold layout/type/
length contracts. The Python caller does not enforce those contracts. Float32
queries, noncontiguous queries, and int64 positions all reach kernels and return
success. The ABI's integer metadata also lacks comprehensive checked-product/grid
validation. These are test-interface deficiencies, not evidence that the guarded
native executable currently exposes this API to user requests.

Required remediation: a typed, contiguous, same-device, same-stream checked wrapper;
exact shape/size checks; explicit aliasing rules; checked launch arithmetic; negative
tests. A raw-pointer test bridge must not become the production integration boundary.

### R4 — Convolution arithmetic contract differs under cancellation (high for parity)

Location: `native/include/lfm/lfm3_ops.cuh::convolution7`.

Finite BF16 inputs with mixed magnitudes (2^25 or 2^30 and unit terms) reproduce
native/checkpoint-Triton-decode differences of up to 2.0. The native code sums seven
products sequentially; the checkpoint decode uses a Triton reduction tree. Ordinary
random fixtures did not reveal this. These are synthetic stress inputs, not a
demonstrated failure with actual model activations.

The checkpoint prefill and decode implementations use different reduction
structures themselves. The reproduction records both to avoid mistaking upstream
arithmetic differences for a change in the mathematical convolution definition.

Required remediation: define the intended prefill/decode numerical contract, then
match it or explicitly bound deviations on real checkpoint activations and logits.
Do not fix this merely by relaxing the current absolute tolerance.

## What survived the attack

- Broader rotary probes at 512 randomized positions in [0,32767] matched exactly
  for both global partial RoPE and local full RoPE in the observed fixtures.
- The 423 original checks remain useful bounded observations; they do not cover
  malformed inputs, arbitrary magnitudes, the full forward pass, or SDPA parity.
- The explicit execution guard prevents these isolated operators from silently
  becoming an allegedly working LFM3 model runtime.

## Additional gates

- CUDA memcheck: exit 0, **0 errors**, on the 141-check seed-42 ordinary fixtures.
- CUDA racecheck: exit 0, **0 errors and 0 warnings**, on the same fixtures. This
  does not validate incorrect input contracts, adversarial numerics, or the full
  model; it only credits the inspected execution paths. Raw logs are retained.
- Real checkpoint activation traces and teacher-forced logits: not implemented.
- Prefill/decode and batch isolation with real weights: not validated.
- SDPA versus eager arithmetic policy: not settled. Current attention test targets
  eager BF16-materialization semantics, not the working sandbox's SDPA backend.
- Actual inference, generation quality and performance: not established.
- LFM2 full-model GPU regression: not run for the copied tree; config tests and
  successful compilation are not a substitute.

## Reproduction

```bash
# All five regressions must pass on the remediated candidate:
PYTHONPATH=tests python3 -m pytest audit/test_host_contracts.py -q --tb=short
```

On Thor, use `audit/redteam_gpu.py` with the same `--library`, `--checkpoint` and
`--output` arguments as the ordinary operator harness. Results are retained in
`validation/redteam-gpu.json` (historical baseline). Fixes should be a separate, reviewable step and
must turn the failing acceptance tests into passes without weakening assertions.
