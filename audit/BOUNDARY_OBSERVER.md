# Producer/reducer boundary observation — diagnostic only

## Question

The prior before/after experiment established physical workspace changes and
post-completion initialization reports. It did not observe the workspace before
the split-K reduction. This experiment inserts a read-only, stream-ordered copy
at launch callbacks to observe that boundary explicitly.

This changes synchronization and timing. It is a separate custom SDK observer,
not builtin initcheck, not a production remedy and not acceptance evidence.

## Documented mechanism and constraints

Source: NVIDIA Compute Sanitizer API guide,
https://docs.nvidia.com/compute-sanitizer/SanitizerApiGuide/index.html
(retrieved2026-11-05; page reports2026.4.0).

- Callback code normally must not call CUDA APIs. This observer calls **no CUDA
  APIs** and uses `sanitizerMemcpyDeviceToHost`, explicitly documented as safe
  within callbacks and returning after the copy completes.
- The callback uses the launch's `Sanitizer_StreamHandle`, not a guessed stream.
- `SANITIZER_CBID_LAUNCH_BEGIN` is the grid-initiation event. Calibration tests
  its effective before/after boundary rather than assuming it is GPU completion.
- The API allows one subscriber. Builtin initcheck is not run concurrently;
  no absence of checker output in the observer process can mean initcheck clean.
- `LD_PRELOAD` is used only in the owned diagnostic child's environment to load
  the SDK client. No API interposition, production preload or global environment
  modification occurs. The client registers through documented SDK APIs and
  does not patch code or write device memory.

Injection must occur at the **final target exec**, not in the ownership launcher's
environment. The first calibration caught that distinction: preloading the Python
signal-unmask launcher created the observer log before exec; the CUDA target could
not create it again and had no valid observation. Its failed evidence and frozen
source are retained in `validation/boundary-observer-v1/`. The corrected runner
executes an absolute `env` utility to install the preload only for the final
binary. A host constructor-marker test verifies exactly one load in the intended
target, not the ownership launcher. Calibration assertions were not weakened.

The NVIDIA sample API usage consulted was `MemoryTracker/MemoryTracker.cpp` at
https://github.com/NVIDIA/compute-sanitizer-samples/tree/1e1fab8e25148127f5dec10210ac70b799d1c36f
The observer implementation is independent; SDK headers/libraries are external
dependencies, not copied into this repository. The dependency notice is below.

## Calibration before cuBLAS

A separate CUDA control allocates one exact8,519,680-byte workspace and two
49,152-byte destinations. It seeds them with A5/5A/3C respectively. Three ordinary
kernels then fill scratch with known FP32 values, double them into output and
copy output to sink. Every final output is checked.

The observer must capture:

1. Before producer: the untouched A5 scratch.
2. Before reduction: the complete known producer result **and** still-poisoned5A
   reduction output. This rejects an observer firing after reduction execution.
3. Before consumer: correct reduction output but still-poisoned3C sink. This
   rejects an observer firing after consumer execution.

A missing-producer negative control must fail the application's complete numerical
comparison and be rejected by the observer's event-order/count checks. Missing
callbacks, duplicate allocation identities, failed memory APIs, nondefault streams,
unexpected kernels, incomplete files or timeout are failures, never observations.
CPU model tests cover state handling only; actual CUDA timing requires calibration.

## GEMM observation

Only after both controls are validated, run the existing asymmetric1-row GEMM
with unchanged default compute/tensor enum/default handle math. The observer
records snapshots before the exact NVJET producer, before split-K reduction,
and before final consumer. All final producer/consumer outputs remain checked
against the independent CPU oracle. Addresses are matched to the probe's own
allocation metadata. The expected workspace partials are derived from integer
dot products, not GPU outputs. No scratch or output initialization is introduced.

Module-load callbacks can copy selected CUBIN images while the SDK-provided data
is valid. Launch records bind the actual CUmodule to those images. Capture is
bounded to16 selected images,128MiB per image and256MiB total. Actual images and
hashes remain local; no vendor code is submitted externally. If images are not
available, boundary observations and missing disassembly evidence remain distinct.
Finding instructions in a captured image alone does not establish dynamic execution.

## Build/run and provenance

The runtime container image is
`sha256:2a9b0054deac9e70b38e31bfdc0b789a4918af50a07e9832263c47bcbc5b08da`.
It does not contain the compiler toolkit. The separately measured compiler image
is `sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f`
(the historical r38.3 CUDA13.0.88/GCC13.3 build tag). These identities must not
be conflated with the host BSP or each other.

Configure the separate diagnostic build with `LIQUID_LFM3_OPS_TESTS=ON` and
`LIQUID_SANITIZER_SDK` pointing to the pinned2026.4 SDK. Build only
`lfm3-boundary-control` and `lfm3-boundary-observer`.

`tools/check_lfm3_launch_boundary.py` requires the observer/control/probe/SDK/library
paths and a new output directory. It first validates calibration, then performs
up to three bounded GEMM observations. All dependency, source, input and tool
identities are pinned. Owned-process cleanup and termination unwinding apply to
every child. The temporary container retains a48GiB hard cap and no extra swap.

Measured frozen sources use the earlier name `check_lfm3_boundary.py`. Final review
found that name belonged to an existing tensor-wrapper negative-test runner. That
original is restored byte-for-byte; the SDK runner is renamed as above. Historical
commands and hashes retain their actual measurement-time names.

## SDK dependency notice

NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE AGREEMENT,
NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE LICENSED DELIVERABLES
FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY
KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED DELIVERABLES,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS
FOR A PARTICULAR PURPOSE. NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY
IN THE LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL,
INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THESE LICENSED DELIVERABLES.

U.S. Government End Users. These Licensed Deliverables are a "commercial item" as
that term is defined at48 C.F.R.2.101 (OCT1995), consisting of "commercial computer
software" and "commercial computer software documentation" as such terms are used
in48 C.F.R.12.212 (SEPT1995) and is provided to the U.S. Government only as a
commercial end item. Consistent with48 C.F.R.12.212 and48 C.F.R.227.7202-1 through
227.7202-4 (JUNE1995), all U.S. Government End Users acquire the Licensed
Deliverables with only those rights set forth herein.

## Completed results

`validation/boundary-observer-v2/` completed GPU positive calibration, the
missing-producer negative control, and three GEMM observations. All six calibration
snapshots match their complete expected bytes, including still-poisoned reduction
output and consumer sink. The negative control reports12,288 incorrect outputs
and is independently rejected by observer order checks.

In each GEMM observation all12,288 FP32 scratch words match the independent oracle
**before reduction**, with23,222 bytes changed since before producer. Zero scratch
bytes change between before-reduction and before-consumer. All complete final
outputs match. Launch geometry, allocation identities and start/end hashes validate.

Actual launched CUBIN hashes are identical across the three runs: producer
`99fed3a84c356fb8b96a32bff51df0808c49131f033de9d139de976b5ef54892`, reducer
`595b0ad14d1aa3bdf2a6e2b13baff6691a203558bf58185ba5222577b30031c5`.
Images remain remote; static disassembly/commands/hash metadata are retained in
`validation/boundary-disassembly-v1/`. Producer0x0c60 is `UTMASTG.4D`;
reported reducer0x1470 is `LDG.E.CONSTANT`. Static instructions are not measured
execution counts. See `TMA_STORE_FINDINGS.md` for the no-cuBLAS follow-up.

### Interpretation limit

The experiment cannot establish an unchanged
initcheck result or exclude every synchronization-sensitive defect in the original
execution. Its direct question is whether correct scratch is present before the
reduction **under the explicitly introduced boundary synchronization**.
