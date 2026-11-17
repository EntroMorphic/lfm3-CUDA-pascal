# Minimal TMA store: initialization-report discrepancy

## Conclusion and limits

A no-cuBLAS program reproduces correct, physically changed global output being
reported uninitialized after a4-D tensor store. Ordinary stores in the same binary
do not produce downstream initialization reports. This is consistent with NVIDIA's
documented unsupported-instruction coverage, not vendor confirmation of the exact
checker mechanism or a production sanitizer pass.

The full-runtime synccheck internal-launch failure is **not reproduced** by this
minimal program. Full32K racecheck remains incomplete. Neither is resolved here.

## Program and controls

`native/probes/lfm3_tma_store_probe.cu` links cudart and the CUDA driver, not cuBLAS.
It uses96 blocks,128 threads,12,288 FP32 words, no output initialization, and rank4
dimensions[128,1,1,96], strides[512,512,512], box[128,1,1,1]. Shared writes precede
an async-proxy fence and block barrier. One thread issues the tensor store,
commits it and waits with **`cp.async.bulk.wait_group 0`**, not a read-only wait.
A second block barrier precedes exit. The same-stream consumer copies every word.
Complete producer and consumer outputs are downloaded and checked bitwise.

Binary SHA256: `034ac1babdd2fca9987921ac78ea6c4f0a4dc8c0ed6e8330b9da5b1bbf588d74`.
Static disassembly shows `UTMASTG.4D` at0x0210, `UTMACMDFLUSH` at0x0220, and
ordinary `STG.E` paths. Metadata binds the binary hash; static code is not an
instruction-execution counter.

## Measured observations

Evidence directories are under `validation/`.

| Directory | Ordinary stores |4-D TMA stores |
| --- | --- | --- |
| `tma-store-minimal-v1` |3 runs, zero initcheck errors |3 runs,13,824 errors each |
| `tma-store-distinct-v2` |bias1025/1026/1027, zero errors |bias1028/1029/1030,13,824 errors each |
| `tma-store-preimage-v3` |bias1025,1,536 deliberate preimage errors only |bias1026,15,360 errors |
| `tma-store-memcheck-v4` |3 runs, zero errors |3 runs, zero errors |
| `tma-store-synccheck-v4` |3 runs, zero errors |3 runs, zero errors |

Every non-missing-producer case matches all12,288 producer words and all12,288
consumer words. The initcheck missing-producer control is detected in each matrix.
It is intentionally omitted from memcheck/synccheck: those do not detect
uninitialized reads. No exclusions or suppressions are used. Memcheck/synccheck
use the **same binary hash** as distinct/preimage initcheck, without recompilation.

Both ordinary and TMA preimages contain49,152 zero bytes. After each producer,
36,532 bytes and **all12,288 FP32 words** change to correct nonzero values. The
missing-producer control changes none. TMA's15,360 canonical reports comprise
12,288 consumer global reads and3,072 host-API reports from preimage and
post-completion downloads. Ordinary stores report only1,536 host-API accesses
from the deliberate uninitialized preimage download. Preimage runs are altered
diagnostics, not clean acceptance attempts. Complete logs are retained losslessly;
the strict distinct-value runs have no preimage read.

The runner rejects incomplete outputs, numerical mismatches, internal checker
errors, missing canonical footers, timeout and identity changes. Its
`measurement-completed` status means completed measurement, **not** zero errors or
acceptance. Earlier runner versions/failures are preserved. All completed matrices
have stable start/end hashes and owned-process cleanup records.

## Official limitation, accurately scoped

NVIDIA Compute Sanitizer release notes, version2026.4.0, last updated Nov04,2026,
section **Known Limitations**:

> Some `tensormap` and `tcgen05` instructions accessing global memory are not
> currently supported, which may result in initcheck false positives.

https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations

Raw HTML/retrieval/hash metadata: `validation/tma-store-documentation-v1/`.
The page separately says2026.2 added **memcheck TMA-load** support and2025.4 added
initcheck support for `cp.async.bulk`/`cp.async.red`. Neither establishes complete
tensor-store tracking. The limitation does not name this exact `UTMASTG.4D`, Thor
configuration or cuBLAS kernel. Do not claim NVIDIA confirmed this reproduction or
that every original report has one cause. Public headers name a TMA-store patch
hook but omit its callback typedef; no guessed ABI was used.

## Remaining gates and cleanup

No scratch initialization, pedantic math, production preload, suppression, golden
change, driver/BSP replacement or waiver was adopted. See `BOUNDARY_OBSERVER.md`
for calibrated synchronized pre-reduction observations. Production initcheck,
unexcluded full-runtime synccheck and full32K racecheck remain non-accepting.
No vendor submission or external publication occurred.

The48GiB diagnostic container `lfm3-boundary-v1` is stopped. Independent host `/proc`
inspection found no matching GPU-probe/sanitizer/runtime processes; production
single/worker and diagnostic binary hashes were unchanged. Evidence:
`validation/boundary-cleanup-v1.json`. Host suite:1245 passed,
`validation/host-boundary-final-v3.log`. This final run follows the filename-collision
repair: the pre-existing tensor-boundary runner is restored byte-for-byte, while
the SDK runner is now `tools/check_lfm3_launch_boundary.py`. Frozen evidence retains
the actual earlier filename. Offline full-output/snapshot/report/source replay and
evidence hashes are recorded in `validation/boundary-replay-v1.json`.
