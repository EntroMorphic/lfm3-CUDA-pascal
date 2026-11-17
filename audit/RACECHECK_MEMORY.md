# Guarded full32K racecheck diagnostic

No final diagnostic result is claimed here. A zero-hazard footer alone is not
success: the prior unfiltered optimized full32K attempt returned tool code 9,
produced no application JSON, and failed despite displaying zero hazards.
The parent collected original host OOM evidence and correlation in
`validation/racecheck32k-bounded-full-v1/kernel-oom-correlation.json`.
Reported host evidence: Nov 04 19:38:53 global OOM killed host PID 118327
(`liquid-lfm3`), anon RSS 116214448 KiB; matching container
`bb3189c1e439a3a2fda695bdba5aaf3eee31c53302a56e6ff10b71f72df57ca4`.
Post-event cgroup values were `memory.peak=119604572160`, `oom_kill=1`,
`memory.max=max`. Container kernel-journal access was denied; host journal access
was available. Do not blindly repeat that unlimited-memory run.

## Execution contract

Run only when the parent authorizes GPU work, in a **new temporary container**
using the same immutable base image ID and exact copied serving CUDA/cuBLAS
libraries (old 13.1, pinned). Use Docker `--memory=48g --memory-swap=48g`;
these are parent/container-launch requirements, not limits installed by this
Python program. Preserve the serving `LD_LIBRARY_PATH`: the guard inherits it
unchanged. The parent must verify image, library, binary e436 and sanitizer 528b
identities against the recorded full hashes; abbreviated labels are not hashes.
Do not transfer files or disturb another active frozen-tree workload.

```sh
python tools/check_lfm3_racecheck_guarded.py \
  --sanitizer /absolute/path/to/compute-sanitizer \
  --binary /absolute/path/to/liquid-lfm3 \
  --checkpoint /absolute/path/to/checkpoint \
  --request /absolute/path/to/original-32768-request.json \
  --expected-result /absolute/path/to/accepted-native-result.json \
  --output /absolute/path/to/NEW-evidence-directory \
  --timeout 900 --num-workers 1 \
  --rss-budget-gib 32 --min-host-available-gib 16
```

`--racecheck-num-workers 1` bounds worker concurrency, not launch coverage.
No launch/kernel filters, clusters, hazard limits, async-disable,
force-blocking, launch skip/count or reduced fixture are introduced. All 32768
input IDs and the accepted teacher argmax must match. The guard pins SHA256 of
binary, sanitizer, request and expected JSON at start/end. Checkpoint, image and
shared-library identities remain the parent's external provenance obligations.

Every approximately one second (via standard `Popen.wait(timeout=...)`) the
guard records owned-PGID process RSS, host MemAvailable and cgroup events,
current/max/peak in `resources.jsonl`; before/after cgroup state is also saved
in `summary.json`. A preflight sample runs before spawning. Group RSS over
32 GiB, host availability below 16 GiB, or any new cgroup OOM event stops and
kills only the owned process group. These are predeclared diagnostic stop
conditions, **not success**, and sampled monitoring cannot prevent every rapid
allocation spike: the dedicated 48 GiB hard cap is required protection.
RSS sums can double-count shared resident pages, intentionally conservatively.

Only normal exit zero, exact canonical zero-racecheck summary, matching valid
native JSON, unchanged hashes, complete group cleanup and no guard/error/timeout
yield `scoped_passed=true`. `full_acceptance` is always false. Missing JSON or
exit 9 fails even with a clean footer. A stopped or capped attempt establishes
neither race freedom nor actual checker coverage through completion. No BSP
support, vendor confirmation or broader acceptance is inferred.

## Host-only tests

```sh
python -m pytest -q tests/test_lfm3_racecheck_guarded.py
```

Tests use fake executable sanitizer/application scripts and simulated memory
snapshots; they do not access a GPU, SSH, Docker or the serving benchmark.

## Parent execution setup

The completed balanced benchmark was collected before this attempt. Three copied
serving libraries match the original full hashes; `ldd` resolves cudart/cuBLAS/
cuBLASLt from the read-only `/pinned-cuda` mount. The immutable image is
`sha256:2a9b0054deac9e70b38e31bfdc0b789a4918af50a07e9832263c47bcbc5b08da`.
Actual hard memory and memory-plus-swap limits are both51,539,607,552 bytes.

The first temporary container omitted driver exposure: its instrumented run failed
before the first instrumented API, and a plain CUDA control confirmed the missing
driver library. Evidence in `validation/racecheck32k-guarded-v1/` is not coverage.
That container was stopped. A fresh v2 explicitly exposes NVIDIA devices/compute
libraries; `cuDriverGetVersion` reports13000 and `cuInit` succeeds. Its dedicated
container manifest is retained in `validation/racecheck32k-guarded-v2/`.
V2 completed as **timeout, not accepted**, after900.04s without native application
JSON. Its sanitizer log contains only the banner, so no completed coverage or
zero-hazard result is established. Start/end artifact hashes are stable, no guard
tripped, and all OOM counters remain zero. Peak aggregate owned-group RSS is
31,331,758,080 bytes (shared pages may be double-counted); cgroup peak is
27,157,561,344 bytes. Minimum observed host MemAvailable is90,010,324,992 bytes.

Owned-group cleanup completed and independent host inspection found no matching
live sanitizer/application descendants. The temporary v2 container was stopped;
both temporary containers and all failed evidence are retained. Production
container, drivers, BSP and serving libraries remain unchanged. This resolves
the unsafe unbounded retry pattern, **not** the full32K racecheck acceptance gate.
Do not repeat a longer blind run or credit incomplete instrumentation as clean.
