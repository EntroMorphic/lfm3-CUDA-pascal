# Thin-GEMM hybrid candidates (v5–v8): decode remediation of the WMMA path

## Current disposition

The v4 ordinary-store WMMA candidate (`WMMA_REMEDIATION.md`) removed the vendor TMA stores but
measured a **3.445809× decode slowdown**. This work replaces only its decode (rows ≤ 8) producer.
The arithmetic is unchanged and verified bitwise against v4's frozen runtime kernels. Prefill
(`wmma_partials`), `wmma_finish`, partition selection and every other path are untouched.

Candidate `thin-hybrid-runtime-candidate-v4` (v5/v8 policy) — single
`327a9096d6897ab1b1bfdf262247cb9dbc83dff687f729e0305fc32732644aec`, worker
`3442a068f619bf7cb82daf54f4520d6140ca0acaf0401e2072749403ec62c112`, paired probe
`6922da33e96c7e97231d810a960fbeff7a037b01999f8c27d5efd04cc26cddca`:

| Gate | Result |
| --- | --- |
| Tail geometry, v8 S8 and v5 U16, all four sanitizers | 192 cases bitwise vs v4 under each tool; canonical clean footers |
| Original eager forward | all 6890 checks, zero tolerances |
| Scoped integrated sanitizers (four-token fixture) | initcheck 3.8 s, unexcluded synccheck 1.8 s, memcheck 4.7 s, racecheck 29.3 s: zero errors/hazards/warnings, exact application output |
| Balanced warmed AB/BA benchmark, 20 samples per binary | median decode 0.3389805 s vs production 0.3996895 s; **median paired decode ratio 0.8450810816** |
| Full context suite through 4096 | passed |
| Paired normal / compact | passed |
| Grouped worker | passed |
| 32K SDPA | passed |
| Full 32768-prefill guarded racecheck | **timed out at 900.6 s**, no application JSON, no OOM (cgroup peak 26.7 GB), cleanup complete |

Binary, tool, runner and library hashes were stable across the run (`gates.json`, runner
`69bc3005…`). **Production is unchanged. Full acceptance remains false.**

The upper racecheck fixture is prefill-only. Every linear in it has 32,768 rows (`"thin":"none"` in
the plan log), so it exercises v4's unchanged `wmma_partials`, not v5/v8. Its timeout matches
v4's (900.605 s) in kind. It is **not** credited, waived or attributed to this change. Prefill
also remains slower than production (median 0.1099 s vs 0.0320 s). The prefill producer is the
next target for both.

## What was measured, in order

1. **Profile (nsys, `build/profile-v1`)** of one decode per binary. In v4, `mma_thin_partials`
   takes 91% of GPU time (2,635 ms over warmup plus measured run), and `wmma_finish` takes 12 ms.
   Every decode shape streamed weights at 36–44 GB/s against cuBLAS's 150–206 GB/s.
2. **Cold-cache probe** (`native/probes/lfm3_thin_v5_probe.cu`, `validation/thin-probe-v1`):
   - Each candidate is compared bitwise with v4's frozen runtime header on 203 random-input cases
     (decode shapes, n = 1/2, tail geometry) and on the real layer-0 conv in_proj decode fixture.
   - Timing rotates through ≥ 2 GiB of weight copies. The first runs reused one copy and reported
     cache-served rates above DRAM peak; they are kept, and marked superseded.
   - The probe now fails any timing above 273 GB/s.
3. **v5** (register prefetch of U K-steps): exact, about 1.5–2.8× v4, still about 2× slower than
   cuBLAS.
4. **v6** (row-contiguous 16-byte loads staged through padded per-warp shared memory): exact, at or
   above cuBLAS on the large shapes.
   - Hybrid v5/v6 passed all gates through 32K, with paired decode ratio **0.9351151315**.
   - Its scoped racecheck **timed out at 300 s**, the cost of instrumenting shared memory.
5. **v7** (same loads, quad rotation transpose in registers, no shared memory): exact; scoped
   racecheck clean in 28.6 s (hybrid v5/v7).
   - It was 25–40% slower than v6.
   - A timing-only diagnostic (`v7diag`, wrong results by design, outside the generator-extracted
     region) recovered v6 speed, which isolated the transpose ALU work as the cost.
6. **v8** (two-stage XOR-butterfly quad transpose): exact; tails racecheck clean in 4–5 s.
   - Interleaved cold-cache GB/s, v8 S8 against v6 S8: LM head 221 vs 209; gate/up 200 vs 190;
     down 196 vs 146; conv in_proj 181–228 vs 156.
   - Policy: v8 S8 for m ≥ 2048, v5 U16 below that.

## Reproduction

`tools/prepare_lfm3_thin_hybrid_runtime.py --large-thin {v6,v7,v8}` generates an isolated mirror.
Only `Runner::linear` dispatch changes (same anchors as v4), and the kernels are copied verbatim
from the two probe sources, with region hashes in the manifest. `tests/test_lfm3_thin_hybrid_runtime.py`
checks isolation, v4-input immutability, region anchors and probe-only exclusions.

The gate matrix runs through `tools/run_lfm3_thin_hybrid_gates.py`:
- the same checker tools and arguments as v4
- stop at first failure
- pinned start/end hashes, a heartbeat and runner identity

It is driven from the workstation by `tools/thor_remote.py` (`sync` verifies SHA-256 on both ends,
`launch` waits for the runner's own identity and a fresh heartbeat, `watch` fails on unreachable
host, stale heartbeat or wrong identity). Builds run as root in compiler image `sha256:3c2b5b8e…`
via `tools/thor/build_candidate.sh` and `tools/thor/build_thin_probe.sh`. Execution uses the
pinned runtime/library environment (`/pinned-cuda`).

Evidence: `thin-hybrid-v4/` (v5/v8, full matrix), `thin-hybrid-v1/` (v5/v6), `thin-hybrid-v3/`
(v5/v7 scoped sanitizers) and `thin-probe-v1/` (probe runs, frozen probe/template revisions). All
757 copied evidence files were SHA-256-verified against Thor. Bulky activation dumps and binaries
stay in `build/` on Thor, following the v4 convention.

## Limits

- **Pinned scope.** The heuristic-ID-66 mapping and the shape policy are measured for this
  library, GPU and checkpoint. They are not portable guarantees.
- **Benchmark scope.** The benchmark is the 15-step greedy fixture of the v4 benchmark. HTTP,
  cancellation, soak and grouped-throughput gates have not been repeated on this candidate.
- **Upper sanitizers.** Upper initcheck, synccheck and memcheck at the 32K fixture are not launched
  after a failed upper racecheck. They are untested, not inferred.
