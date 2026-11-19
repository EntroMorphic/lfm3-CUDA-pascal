# speed-baseline-v1

Phase 2's starting point (2026-11-18): the shipping engine as of red-team-1 (`text-v10`, `a9e3912d…`) on the cases
phase 2 is judged by, and the card's own ceilings. Measurement only; no rules. Records: [records/](records/)
(`SHA256SUMS` verified after the copy).

## The engine

`tools/pascal/speed.sh`, warm (one warm-up run), no tracing, median of 3 runs (the spread was under 1% everywhere but
prefill-32768, 1.2%).
Prompts are the first tokens of window 0 of the 32,768-token perplexity windows; decode cases teacher-force.

| Case | Capacity | Prompt | Steps | Median | Peak device memory |
|---|---:|---:|---:|---:|---:|
| prefill-32 | 4,096 | 32 | | 0.423 s | 4,903 MiB |
| prefill-257 | 4,096 | 257 | | 0.916 s | 4,903 MiB |
| prefill-1024 | 4,096 | 1,024 | | 2.277 s | 4,903 MiB |
| prefill-4096 | 4,096 | 4,096 | | 9.287 s | 4,903 MiB |
| prefill-32768 | 32,768 | 32,768 | | 89.43 s | 6,451 MiB |
| decode-1k | 1,024 | 300 | 700 | 15.69 ms per step | 4,421 MiB |
| decode-8k | 8,192 | 7,000 | 1,000 | 16.25 ms per step | 4,813 MiB |
| decode-32k | 32,768 | 31,000 | 700 | 20.58 ms per step | 6,451 MiB |

The 32,768-token prefill was measured in the second job: the first asked for one token beyond the capacity and the
engine refused it.

## The card

`native/bench/card_ceilings.cu`, best pass after a 2 s warm-up:

| | Measured |
|---|---:|
| FP32 multiply-add (2 FLOP each) | 9.7 and 9.4 TFLOPS in two runs; 8.8 in a third, after the engine runs |
| Reading device memory, 16-byte loads | 248 to 251 GB/s |

The read figure is not a ceiling: the engine's own decode kernel read weights at 258 GB/s under the profiler
(decode-atoms-v1), so this test leaves some rate unused. The memory clock under load is 4,513 MHz (state P2), as
recorded in decode-atoms-v1.

The plain prefill kernel (`pascal_gemm_bench`) runs at 0.79 to 1.60 TFLOPS: a sixth of the card's multiply-add rate.

## Decode is 1 ms slower than recorded, in every build

prefill-speed-v1 recorded 14.71 ms per step for this same request (decode-1k) with `text-v1` and `text-v4`. Today,
two runs each:

| Build | ms per step |
|---|---:|
| `text-v1` (`e30023b7…`) | 15.725 |
| `text-v4` (`a7037d74…`) | 15.745 |
| `text-v6` (`7464f580…`) | 15.710 |
| `text-v10` (`a9e3912d…`) | 15.693 |

The same binaries that ran 14.7 ms then run 15.7 ms now, so the difference is in the card or the host, not the code.
Not explained. Since then the host was rebooted and the persistence daemon's mode changed (decode-atoms-v1); the
clocks sampled during the runs are the same as recorded then (SM about 1,950 MHz, memory 4,513 MHz). Phase 2 is
measured against today's figures, with the baseline build re-measured beside each candidate.

## Seen before the bar was set

`gemm-bench-wide2.txt`: a new prefill kernel (128 × 128 tiles, 8 × 8 outputs per thread, 16-byte loads, two shared
buffers) at 4.4 to 7.2 TFLOPS on the benchmark's shapes, with the same checksums as the plain kernel; and
`wide2_check.txt`, 94 cases bit-identical to the plain kernel. Kernel benchmarks only: no engine run with it.
