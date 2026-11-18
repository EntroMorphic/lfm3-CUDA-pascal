# prefill-speed-v1

The prefill GEMM kernels re-tiled (2026-11-17), after the project owner lifted the plan's speed guardrail, "first for
a faster exact prefill kernel". No speed bar was set in advance: this round records what changed, that nothing but
speed changed, and the numbers.

## What changed (`native/include/lfm/gemm.cuh`, commit `21496fe`)

- **Shared-memory layout.** The first Pascal kernel kept the B tile as `[column][k]` with 32 k per row, so the 16
  columns a warp reads at one k were 64 floats apart: one bank, every read serialized. Tiles are now `[k][row]` and
  `[k][column]`; a warp reads 4 addresses 8 floats apart and 8 addresses 4 apart, all in different banks.
- **Register patches.** Each thread owns 8 × 4 outputs (plain) in registers, in place of 2 × 2; the block's height
  follows the row count (32 to 256 rows) so a short prefill does not compute padding.
- **Global loads** run along k within a row (coalesced).
- **Exact kernel** (`-DLFM_GEMM_TENSOR_CORE_MODEL`): the same layout with a 2 × 2 patch, which measured fastest (it
  is bound by its arithmetic; at 16 or more outputs per thread it ran 5 times slower); the 16 products of a step
  are summed in 32 bits (their sum is below 2^31) and the accumulator added in 64; the pass over the 16 exponent
  sums is skipped where the accumulator's exponent already bounds them.

The arithmetic of every output is unchanged in both kernels: plain, FMA in k order; exact, the tensor-core model.

## Nothing but speed changed

| Check | Result |
|---|---|
| Benchmark output checksums, 5 shapes, plain and exact, before and after | identical |
| Parity gate, exact kernel (`6d598e8c…`), tree `27ad2d65…` | 7,826 of 7,826 trace files equal to the Thor pins |
| Parity gate, exact kernel with the integer reference step (`ea9adfa3…`) | equal |
| Plain kernel (`text-v4`, `a7037d74…`) against `text-v1`: fixtures (pinned ids, reset), with compact windows off and on | pass |
| …long decode, 701 steps | FP32 logits byte-identical (digest `7b36221a…`) |
| …NLL subsets at 256, 257, 4,096 | byte-identical |
| Builds | 0 warnings |

`verify-and-timing.log` is the second half of that run. Not re-run with these kernels: the full NLL files and
32,768 tokens (text-correct-v1, memory-v1); the host suite (no host code changed except the new header, which the
host check compiles).

## Numbers (GTX 1080, `native/bench/pascal_gemm_bench.cu`, best of 3 to 5 after a 2 s warm-up)

GFLOP/s, counting 2 per multiply-add:

| Shape (rows × columns × K) | Plain, before | Plain, now | Exact, before | Exact, now | Exact, reference step |
|---|---:|---:|---:|---:|---:|
| 257 × 6,144 × 2,048 | 133 | 728 | 34 | 246 | 79 |
| 256 × 7,680 × 2,048 | 149 | 1,482 | 109 | 300 | 88 |
| 256 × 2,048 × 7,680 | 147 | 1,206 | 126 | 299 | 83 |
| 4,096 × 2,048 × 2,048 | 449 | 1,463 | 139 | 301 | 88 |
| 4,096 × 7,680 × 2,048 | 577 | 1,638 | 138 | 302 | 87 |

"Before" was measured without the warm-up, which the small shapes' figures probably reflect (the first shape of
each run was slowest); the 4,096-row figures are the fair comparison: plain 2.8×, exact 2.2×.

End to end, no tracing, one run (process start to first logits, fixed costs included):

| | Plain, before | Plain, now | Exact, first version | Exact, now |
|---|---:|---:|---:|---:|
| Prefill, 34 rows | 1.24 s | 0.86 s | 3.77 s | 1.69 s |
| Prefill, 257 rows | 2.56 s | 1.75 s | 16.23 s | 3.94 s |
| Scorer, 64 windows of 256 | 108 s | 43 s | | |
| Scorer, 2 windows of 4,096 | 58 s | 24 s | | |

## Reading

- The exact kernel costs about 5.4 times the plain kernel per multiply-add (302 against 1,638). Per product it
  does two multiplies, a float-to-integer conversion and an integer add, with the exponent work on top, against
  one FMA; a ratio much under 4 is not in sight with this formulation.
- End to end at 257 rows the exact engine takes 2.3 times the plain engine's prefill time, since the GEMMs are not
  all of prefill.
- The plain kernel is at 1.6 TFLOPS. The card's FP32 ceiling by its specification is several times that; unmeasured
  here.

## Decode, steady state (measured after the note below was first written)

`liquid-lfm3`, greedy, batch one, no tracing, two runs each (identical to within 0.1%):

| Request | `text-v1` | `text-v4` (re-tiled plain) | `parity` (exact) |
|---|---:|---:|---:|
| 300-token prompt, 700 teacher-forced steps, capacity 1,024, one warm-up run | 14.71 ms per token (68.0 tokens/s) | 14.72 (67.9) | 14.75 (67.8) |
| `baseline` fixture, 15 steps, two warm-up runs | 14.4-14.5 | 14.4-14.5 | 14.5 |
| `baseline` fixture, 15 steps, cold (no warm-up) | 15.0-15.2 | **22.3-22.5** | 14.5 |

Warm prefill of the 34-token fixture: 0.367 s, 0.245 s and 0.694 s.

- Steady-state decode is 14.7 ms per token, 68 tokens per second, at about 1,000 tokens of context, and is the same
  in all three builds, as it must be: decode runs none of the changed kernels.
- The slow figure is a cold-start effect: with warm-up runs it is gone. Explained in
  [decode-atoms-v1](../decode-atoms-v1/): with no process holding the card, the driver re-initializes it for the next
  one and its clocks climb over about 2 s; the faster prefill starts decode earlier in that climb. The driver's
  persistence mode removes it.
- For reference, the Thor engine's README gives 62 tokens per second at batch one for its decode loop. The
  conditions of that figure are not matched here, so it is context, not a comparison.
