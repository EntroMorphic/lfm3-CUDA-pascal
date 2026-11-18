# decode-atoms-v1

What a decode step costs on the GTX 1080, down to the kernel and the byte, and why a cold process decodes slower
(2026-11-17, at the project owner's request). Measurement only; no engine change and no rules.

Method: Nsight Systems 2025.1.3 (`nsys profile -t cuda`) on `liquid-lfm3` (`text-v4`, `a7037d74…`), exported to
SQLite and read by `tools/pascal/decode_atoms.py`; `nvidia-smi` sampling the card's performance state and clocks
every 50 to 100 ms. Greedy, batch one. Steady state: a 300-token prompt and 700 teacher-forced steps at capacity
1,024, steps 300 to 700 (`atoms-steady.txt`).

## A steady-state step: 14.86 ms

| | ms per step | Share |
|---|---:|---:|
| Projection GEMMs (the `thin` kernel reading weight matrices) | 13.375 | 90.0% |
| Every other kernel | 1.010 | 6.8% |
| Not in any kernel (between launches, host work) | 0.478 | 3.2% |
| **Wall** | **14.863** | |

399 kernel launches per step. Min 14.76, max 16.02 ms over 401 steps. (Under the profiler; unprofiled, the same
request runs at 14.72 ms per step.)

### The projections are the weights going by

Each decode step reads every weight matrix of the language model exactly once:

| Weights | Calls per step | Bytes per call | Time per call | Effective rate | ms per step | MB per step |
|---|---:|---:|---:|---:|---:|---:|
| MLP `w1`, `w3` (7,680 × 2,048) | 48 | 31,457,280 | 120.1 µs | 262 GB/s | 5.765 | 1,509.9 |
| MLP `w2` (2,048 × 7,680) | 24 | 31,457,280 | 121.0 µs | 260 GB/s | 2.904 | 755.0 |
| `lm_head` (128,000 × 2,048) | 1 | 524,288,000 | 1,967.9 µs | 266 GB/s | 1.968 | 524.3 |
| conv `in_proj` (6,144 × 2,048) | 13 | 25,165,824 | 96.4 µs | 261 GB/s | 1.253 | 327.2 |
| conv `out_proj`, attention `q`, `o` (2,048 × 2,048) | 35 | 8,388,608 | 34.0 µs | 247 GB/s | 1.190 | 293.6 |
| attention `k`, `v` (512 × 2,048) | 22 | 2,097,152 | 9.9 µs | 212 GB/s | 0.219 | 46.1 |
| attention gate (16 × 2,048) | 11 | 65,536 | 6.9 µs | — | 0.076 | 0.7 |
| **All** | **154** | | | **258 GB/s** | **13.375** | **3,456.8** |

3,456.8 MB is the language model: 3,297 MiB of BF16 weights. The step time is, to 90%, the time to move those
bytes through the card's memory at 258 GB/s. During the run `nvidia-smi` reports memory utilization 100% and the
card in performance state P2 with its memory clock at 4,513 MHz.

The card's memory is nominally 320 GB/s (256-bit GDDR5X at 10 Gb/s, from its specification, not measured here) at
its full memory clock of 5,005 MHz, which `nvidia-smi` shows only in state P0, with no load. Scaled to 4,513 MHz
that is about 289 GB/s; the engine's projections run at about 90% of it.

### The rest of the GPU time: 1.01 ms

| Kernel | Calls | ms per step |
|---|---:|---:|
| `norm` (RMS norms, three block shapes) | 119 | 0.430 |
| attention P·V (`split_kn_partial`, `_finish`) | 22 | 0.237 |
| `probabilities` (softmax) | 11 | 0.088 |
| attention Q·Kᵀ (`thin_k128`) | 11 | 0.071 |
| `rotary_from_frequencies` | 22 | 0.056 |
| `swiglu` | 24 | 0.048 |
| `convolution7` | 13 | 0.044 |
| `scatter`, `head_gate`, `embed` | 23 | 0.045 |

At about 1,000 tokens of context, attention over the cache costs 0.4 ms of the 14.9; it grows with context, the
weight reads do not.

### Outside the kernels: 0.48 ms

`cudaLaunchKernel` takes 1.06 ms of host time per step for the 399 launches (2.7 µs each), overlapping the GPU's
work; the 0.48 ms is what the GPU spends between kernels and the host between steps (the 512 KB logits copy, the
argmax over 125,033 tokens). The blocking logits copy shows as 13.4 ms of `cudaMemcpy`: that is the wait for the
step's kernels, not copy time.

## Why a cold process decodes slower

The card lowers its clocks when idle and raises them only gradually under load (`decay-changes.txt`):

| Since load began | State | Core clock | Memory clock |
|---|---|---:|---:|
| idle | P8 | 139 MHz | 405 MHz |
| 0.1 s | P2 | 278 MHz | 4,513 MHz |
| 0.5 s | P2 | 493 MHz | 4,513 MHz |
| 1.0 s | P2 | 1,303 MHz | 4,513 MHz |
| 1.4 s | P2 | 1,721 MHz | 4,513 MHz |
| 1.8 s | P2 | 1,999 MHz | 4,513 MHz |

After the load ends the card holds its clocks for about 2.6 s, then steps down about every 2.7 s: 1,265 MHz, then
772 MHz with the memory clock at 810 MHz after 8 s, and 139 MHz from about 13 s.

Profiles of the cold `baseline` fixture (34-token prompt, 15 decode steps):

| Build | Prefill (kernel time) | Decode starts at | Step time | MLP weights read at |
|---|---:|---:|---:|---:|
| `text-v1` (original plain kernel) | 1.15 s | 1.15 s | 15.7 falling to 14.9 ms | 255 rising to 261 GB/s |
| `text-v4` (re-tiled plain kernel) | 0.82 s | 0.82 s | 22.6-22.9 ms | 179-181 GB/s |

During the `text-v4` run the core clock was between 250 and 580 MHz. At that clock the projection kernel cannot keep
up with the memory: it reads weights at 180 GB/s, not 260, in every call of every step. `text-v1`'s slower prefill
ended later in the ramp, so its decode ran nearly at full clock. With warm-up runs all builds decode the fixture at
14.5 ms per step ([prefill-speed-v1](../prefill-speed-v1/)).

So the slow figure that prompted this was the card's clock ramp, which the faster prefill exposed. Nothing in the
engine changed in decode.

It is not only a first-request effect: after about 3 s without work the clocks begin to fall, and after about 13 s
the next request starts from the bottom and spends its first 2 s on a slow card.

## What could make a step cheaper (none of it done; estimates are arithmetic, not measurements)

Without changing any output:

| Lever | What it touches | Bound |
|---|---|---|
| Run with the memory clock at 5,005 MHz instead of 4,513 | the 13.4 ms of weight reads, if they scale with the clock | about 1.3 ms |
| Fuse or batch the small kernels (119 norms are 0.43 ms; 399 launches leave 0.48 ms between kernels) | the 1.5 ms that is not weight reads | under 1 ms |
| Stop `lm_head` at the last live token (125,033 of 128,000 rows) | 2.3% of 1.97 ms | 0.05 ms |
| Keep the clocks up between requests (a host power setting, or a trickle of work from the server) | the first 2 s after an idle period: 22.8 against 14.9 ms per step, and a slower prefill | cold-start only |

The first and last are properties of the host's driver and power management, not of the engine, and changing them is
a decision about the machine.

Past that, a single stream cannot go faster while each token reads 3.46 GB at the memory's rate. What changes the
arithmetic is reading the weights once for several rows: the batch decoder and the tree-verify decoder of the
imported engine (phase 3). Storing the weights in fewer bits would also cut the bytes, and would change outputs.
