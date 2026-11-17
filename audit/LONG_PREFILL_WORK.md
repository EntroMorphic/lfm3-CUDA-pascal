# Long-prefill key-range optimization (host-verified, GPU acceptance pending)

## Scope and status

The shared `Runner::attention_tile` now obtains its key range from the pure host
planner in `native/include/lfm/text_attention_tiles.hpp`. The optimization is
enabled only when **both** `total_rows > 4096` and
`total_rows > scratch_rows(context)`. There is no new request flag or runtime
fork. Existing long-context experimental gating is unchanged. This source
change is provisional, not a numerically accepted replacement for the frozen
remote runtime. No GPU, SSH, remote transfer/build, or remote workload was used
for this work. The existing remote HTTP soak uses the original frozen binary.

The observed 900-second stopping interval was localized by the parent work to
actual B prefill or cache capture (reference B32764 had completed). Excess masked
attention work is a motivated optimization hypothesis, **not** a demonstrated
single-kernel cause or proof that the 900-second budget will now be met.

## Exact change

For an optimized tile, let `a = cursor + first`, `r = tile_rows`, and `w` be the
window (zero denotes full attention). The planner returns the half-open absolute
cache interval:

- full attention: `[0, a+r)`;
- sliding attention: `[max(0, a-w+1), a+r)`.

The existing GEMMs use this range's span and offset for K/V. The existing
`probabilities` / `probabilities_large` calls continue to receive query start
`cursor+first` and now receive the selected absolute key start. Kernel code,
causal/window masks, normalization, RoPE, scatter order, query tiling, trace
labels, and boolean request parsing are untouched. Probability trace payloads
for optimized tiles reflect the reduced key span, as do their GEMM dimensions.

All prompts of 4096 rows or fewer retain the old key range at **every** capacity,
including 257 rows at capacity 32768 and 4096 rows at capacity 4112. This includes
the old 128-row query tiling at long capacities. One-token decode retains the
original dynamic sliding-cache start `max(0, cursor+1-window)` and full-attention
range. No short-prompt arithmetic or launch schedule is intentionally changed.

## Mask-coverage proof (not a floating-point-equivalence proof)

For each query `q` in `[a,a+r)`, its unmasked keys are exactly `[L(q),q]`, where
`L(q)=0` for full attention and `L(q)=max(0,q-w+1)` for sliding attention.
Monotonicity gives `L(q)>=L(a)` and `q<a+r`. Thus every visible key is retained.
Any excluded left key satisfies `k<L(a)<=L(q)` and is window-masked for every
query. Any excluded right key satisfies `k>=a+r>q` and is causal-masked for every
query. The original mask is still applied inside the retained interval, so keys
needed only by later queries in a tile remain masked for earlier queries.
Ranges are nonempty because each query's own key is included. All referenced
keys/values have already been scattered before attention starts.

## Score-position reduction at 32768 rows, 16 query heads

Counts are per attention layer, counting score elements, not bytes or measured
runtime. Both score and value GEMMs use the narrower key dimension.

| Attention | Original | New |
| --- | ---: | ---: |
| Full | 17,179,869,184 | 8,623,489,024 |
| Sliding window 256 | 17,179,869,184 | 200,019,968 |

Original count: `16 * 32768 * 32768`. Full count:
`16 * 128 * 128 * (256*257/2)` (about 49.80% fewer).
Sliding spans are `128, 256`, followed by 254 spans of `383`, so the count is
`16 * 128 * (128+256+254*383)` (about 98.84% fewer). Scratch allocation remains
unchanged and bounded by the existing capacity policy.

## Host evidence

Command, run locally:

```sh
python -m pytest -q tests/test_text_attention_tiles.py tests/test_text_serving_limits.py
```

Result: **5 passed in 10.68s**. The new tests compile the actual header with
`g++ -std=c++17 -O2 -Wall -Wextra -Werror`; missing compiler is a failure, not a
skip. They cover:

- exhaustive query starts, tile lengths, queries, and keys for lengths 1..145
  with windows 0, 1, 2, 7, 128, 256;
- every feasible start and tile length 1..128 at length 4097, checking exact
  interval inclusion/exclusion for windows 0/256;
- every feasible tile length 1..128 at each of the final 512 starts for lengths
  4097, 32767, 32768, plus full key enumeration at adversarial early/window-edge
  and end-of-context starts;
- all eager prefill lengths at all capacities 1..4096, all short prompt lengths
  at long capacities 4097/4112/32768, all decode positions at representative
  eager and long capacities, and explicit 4097/4098 optimized boundary cases;
- invalid capacity, cursor, tile, window, and nonempty-cache multirow inputs;
- exact original/new position counts above.

Independent review found no concrete bounds/mask/alias defect, but noted that
the exhaustive small-window enumeration above exercises the disabled branch.
Parent follow-up adds optimized4097/4098/32764/32767/32768 cases with windows
0/1/2/7/128/256/65536, full key enumeration and partial tiles. It also explicitly
covers full-attention spans2048/2176. Changing absolute key start can reassign
reduction lanes even without changing softmax dispatch; independent SDPA remains
mandatory. Parent CUDA13/GCC12 compile-only of the modified worker succeeded.

The first run passed coverage, legacy, invalid, and existing serving limits but
failed my initial hard-coded sliding count by 2048. The expected value was
corrected using `128+256+254*383`; no numerical tolerance or coverage check was
relaxed. The second run passed all five tests.

## Required new GPU evidence before acceptance (parent-owned)

The key-span change can change cuBLAS algorithms and softmax reduction schedules
(including the 2048 threshold), and thus BF16 rounding. Mathematical masking
equivalence does **not** imply bitwise equivalence for long prompts. Existing
binary results do not validate this modified source. Still required:

- rebuild the affected standalone/shared runtime binaries;
- rerun existing strict SDPA gates unchanged: max logit error 0.5 / relative RMS 0.02,
  exact argmax, and exact reset checks;
- rerun all 6890 short-path bitwise checks and native grouped 73 bitwise checks;
- rerun all five standalone paired fixtures in each normal/compact mode;
- new numerical and sanitizer evidence for the modified native algorithm,
  including actual upper-context prefill/memcheck and its wall-clock budget.

No CUDA binary was compiled here. No new GPU numerical, sanitizer, performance,
or 900-second completion evidence is claimed. No fixture/operator checks were
skipped, suppressed, zero-filled, made adaptive, or assigned looser tolerances.

## Parent-session numerical verification

The separate Thor build `build/cmake-bounded-prefill-v1/` leaves original accepted
binaries intact. Single `e436191d...` passes all6890 original short-path bitwise
checks in `validation/forward-bounded-prefill-v1/`. It also passes the unchanged
independent SDPA gates at8192,32767+1 and32768 in
`validation/context32k-bounded-prefill-v1/`: maximum observed logit error0.15625,
maximum relative RMS0.003109956, every argmax exact and reset exact. The declared
limits remain0.5/0.02; same-source serial parity was not substituted for SDPA.

Worker `1aa1e0c2...` separately passes73 full-vocabulary bitwise grouped-vs-serial
steps, independent non-greedy replay, retirement/reuse/EOS/cancellation/recovery
and aggregate32K in `validation/grouped-worker-bounded-prefill-v2/`. This rerun
strengthens live cancellation to retain a **nonempty67-token prior committed
history**, while physical state is discarded. Raw trace hashes and independent
postvalidation are preserved. The full host suite at this point passes947 tests
in `validation/host-acceptance-followup-v2.log`.

Both standalone reference-mode matrices now pass all five fixtures,209 full-bit/
full-cache comparisons each, on probe `345f1ded...`:
`validation/decode-batch-bounded-normal-v1/` and
`validation/decode-batch-bounded-compact-v1/`. Normal upper completes in18.76s
versus43.99s in the earlier unoptimized normal fixture; these are finite fixture
times, not a serving throughput promise.

Strict no-dump memcheck also passes **both scratch and unchanged upper fixtures
with zero errors** in `validation/decode-batch-bounded-memcheck-v1/`:14.94s and
177.11s respectively. Normal mode retains four independent GPU owners and every
cache/rejection/retirement assertion. This closes that new binary's standalone
upper memcheck gate without excluding kernels, reducing fixtures or changing
numerical thresholds. Original600/900s timeouts remain preserved under their
original binaries. Integrated sanitizer checks, including worker dispatch, and
the initcheck/tooling blocker remain separate.
