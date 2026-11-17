# Host text sampler integration

`native/include/lfm/text_sampling.hpp` is standalone C++17 (no JSON/CUDA).
Construct `lfm::TextSamplingOptions`, then one `lfm::TextSampler` **per generation**.
Pass the same sampler through all decode steps to preserve its RNG stream:

```cpp
lfm::TextSamplingOptions options;
options.do_sample = true;  // Default false: greedy, lowest ID breaks equal scores.
options.temperature = 0.8;
options.top_k = 40;
options.top_p = 0.95;
options.min_p = 0.05;
options.repetition_penalty = 1.1;
options.seed = 42;
lfm::TextSampler sampler(options);
int next = sampler.sample(host_float_logits, vocab_size, history_ids, suppressed_ids);
```

The caller parses metadata strictly (reject wrong JSON types, fractional integer
fields, negative/out-of-range seeds before conversion). `seed` is uint64;
`top_k` is int64, zero disables it, otherwise 1..vocab_size. Temperature and
repetition penalty must be finite and strictly positive; top_p in (0,1]; min_p
in [0,1]. All options are validated even for greedy decoding. Zero temperature
is invalid: request greedy with `do_sample=false`, not temperature=0.

`sample(const float*, size_t, const vector<int>& history={}, const vector<int>&
suppressed={}) -> int` never mutates logits/history. History should include the
the intended repetition history; duplicate IDs apply the penalty once. The native
Omni integration uses generated IDs only, matching the checkpoint's text-history
policy rather than penalizing the input prompt.
Positive seen logits are divided by the penalty; negative ones multiplied.
Penalties below one are valid (encourage repetition). Suppression/history IDs
must be in bounds even when the penalty is one. Duplicate suppression is benign.
Empty/null logits, vocabulary larger than INT_MAX, any NaN/infinite logit (even
suppressed), top_k larger than vocabulary, or all-suppressed input throw
`std::invalid_argument`. Do not encode suppression as negative infinity.

Policy order: repetition penalty, suppression, score-descending sort (ties by
ascending token ID), temperature softmax, top_k, min_p relative to largest
probability, top_p nucleus, renormalize. Top_k is implemented before softmax,
equivalently avoiding work on discarded scores. Top_p is evaluated on remaining
mass after top_k/min_p, includes its crossing token, and always retains at least
one token. Greedy returns the maximum adjusted allowed score; filters cannot
remove that maximum. There is no implicit EOS or special-token suppression.

`distribution(...) const -> vector<TextCandidate>` exposes final positive
probabilities (`long double`) in the same sorted order and never consumes RNG.
Softmax subtracts the maximum before temperature division. Long-double penalty
arithmetic covers the full finite float-logit/double-parameter range on Linux
x86 extended-precision hosts; narrower hosts throw `std::overflow_error` if a
penalty-adjusted score is unrepresentable. Exponential underflow removes zero
mass candidates and cannot empty support. Allocation failures propagate.

Determinism: stochastic calls consume exactly one mt19937_64 output, including
singleton support; greedy and rejected calls consume none. The top 53 random
bits map to [0,1), and inverse CDF uses the first strictly greater cumulative
probability. Identical inputs/options/seed/call sequence on the same host build
reproduce exactly; copying a sampler copies its stream. This is **not bitwise
PyTorch RNG parity**, nor a promise across different libm/long-double platforms.
The sampler is mutable and requires external synchronization if shared.

## Native request integration

The runtime parses `do_sample`, `temperature`, `top_k`, `top_p`, `min_p`,
`repetition_penalty`, and `seed` before constructing the GPU runner. The current
JSON boundary accepts seeds0..2147483647 (the standalone header supports uint64).
Teacher argmax diagnostics never consume stochastic RNG. Each warmup starts the
same seeded sampler as the measured request, so warmup cannot advance its stream.
Default greedy/penalty1 retains a direct argmax fast path. Reset verification
replays token inputs and requires exactly equal raw logits, independently of RNG.

`tools/check_lfm3_sampling.py` tests top-k1/greedy equality, repeated seed42,
warmup isolation, suppression/stopping, exact reset and fourteen invalid metadata
cases rejected against a config-only directory before weights can be loaded.
The first native suite passed those controls, but independent review found that
they could all pass if runtime sampling were silently bypassed. The remediated
gate now records every sampled step's logits and replays them in a standalone
`text-sampling-oracle` process with the identical policy, seed and generated-only
history. Every emitted ID must match and the fixed fixture must contain a
non-greedy choice. Six host oracle tests explicitly reject greedy bypass, wrong
IDs, missing/truncated traces and empty output.

The strengthened GPU integration suite passed for binary `85028f2a...`, including
independent logit replay and a demonstrated non-greedy choice. All6890 short
activation/logit comparisons remain bitwise equal for that candidate.

An independent Python arithmetic oracle additionally matches token support and
probabilities to1e-12 over80 deterministic policy combinations (temperature,
top-k/top-p/min-p, sign-correct repetition penalty, duplicate history/suppression,
and ties). This validates the documented filter order, not torch RNG equivalence.

## Linear deterministic selection fast paths

`TextSampler::sample()` now scans for the maximum adjusted allowed score when
`do_sample=false` or `top_k==1`. These paths avoid candidate sorting, exponentials
and normalization; time is O(vocabulary + history + suppression), with two
vocabulary-sized byte flag arrays. `distribution()` and the arithmetic/filter
order for all other stochastic policies are unchanged.

The scan retains the full validation contract: vocabulary/null checks, top_k
bounds even for greedy calls, every history/suppression ID (duplicates remain
benign), and every logit's finiteness even when suppressed. Allowed history
entries receive the sign-correct long-double penalty once, with the same overflow
check as `distribution()`. Suppressed entries are not penalty-adjusted, matching
the existing contract. Ascending traversal with a strict greater-than comparison
retains the lowest-ID maximum. Positive temperature and valid top_p/min_p cannot
remove that maximum, so selection matches `distribution().front().token_id`.

Only after successful validation does stochastic top-k1 explicitly discard one
`mt19937_64` output. Greedy and rejected calls consume none. No mutable-policy or
RNG-inspection API was added. Since fixed top-k1 outputs cannot reveal RNG
position through the public API, its exact draw is guarded by a source-contract
test; runtime reference-engine tests cover the general stochastic path, including
singleton support followed by wider support and intervening rejected calls.

Host validation: `python -m pytest -q tests/test_text_sampling.py
tests/test_text_sampling_distribution.py tests/test_text_sampling_oracle.py
tests/test_text_sampling_fastpath.py` passed **25 tests**. The new probes run both
optimized and with UBSan (`-fsanitize=undefined -fno-sanitize-recover=all`), compare
against the unchanged distribution and an independent unsorted argmax oracle,
and exhaust all 5^3 small score tuples, history subsets and nonempty allowed
subsets across greedy/top-k1 policies. Additional cases cover finite arithmetic
extremes, signed-zero ties, repeated IDs, invalid inputs and 128K vocabulary.
This is host correctness evidence, not an HTTP throughput result: the reported
13.1 tok/s HTTP soak and older roughly 39 tok/s standalone measurement are not
comparable benchmarks. Native logit replay and like-for-like performance checks
remain integration work; no GPU execution was performed for this change.
