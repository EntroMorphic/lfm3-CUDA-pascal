# Finite resource and latency acceptance

`tools/check_lfm3_soak.py` starts a temporary authenticated loopback listener and
runs alternating greedy/top-k1 seeded requests against an accepted86-token
control. Every response must match the reference IDs. It also repeats invalid
metadata, reset, close/recreate and two-resident-session paths. Final cleanup
requires no session records and a reaped worker. Sampling policy here is top-k1,
not evidence of non-greedy sampling; the resident logit-replay gate covers that.

Limits are fixed before execution: after warmup, neither process may gain more
than32MiB RSS or two file descriptors. Samples and failure evidence are saved
before assertions. These Linux metrics do **not** measure GPU-memory allocation.
Six host tests reject empty coverage and over-budget RSS/FD samples.

## Initial measured run

Evidence: `validation/http-soak-round2-v1/`, actual response copied to
`../tmp/http-soak-round2-v1/control-response.txt`. Protocol2 worker `1f3401db...`;
source hashes are in the summary. No GPU workload was run concurrently.

- 30 measured cycles pass over468.44 seconds including warmup (about7.8 minutes).
- 60 measured86-token responses match exactly, in addition to warmup/lifecycle
  requests. Maximum post-warm RSS growth: server1,835,008 bytes,
  worker22,413,312 bytes. Maximum descriptor growth: zero for both.
- Greedy median6.5636 seconds, p956.6510 seconds; top-k1 sampled median6.5655,
  p956.6496. Median end-to-end throughput is about13.1 tokens/second for each.

These include HTTP, formatting, sampling and prefill, and are **not comparable**
to the earlier standalone decode-only benchmark. They make no universal speedup,
SLA, long-duration stability, GPU-memory plateau or batched-throughput claim.
Further runs must keep tool/source/binary hashes and workloads explicit.

Inspection found the worker's deterministic choices unnecessarily used the full
128K long-double sort/softmax sampler path. A separately tested O(vocabulary)
argmax fast path is being developed; any throughput improvement must be measured
on the same workload after numerical replay, not inferred from the code change.

## Same-workload fast-path confirmation

`validation/http-soak-fastpath-v2/` passes the unchanged30-cycle workload and
fixed resource bounds on worker `842a46f1...`, after cancellation and deterministic
sampling fast paths. Native stochastic logit replay,32K budget checks and all6890
batch-one exact comparisons also pass for that candidate/runtime increment.

- Elapsed 180.48 seconds including warmup. Greedy median 2.4689 seconds,
  p95 2.5126; top-k1 sampled median 2.4674, p95 2.4953. Median end-to-end throughput:
  34.83 and 34.86 tokens/second respectively.
- Maximum warm RSS growth: server131,072 bytes, worker0 bytes. No descriptor
  growth for either process. Every86-token response remains exact.
- Observed median speed ratio versus the first identical-workload run:2.659x.
  Server/client cancellation code changed too, so this is not isolated causal
  attribution to the sampler or a universal performance claim.

The greedy/top-k1 sample path now scans logits instead of sorting/normalizing the
entire vocabulary; its strict validations and RNG draw contract remain verified.
Actual text: `../tmp/http-soak-fastpath-v2/control-response.txt`. Full-serving GPU
batching, longer soak, GPU-memory/resource and sanitizer acceptance remain open.
