# Fair explicit-group HTTP benchmark

`tools/check_lfm3_group_benchmark.py` is a dedicated acceptance/measurement
runner. Host tests validate its contract; the completed parent GPU run is recorded below.

The previous soak's five early serial observations versus 561 group cycles
are observational only, not a balanced end-to-end speed comparison. Do not
reuse those timings as this benchmark's results.

## Exclusive run

After the separate diagnostic has finished, run on the intended GPU host with
the existing serving virtual environment (CUDA 13.1 libraries); use an unused
output directory:

```sh
python tools/check_lfm3_group_benchmark.py \
  --checkpoint /absolute/path/to/checkpoint \
  --worker /absolute/path/to/worker \
  --control-result /absolute/path/to/accepted-control.json \
  --numerical-evidence /workspace/liquid-CUDA/build/grouped-worker-bounded-prefill-v2/summary.json \
  --output /absolute/path/to/new-benchmark-output \
  --samples 20
```

Samples default to 20 and are bounded to 5..100. No SSH, shared endpoint,
external client, automatic microbatching, or CPU stand-in is used by this
runner. It starts its own IPv4-loopback HTTP server at a random port with an
unrecorded random bearer. Worker cancellation and explicit batching are both
enabled. The `Service` and checkpoint tokenizer frontend are the real serving
path. It prepends the current virtual environment's
`lib/python3.12/site-packages/nvidia/cu13/lib` to worker `LD_LIBRARY_PATH`, like
the existing HTTP gate, and records the effective path. This is **not** evidence
that the default compatibility library environment changed or was validated.

## Fixed comparison

Two capacity-512 sessions persist across both modes. Both use the existing
control question and identical messages: lane A greedy, lane B sampled with
`top_k=1`, seed 42. Every lane must return the first 32 accepted control IDs,
so each measured mode completes exactly two sequences / 64 output IDs.
The repeated prompt is shorter than the previous generated history: each
request must report zero reused tokens, the exact expected consumed count,
the expected stop/length reason, valid cache and no cancellation. No reset or
session recreation is inserted selectively between modes.

Three complete rounds of **both** modes are warmed before any measurement;
warmup records are real completed requests, not synthesized counters.
Within each measured round, even rounds run serial-pair then group; odd rounds
reverse the order. Serial-pair comprises two sequential `/generate` calls;
group is one `/generate_batch` call. Group metadata must report actual GPU
batching and exactly 31 paired decode calls, on the envelope and both lanes.
Every serial lane must report GPU batching false. All warmup and measurement
responses undergo the same strict output/count/stop/policy validation.

Wall time starts before the first HTTP request and ends after complete response
bodies are read and parsed. It includes frontend rendering/tokenization,
prefill, sampling, all generation, decoding, HTTP and connection overhead;
it excludes session creation, cleanup, JSON report writes and audit assertions.
This is not a decode-only throughput comparison.

## Evidence and acceptance

`--numerical-evidence` is required: supply the existing native grouped gate's
JSON `summary.json`. Before starting the worker, the runner requires
`status == "passed"`, `worker_sha256` equal to the actual worker executable's
SHA-256, integer (not boolean) `bitwise_logit_steps >= 73`,
`independent_non_greedy_logit_replay is true`, and
`aggregate32k_checked is true`. Matching HTTP token IDs and batching metadata
alone cannot substitute for this numerical acceptance evidence. Evidence for
another binary is rejected even if it emits the same IDs; do not reuse another
binary's old acceptance/control artifacts.

This is **trusted artifact binding, not attestation**: the JSON is assumed to
come from the trusted native acceptance run. The benchmark does not prove
that an untrusted artifact author ran the gate or authenticate a forged JSON.

`summary.json` preserves all warmup and measured responses, each per-mode
latency, round and AB/BA order, and intermediate records even when validation
fails. The default run contains all 20 serial-pair and 20 group observations.
Summaries include median wall seconds, nearest-rank p95, aggregate output rate
`64 * samples / sum(wall_seconds)`, each paired serial/group wall ratio and
the median paired ratio. Ratios above one favor grouping for this fixture;
there is no required speedup, adaptive threshold, universal SLA or statistical
significance claim.

The executable, benchmark/server/client/frontend/ownership/chat wrappers
(all local tools Python files), and native C++/CUDA/header sources are hashed
before worker startup and again after cleanup. Both the accepted control file
and numerical-evidence JSON are included in that same start/end hash map;
mutating either fails acceptance. Their resolved paths, hashes and supplied
numerical evidence, module load paths and Python executable are recorded. Passing requires the
same source set and matching hashes, complete strict warmup/measurement
coverage, successful session closure, stopped HTTP/owner threads and successful
worker process-group cleanup. Hash drift or cleanup errors fail the run and
retain the output directory. Do not call a running or failed report acceptance.

Host validation:

```sh
python -m pytest -q tests/test_lfm3_group_benchmark.py
```

## Completed balanced measurement

`validation/http-group-benchmark-v1/` passes20 warmed samples per mode on worker
`1aa1e0c2...`, bound to its accepted native numerical evidence. Local postvalidation
rechecks all responses, AB/BA coverage, start/end identities and recomputed statistics.

| Mode | Median wall | p95 wall | Aggregate output rate |
| --- | ---: | ---: | ---: |
| Serial pair | 1.73835s | 1.75319s | 36.739 tokens/s |
| Explicit group | 1.06987s | 1.08031s | 59.734 tokens/s |

Median paired serial/group wall ratio is1.62445. This is a same-workload observation,
not a universal speedup or SLA. Actual response text is copied to
`../tmp/http-group-benchmark-v1/actual-responses.txt`.
