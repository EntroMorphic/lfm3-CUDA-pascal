# Explicit grouped text serving: acceptance ledger

This increment adds native `generate_batch`, a correlated client group RPC and
authenticated HTTP `POST /generate_batch {"requests":[A,B]}`. It is a bounded
fixed-two-lane scheduler with serial prefill and genuine two-row dense decode;
it is **not automatic coalescing** of unrelated `/generate` requests. The original
single endpoint remains serialized and truthful about batch participation.

Each lane has its own lease, sampler/RNG, stop decision, committed transcript,
physical cache and cancellation identity. Group cancellation uses tagged frames
separate from legacy single-RPC cancellation, with one shared pipe/partial-frame
reader. A retired lane is not forwarded again. Pre-dispatch cancellation can
leave only single-lane work, which must report zero paired calls.

## Red-team and host evidence

- Native scheduler/reader host fakes, strict C++17/UBSan and legacy regressions:
  full suite **905 passed**, `validation/host-grouped-serving-v1.log`.
- Independent review reproduced seven acceptance-helper gaps: trace aliasing,
  vacuous references/coverage, impossible counts, boolean counts and invalid cache
  metadata. Original failures: `validation/redteam-group-gates-before.log`.
  All are now mandatory passing regressions. The stochastic native gate also
  requires independent per-step sampler replay and a demonstrated non-greedy
  choice, not merely agreement between two potentially greedy paths.
- Four service failures exposed wrong-session lifecycle acknowledgments and
  impossible paired counts. Original failures:
  `validation/redteam-service-batch-before.log`. Present-but-mismatched lifecycle
  IDs now fail closed; legacy omitted IDs remain accepted. Actual native RPC
  correlation remains mandatory. Group counts must agree with completed work.
- Independent native source review found no new concrete defect. This is not GPU
  proof. The Thor worker compiled as `184f14b7...`; GPU acceptance is pending.

## GPU verification and resource gates

`check_lfm3_batch_worker.py` requires independent serial/group trace identities,
all full-vocabulary BF16 bits, IDs, policies, counts and real pair calls. Its
fixtures cover mixed context/policy, both retirement orders, reuse, pre-mutation
rejection, EOS, observed live lane cancellation, peer continuation, recovery,
stale tags, and aggregate capacities3+32765. Stochastic logit replay is mandatory.

`check_lfm3_http_batch.py` checks authenticated explicit grouping, live targeted
cancellation and recovery, then the same exact32-token output per lane under
fixed warm RSS/FD/device-wide memory limits. Both HTTP policies are deterministic
greedy/top-k1 controls; non-greedy acceptance belongs to the native replay gate.
The planned extended run requires at least60 cycles and600 measured seconds,
with serial-pair timing as a same-workload observation, not a speed promise.
Post-close device free memory must recover against an empty-session warm baseline
within a fixed16MiB allowance. Device-wide CUDA free memory on Thor is not a
per-worker allocation or physical-residency attribution.

Neither these tools nor host tests waive sanitizer gates. Initcheck remains
unresolved; no supported vendor BSP/tool pairing or bug confirmation is claimed.
The checkpoint-free reproduction packet is prepared locally, not published.
Actual GPU results must be appended before this increment is accepted.

## Initial integrated GPU evidence

Worker `184f14b7...` passes `validation/grouped-worker-v1/`:73 full-vocabulary
BF16-bit comparisons against independently materialized serial traces, exact
IDs, independent stochastic logit replay with a verified non-greedy choice,
mixed lengths/policies, both retirement orders, reuse, EOS, invalid-second-lane
pre-mutation rejection, observed live B cancellation while A completes, fresh
recovery, pre-dispatch A cancellation, stale tagged-control rejection and
aggregate capacities3+32765.347 raw traces remain on Thor; their full sizes and
SHA256 are copied in `trace-manifest.json`. Local postvalidation independently
matches all73 serial/group hash pairs without aliasing trace identities.

Authenticated HTTP smoke also passes in `validation/http-grouped-smoke-v1/`,
with exact32-token outputs, live per-lane cancellation, peer preservation and
recovery, fixed warm RSS/FD/device-free budgets and post-close recovery. Five
measured groups show median0.8612s versus1.3238s for a serial pair (74.3 versus
48.3 aggregate tokens/s). These are short same-workload observations, not a
universal batching speedup or sustained-load certificate. Actual text is copied
to `../tmp/http-grouped-smoke-v1/`.

The extended60-cycle/600-second run is recorded separately below. Upper-bound memcheck and
integrated racecheck/initcheck remain open regardless of these numerical passes.

Follow-up gate strengthening: the initial live group-cancel fixture began after
reset, so its retained committed history was empty. It established cancellation
and peer behavior, but not preservation of a nonempty prior group-lane history.
The gate now keeps a completed serial history before cancellation and requires
unchanged nonzero transcript/committed counts alongside zero physical state.
Six host counterexamples cover that distinction. Fresh GPU revalidation is
required for the strengthened assertion; original v1 evidence is not rewritten.

## Extended grouped soak: accepted finite workload

`validation/http-grouped-soak-v2/` passes **610 cycles over600.15 measured
seconds** on worker `184f14b7...`. Every measured group returns the exact32-token
control for each lane; reset/invalid-second-lane and live cancellation/recovery
paths are also checked. Fixed predeclared budgets were not changed.

- Maximum warm RSS growth: server262,144 bytes, worker0. Descriptor growth0
  for both processes.
- No observed CUDA-free-memory decrease from the warm resident baseline. After
  closing both sessions, free memory is60,973,056 bytes above the empty-session
  warm baseline, within the fixed recovery requirement. These are device-wide
  Thor/unified-memory observations, not precise per-worker allocation attribution.
- Median measured group0.8679s,73.74 aggregate tokens/s over610 samples. Five
  serial-pair baseline samples have median1.3355s,47.92 aggregate tokens/s.
  Workload and source/binary hashes are explicit; no universal throughput/SLA
  claim or automatic batching of independent arrivals is implied.

Actual outputs are copied to `../tmp/http-grouped-soak-v2/`. This closes the
specified ten-minute grouped resource workload, not arbitrary-duration stability
or the unresolved integrated sanitizer gates.

## Final default-math candidate resource confirmation

After the long-prefill change, worker `1aa1e0c2...` passes the same fixed-duration
workload independently in `validation/http-bounded-grouped-soak-v4/`:561 exact
groups over600.12 measured seconds. Maximum warm RSS growth is1,441,792 bytes
for the server and22,282,240 bytes for the worker, both below the unchanged32MiB
budget. Neither process grows descriptors; observed device-wide free memory does
not drop from the warm baseline. Closing both sessions recovers free memory to
24,215,552 bytes above the empty-session warm baseline. Actual responses are in
`../tmp/http-bounded-grouped-soak-v4/`.

The same candidate passes legacy resident32K/sampling, scalar cancellation and
three eight-client HTTP admission rounds in `validation/worker-bounded-regression-v1/`,
`validation/cancellation-bounded-regression-v1/` and
`validation/http-bounded-regression-v1/`. Its actual grouped worker protocol also
passes memcheck (`INTEGRATED_SANITIZER_RUNNER.md`). Transport termination now waits
for quiet live helper groups, and both grouped gate CLIs unwind termination safely.

Soak timing remains observational: five early serial-pair samples and561 group
samples have different sample counts/timing. A separate warmed AB/BA benchmark
with matched counts and numerical-evidence binding is being verified; the soak
alone is not a balanced performance experiment. Full acceptance remains false
while required unexcluded sanitizer coverage is unresolved.
