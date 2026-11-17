# Serving red-team round 2

Starting commit: `15cb8e8`. Both independent reviews and host counterexamples
were used; no new cache-corruption defect was established by source review.

## Reproduced defects

1. A worker could send duplicate acknowledgments and falsely satisfy a later RPC.
   The transport previously retained surplus frames without correlation. Internal
   protocol **2** now requires monotonically increasing positive `rpc_id` values,
   validates exact reply identity/type, and fails closed on trailing data. Native
   mutation checks the next expected ID first. Tests reject coalesced/delayed
   duplicates, wrong/missing/boolean IDs, and transport-owned ID injection.
   Protocol1 workers are intentionally incompatible and must be rebuilt together
   with their clients. Never clear unsolicited bytes to conceal corruption.
2. Request deadlines previously began after synchronous encoding. One absolute
   deadline now covers encoding, owner admission/waiting and decoding. A single
   bounded daemon frontend worker keeps slow nonpreemptible CPU work off handler
   threads; late results cannot reach native inference or release a newer lease.
   This bounds caller waiting, not the CPU instruction stream itself. Tests cover
   slow encode/decode, queue expiration and completion/lease races.

Original failures: `validation/redteam-http-round2-before.log`. They are retained
and were not rewritten as passes. Complete host suite including new soak resource
counterexamples:581 passes in `validation/host-redteam-round2-v1.log`.
The final red-team-only scope passes575 tests, explicitly excluding pending
soak/cancellation suites: `validation/host-redteam-round2-final.log`.

## Acceptance false-pass remediation

- Duplicate-key HTTP rejection now duplicates a field of an otherwise valid
  live-session request, followed by a matching exact-ID positive control.
- Split-delimiter input now uses fragments that are individually benign but
  reserved after concatenation. A benign multipart positive control must match
  normal token IDs and output. Missing-guard counterexamples fail the gate.
- The resident worker now supports explicitly opted-in startup audit tracing:
  `--sampling-trace-root NEW_DIRECTORY`, with per-request `trace_sampling:true`,
  at most32 output steps, and paths derived solely from RPC IDs. No request may
  select a filesystem path, and the HTTP service never enables this debug option.
  The resident gate replays every sampled step through the independent sampler
  process, requires a non-greedy choice, and tests reset/interleaving reproducibility.
  This closes the greedy-bypass gap in the worker's separate option-parsing path.

The strengthened resident gate passes, including32K budget/overflow, exact IDs/
logits/counts, A/B isolation and independent stochastic replay:
`validation/worker-redteam-v4/`. Worker hash `1f3401db...`; single reference remains
`faf3d2df...`, sampling oracle `3026595f...`. Native HTTP revalidation is recorded
separately, not inferred from the resident result.
The strengthened native HTTP gate also **passes** three eight-client rounds,
exact-output recovery, both corrected negative/positive fixtures and fault/reset
checks: `validation/http-redteam-round2-v4/`. Its worker hash matches the resident
sampling/32K gate. Actual response is copied to
`../tmp/http-redteam-round2-v4/control-response.txt`.

## Remaining gates

Cooperative token-boundary cancellation comes before fixed two-lane GPU decode
batching. Cancellation during prefill will not preempt that prefill; destroying
one cancelled cache must not affect its peer. Two sequential forwards must never
be labeled GPU batching. Host/worker resource plateau and end-to-end latency are
being measured separately from GPU memory and kernel throughput. Integrated
racecheck timeout and cuBLAS initcheck tracking remain open until real evidence
closes them; no isolated-probe substitution or zero-fill workaround is permitted.
