# Integrated sanitizer runner

`tools/check_lfm3_sanitizer.py` uses `OwnedJob` for every sanitizer invocation
and `owned_main` for signal unwinding. All work after spawn is inside an
unconditional ownership `finally`. Successful leader exit, timeout, SIGTERM,
SIGINT, and postspawn exceptions all quiesce the owned process group, including
launcher descendants. Cleanup failure raises instead of reporting acceptance.

## Default gate

Without selectors the runner still executes memcheck, racecheck, synccheck,
positive/negative initcheck controls, the cuBLAS producer control, and (only
after successful controls) native initcheck. Timeout is per invocation and must
be in `(0, 3600]` seconds. `--print-limit 4` limits reporting only; error-exitcode
86 remains mandatory and no launch limits or new kernel filters are added.

Native runs must produce parseable application JSON with the native backend,
identical request input IDs, valid vocabulary IDs, correct generated/teacher
lengths, legal early stopping and suppression, and successful reset checking
when requested. Results must match across instrumented executions. Supply
`--expected-result` to additionally pin input IDs, generated IDs and teacher
argmax to an independently accepted native result for the identical request.
Cross-tool consistency without that option is not an independent numerical
accuracy oracle; existing reference/logit acceptance remains necessary.

`passed` and `full_acceptance` require all default tools and controls, valid
matching application results, and no nvjet exclusion. The historical
`--exclude-nvjet-synccheck` option remains explicit, is recorded, and can never
produce full acceptance. A failed normal gate exits 86.

## Independent full-native diagnostics

`--tools` accepts one or more of `memcheck racecheck synccheck` and requires
`--diagnostic-only`. Diagnostics skip initcheck and its controls, preserve the
entire supplied request, and never substitute a smaller fixture. They report
`scoped_passed`, `selected_tools`, `controls_ok: false`,
`initcheck_status: not-run-diagnostic-only`, and **always** `passed: false` and
`full_acceptance: false`. Exit 0 means only scoped success. Even diagnostics
with all three tools are not a full gate.

From the repository root, with these variables pointing to the owner's existing
full-32768 prefill artifacts (do not edit or regenerate the request):

```bash
python tools/check_lfm3_sanitizer.py \
  --sanitizer "$SANITIZER" --probe "$PROBE" --binary "$BINARY" \
  --checkpoint "$CHECKPOINT" --request "$REQUEST_32K" \
  --expected-result "$EXPECTED_32K" --output "$SYNCHECK_OUTPUT_NEW" \
  --timeout 3600 --diagnostic-only --tools synccheck
```

```bash
python tools/check_lfm3_sanitizer.py \
  --sanitizer "$SANITIZER" --probe "$PROBE" --binary "$BINARY" \
  --checkpoint "$CHECKPOINT" --request "$REQUEST_32K" \
  --expected-result "$EXPECTED_32K" --output "$MEMCHECK_OUTPUT_NEW" \
  --timeout 3600 --diagnostic-only --tools memcheck
```

Output directories must not already exist. The probe argument remains required
for CLI/provenance compatibility, but diagnostics do not execute it. Omit
`--diagnostic-only --tools ...` to run the complete default gate. Scoped passes
cannot be combined to claim that the default controls or integrated initcheck
passed.

## Host tests and compatibility

```bash
python -m pytest -q tests/test_lfm3_integrated_sanitizer_owned.py
```

These tests use an executable fake sanitizer (no GPU) and cover orphaned
descendants after leader exit, timeout, SIGTERM, postspawn exceptions, bad or
missing application JSON, expected-result mismatch, selector guards and excluded
scope. The initial legacy fake in `tests/test_lfm3_sanitizer_harness.py` wrote
an invalid request and no native JSON, exposing
two old false passes. The fixture was subsequently corrected to provide valid
requests and application JSON; missing/malformed/mismatched results are now
explicit rejection cases. No production validation was relaxed to keep tests green.

No GPU execution or remote acceptance is established by these host tests.

The runner also pins start SHA256 identities for binary, probe, sanitizer, request
and optional expected result. Changed or missing artifacts stop further launches,
force every acceptance flag false, and return86. Valid-output counterexamples
mutate each pinned input and remove tool/request files; they cannot pass merely
because the sanitizer text and application JSON look clean.70 related host tests
pass after fixture/identity remediation. Group acceptance CLI entrypoints now
unwind SIGTERM/SIGINT through cleanup, and the transport verifies quiet live
process groups after final kill, propagating incomplete cleanup to repeated callers.

## Parent GPU evidence

`validation/sanitizer32k-bounded-memsync-v1/` runs the full unchanged32768-token
request on single `e436191d...`. Memcheck is clean with the expected application
result. Unexcluded synccheck produces the correct result but reports internal
NVJET launch-tracking errors; it is **not accepted**. That historical invocation
preceded the new start/end identity fields; it is not retroactively labeled as
having performed those checks.

`validation/grouped-worker-memcheck-v1/` instruments the actual worker protocol,
not only a standalone kernel. A recorded exec-only adapter runs real worker
`1aa1e0c2...` under memcheck, forwarding inherited cancellation and JSON pipes.
The complete grouped gate passes73 bitwise-logit steps, non-greedy replay, live
and pre-dispatch cancellation, retained nonempty history, recovery and aggregate
32K, then clean shutdown. The tool finishes with **zero errors**. Adapter, actual
worker and sanitizer hashes are distinguished; this does not establish initcheck,
racecheck or synccheck acceptance.
