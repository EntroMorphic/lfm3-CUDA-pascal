# Owned GPU validation jobs

The first paired upper-bound memcheck used `subprocess.run(timeout=600)`. That
killed only the sanitizer leader, leaving its TreeLauncherSubreaper and native
CUDA child alive. The shell then continued into a32K numerical run. This violated
the serialized-GPU requirement. The two matching owned descendants were verified
by command/group and killed as root inside their container; subsequent process
inspection found none. No unrelated process was killed.

Evidence is retained in `validation/decode-batch-memcheck-v1/`. Its scratch case
was clean; the upper case is a **timeout, not a pass**. The numerical output in
`validation/context32k-fastpath-v3/` is retained but **not accepted** because of
overlap. A fresh serialized rerun passes in `validation/context32k-fastpath-v4/`
for `ad757f34...`, with unchanged strict SDPA tolerances and exact reset/argmax.

## Cleanup and lifecycle remediation

`lfm3_owned_process.py` now owns a new process session, restores the child's
original signal mask before exec, and requires the leader and every live
same-group descendant to stop before another job starts. Zombies are not running
CUDA work. Cleanup failure propagates instead of permitting the next fixture.

Independent review then found SIGTERM and post-spawn report/observer setup could
bypass that cleanup. Original host failures are preserved in
`validation/redteam-probe-launcher-before.log`. `OwnedJob` retains ownership before
spawning under a signal mask and is closed unconditionally around all setup,
waiting and teardown. CLI SIGTERM/SIGINT unwind once; repeated termination cannot
interrupt cleanup. Cleanup itself masks interruption and serializes its state.
SIGKILL cannot run Python cleanup; a supervisor must still terminate the whole
owned job group in that case. Same-group children/readable Linux `/proc` are
explicit assumptions, not a claim against a hostile escaping executable.

Tests inject startup SIGINT, single/repeated SIGTERM, post-spawn file-write errors,
sleeping helper descendants, exited/erroring leaders, cleanup failure and an
invalid shared group. The paired wrapper also fails fast after a timeout, retaining
its unrun fixture list. CPU/RSS/system-memory observations and launch dumps are
diagnostics, not sanitizer success criteria.

## Acceptance-record hardening

The paired wrapper now requires the requested normal/compact reference mode,
four/two GPU owners, exact fixture geometry/counts, one matching coverage record,
all required coverage flags, `float_bits` on every logit comparison, full-cache/
retired/reset checks, and `worker_integration:false`. False mode, weaker float
comparison and fabricated coverage no longer pass. Actual normal and compact GPU
records are separately postvalidated with the strengthened validator; both retain
209 comparisons,62 paired calls,55 pre-mutation rejections and67 retired checks.

The complete host suite passes722 tests in
`validation/host-owned-job-remediation-v2.log`. Host counterexamples are not GPU
proof. Standalone batch kernels remain unintegrated into worker/service serving,
and upper-bound memcheck and integrated racecheck/initcheck remain open.

The final runner's actual scratch smoke passes in
`validation/decode-batch-final-runner-v5/` with complete group cleanup. The later
300-second racecheck timeout was also cleaned up; post-run process inspection
found no matching live descendants. Its resource log missed reparented children,
so racecheck now uses process-group rather than parent-tree sampling. Unlimited
launch reporting is now explicit in both diagnostic wrappers;100 printed launches
in older logs must not be treated as a complete launch ledger.

Two additional regressions verify same-group resource observation after parent
exit and unlimited racecheck reporting/cleanup. The final full suite passes
**724 tests** in `validation/host-owned-job-diagnostics-v3.log`. The corrected
60-second diagnostic records all three owned processes and complete cleanup in
`validation/racecheck32k-diagnostic-v4/`; its timeout remains non-accepting.

Later transport/gate hardening adds quiet-group verification after final kill,
propagating incomplete cleanup to repeated/concurrent callers while still closing
pipes. Native/HTTP grouped gate entrypoints now unwind SIGTERM/SIGINT through
their cleanup blocks. The integrated sanitizer runner also pins start/end artifact
identities and requires valid native application results; scoped runs and excluded
coverage cannot claim full acceptance.94 targeted lifecycle tests pass in
`validation/host-transport-sanitizer-lifecycle-v1.log`. The combined suite including
diagnostics has1021 passing tests in `validation/host-acceptance-followup-v5.log`.
