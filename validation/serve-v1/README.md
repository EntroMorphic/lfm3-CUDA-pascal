# serve-v1: phase 3's bar

Status: **rules pass** (2026-11-19); red team to follow. On the rule run (`job.sh`, tree `cdaa0ad`, builds `text-v33`
and `parity-v33`) rules 1 to 11 pass, and rule 12 passes with its full-queue case judged unreachable by the owner
(below). Records: [records/](records/) (`SHA256SUMS` verified after the copy; the reference checks' logit traces are left on the card).

Phase 3's exit gate (PLAN): "the reference's server checks pass; requests that would exceed memory are refused,
never crash". Decision 8 (PLAN) sets how: both builds, a text-only server build, paired decode on the compact caches.

## Seen before the rules were set

- The imported server and worker do not compile for `sm_61`: `lfm3_worker.cu` includes the speech, batch-decoder
  and vision code (27 errors, red-team-1 `job2.log`). A scratch copy with those left out compiled with 0 errors (nvcc
  front end only, not on the card).
- The pinned control (the reference's 86-token reply, `validation/frontend-remediation/post-remediation-native` in
  the reference repository): `parity-v32` gives its ids exactly; `text-v32` first differs at token 58 (87 tokens).
- Admission limits are token counts (2 sessions, 32,768 tokens in total); nothing compares a request with free
  memory. A failed allocation makes the server answer 503 to every later request until it is restarted.
- Paired decode refuses the compact window caches, the default above 4,096 tokens of capacity.

## Rules

Judged on the final phase 3 tree. "Both builds": the shipping configuration and the parity configuration (decision 4).

**Outputs unchanged** (every one must hold):

1. Parity gate 7,826 of 7,826; fixtures with the compact windows off and on; NLL files at 256, 257, 4,096 and 32,768
   byte-identical to text-correct-v2's; long-decode-v1's five cases identical; boundary sweep 33 of 33; host suite as
   recorded. If `liquid-lfm3`'s device code is unchanged from `text-v32` function for function, the NLL and long
   decode runs may be replaced by that comparison.
2. The host side as imported: `lfm3_http.hpp`, the tokenizer, the chat template and the other host front ends
   byte-identical to the import (SHA-256 list); the text-only switch's values checked (`#error` on any other).

**The reference's server checks** (`tools/thor/run_native_gates.sh`'s serving stages, run through a Pascal runner
with `LFM_ALLOCATION_METRICS=1`), on both builds, every one must pass:

3. `check_lfm3_sampling.py` (the sampling oracle).
4. `check_lfm3_worker.py`, with the reference's long request (32,768).
5. `check_lfm3_batch_worker.py`: paired decode equal to serial in every logit bit, including its pair of
   capacities 3 and 32,765 (compact caches).
6. `check_lfm3_server.py`, `--concurrency-rounds 3`.
7. `check_lfm3_cancellation.py` and `check_lfm3_http_cancellation.py`.
8. `check_lfm3_http_batch.py` with `--require-exact-accounting`, its 600 s soak included.
9. `check_lfm3_native_server.py`: the single-binary server, against the Python server and worker.

Control: the parity build with the reference's pinned files (SHA-256 checked as the reference checks them); the
shipping build with `text-vN`'s own output on the pinned request, recorded by the CLI before the checks run.

*Adaptations to the card (development runs, 2026-11-19), each a copy in `tools/pascal/` with its whole diff here:*

- The checks' Python front end runs the reference's pinned stack (Transformers 5.22.1, tokenizers 0.23.3, Jinja2
  3.1.6, Python 3.12.3) in its own environment on the card (`/workspace/build/venv-ref`); the container had none.
- `check_lfm3_worker.py` ran the CLI beside the resident worker; both cannot hold the weights in 8 GB. The copy
  computes the same CLI references before the worker loads them (`check_lfm3_worker.diff`: the calls moved, still
  inside the check's `preflight_only` guard, and an import path; every comparison the same).

*Amendment to rules 3 to 9 (owner, 2026-11-19, after development runs).* The parity build's prefill is 15 times the
shipping build's (429 s against 28.6 s at 32,763 tokens). On the reference's timing it failed three checks only by
time: the worker check's 300 s limit on its 32K CLI reference, the grouped-worker check's 180 s worker deadline on its
32K pair, and the native-server check's batch cancellations (0.05 to 0.5 s), which all landed before its prefill
ended. It passed the other five, four of them against Thor's pinned control (the fifth such check is the
native-server one). The figures are from development runs (`records/development/`). For the parity build only,
those three run with their timing scaled: 1,800 s limits, and cancellation delays from 1.0 to 2.0 s in steps of
0.05 s after the reference's four (a first try with 1, 2, 4 and 8 s missed: decode ran between 1.0 and 2.0 s) (`*_parity.diff`). Every
comparison is unchanged. The shipping build runs the reference's timing.

**Paired decode on the compact caches:**

10. Paired decode equal to serial decode in every logit bit, and the same ids, for 600 steps: a session of capacity
    4,097, 8,192 or 28,672 (prompts of 3,400, 7,500 and 28,000 tokens) paired with one of 4,096 (prompt 257). The
    256-slot windows wrap several times during the steps.

**Memory** (requests that would exceed memory are refused, never crash):

11. At the limits the server admits, nothing fails to allocate. Run through the server, each filled to its capacity:
    one session of 32,768; two of 4,096 and 28,672; two of 2,048 and 30,720; two of 16,384; and `/generate_batch` on
    two of 4,096 and 28,672. Peak device memory (whole card, sampled) at most 7,740 MiB (the 7,996 free less 256).
    If a configuration exceeds that, the default total capacity is lowered to the largest multiple of 1,024 at which
    every configuration above (scaled) passes, and the server's `ready` reports it.
12. Past the limits, refused and still serving: a third session, a capacity over the total, a prompt plus
    `max_new_tokens` over a session's capacity, and a full queue each get the reference's 400 or 429, and the control
    then gives the same ids as before.

Not judged: speed (the reference sets no serving speed). Recorded: decode ms per step through the server against the
CLI.

## Predictions

- Rules 1 and 2 hold: the serial path is not touched.
- The checks pass on both builds; the shipping build's control differs from the reference's from token 58.
- The reference's limits fit: the worst configuration about 6.9 GB (a survey's estimate; 6,580 MiB).

## Results

The rule run: `job.sh` (tree `cdaa0ad`), log `records/jobs/serve-v1.log`. The host suite on the same tree:
`records/host-suite-cdaa0ad.txt`.

| # | Rule | Result |
|---|---|---|
| 1 | Serial path unchanged | **Pass.** Parity gate 7,826 of 7,826. `liquid-lfm3` and `lfm3-score`: device code identical to `text-v32` in all 93 and 94 functions (`sv1-serve/*.sass-compare.txt`; `text-v32`'s NLL and long-decode results stand). Fixtures pass with the compact windows off and on; the sanitizer reports 0 errors on the fixtures and the 257- and 513-token compact sweep; boundary sweep 33 of 33. Host suite as recorded (2,397 passed, 5 skipped, 1 needing Transformers 5.x) |
| 2 | Host side as imported | **Pass.** 21 host files (`lfm3_http.hpp`, the tokenizer, the text front ends, sessions, limits, cancellation, image front end, `lfm3_server.cu`, the Python server and front end) byte-identical to the import (`git diff fc130a3`: none). `LFM3_TEXT_ONLY=1` builds; `2`, `on`, `01` and empty stop the build |
| 3-9 | The reference's serving checks | **Pass,** all eight on both builds (`sv1-serve-shipping/status.txt`, `sv1-serve-parity/status.txt`). The shipping build against its own control (87 tokens, differing from the pinned control from token 58, as predicted); the parity build against Thor's pinned control, with the amended timing |
| 10 | Paired decode on the compact caches | **Pass.** Both lanes of all three pairs (the session of 4,097, 8,192 or 28,672 on the compact caches; its partner of 4,096 not): 600 steps, 581 of them paired (the worker traces at most 32 tokens per request, so the run is 19 continuations, each opening with one serial step); ids and every BF16 and FP32 logit equal to serial decode |
| 11 | Memory at the limits | **Pass,** for the five configurations tried (greedy, 32 new tokens; prompts 33 to 76 tokens short of capacity), no failure. Peak: 6,451 MiB (one of 32,768), 6,833 (4,096 + 28,672), 6,841 (2,048 + 30,720), 6,469 (16,384 + 16,384), 6,845 (`/generate_batch` on 4,096 + 28,672); limit 7,740 |
| 12 | Past the limits | **Pass** (owner, 2026-11-19: the full-queue case unreachable). Third session: 429 `session_quota`; capacity over the total: 429 `session_quota`; prompt plus `max_new_tokens` over capacity: 400 `invalid_request`; after each, the control gives the same ids. Full queue: with `--queue-size 1`, while one session generated and the other waited, the only further text request went to a busy session and was refused with 409 `session_busy`; the control then gave the same ids |

Recorded, not judged (`sv1-serve/speed.json`): the control through the CLI, 0.218 s of prefill and 15.3 ms per
decode step (2.6 s of process wall time with the weights' load); through the server, warm, 1.58 s per request for the
same 87 tokens (2.19 s for the first request after start).

### Rule 12's full queue (owner, 2026-11-19)

A full queue cannot occur in the text-only server at the reference's limits. The imported HTTP layer accepts
`--max-sessions` 1 to 2 only (`lfm3_http.hpp`, held as imported by rule 2); a job leaves the queue when it starts
running, so only waiting jobs count against `queue_size` (at least 1); every queued text operation holds its
session's lease or, for `create`, a place in the session quota; and the text-only build refuses `/speech` with 501
before the queue. So while one session's job runs, at most one job (the other session's) can wait, and any further
request is refused first: 409 `session_busy`, 429 `session_quota` or 501. The owner first chose to force a full queue;
that needs more than two sessions or speech, which the host side as imported does not allow, and the owner then
judged the case unreachable. 429 `queue_full` remains covered by the host suite's server tests on a fake engine.

### Predictions against the measurements

- Rules 1 and 2 held, and the serial path's device code did not change at all.
- The checks pass on both builds, after the adaptations and the amendment above; the shipping build's control
  differs from token 58, as predicted.
- The worst configuration peaked at 6,845 MiB (predicted about 6,580 MiB, the survey's 6.9 GB).
