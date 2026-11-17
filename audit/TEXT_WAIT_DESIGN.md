# Text waits behind speech: design for the next round

Status: design only (2026-11-11). No engine or server code is changed by this note. It closes nothing; it prepares
the round Tripp scheduled after audio-assurance-v1 ("fix the text wait next round"). Nothing here was measured on Thor
for this note; every claim about today's behavior is read from the code (cited) or from earlier measured records.

## 1. Why a text request waits today

One engine-owner thread makes every engine call, and while any speech row is running it makes only speech calls.

- **One owner thread.** `Service` starts a single `owner_` thread running `run()` (`native/include/lfm/lfm3_http.hpp:465`).
  Every engine operation (`create`, `reset`, `close`, `generate`, `generate_batch`, `speech_admit`, `speech_step`)
  is an `rpc()` made from that thread (`lfm3_http.hpp:1413`). The worker is compiled into the server and runs in
  process (`native/src/lfm3_server.cu:19-20`, `CoreBackend::call`), so there is no second engine to send text to.
- **Running speech rows own the loop.** `run()` chooses, each pass, between one speech cycle and one queued job:
  `speech = !running_.empty() || speech_first();` (`lfm3_http.hpp:1719`). While `running_` (the admitted speech rows)
  is not empty, the owner calls `speech_cycle()` and never pops `queue_`, where text jobs wait (`lfm3_http.hpp:1719-1728`).
- **The batch drains for a text job; it does not pause for one.** `speech_first()` admits queued speech only if it
  arrived before the oldest queued job (`lfm3_http.hpp:1523-1525`). So a waiting text job stops new speech from being
  admitted, but the rows already running continue to the end ("speech rows run on until they finish",
  `lfm3_http.hpp:1717-1718`; a ticket "runs to its end ... whatever its deadline", `lfm3_http.hpp:1014-1016`).
- **How long a drain can be.** A speech request is bounded by `kSpeechStepBudget = 6144` steps
  (`native/src/lfm3_worker.cu:531`). At about 37 ms an 8-row step (validation/speech-speed-v1/README.md:85), one
  drain can last about 230 s on a quiet Thor, and longer on a shared one.
- **Every text request waits for a drain, and a text client makes two.** A client creates a session and then
  generates; both are jobs in `queue_` (`create` `lfm3_http.hpp:504-533`, `generate` `lfm3_http.hpp:570-601`), and
  each waits for the rows running when it arrived. Measured before: text latency was 34.0 s while the rows running
  at its arrival finished 15.0 s later (validation/speech-batch-v1/README.md:114-117).
- **Past its deadline.** A text request's deadline is `request_timeout` (330 s, `lfm3_http.hpp:429`) from arrival.
  Two drains near the step budget exceed it, and the job is answered 504 `queue_deadline` (`lfm3_http.hpp:1733`).
  That is red-team finding 5.
- **The reverse wait.** A text generation is one engine call: op `generate` runs prefill and the whole decode loop
  before it returns (`native/src/lfm3_worker.cu:428-505`). Speech that arrives during it waits for all of it. While
  speech rows run this cannot happen, because text is not started then. When no rows run, a long text generation
  delays the first audio of new speech by its full length.

**Not a GPU limit.** Text and speech use separate engine state: each session has its own `Runner` (its caches and its
cuBLAS handle and workspace; `lfm3_worker.cu:454`, `native/src/lfm3_main.cu:397`), and speech has its own
`speech_runner` and `Batch` (`lfm3_worker.cu:635-639`, `679-692`). The wait comes from how the owner thread schedules
work, not from contention for the GPU.

## 2. Designs

The hard rule: every reply, text and speech, stays bit-identical to the same request alone. All three designs below
keep the engine's arithmetic. They differ only in when the owner thread makes its calls. Design C is the exception:
it changes the arithmetic, and that is why it is rejected.

### A. Run short jobs between speech steps (scheduler only)
Let `run()` pop session operations (`create`, `reset`, `close`) between speech cycles even while rows run. Each takes
milliseconds. Leave `generate` as it is today.
- **Exact:** the same calls with the same inputs; only their order relative to speech steps changes (argument in §2.D).
- **Effect:** a text client then waits for one drain instead of two, so the wait roughly halves. The drain itself
  remains.
- **Complexity:** small. The `run()` choice and the `speech_first` rule apply only to `generate` and `generate_batch`.
- **Verdict:** a useful first step. On its own it is not the fix.

### B. Sliced text generation, interleaved token by token with speech steps (recommended)
Split op `generate` into resumable steps in the worker, and have the owner thread interleave them with speech steps.

Worker (`lfm3_worker.cu`):
- `text_begin {op generate's fields}` does everything op `generate` does before its decode loop (`:428-469`):
  validation, the ledger lease, the sampler, and the prefill or reuse continuation. It keeps a `PendingText` in the
  session: the sampler with its RNG state, `generated`, `t`, the lease, and the options.
- `text_step {"session_id", "tokens":k, "cancel":bool}` runs up to k iterations of the existing loop (`:472-488`) and
  returns `{"busy":true}`, or the reply op `generate` returns today (`:489-503`), or the cancelled reply (`:504-512`).
- Op `generate` becomes `text_begin` followed by `text_step` until finished. That keeps one code path: today's alone
  path is the sliced path with k equal to the token count, so "alone" is computed exactly as the sliced path computes
  it.
- A session with a `PendingText` refuses every other op (defense in depth; the Service lease already serializes
  sessions).

Service (`lfm3_http.hpp`):
- The owner thread keeps at most one active text generation (`text_`), served first in, first out, as text is today.
- Each pass of `run()`:
  1. reap idle sessions;
  2. run short session ops at the head of `queue_`;
  3. start the next `generate` with `text_begin` if no text is active;
  4. admit queued speech into free rows;
  5. one `speech_step` if rows run;
  6. one `text_step` if text is active.
- The slice is k = 1 token while speech rows run, and a long slice (for example 256 tokens) when none do, so text
  alone keeps its speed.
- The schedule is deterministic. The design does not need that for exactness, but it makes the tests reproducible.
- Text no longer blocks speech admission, so the drain rule (`speech_first`) is retired for `generate`. It stays for
  `generate_batch`, which this round leaves as one whole call; see red-team item 11.
- Cancellation and deadlines for text are checked between slices by the owner. A gone client or a passed deadline
  sends `"cancel":true`, and the worker takes today's cancellation path (cache reset, `cache_valid:false`).

Expected effect (estimates from earlier measurements; the rules decide):
- **Each speech step gets longer:** an 8-row step goes from about 37 ms to about 55 ms. The added ~18 ms is one
  single-row LM decode step, 16.3 ms (validation/speech-speed-v1/README.md:7), plus host sampling.
- **Still faster than real time for TTS:** 55 ms is less than the 80 ms of audio a frame step produces.
- **Text under 8 rows:** about 18 tokens/s instead of waiting for a drain. 128 tokens take about 7 s end to end
  instead of up to minutes.
- **Interleaved streams have less headroom.** They already showed stalls on a busy GPU. Rule 3 below tests them
  explicitly.

Cost:
- **Complexity:** moderate. The worker refactor is mechanical: the loop's locals move into a struct. The scheduler
  change is the risky part, and the red-team covers it.
- **Prefill is not sliced.** `Runner::forward` supports only an empty-cache prefill in one call, or one-token decode
  (`lfm3_main.cu:540-543`). Splitting a prefill would be new arithmetic, so a prefill stays one call and speech pauses
  for its duration (red-team item 9).

### C. Fold text rows into the speech batch's decoder (rejected for this round)
Run a text session as an extra row of `lfm3_batch::BatchDecoder`, so the LM step for 8 rows covers it at almost no cost.
- **Not exact:** the batched decoder "is a new arithmetic path, not bitwise the serial decode"; the two agree for 19
  to 128 tokens before diverging (validation/batch-decode-v1/README.md:36-37). Text alone uses `Runner::forward`.
  Moving text to the batch path would change every text reply compared with today's validated references (text gate,
  depth tripwire, the HF comparisons).
- It would also need session capacity up to 32768 inside a batch built for `kSpeechCapacity = 4096`
  (`lfm3_worker.cu:528`), plus prefix reuse inside the batch.
- **Verdict:** this is the fastest design, but it only makes sense as a separate, re-validated text path, if speed
  under mixed load becomes the goal.

### D. Why interleaving is exact (A and B)
1. **Disjoint state.** A session's `Runner` (caches, cuBLAS handle, cuBLAS workspace) is touched only by that
   session's ops. The speech `Batch` (decoder, depth decoder, codecs and their streams, logits and mask buffers) is
   touched only by speech ops. The weights are shared but read-only.
2. **Host-serialized ops.** The owner thread makes one call at a time, and each op runs to completion before the next
   starts.
3. **No speech GPU work is in flight at the end of a step.** `Batch::step` synchronizes the device after the LM step
   (`native/src/lfm3_speech.cuh:461`) and collects every codec chunk submitted that step before returning
   (`lfm3_speech.cuh:467-475`); `finish_row` synchronizes too (`lfm3_speech.cuh:295`). This must still be verified
   (rule 1's `CUDA_LAUNCH_BLOCKING` run, as rule 12 of audio-assurance-v1 does for codec streams).
4. **Same calls in the same order.** Each request's own engine calls (prefill, then one `forward` per token, the
   sampler on the same logits with the same RNG) are identical whether or not another request's calls come between
   them. The kernels are deterministic for a given input and shape; none depends on what ran before.
5. **One process-global scratch: `split_workspace`** (`native/src/lfm3_batch.cuh:61-70`). Split-K is off by default
   (`k_splits` returns 1 unless `LFM3_BATCH_SPLITK` is set, `:56-59`), so the workspace is unused. Even with split-K
   on, its contents are produced and consumed inside one op, and ops are host-serialized. Red-team item 1 makes this
   explicit.

## 3. Rules, pre-registered (to move into `validation/text-wait-v1/README.md` before any data)

Builds: the text-wait build (twa-N) against assure-s2 (server a8615c00…) and assure-a2 (engine). Thor must be quiet
for every timing rule, with nothing else on the GPU (memory note: timing checks need a quiet Thor). Speed is labeled
decode loop or end to end.

1. **Exact.**
   - Text: the 3 soak text bodies, the text gate's prompts and a sampled body (T 0.528, seed 42) give `generated_ids`,
     text and `return_logits` bit-identical:
     - alone on assure-s2;
     - alone on twa;
     - under 8 streaming speech rows on twa.
   - Speech: all 42 cases, unstreamed and streamed, with every event identical to alone, while a text client
     generates continuously.
   - The same check with `CUDA_LAUNCH_BLOCKING=1`.
   - `lfm3-speech` regression 36/36, engine exactness 41/41, the text gate and the depth tripwire are unchanged.
   - Pass: 0 differences.
2. **Text no longer waits for a drain (end to end).**
   - Setup: 8 rows of long-form TTS streaming, each with at least 60 s left. A new client then creates a session and
     generates 128 tokens of the reflective prompt (stop disabled).
   - Pass: the median of 10 arrivals is at most 3.5 × its median alone and at most 10 s, and no arrival exceeds 15 s.
   - Today's code must fail this rule. It is run on assure-s2 first, and its failure is recorded.
3. **Speech stays real time under text load.**
   - The stream check's real-time rule, at 8 concurrent streams (TTS and interleaved), while one text client
     generates back to back (128-token requests).
   - Pass: no player runs dry.
   - First audio of a TTS request arriving under that load: median at most 0.6 s over 10 arrivals (0.445 s without
     text load, validation/speech-batch-v1/README.md:112).
4. **Speech is not delayed by a long text generation.**
   - Setup: one text generation of 2,000 tokens is running; a TTS request arrives.
   - Pass: its first audio arrives within 1.0 s. Today it waits for the whole generation, so this fails on assure-s2.
5. **Nothing slower alone.** Paired medians, twa against assure-s2, interleaved runs, 5 pairs each:
   - text alone, decode loop and end to end: ratio at least 0.99;
   - speech at 1 row and at 8 rows (engine, decode loop): ratio at least 0.99.

   Speed round 2 failed on a 2.8% text slowdown, so 1% is the line.
6. **Deadlines and cancellation.**
   - Server started with `--request-timeout 20`; a text request arrives behind 8 rows that each have 60 s left.
     Pass: 200. Today: 504 `queue_deadline`.
   - A text client that resets mid-generation: its session's cache is reset, and the next generate on the session is
     identical to a fresh one.
   - A text request that passes its deadline mid-generation: cancelled at the next slice; its reply is 504 and the
     cache is reset.
   - SIGTERM with a text generation active and 8 rows running: exit 0, and every reply is either complete or an error.
7. **Hostile input and soak.** The audio-assurance-v1 fuzz (rule 5) and a 2 h soak (rule 4), unchanged except that the
   soak's text share rises from 1 in 10 to 1 in 4. Same pass rules.
8. **Sanitizers.** memcheck, racecheck, synccheck and initcheck on the engine test that interleaves `text_step` and
   `speech_step` (4 speech rows and one text session). Pass: 0 errors and outputs identical, with the accepted
   initcheck exception for unloaded rows only.
9. **Timeout defaults** (the audio-assurance-v1 finding):
   - After this change, the engine timeout watches slices: one speech step, one text slice or one prefill. The
     defaults stay 300 s and 330 s; the round records the longest single call measured, which should be a 32k-token
     prefill.
   - The session idle timer must not count time a session spends queued or generating. A test idles a session for
     310 s behind work and expects 404; one queued for 310 s expects its reply.

**Tests that fail on today's code, written first:**
- `tests/test_lfm3_server_text_wait.py`: rules 2, 4 and 6 (first case) as fast black-box checks with short speech;
  run on the harness with `--speech-rows 2`.
- A scheduler unit test on `lfm3_http_harness` (a fake backend that records the call order): with rows running and a
  queued `generate`, the recorded calls interleave `speech_step` and `text_step`.
- `native/tests`: `text_begin` + k × `text_step` equals op `generate`, bitwise (ids and logits), for greedy, sampled,
  full prefill, reuse and cancelled requests.

## 4. Red-team of design B (on paper)

1. **Shared scratch.** `split_workspace` is process-global (`lfm3_batch.cuh:61`). It is safe while ops are
   host-serialized. Fix: make it a member of each `BatchDecoder`, or assert split-K off in the server; add a test
   with `LFM3_BATCH_SPLITK=2`.
2. **Codec streams.** If a later speed change leaves codec work in flight across a step boundary (the obvious next
   optimization), text kernels would overlap it. The state is still disjoint, so it would still be exact, but the
   reasoning in §2.D point 3 would no longer hold. Rule 1's `CUDA_LAUNCH_BLOCKING` run keeps this honest; record the
   assumption in `lfm3_speech.cuh`.
3. **Sampler state across slices.** The RNG must live in `PendingText`, not on the stack. The test is §3's
   begin+steps = generate check for sampled requests, with k = 1, 3 and 256.
4. **The ledger across slices.** The lease begins at `text_begin` and commits at the finish (`lfm3_worker.cu:442`,
   `490-491`). Any error or cancel must call `lease.abort()` and reset the runner, exactly as `:504-515` do. A
   `PendingText` left behind by an engine failure is dropped with the session when the service fails closed.
5. **Session ops while text is pending.** The Service lease already prevents them. The worker refuses them too,
   because if it didn't, a `reset` arriving mid-generation would corrupt `consumed`.
6. **The watchdog.** It watches each `rpc()` (`lfm3_http.hpp:1433-1434`, `1666-1685`). Calls get shorter, so it trips
   only on real hangs or very long prefills. Cancellation targets (`active_`) are cleared at every call
   (`:1420-1425`), so text cancellation cannot rely on the pipe mid-slice; the owner checks between slices instead.
   That is at most one token late.
7. **The idle reaper.** `reap_idle` closes sessions that are not busy (`lfm3_http.hpp:1687-1699`). A session with
   queued or active text is leased (`lease(sid, ...)`, `:576`), so it is busy. Rule 9's test pins this down.
8. **The 429 queue.** The limit counts `queue_ + speech_queue_` (`:1653`, `:1492`). An active text generation leaves
   `queue_`, as an admitted speech row leaves `speech_queue_`, so one more request is in flight than today. Document
   it; the fuzz's 429 counts change.
9. **Prefill pauses speech.** A long prompt's prefill is one call (`lfm3_main.cu:543`). At 32k tokens it can take
   seconds, and every streamed row would stall that long. Options:
   - measure and document the prompt length above which streams can stall (rule 3 uses short chat prompts; add one
     4k-token prompt);
   - later, start a long prefill only when streamed rows have enough buffered audio. The server knows each streamed
     ticket's audio sent against its wall clock.

   Speech admission already pauses rows the same way (an ASR prompt's audio encoder) and passed rule 3.
10. **Interleaved speech headroom.** Interleaved rows spend steps on text slots, so their audio per step is lower.
    Adding about 18 ms per step may make them run dry where TTS rows do not. If rule 3 fails for interleaved rows,
    run text slices every other step when an interleaved row streams. That is a schedule change, not an arithmetic
    one, so exactness is not affected.
11. **`generate_batch`.** It stays one whole call, keeping the drain rule, so its waits are unchanged this round.
    Document this, or slice it the same way in a later round. It needs exactly two sessions and is rare.
12. **Shutdown.** `run()` returns when the queues and `running_` are empty (`:1707`). It must also wait for the active
    text. `cancel_all_` (`:706`) must cancel the active text at its next slice.
13. **Engine failure in a slice.** `fatal()` must settle the active text with a 503, as `speech_cycle` settles every
    ticket (`:1607-1616`). Otherwise a handler waits until its deadline.
14. **A behavior change.** Today a text request that passes its deadline while running finishes, and its client gets
    504 `request_deadline_outcome_unknown` while the cache stays valid (`:1750-1757`). Now it is cancelled and its
    cache reset. That is more predictable, but it is a change: document it in the server's header and in
    `tests/test_lfm3_http_native.py`.
15. **Memory.** No new GPU allocations: session runners and the speech batch exist today. `PendingText` holds host
    data only: the sampler, at most `count` ids, and the 65536-float `last_scores` the session already keeps.

## Note found while reading (outside this design)
The 503 in today's batch-server run (`budget-truncated-s1`, assure-a2/assure-s2 run) came from the check's reference
server, the single-request server-stream-v3 started at `tools/check_lfm3_speech_batch_server.py:102`, not from
assure-s2. That server runs a /speech request as one engine call (op `speech`, `lfm3_http.hpp:1097-1107`), which the
300 s watchdog times as a whole. In the current server with speech rows, the watchdog times each step.
