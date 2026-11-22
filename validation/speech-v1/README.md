# speech-v1: phase 4's bar

Status: **rule run done: rules 1 to 8 and 10 to 14 pass on the final tree; rule 9, the owner's listening, pending**
(2026-11-21). **Then red-team-2 found rule 12's audio-in clip 11 s short of the request limit: with the full clip,
30,720 peaks at 7,759 MiB, so rule 12 failed; the default capacity is lowered as the rule says (29,696; 7,707 MiB),
and the rules the change reaches were run again there and pass ([red-team-2](../red-team-2/)).** The rules were set (2026-11-20) before any speech output of this engine had been generated on the
card.

Phase 4's exit gate (PLAN): "the reference's speech rounds: word error rate against HF, and listening". Decision 9
(PLAN) sets how.

## What it will be

Speech out (text in, text and speech out: the language model's speech loop, the depth decoder, the frame sampler, the
mode machine and the codec) and audio in (the audio encoder), in `lfm3-speech` (the CLI) and in the server
(`/speech`, streamed, with concurrent requests in the speech batch), in both builds. Out of scope: images (phase 5;
image parts are refused), the multilingual round (shelved in the reference, never run), speed (recorded, not judged:
the reference's speed rounds set no quality gate).

## Decisions

Decision 9 (PLAN, 2026-11-20). The owner, on where the speech scorer runs: "Let's run it on the GTX while the GTX is
free." Whisper large-v3 at the reference's pinned revision runs on the card, never while the engine runs.

Decision 10 (PLAN, 2026-11-20, after development 11). The owner, on the reference checks' timing limits: record them,
do not judge them (see the amendment under rule 11).

Decision 11 (PLAN, 2026-11-21, before the rule run). The owner, on the rule run's 16 hours: "Time is life. Let's save
time everywhere we can without cutting corners." The rules are judged on the final tree, as set. A result from a
development run is carried to the final tree only when nothing changed since could reach it, shown in the rule run
itself: (E1) the device code of every final binary equals that development build's function for function
(`sass_compare.py`), and the host sources that differ are listed and are off that result's path; (E2) the final
binaries, run again on a sample of that result's inputs, give byte-identical outputs (every run behind rules 7 and 8,
the 24 escalation runs, the 19 committed LibriSpeech utterances and every 32nd other on both builds; rule 3's 15 clips
on the parity build). Everything a change could reach runs again on the final binaries. Phase 3 set the precedent
(serve-v1 rule 1: identical device code in place of the NLL and long-decode runs).

## Seen before the rules were set

- **What does not compile for `sm_61`** (`nvcc` on `native/apps/lfm3_speech.cu`, 2026-11-20): the speech batch
  decoder `lfm3_batch.cuh`, which runs every speech step of the language model, even a lone request's, through the
  Thor engine's tensor-core kernel in a tile table (17 errors); its optional compressed weights (`gemm_lossless.cuh`,
  `cp.async`, 8); the vision encoder (`ldmatrix`, 2); and, at `ptxas`, the FP32 kernel's staged path
  (`fp32_ops.cuh`, `cp.async` and 64 KiB of shared memory a block), used by the depth decoder, the codec and the audio
  encoder. The staged path does the plain path's arithmetic exactly (its own comment; validation/speech-speed-v1 in
  the reference).
- **The port** (development, 2026-11-20): the batch decoder's projections go through `lfm::gemm`'s prefill kernel at
  every row count (`Problem::wide`): the tensor-core model in the parity build, the plain kernel in the shipping build;
  the compressed weights are left out (`LFM3_LOSSLESS=1` refused); `LFM_FP32_STAGED=0` builds the plain FP32 kernel
  only; `LFM3_VISION=0` puts the vision encoder's interface in place with every call refused
  (`lfm/vision_absent.hpp`). Both builds compile with 0 errors and 0 warnings. Nothing has run.
- **Memory.** Speech adds 545 MiB of weights (audio encoder 209, depth decoder 160, codebook embeddings 95, codec
  decoder 70, audio projector 11). The text server peaked at 6,845 MiB at its limits (serve-v1); the server's default
  is 8 speech rows (1 to 16), each with a 4,096-token cache. By arithmetic, not measured, the sum may exceed 7,740.
- **The reference data on this side** (read-only clone of the reference's repository and its untracked outputs):
  - The committed Whisper transcripts and WER of HF and of the Thor engine: speech-e2e-v1's `rounds/e2e` (10 cases,
    seeds 1 to 3), `rounds/long` (seeds 4 to 15 of `tts-numbers-female` and `tts-long-male`) and `rounds/fridge`
    (`il-fridge`, seeds 1 to 15), all imported here; speech-long-form-v1's `asr/hf-bf16-split.json`.
  - The blind listening sets with their keys: `outputs/listening-c3-v1` (seed 1 of the ten cases, HF and the Thor
    engine; `tts-long-male` the engine split by sentence) and `outputs/listening-split-v1` (the split lighthouse
    paragraph, seeds 1 to 3, and the bread paragraph, engine only). 24 kHz 16-bit WAVs as the engines wrote them.
  - Audio in: audio-in-v1's `round2-full-check.json`, the summary over LibriSpeech test-clean (2,620 utterances,
    53,027 words: HF 2.606%, the Thor engine 2.591%) and the 19 utterances whose word errors differ between the two,
    with both transcripts. The full per-utterance table is on Thor only. LibriSpeech test-clean is on the development
    machine (688 MB, with its SHA-256).
  - Not here: Thor's raw speech outputs (frames, FP32 waveforms), HF's audio beyond seed 1, the depth decoder's and
    codec's FP32 and FP64 anchors, and the audio encoder's stage dumps.
- **The scorer**: `tools/lfm3_asr.py` as imported (Whisper large-v3 in FP16, greedy, English). On the card in its own
  environment (`tools/pascal/setup_asr.sh`: torch 2.7.1 for CUDA 12.6, whose kernels include `sm_60`, Transformers
  5.22.1); the weights at the reference's pinned revision. A 64 x 64 FP16 product ran on the card; nothing has been
  transcribed.

## Development (after the rules were set)

The `dev*.sh` scripts here say what each development run did; its records stay on the card's host (the rule run's are
committed with the results).

- **The codec checkpoint, shared** (development 9). Every row of the speech batch built its own codec decoder, and each
  loaded its own copy of the codec checkpoint (147 MB; harmless on Thor's unified memory). On the card that made each
  row cost about 245 MiB: the CLI ran out of memory past 12 rows, and through the server no number of rows passed
  rule 12. Now one checkpoint is loaded and shared by the rows' decoders (`lfm/codec.cuh`, a constructor taking the
  loaded checkpoint; `lfm3_speech.cuh`); the weights are only read, and each decoder keeps its own buffers. Host code
  only. Both builds give the same 15 rule-3 WAVs byte for byte as before; the CLI's 41 concurrent requests at 8, 12
  and 16 rows equal their alone runs, at 5,933, 6,287 and 6,703 MiB. A row now costs about 102 MiB.
- **Rule 12's outcome** (developments 8 to 10, 14 and 15, `tools/pascal/speech_memory.py`). With text at its limits,
  8 rows peak at 8,073 MiB, and even one row exceeds the limit at a total capacity of 32,768 (`/generate_batch` on
  4,096 + 28,672: 7,745 MiB). Lowered as the rule says: one speech row, and the total capacity to 31,744 (development
  10; the owner kept the rule as written, 2026-11-20: "Let's roll with it as is"); then to 30,720 after the red-team
  review's configuration, audio in at its limit while the text session at its limit is open, peaked at 7,781 MiB at
  31,744 (development 14). At 30,720 every configuration passes beside one row (the highest, that audio-in
  configuration, 7,697 MiB; development 15). Each 1,024 tokens of capacity is about 52 MiB and a row about 102 MiB,
  so each further row takes 2,048 tokens from the default, down to 2,048: 8 rows at 16,384 pass (7,461 MiB at the
  highest) and 16 rows at 2,048 pass (7,709 MiB). The server defaulted to `--speech-rows 1` and, when it serves
  speech, a total capacity of 30,720 (32,768 without speech, as serve-v1 measured); its ready line reports both
  (`{"ready":true,"port":N,"total_capacity":30720,"speech_rows":1}`). *Superseded by red-team-2 (D1): that audio-in
  configuration's clip was 11 s short of the request limit; with the full clip the defaults are 29,696 at one row,
  2,048 less a row, at least 1,024.* An explicit `--total-capacity` is the
  operator's.
- **`/generate_batch` and speech.** A batch is an exclusive job: the server lets the running speech rows finish before
  it runs (`lfm3_http.hpp`, `speech_first`). So that configuration of rule 12 cannot run beside generating rows; the
  check counts it when the rows ran first, their batch still resident.
- **Adaptations to the card** (rule 11): four copies in `tools/pascal/`, each with its whole diff against the
  reference's check here (`check_lfm3_*.diff`). `check_lfm3_audio_in_server.py` runs every `lfm3-speech` reference
  before the server starts (a second copy of the model does not fit) and records each case's mode;
  `check_lfm3_speech_batch_server.py` runs its alone server as the same binary with `--speech-rows 1` (there is no older
  single-request server), gives `--speech-rows 8` to the servers that relied on the reference's default of 8, and
  reports every server's exit code; decision 10's longer deadlines in those two and in `check_lfm3_speech_server.py`
  (also a longer session idle timeout) and `check_lfm3_speech_stream.py`. Every comparison is the reference's.
- **The red-team review** (2026-11-21, after development 13) found, and these were changed:
  rule 12's audio-in configuration had never been resident beside the text session (the clip was sent 20 s before the
  session existed; now the clip goes three times while the filled session is open); `judge_speech_server.py` dropped
  two non-timing criteria (audio-in: text present unless audio; the batch check's other servers' exit codes) and did
  not check that the reports were of the build judged or complete (now it checks the binaries' and references' hashes
  and every case count); rule 6's comparison passed on a subset (now all 25 clips); `--speech-rows` above 1 kept the
  one-row capacity, which could exhaust the card (now the default capacity falls by 2,048 tokens per further row);
  rule 13's text case also broke the chunk limit (now within every other limit); the imported reference records were
  not in git (a `.gitignore` pattern; now anchored, and the records committed); the listening set's index printed
  durations and HF's texts, which identify HF against listening-c3-v1 (now neither; see rule 9 below); the rule run
  lacked rule 3's rendering clause and escalated only two cases (now both).

## Rules

Judged on the final phase 4 tree. "Parity build" and "shipping build" as in decision 4.

**Outputs unchanged:**

1. Text. Parity gate 7,826 of 7,826; the text CLI's and scorer's device code identical to `text-v33`'s function for
   function (or, if not, rule 1 of serve-v1 in full); fixtures; host suite as recorded.
2. The reference's serving checks (serve-v1 rules 3 to 9) pass on both builds with the server built with speech, and
   serve-v1's rule 10 (paired decode) still holds.

**Parity: the speech engine is the Thor engine's** (the parity build):

3. Speech out. `lfm3-speech` reproduces the Thor engine's WAVs byte for byte: seed 1 of the nine single-request cases
   of `prompts.json` (`listening-c3-v1`, the native side by its key), and the lighthouse paragraph split by sentence,
   seeds 1 to 3 (`listening-split-v1`, the native side by its keys; seed 1 is `listening-c3-v1`'s `tts-long-male`).
   The requests are rendered as the reference rendered them (`tools/lfm3_speech_hf.py requests`, and the chunked
   tool's requests for the split runs); the native chat rendering must give the same ids.
4. Audio in. `lfm3-speech` over all 2,620 LibriSpeech test-clean utterances, as audio-in-v1 ran them: the corpus's
   word errors equal the Thor engine's (2.591% of 53,027 words: 1,374), and each of the 19 committed utterances'
   transcript equals the Thor engine's.
5. The speech-only device code is the same in both builds: every function of `lfm3-speech` that is not the language
   model's GEMM (`lfm::gemm`) is identical between the parity and the shipping binaries (`sass_compare.py`). So the
   depth decoder, the codec, the sampler and the audio encoder compute in the shipping build what they compute in the
   parity build, and rules 3 and 4 carry the reference's tier A and B results (depth decoder, codec, feedback
   embedding, sampler, mode machine; audio-in-v1's rules 1 and 2) to both.

**The shipping build's speech, judged as the reference judged its own** (journal/lfm3-tts-quality-bar.md):

6. The scorer. Before it judges anything, Whisper on the card reproduces the committed transcripts: on the HF and
   Thor-engine seed-1 clips of `listening-c3-v1`, each transcript equals the committed one for that clip
   (`rounds/e2e/asr-hf.json`, `asr-native-cases.json`; the split `tts-long-male` against
   `speech-long-form-v1`'s record of it). If any differs, nothing is judged by it until the owner decides.
7. Intelligibility (C1), per case, over `prompts.json`'s ten cases at seeds 1 to 3 (`tts-long-male` split by
   sentence, as the reference ships long text, against `hf-bf16-split.json`): a case is flagged when the shipping
   build's mean WER exceeds HF's worst seed. A flagged case runs seeds 1 to 15 and fails if the shipping build's mean
   is higher and a two-sided Mann-Whitney test gives p < 0.05, against HF's committed 15 seeds where they exist
   (`tts-numbers-female`, `tts-long-male`, `il-fridge`) and otherwise against the parity build's 15 seeds (by rule 3
   the Thor engine, which passed this rule against HF). For the interleaved cases the WER is against the build's own
   text (C4).
8. Endings and length (C2). Every utterance ends by the model's own EOS, and on each case the shipping build's mean
   frames per utterance is within 15% of HF's (the reference measured 0.95 to 1.05 outside the long-form loops).
9. Listening (C3), the final gate. The owner, blind, the shipping build against HF: seed 1 of the ten cases (HF's from
   `listening-c3-v1`), A and B by a draw from the OS's entropy, the key opened after every verdict. Pass: the owner's.

   *The set* (`tools/pascal/listening_set.py`, on the development machine, 2026-11-21; `outputs/listening-speech-v1`,
   not committed): the shipping clips are development 4's runs (`sp-refs-text-s1`), byte-identical to every later
   build's; the rule run checks the final build's against the key's hashes. HF's clips are the same files as
   listening-c3-v1's HF side, whose key the owner has seen; the index gives only each case's prompt, and the owner
   listens without consulting that set.
10. Audio in. Over the 2,620 utterances, the shipping build's corpus WER minus HF's has a paired-bootstrap 95% upper
    bound of at most +0.25 points (audio-in-v1's rule 3). HF's per-utterance word errors: the parity build's,
    except the 19 committed utterances, where HF's own; valid only if rule 4 passes.

**Serving:**

11. The reference's speech server checks pass on both builds: `check_lfm3_speech_server.py`,
    `check_lfm3_speech_stream.py`, `check_lfm3_speech_batch_server.py`, `check_lfm3_speech_concurrent.py` and
    `check_lfm3_audio_in_server.py`, each against the same build's `lfm3-speech` (a served reply equals the CLI's,
    streamed equals whole, batched equals alone). Adaptations to the card, if any, as in serve-v1: a copy in
    `tools/pascal/` with its whole diff.

    *Amendment (decision 10, owner, 2026-11-20, after development runs).* On the shipping build every output of the
    five checks was exact (development 11), and three failed only by time, on limits set for Thor:
    `check_lfm3_speech_server.py`'s worst case (512 chunks, 6,144 frames: 488 s against 240 s; its text control
    then got 404 because the control's session sat idle past the 300 s session timeout), `check_lfm3_speech_stream.py`'s
    no-gaps rule (the 14 interleaved cases, 0.3 to 1.4 s stalled each: interleaved runs a little slower than real
    time on the card; TTS never stalls), and `check_lfm3_speech_batch_server.py`'s first audio under load (median
    1.16 s against 1.0 s) and real time at 8 rows (one stall in one of 8 streams). The parity build is about three
    times slower. Rule 11 is judged on everything but timing: `tools/pascal/judge_speech_server.py` reads each
    check's report, applies its own criteria without the timing terms, and records every timing limit with its
    figure. `tools/pascal/check_lfm3_speech_server.py` runs the server with `--session-idle-timeout 3600` and a
    3,600 s client deadline on the worst cases, so its text control tests text rather than the idle timeout.
12. Memory at the limits. Through the server: serve-v1 rule 11's five text configurations with every speech row
    generating at the same time, and audio in at the request limits. Peak device memory at most 7,740 MiB. If not,
    the default speech rows are lowered to the largest number at which every configuration passes; if one row does
    not fit at a total capacity of 32,768, the total capacity is lowered as serve-v1 rule 11 says. The server's
    `ready` reports what it serves.
13. Past the limits: a speech request over its limits (prompt, chunks, clips, body) and one more than the queue holds
    are refused with the reference's status, and the control (text and speech) then gives the same bytes as before.
14. Memory safety: `compute-sanitizer` memcheck on `lfm3-speech` (one TTS case, one interleaved, one audio in, both
    builds): 0 errors, outputs identical to the plain run's.

Not judged: speed (nor, by decision 10, the reference checks' timing limits). Recorded: the real-time factor of each case through the CLI, and through the server with 1 and 8
rows.

## Predictions

- Rules 1, 2 and 5 hold: the text path's device code does not change, and the speech kernels do not depend on the GEMM
  configuration.
- Rule 3 holds: the batch decoder's projections are the tensor-core model's, which reproduces the Thor engine's
  prefill, and every other speech kernel is plain FP32 arithmetic in a fixed order.
- Rule 12 needs fewer speech rows than 8 at a total capacity of 32,768.

## Results

The rule run: `job.sh` (tree `50acd20`; builds `text-v35` and `parity-v35`), 2026-11-21 06:49 to 12:23, log
`records/jobs/speech-v1.log`, records `records/sp-v1/` (WAVs, raw frames and SASS left out; `SHA256SUMS` over every
record). The development runs' job logs: `records/development/jobs/`. The host suite on the same tree:
`records/host-suite-50acd20.txt`. Rules run again on the final binaries are marked *rerun*; results carried by
decision 11 are marked *carried*, with the development run they come from.

**Decision 11's evidence** (both hold, so every carried result stands):

- E1. Device code of every final binary equals the development builds' function for function: `lfm3-speech` against
  s1 and s4 (134 functions shipping, 140 parity), `liquid-lfm3-server` and `liquid-lfm3-worker` against s4; 0
  differing, 0 only in one (`sp-v1/e1/`). Host sources differing: from s4, `lfm3_server.cu` only (rule 12's capacity
  default and ready line); from s1, also `codec.cuh` and `lfm3_speech.cuh` (the shared codec checkpoint). The job also
  listed `lfm3_decode_batch.cuh`, `lfm3_worker.cu` and `text_sampling_oracle.cpp` against s1: those files are
  unchanged since `86e6afd`; s1 was built before `build_speech.sh` built the worker and server, so its source list
  never hashed them.
- E2. The final `lfm3-speech`, run again on the carried results' inputs: 280 runs, every output byte-identical. The
  shipping build's 30 runs behind rules 7 and 8, the 3 split runs and the 24 escalation runs (57); 101 LibriSpeech
  utterances on each build (the 19 committed and every 32nd other) against development 3's; rule 3's clips and
  rendering below are rerun outright.

| # | Rule | Source | Result |
|---|---|---|---|
| 1 | Text unchanged | rerun | **Pass.** Parity gate 7,826 of 7,826. `liquid-lfm3` and `lfm3-score`: device code identical to `text-v33` in all 93 and 94 functions. Fixtures pass. Host suite: 2,392 passed, 5 skipped, 1 failed (`test_every_validation_prompt_matches_hf_ids`, which needs Transformers 5.x: the same failure and skips as phase 3). Phase 3 recorded 2,397 passed; `tests/` is unchanged since, and the five are not explained (phase 3's working tree cannot be reconstructed to collect it again) |
| 2 | The reference's serving checks | native-server rerun; the rest carried (development 13) | **Pass.** serve-v1 rules 3 to 9 on both builds with the server built with speech: `native-server` (the only stage on `liquid-lfm3-server`, whose host code changed) rerun, 109 s shipping and 197 s parity (`sp-v1-serve-*/status.txt`); sampling, worker, grouped worker, server, cancellation, HTTP cancellation and `/generate_batch` from development 13 (`sp-v1/carried/serve-*-dev13/`), the worker and CLI unchanged since (E1). serve-v1's rule 10: paired decode, all three pairs, both lanes, 600 steps, ids and traces equal (`carried/paired.log`) |
| 3 | Speech out = the Thor engine's | rerun | **Pass.** 16 of 16 clips (15 runs: nine cases at seed 1, the lighthouse split at seeds 1 to 3) byte-identical to the Thor engine's; the native chat rendering gives the reference's ids in 15 of 15 (`rule3.txt`, `rule3-native.txt`) |
| 4 | Audio in = the Thor engine's | carried (development 3) | **Pass.** 2,620 utterances: 1,374 word errors of 53,027 (2.591%), the Thor engine's; 19 of 19 committed transcripts equal (`carried/rule4.json`) |
| 5 | Speech device code the same in both builds | rerun | **Pass.** Outside `lfm::gemm` 57 functions identical, 0 not (`rule5.txt`) |
| 6 | The scorer | rerun | **Pass.** Whisper on the card: 25 of 25 transcripts equal to the committed ones, all 25 clips prepared (`calibration.json`) |
| 7 | Intelligibility (C1) | carried (developments 5 and 6) | **Pass.** Two cases flagged at seeds 1 to 3, both passed at 15 seeds: `tts-numbers-female` (mean WER 17.5% against HF's 19.6%, p = 0.23) and `tts-long-male` split (0.14% against 0%: two seeds with one word of 97; p = 0.16). Every other case at or under HF's worst seed (`carried/c1c2.json`, `carried/escalate/`) |
| 8 | Endings and length (C2) | carried (development 5) | **Pass.** Every utterance ended by EOS; mean frames per utterance 0.93 to 1.07 of HF's (`c1c2.json`) |
| 9 | Listening (C3) | the owner | **Pending:** the owner's blind verdicts. The set's ten shipping clips are byte-identical to the final build's seed-1 clips (10 of 10 of the key's hashes, checked without revealing A/B: `rule9-key-check.txt`) |
| 10 | Audio in, against HF | carried (development 3) | **Pass.** Shipping 2.601% against HF's 2.606%: difference −0.006 points, paired-bootstrap 95% interval −0.030 to +0.017 (limit +0.25); 13 utterances better, 13 worse (`carried/rule10.json`) |
| 11 | The reference's speech server checks | audio-in and batch-server rerun; the other three carried (developments 11 and 12) | **Pass** on both builds by decision 10 (`text-rule11.json`, `parity-rule11.json`): every judged criterion holds, the reports are of these builds, every case ran. The two batch-server checks report FAIL on their own verdict by timing terms only (first audio under load, real time at 8 rows); see below |
| 12 | Memory at the limits | rerun at 1 row; 8 and 16 rows carried (development 15) | **Superseded (red-team-2 D1): the audio-in clip was 86.9 s, the request allows 98.2 s; with it, 1 row at 30,720 peaks at 7,759 MiB and 16 rows at 2,048 at 7,771, both over. The defaults are now 29,696, 2,048 less a row, at least 1,024 (7,707, 7,471 and 7,417 MiB at 1, 8 and 16 rows).** As recorded then: **Pass.** Ready line with the defaults: `{"total_capacity":30720,"speech_rows":1}`. At one row and 30,720 every configuration passes, highest 7,697 MiB (audio in at its limits beside the filled text session); the five text configurations 7,215 to 7,615 MiB; limit 7,740. 8 rows at 16,384: highest 7,461 MiB; 16 rows at 2,048: 7,709 MiB (`memory.log`, `carried/memory-rows*.log`) |
| 13 | Past the limits | rerun | **Pass** on both builds: text, chunks, prompt, capacity and clips 400 `invalid_request`; body (4 MiB + 1) and declared 16 MiB + 1 413 `body_limit`; one over the queue 429 `queue_full`, the five held requests 200; the controls (text and speech) identical before and after (`limits-*.log`) |
| 14 | Memory safety | carried (development 13) | **Pass.** `compute-sanitizer` memcheck, one TTS, one interleaved, one audio-in case on each build: 0 errors, outputs 3 of 3 identical to the plain run's (`carried/rule14/`, `development/jobs/speech-dev13.log`) |

The rule run's copy of the carried records put development 6's escalation files beside development 5's, and its
`asr-split.json` collided with development 5's and was refused; they were copied to `carried/escalate/` by hand after
the run (`job.sh` now does that).

**Recorded, not judged** (decision 10, `*-rule11.json`; and the CLI, from the job log):

| | Shipping | Parity |
|---|---|---|
| Real-time factor, CLI (audio seconds per wall second; mean of 3 seeds) | TTS 1.35 to 1.38, interleaved 1.06 to 1.08 | TTS 0.49 to 0.50, interleaved 0.38 to 0.39 |
| Real-time factor, server, one request (the stream check's done time) | TTS median 1.32, interleaved median 1.05 | TTS median 0.47, interleaved median 0.38 |
| First audio, streamed (limits: TTS median 1.0 s, max 1.5 s; interleaved median 1.5 s) | TTS median 0.91 s, max 0.95 s; interleaved median 1.26 s | TTS median 2.64 s, max 3.21 s; interleaved median 3.64 s |
| Streams with gaps (limit: none) | 14 of 42: the interleaved cases, 0.3 to 1.4 s stalled each | 39 of 42, 2.1 to 416 s stalled each |
| Cancel delay (limit 1.0 s); stream cost (limit 1.03) | 0.76 s; 1.002 | 2.10 s; 1.001 |
| Worst case, 512 chunks (limit 240 s); large cap | 490 s; 135 s | 2,274 s; 382 s |
| 90 s clip in (limit 240 s) | 19.6 s | 62.2 s |
| First audio under load (limit: median 1.0 s) | median 1.15 s | median 3.18 s |
| Streams with gaps at 8, 12, 16 rows; 7 + 5 | 1 of 8; 12 of 12; 16 of 16; 8 of 12 | all |

The shipping build speaks faster than real time through the CLI and one request at a time through the server; at
8 rows the speech batch is at real time on the card (one stall in one stream).

### Predictions against the measurements

- Rules 1, 2 and 5: held as predicted.
- Rule 3: held as predicted, byte for byte on all 16 clips.
- Rule 12: held, and further than predicted: not even one row fit at a total capacity of 32,768, so the total
  capacity was lowered to 30,720.
