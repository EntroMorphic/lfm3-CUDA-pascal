# speech-v1: phase 4's bar

Status: **rules set** (2026-11-20), before any speech output of this engine has been generated on the card.

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

Records in `records/` (the `dev*.sh` scripts here say what each run did).

- **The codec checkpoint, shared** (development 9). Every row of the speech batch built its own codec decoder, and each
  loaded its own copy of the codec checkpoint (147 MB; harmless on Thor's unified memory). On the card that made each
  row cost about 245 MiB: the CLI ran out of memory past 12 rows, and through the server no number of rows passed
  rule 12. Now one checkpoint is loaded and shared by the rows' decoders (`lfm/codec.cuh`, a constructor taking the
  loaded checkpoint; `lfm3_speech.cuh`); the weights are only read, and each decoder keeps its own buffers. Host code
  only. Both builds give the same 15 rule-3 WAVs byte for byte as before; the CLI's 41 concurrent requests at 8, 12
  and 16 rows equal their alone runs, at 5,933, 6,287 and 6,703 MiB. A row now costs about 102 MiB.
- **Rule 12's outcome** (developments 8 to 10, `tools/pascal/speech_memory.py`). With text at its limits, 8 rows peak
  at 8,073 MiB, and even one row exceeds the limit at a total capacity of 32,768 (`/generate_batch` on 4,096 + 28,672:
  7,745 MiB). Lowered as the rule says: one speech row, and the total capacity to 31,744, where every configuration
  (scaled) passes beside one row (the batch at 7,693 MiB). Each 1,024 tokens of capacity is about 52 MiB, so each
  further row would cost about 2,048 tokens of capacity; the owner kept the rule as written (2026-11-20, "Let's roll
  with it as is"). The server defaults to `--speech-rows 1` and, when it serves speech, `--total-capacity 31744`
  (32,768 without speech, as serve-v1 measured); its ready line reports both
  (`{"ready":true,"port":N,"total_capacity":31744,"speech_rows":1}`). `--speech-rows` still accepts 1 to 16; more rows
  than one fit beside smaller text sessions, not beside text at its limits.
- **`/generate_batch` and speech.** A batch is an exclusive job: the server lets the running speech rows finish before
  it runs (`lfm3_http.hpp`, `speech_first`). So that configuration of rule 12 cannot run beside generating rows; the
  check counts it when the rows ran first, their batch still resident.
- **Adaptations to the card** (rule 11, copies in `tools/pascal/` with their whole diffs):
  `check_lfm3_audio_in_server.py` runs every `lfm3-speech` reference before the server starts (a second copy of the
  model does not fit); `check_lfm3_speech_batch_server.py` runs its alone server as the same binary with
  `--speech-rows 1` (there is no older single-request server) and gives `--speech-rows 8` to the servers that relied on
  the reference's default of 8.

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
