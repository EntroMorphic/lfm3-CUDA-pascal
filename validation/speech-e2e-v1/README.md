# #2 step 4: native text in -> text and speech out, against HF (C1, C2, C4)

`lfm3-speech` (`native/src/lfm3_speech.cuh`, driver `native/tests/lfm3_speech.cu`) against HF BF16 SDPA
(`tools/lfm3_speech_hf.py`) on `prompts.json`: 10 cases (TTS female and male: fox, museum, numbers; a long
paragraph; three interleaved replies), seeds 1-3, the same prompt ids. Round: `tools/thor/speech_e2e_round.sh`;
ASR: Whisper large-v3 (`tools/lfm3_asr.py`); tables: `tools/lfm3_speech_report.py`. Bar:
`journal/lfm3-tts-quality-bar.md`. Records: `rounds/{e2e,long,fridge}/` (report.md, report.json, the ASR jsons).

Corrected in the step-4 red-team (2026-11-09):
- The scorer counted a hyphenated compound spoken as separate words as errors. `lfm3_asr.py` now folds compounds
  (a rule fixed before rescoring). All committed ASR records were rescored in place, with the earlier values kept as
  `wer_before`. Exactly 11 runs changed, all il-fridge, all to 0.000.
- The round reports were regenerated with the current report tool: the per-case rule, and a CI and Mann-Whitney p for
  every case. The committed versions had still printed the first, pooled rule.
- Speed is now stated like for like.

## C1, intelligibility (per case)
The rule, per case: a case is flagged when native's mean WER over the seeds exceeds HF's worst seed on that case; a
flagged case escalates to 15 seeds and a Mann-Whitney test. (The first version pooled all cases and passed while one
case was 0.89 against 0.16; replaced.) Seeds 1-3 (`rounds/e2e/report.md`) flag two cases:
- `tts-numbers-female`: seeds 4-15 (`rounds/long`) native 0.189 against HF 0.206 (p = 0.26); seeds 1-15 0.204 against
  0.196 (p = 0.55; `../speech-long-form-v1/report.md` section 0). **Clears.** Both engines share a floor: in every clip
  of either engine, "7:45" normalizes to "7 45" against Whisper's "7.45" (2 of 19 words). The rest are real misreadings
  of "389", by both engines.
- `tts-long-male`: native 0.418 against HF 0.163 over seeds 1-15 (p = 0.019); on the fresh seeds 4-15 alone 0.301
  against 0.164 (p = 0.053). **The model's own failure** (the FP32 model is no better), **fixed by splitting long text
  into sentences:** `../speech-long-form-v1` (native split: WER 0.000 on 30 of 30 runs).
- `il-fridge` was flagged under the old scoring: hyphenated compounds counted as errors, which made seeds 1-3 0.037
  against 0.009. Rescored, it is 0.000 in both engines at seeds 1-3. Over 15 seeds it is 0.002 against 0.001
  (`rounds/fridge`), one or two words in 15 clips per engine. Not flagged.
The other seven cases are not flagged (fox and interleaved 0.000 in both engines; museum 0.028 in both).

## C2, duration and endings
Every utterance of both engines ended by the model's own EOS. Frames per utterance, native/HF ratio of means by
case: 0.95-1.05 except il-fridge 1.14 at seeds 1-3 (1.03 at seeds 4-15) and tts-long-male 1.74 (the loops above).

## C4, interleaved: speech against its own text
il-reflective, il-study and il-fridge: 0.000 WER at seeds 1-3, against the text each engine wrote.

## Speed (BF16 weights, batch 1, Thor; audio seconds per second, mean over runs)
| round | native, decode loop | native, end to end | HF, end to end |
|---|---|---|---|
| e2e (10 cases, seeds 1-3) | 3.24 | 3.20 | 1.91 |
| long (2 cases, seeds 4-15) | 3.45 | 3.43 | 2.06 |
| fridge (il-fridge, seeds 4-15) | 2.85 | 2.84 | 2.08 |
| e2e again, MAXN (2026-11-09, `build/speech-e2e-maxn-v1`) | 3.05 | 3.01 | 1.90 |

"End to end" includes the prefill; HF's time always does. The rounds ran on different days, and Thor's power state
was not recorded or held fixed, so compare engines within a round (native 1.4-1.7x HF), not rounds with each other.
Native split long-form, end to end: 3.42; again in MAXN, 3.53 (30 runs, range 3.45-3.58; WER 0.000 on 30 of 30, 150 of
150 chunks ended by EOS; `build/speech-split-maxn-v1`).
Since 2026-11-09 Thor runs in MAXN, and every round records its power state (`tools/thor/power_state.sh`; it refuses to
run outside MAXN). The clocks were not locked for the MAXN rounds above, so the GPU clock scaled with load. Since 2026-11-09 17:20
(`tools/thor/pin_power.sh`, run by Tripp) they are: GPU 1575/1692 MHz, CPU 2601 MHz fixed, kept across reboots by
`jetson-clocks.service`. power_state.sh records `clocks_locked: true`. Speed measured before that is not comparable. The MAXN e2e
rerun produced bit-identical native outputs (30 of 30) and identical WERs in both engines, yet ran slower than the
first e2e round. The outputs are bit-identical, so the code is not the cause; the clock governor is the
likely one (not measured).

## C3 (listening): PASS
Blind A/B, seed 1 of every case (`outputs/listening-c3-v1`, not in the repo; key in `listening-c3-v1.key.json`), Tripp,
2026-11-08 and 2026-11-09, unblinded only after every call was in:
- fox (female, male), museum (female, male), il-study, il-reflective, il-fridge: both clips good. Tripp heard B as
  slightly more expressive, but B was native in 3 of these 7 pairs and HF in 4, so this does not favour an engine.
- tts-numbers (female, male): both engines stutter on the numbers. Whisper hears "389" wrong in all four clips
  (HF $289.50 and $809.50 plus "gate 20"; native $409.50 and $890.50). This is the model's limit with digits.
- tts-long-male, rebuilt after the fix, with native split against HF BF16 unsplit: HF murmurs, native solid. Whisper finds
  HF's last sentence looping ("visitors climb them tadvitas climb them tuluncu..."). See `../speech-long-form-v1`.
Blinding: the order came from a seeded draw (9 pairs from the tool; the rebuilt pair's key was written by hand). It
was blind in practice: the order was not looked up. The tool now draws from the operating system's entropy.
