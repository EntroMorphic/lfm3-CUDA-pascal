#!/usr/bin/env python3
"""Audio input (#3): the speech-to-speech listening set for Tripp (validation/audio-in-v1, rule 4).

prepare WORK_DIR: spoken questions as 16 kHz WAV files: three LibriSpeech test-clean utterances (real human read
  speech, LIBRI_WAV_DIR) and five questions spoken by the native engine's own TTS (WORK_DIR/tts-q*.wav, 24 kHz, made by
  lfm3-speech from WORK_DIR/tts-q*.req.json, written by "tts" first) resampled to 16 kHz with librosa (soxr_hq). Writes
  WORK_DIR/requests/*.req.json for lfm3-speech: each question answered in interleaved text and speech, and in speech
  alone, seed 1, as the model card's voice chat example prompts it.
  Usage (lfm3-omni-dev): lfm3_audio_in_listening.py tts WORK_DIR      (the TTS requests)
                         lfm3_audio_in_listening.py prepare WORK_DIR LIBRI_WAV_DIR
"""
import json
import sys
from pathlib import Path

QUESTIONS = [
    "What's the tallest mountain in the world, and how tall is it?",
    "Can you give me three quick tips for sleeping better?",
    "How does a lighthouse warn ships at night?",
    "I'm making pasta for four people. How much dry pasta should I cook?",
    "Tell me a short, friendly joke about computers.",
]
LIBRI = ["1580-141084-0005", "4446-2275-0004", "8463-294828-0027"]
SYSTEMS = {"interleaved": "Respond with interleaved text and audio.", "audio": "Respond with audio."}


def tts(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(QUESTIONS):
        (work / f"tts-q{i}.req.json").write_text(json.dumps(dict(
            messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                      {"role": "user", "content": q}], mode="audio", max_new_tokens=512, seed=1, split=False)))


def prepare(work, libri):
    import librosa
    import soundfile as sf
    work = Path(work)
    (work / "requests").mkdir(parents=True, exist_ok=True)
    clips = {}
    for i, q in enumerate(QUESTIONS):
        x, _ = librosa.load(work / f"tts-q{i}.wav", sr=16_000, res_type="soxr_hq")
        sf.write(work / f"q{i}-16k.wav", x, 16_000, subtype="PCM_16")
        clips[f"q{i}"] = (str(work / f"q{i}-16k.wav"), q)
    for u in LIBRI:
        clips[u] = (str(Path(libri) / f"{u}.wav"), None)
    for name, (path, _) in clips.items():
        for mode, system in SYSTEMS.items():
            (work / "requests" / f"{name}-{mode}.req.json").write_text(json.dumps(dict(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": [{"type": "audio", "audio": path}]}],
                mode=mode, max_new_tokens=1024, seed=1)))
    (work / "questions.json").write_text(json.dumps({k: v[1] for k, v in clips.items()}, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "tts":
        tts(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
