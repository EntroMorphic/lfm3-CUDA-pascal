#!/usr/bin/env python3
"""Audio input (#3): LibriSpeech test-clean as speech recognition cases for tools/lfm3_audio_in_hf.py and lfm3-speech.

Reads LIBRI_DIR (the extracted LibriSpeech/test-clean: SPEAKER/CHAPTER/*.flac and *.trans.txt), writes each utterance
as 16-bit PCM WAV under WAV_DIR (the FLAC files are 16-bit, so the samples are unchanged), and writes CASES.json: the
model card's ASR prompt (system "Perform ASR in English.", the clip as the user turn), generation_mode "text"
(greedy), max_new_tokens 512, the reference transcript. All utterances, sorted by id. For each OFFSET, DUMP of them,
evenly spaced (cases[OFFSET::len//DUMP][:DUMP], a fixed selection), carry "dump": true for the stage-by-stage check.
validation/audio-in-v1 used OFFSETS 0,65: round 1's clips (offset 0) and round 2's held-out clips (offset 65).
Usage (lfm3-omni-dev): librispeech_cases.py LIBRI_DIR WAV_DIR CASES.json [DUMP=20] [OFFSETS=0]
"""
import json
import sys
from pathlib import Path

import soundfile as sf


def main(libri_dir, wav_dir, cases_json, dump=20, offsets="0"):
    libri, wavs = Path(libri_dir), Path(wav_dir)
    wavs.mkdir(parents=True, exist_ok=True)
    refs = {}
    for t in sorted(libri.glob("*/*/*.trans.txt")):
        for line in t.read_text().splitlines():
            uid, text = line.split(" ", 1)
            refs[uid] = text
    cases = []
    for uid in sorted(refs):
        spk, chap, _ = uid.split("-")
        flac = libri / spk / chap / f"{uid}.flac"
        x, rate = sf.read(flac, dtype="int16")
        assert rate == 16_000 and x.ndim == 1, (uid, rate, x.shape)
        out = wavs / f"{uid}.wav"
        if not out.exists():
            sf.write(out, x, 16_000, subtype="PCM_16")
        cases.append(dict(case=uid, audio=[str(out)], system="asr", mode="text", max_new_tokens=512, seed=0,
                          reference=refs[uid], seconds=len(x) / 16_000))
    step = max(1, len(cases) // int(dump))
    for offset in map(int, str(offsets).split(",")):
        for c in cases[offset::step][:int(dump)]:
            c["dump"] = True
    Path(cases_json).write_text(json.dumps(dict(source=str(libri), systems={"asr": "Perform ASR in English."},
                                                cases=cases), indent=1) + "\n")
    print(len(cases), "cases,", sum(c["seconds"] for c in cases) / 3600, "hours,",
          sum(1 for c in cases if c.get("dump")), "dumped")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6):
        sys.exit(__doc__)
    main(*sys.argv[1:])
