#!/usr/bin/env python3
"""Audio input (#3): edge cases for the stage-by-stage check (validation/audio-in-v1 round 3), for
tools/lfm3_audio_in_hf.py and lfm3-speech, all dumped.
Clips cut from LibriSpeech test-clean 1089-134686-0000 at 1.0 s: 1, 300, 399, 400, 401 and 1681 samples (a single
feature frame, the window and hop edges, and a 2-token clip); the whole utterance as a 32-bit float WAV; two clips in
one turn with text after them; three clips of mixed lengths, one of them 300 samples.
Usage (lfm3-omni-dev): audio_in_edge_cases.py LIBRI_WAV_DIR OUT_DIR CASES.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf


def main(libri, out, cases_json):
    libri, out = Path(libri), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x, rate = sf.read(libri / "1089-134686-0000.wav", dtype="int16")
    assert rate == 16_000
    cases = []

    def case(name, clips, text=None):
        cases.append(dict(case=name, audio=[str(p) for p in clips], system="asr", text=text, mode="text",
                          max_new_tokens=64, seed=0, dump=True))
    tiny = {}
    for n in (1, 300, 399, 400, 401, 1681):
        p = out / f"cut-{n}.wav"
        sf.write(p, x[16_000:16_000 + n], 16_000, subtype="PCM_16")
        tiny[n] = p
        case(f"cut-{n}", [p])
    f32 = out / "float32-1089-134686-0000.wav"
    sf.write(f32, x.astype("float32") / 32768, 16_000, subtype="FLOAT")
    case("float32", [f32])
    case("two-clips", [libri / "4446-2275-0004.wav", libri / "1580-141084-0005.wav"], "Transcribe both clips.")
    case("three-clips", [libri / "3729-6852-0021.wav", tiny[300], libri / "4446-2275-0004.wav"])
    Path(cases_json).write_text(json.dumps(dict(systems={"asr": "Perform ASR in English."}, cases=cases), indent=1) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
