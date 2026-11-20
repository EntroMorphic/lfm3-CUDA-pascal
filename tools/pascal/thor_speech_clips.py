#!/usr/bin/env python3
"""The Thor engine's speech clips from the reference's blind listening sets (speech-v1 rule 3), by their keys.

Usage: thor_speech_clips.py OUTPUTS_DIR OUT.json
OUTPUTS_DIR: the reference's untracked outputs/ (listening-c3-v1, listening-split-v1 and their keys). Writes, for each
clip, the run name lfm3-speech's requests use (CASE-sSEED), which side was the engine, the file, and its SHA-256:
seed 1 of the ten cases of listening-c3-v1 (tts-long-male: the engine split by sentence), tts-long-male split at
seeds 1 to 3 (listening-split-v1) and the bread paragraph (probe-other-long) split at seeds 1 to 3, engine only.
"""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(outputs, out_json):
    o = Path(outputs)
    clips = []
    key = json.loads((o / "listening-c3-v1.key.json").read_text())
    assert key["seed"] == 1
    for case, sides in sorted(key["key"].items()):
        side = next(s for s, engine in sides.items() if engine == "native")
        path = o / "listening-c3-v1" / case / f"{side}.wav"
        clips.append(dict(run=f"{case}-s1", set="listening-c3-v1", side=side, file=str(path.relative_to(o)), sha256=sha(path)))
    for seed in (1, 2, 3):
        key = json.loads((o / f"listening-split-v1.key-seed{seed}.json").read_text())
        assert key["seed"] == seed
        side = next(s for s, engine in key["key"]["tts-long-male"].items() if engine == "native")
        path = o / "listening-split-v1" / f"seed{seed}" / "tts-long-male" / f"{side}.wav"
        clips.append(dict(run=f"tts-long-male-s{seed}", set="listening-split-v1", side=side, file=str(path.relative_to(o)), sha256=sha(path)))
        path = o / "listening-split-v1" / "bread-native-split" / f"seed{seed}.wav"
        clips.append(dict(run=f"probe-other-long-s{seed}", set="listening-split-v1", side="engine only", file=str(path.relative_to(o)), sha256=sha(path)))
    Path(out_json).write_text(json.dumps(clips, indent=1) + "\n")
    for c in clips:
        print(c["run"], c["set"], c["side"], c["sha256"][:16])


if __name__ == "__main__":
    main(*sys.argv[1:])
