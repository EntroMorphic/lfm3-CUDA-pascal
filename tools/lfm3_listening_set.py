#!/usr/bin/env python3
"""C3 (journal/lfm3-tts-quality-bar.md): Tripp's blind listening set, native against HF.

The fixed rule, no selection: seed 1 of every case in PROMPTS.json, both engines. Each case's two clips are named
A.wav and B.wav in an order drawn from the operating system's entropy (not reproducible from the code or the seed:
the step-4 red-team found the earlier seeded draws predictable); the key is written separately (KEY_JSON), the only record.
OUT_DIR/index.md lists each case's prompt, mode and the text each clip's model wrote (labelled A/B), so interleaved
speech can be checked against its text by ear.
Usage: lfm3_listening_set.py PROMPTS.json NATIVE_CASES HF_CASES OUT_DIR KEY_JSON [SEED (default 1)]
"""
import json
import random
import shutil
import sys
from pathlib import Path


def main(prompts, native_dir, hf_dir, out_dir, key_json, seed="1"):
    spec = json.load(open(prompts))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.SystemRandom()  # unpredictable; a seeded draw could be recomputed from this file
    key, lines = {}, ["# C3 listening set (blind)", "", f"Seed {seed} of every case. Each case has A.wav and B.wav: one is the "
                      "native engine, one is HF. The key is kept separately; please listen before opening it.", ""]
    for c in spec["cases"]:
        name = f"{c['case']}-s{seed}"
        clips = [("native", Path(native_dir) / name), ("hf", Path(hf_dir) / name)]
        rng.shuffle(clips)
        d = out / c["case"]; d.mkdir(exist_ok=True)
        key[c["case"]] = {}
        lines += [f"## {c['case']} ({c['mode']}, {spec['systems'][c['system']]})", "", f"Prompt: {c['user']}", ""]
        for label, (engine, src) in zip("AB", clips):
            shutil.copy(src / "response.wav", d / f"{label}.wav")
            key[c["case"]][label] = engine
            s = json.load(open(src / "summary.json"))
            lines.append(f"- {label}: {s['audio_seconds']:.1f} s" + (f"; text: {s['text'].strip()}" if c["mode"] != "audio" else ""))
        lines.append("")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    Path(key_json).write_text(json.dumps(dict(seed=int(seed), key=key), indent=1) + "\n")
    print(f"{len(key)} cases in {out}; key in {key_json}")


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        sys.exit(__doc__)
    main(*sys.argv[1:])
