#!/usr/bin/env python3
"""speech-v1 rule 6: the scorer on the card must reproduce the committed transcripts before it judges anything.

  prepare OUTPUTS_DIR REFERENCE_VALIDATION_DIR WORK_DIR
    For every clip of the reference's listening sets whose transcript is committed, a case directory in the layout
    tools/lfm3_asr.py reads (WORK_DIR/{hf,engine}/CASE/{response.wav, summary.json}); summary.json carries the case,
    the mode, the audio's length and the intended text from the committed record (the user's text for TTS, the engine's
    own text for interleaved replies), so the scorer's reference text is the committed one. Clips:
      HF, seed 1, the ten cases (listening-c3-v1 by its key; tts-long-male unsplit): speech-e2e-v1 rounds/e2e/asr-hf.json
      the Thor engine, seed 1, the nine single-request cases (listening-c3-v1): rounds/e2e/asr-native-cases.json
      the Thor engine split, lighthouse and bread, seeds 1-3 (listening-split-v1): speech-long-form-v1 asr/native-split.json
  compare WORK_DIR ASR_HF.json ASR_ENGINE.json REFERENCE_VALIDATION_DIR OUT.json
    Each new transcript against the committed one, character for character; exits 1 unless every one is equal.
"""
import json
import shutil
import sys
import wave
from pathlib import Path

E2E = "speech-e2e-v1/rounds/e2e"
SPLIT = "speech-long-form-v1/asr/native-split.json"


def records(validation):
    v = Path(validation)
    load = lambda p: {r["case"]: r for r in json.loads((v / p).read_text())["results"]}  # noqa: E731
    return load(f"{E2E}/asr-hf.json"), {**load(f"{E2E}/asr-native-cases.json"), **load(SPLIT)}


def clips(outputs):
    o = Path(outputs)
    key = json.loads((o / "listening-c3-v1.key.json").read_text())["key"]
    for case, sides in sorted(key.items()):
        for side, engine in sides.items():
            if engine == "native" and case == "tts-long-male":
                continue  # the split clip; listening-split-v1's seed 1 below
            yield ("hf" if engine == "hf" else "engine"), f"{case}-s1", o / "listening-c3-v1" / case / f"{side}.wav"
    for seed in (1, 2, 3):
        k = json.loads((o / f"listening-split-v1.key-seed{seed}.json").read_text())["key"]["tts-long-male"]
        side = next(s for s, e in k.items() if e == "native")
        yield "engine", f"tts-long-male-s{seed}", o / "listening-split-v1" / f"seed{seed}" / "tts-long-male" / f"{side}.wav"
        yield "engine", f"probe-other-long-s{seed}", o / "listening-split-v1" / "bread-native-split" / f"seed{seed}.wav"


def prepare(outputs, validation, work):
    hf, engine = records(validation)
    n = 0
    for kind, case, wav in clips(outputs):
        rec = (hf if kind == "hf" else engine)[case]
        d = Path(work) / kind / case
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, d / "response.wav")
        with wave.open(str(wav)) as w:
            seconds = w.getnframes() / w.getframerate()
        summary = dict(case=case, mode=rec["mode"], audio_seconds=seconds)
        if rec["mode"] == "audio":
            summary["conversation"] = [{"role": "user", "content": rec["intended"]}]
        else:
            summary["text"] = rec["intended"]
        (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        n += 1
    print(f"prepared {n} clips")


def compare(work, asr_hf, asr_engine, validation, out_json):
    hf, engine = records(validation)
    rows = []
    for kind, path, committed in (("hf", asr_hf, hf), ("engine", asr_engine, engine)):
        for r in json.loads(Path(path).read_text())["results"]:
            c = committed[r["case"]]
            rows.append(dict(kind=kind, case=r["case"], equal=r["transcript"] == c["transcript"],
                             transcript=r["transcript"], committed=c["transcript"], wer=r["wer"], committed_wer=c["wer"]))
    # every prepared clip transcribed exactly once (red-team review: a missing clip must not pass on the subset): the 25
    # seed-1 clips, ten HF and fifteen Thor-engine (the split tts-long-male among them, as prepare() writes them)
    prepared = sorted((k, d.name) for k in ("hf", "engine") for d in (Path(work) / k).iterdir() if d.is_dir())
    got = sorted((r["kind"], r["case"]) for r in rows)
    complete = got == prepared and len(prepared) == 25
    ok = complete and all(r["equal"] for r in rows)
    Path(out_json).write_text(json.dumps(dict(clips=len(rows), prepared=len(prepared), complete=complete,
                                              equal=sum(r["equal"] for r in rows), pass_=ok, rows=rows),
                                         indent=1) + "\n")
    for r in rows:
        print(f"{r['kind']:6s} {r['case']:24s} {'equal' if r['equal'] else 'DIFFERS'}  wer {r['wer']:.3f} (committed {r['committed_wer']:.3f})")
    print(f"rule 6 (the scorer): {sum(r['equal'] for r in rows)} of {len(rows)} transcripts equal, {len(prepared)} clips prepared"
          f"{'' if complete else ' (INCOMPLETE)'}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare" and len(sys.argv) == 5:
        prepare(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "compare" and len(sys.argv) == 7:
        compare(*sys.argv[2:])
    else:
        sys.exit(__doc__)
