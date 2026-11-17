#!/usr/bin/env python3
"""Concurrent speech at the engine (validation/speech-batch-v1, rule 1 below the server): every request run alone
(lfm3-speech --batch) and in batches (lfm3-speech --concurrent ROWS, at once and staggered), each output compared byte
for byte: OUT.wav, OUT.f32, OUT.frames.i32, OUT.raw.i32, and OUT.json without its timings.

Requests (41): the 30 of speech-e2e-v1, the two long-form split cases (lighthouse, bread: speech-chunk-native-v1), the
numbers paragraph normalized, the five spoken questions answered interleaved (audio-in-listen) and three ASR clips.
Usage (lfm3-profile-v1): check_lfm3_speech_concurrent.py LFM3_SPEECH MODEL_DIR BUILD_DIR LIBRI_WAV_DIR WORK_DIR OUT_JSON
                         [REFERENCE_ALONE_DIR]
REFERENCE_ALONE_DIR (validation/speech-speed-v1): another binary's `alone` outputs (a previous WORK_DIR/alone); this
binary's alone outputs must equal them too, file for file.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

TIMINGS = {"encode_seconds", "prefill_seconds", "generate_seconds", "lm_step_seconds", "text_pick_seconds",
           "depth_seconds", "sample_host_seconds", "codec_seconds", "real_time_factor", "real_time_factor_end_to_end"}
NUMBERS = ("The meeting starts at 7:45 and ends by 9:00. Order number 389 cost $3.50, about 2.75% more than in 1984. "
           "Dr. Patel met Mr. Lee at St. Mary's on Nov. 9, 2026.")
RUNS = [("conc4", 4, None), ("conc8", 8, None), ("conc8-staggered", 8, "11"), ("conc16-staggered", 16, "12")]


def requests(build, libri, work):
    b, w = Path(build), Path(work) / "requests"
    w.mkdir(parents=True, exist_ok=True)
    out = []
    e2e = sorted((b / "speech-e2e-v1/requests").glob("*.req.json"))
    assert len(e2e) == 30, len(e2e)
    out += [(p.name[:-9], p) for p in e2e]
    out += [("lighthouse-split-s1", b / "speech-chunk-native-v1/requests/tts-long-male-s1.req.json"),
            ("bread-split-s1", b / "speech-chunk-native-v1/requests/probe-other-long-s1.req.json")]

    def write(name, body):
        p = w / f"{name}.req.json"
        p.write_text(json.dumps(body))
        out.append((name, p))
    write("numbers-normalized-s1", dict(messages=[{"role": "system", "content": "Perform TTS in American English using a female voice."},
                                                  {"role": "user", "content": NUMBERS}],
                                        mode="audio", max_new_tokens=512, seed=1, split=True, normalize=True))
    for i in range(5):
        write(f"spoken-q{i}", dict(messages=[{"role": "system", "content": "Respond with interleaved text and audio."},
                                             {"role": "user", "content": [{"type": "audio", "audio": str(b / f"audio-in-listen/q{i}-16k.wav")}]}],
                                   mode="interleaved", max_new_tokens=1024, seed=1))
    for u in ("1089-134686-0000", "4446-2275-0004", "121-123859-0002"):
        write(f"asr-{u}", dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                                         {"role": "user", "content": [{"type": "audio", "audio": str(Path(libri) / f"{u}.wav")}]}],
                               mode="text", max_new_tokens=512, seed=0))
    return out


def outputs(stem):
    files = {}
    for ext in (".wav", ".f32", ".frames.i32", ".raw.i32"):
        p = Path(str(stem) + ext)
        files[ext] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    j = Path(str(stem) + ".json")
    files[".json"] = {k: v for k, v in json.loads(j.read_text()).items() if k not in TIMINGS} if j.exists() else None
    return files


def main(speech, model, build, libri, work, out_json, reference_alone=None):
    work = Path(work)
    reqs = requests(build, libri, work)
    report = dict(speech_sha256=hashlib.sha256(Path(speech).read_bytes()).hexdigest(), requests=len(reqs), runs={})

    def run(tag, args):
        d = work / tag
        d.mkdir(parents=True, exist_ok=True)
        lst = d / "list.txt"
        lst.write_text("".join(f"{p} {d / name}\n" for name, p in reqs))
        t0 = time.time()
        r = subprocess.run([speech, model, *args(lst)], capture_output=True, text=True)
        wall = time.time() - t0
        summary = [json.loads(line) for line in r.stdout.splitlines() if line.startswith('{"rows"')]
        return d, dict(exit=r.returncode, wall_seconds=round(wall, 1), stderr=r.stderr[-1500:],
                       summary=summary[0] if summary else None)

    ref_dir, info = run("alone", lambda lst: ["--batch", str(lst)])
    report["runs"]["alone"] = info
    ref = {name: outputs(ref_dir / name) for name, _ in reqs}
    ok = info["exit"] == 0 and all(v[".wav"] for v in ref.values())
    print("alone", json.dumps({k: v for k, v in info.items() if k != "stderr"}), flush=True)
    if not reference_alone:  # rule 1 of speech-speed-v1 needs it: without it this binary is compared only to itself
        report["alone_vs_reference"] = None
        print("alone vs reference: NONE GIVEN (the batches are compared with this binary alone only)", flush=True)
    else:
        differ = [name for name, _ in reqs if outputs(Path(reference_alone) / name) != ref[name]]
        report["alone_vs_reference"] = dict(reference=str(reference_alone), identical=len(reqs) - len(differ),
                                            differ=differ, pass_=not differ)
        ok &= not differ
        print("alone vs reference", json.dumps(report["alone_vs_reference"]), flush=True)
    for tag, rows, seed in RUNS:
        d, info = run(tag, lambda lst: ["--concurrent", str(rows), str(lst)] + ([seed] if seed else []))
        differ = [name for name, _ in reqs if outputs(d / name) != ref[name]]
        info.update(rows=rows, stagger_seed=seed, identical=len(reqs) - len(differ), differ=differ)
        info["pass"] = info["exit"] == 0 and not differ
        ok &= info["pass"]
        report["runs"][tag] = info
        print(tag, json.dumps({k: v for k, v in info.items() if k != "stderr"}), flush=True)
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print("CONCURRENT", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
