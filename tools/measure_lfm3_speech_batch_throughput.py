#!/usr/bin/env python3
"""Rule 2 of validation/speech-batch-v1: aggregate seconds of audio generated per wall second of the batch loop
(lfm3-speech --concurrent N), the 21 TTS requests of speech-e2e-v1 (the README said 15: a miscount) looped (max(42, 4N) requests, so N rows stay busy),
at N = 1, 2, 4, 8, 16; one untimed warm-up, then two runs each. Pass: N = 8 at least 4 times N = 1 (medians).
Usage (lfm3-profile-v1): measure_lfm3_speech_batch_throughput.py LFM3_SPEECH MODEL_DIR BUILD_DIR WORK_DIR OUT_JSON
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main(speech, model, build, work, out_json):
    work = Path(work)
    tts = sorted(p for p in (Path(build) / "speech-e2e-v1/requests").glob("tts-*.req.json"))
    assert len(tts) == 21, len(tts)

    def run(n, count, tag):
        d = work / tag
        d.mkdir(parents=True, exist_ok=True)
        lst = d / "list.txt"
        lst.write_text("".join(f"{tts[i % len(tts)]} {d / f'r{i}'}\n" for i in range(count)))
        r = subprocess.run([speech, model, "--concurrent", str(n), str(lst)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        return [json.loads(x) for x in r.stdout.splitlines() if x.startswith('{"rows"')][0]

    run(4, 8, "warmup")
    report = {}
    for n in (1, 2, 4, 8, 16):
        runs = [run(n, max(42, 4 * n), f"n{n}-{k}") for k in range(2)]
        report[n] = dict(runs=runs, median=statistics.median(r["audio_per_wall_second"] for r in runs))
        print(n, json.dumps(report[n]["median"]), flush=True)
    ratio = report[8]["median"] / report[1]["median"]
    out = dict(by_rows=report, ratio_8_to_1=ratio, pass_=ratio >= 4.0)
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    print("THROUGHPUT", "PASS" if out["pass_"] else "FAIL", round(ratio, 2), flush=True)
    return 0 if out["pass_"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
