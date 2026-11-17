#!/usr/bin/env python3
"""Compare two lfm3-speech output directories (validation/audio-assurance-v1).
Every file of A must exist in B with the same bytes; a .json is compared without its timing fields. With --slots-only,
only each CASE.json's "slots" (emitted token sequence) is compared.
Usage: compare_speech_outputs.py A_DIR B_DIR OUT_JSON [--slots-only]"""
import hashlib
import json
import sys
from pathlib import Path

TIMINGS = {"encode_seconds", "front_end_seconds", "prefill_seconds", "generate_seconds", "lm_step_seconds", "text_pick_seconds",
           "depth_seconds", "sample_host_seconds", "codec_seconds", "real_time_factor", "real_time_factor_end_to_end"}


def content(path):
    if path.suffix == ".json":
        j = json.loads(path.read_text())
        return json.dumps({k: v for k, v in j.items() if k not in TIMINGS}, sort_keys=True).encode()
    return path.read_bytes()


def main(a, b, out, slots_only=False):
    a, b = Path(a), Path(b)
    files = sorted(p for p in a.iterdir() if p.is_file() and (p.suffix == ".json" or not slots_only))
    differ, missing = [], []
    for p in files:
        q = b / p.name
        if not q.exists():
            missing.append(p.name)
            continue
        if slots_only:
            same = json.loads(p.read_text())["slots"] == json.loads(q.read_text())["slots"]
        else:
            same = hashlib.sha256(content(p)).digest() == hashlib.sha256(content(q)).digest()
        if not same:
            differ.append(p.name)
    extra = sorted({p.name for p in b.iterdir() if p.is_file()} - {p.name for p in a.iterdir() if p.is_file()})
    if slots_only:
        extra = [x for x in extra if x.endswith(".json")]
    report = dict(a=str(a), b=str(b), slots_only=slots_only, files=len(files), identical=len(files) - len(differ) - len(missing),
                  differ=differ, missing=missing, extra=extra, pass_=not differ and not missing and not extra)
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in report.items()}))
    return 0 if report["pass_"] else 1


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--slots-only"]
    if len(args) != 3:
        sys.exit(__doc__)
    sys.exit(main(*args, slots_only="--slots-only" in sys.argv))
