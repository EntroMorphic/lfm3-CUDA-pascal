#!/usr/bin/env python3
"""Audio input (#3): cases for native/tests/audio_prompt_check.cpp (the native renderer against HF's processor).

Writes OUT.jsonl from (1) a LibriSpeech round (tools/librispeech_cases.py CASES.json and the HF tool's
REQUESTS_DIR/CASE.req.json, whose prompt_ids HF's processor rendered) and (2) the render cases of
tools/check_lfm3_audio_in_misc.py (MISC_DIR/expect.json and MISC_DIR/CASE.req.json).
Usage: audio_prompt_cases.py CASES.json REQUESTS_DIR MISC_DIR OUT.jsonl
"""
import json
import sys
from pathlib import Path


def main(cases_json, requests_dir, misc_dir, out_jsonl):
    spec = json.load(open(cases_json))
    n = 0
    with open(out_jsonl, "w") as out:
        for c in spec["cases"]:
            r = json.load(open(Path(requests_dir) / f"{c['case']}.req.json"))
            content = [{"type": "audio", "audio": p} for p in c["audio"]]
            if c.get("text"):
                content.append({"type": "text", "text": c["text"]})
            msgs = ([{"role": "system", "content": spec["systems"][c["system"]]}] if c.get("system") else []) + \
                [{"role": "user", "content": content}]
            out.write(json.dumps(dict(case=c["case"], messages=msgs, prompt_ids=r["prompt_ids"])) + "\n")
            n += 1
        for name, e in json.load(open(Path(misc_dir) / "expect.json")).items():
            if e["kind"] == "render":
                req = json.load(open(Path(misc_dir) / f"{name}.req.json"))
                out.write(json.dumps(dict(case=name, messages=req["messages"], prompt_ids=e["prompt"])) + "\n")
                n += 1
    print(n, "cases")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
