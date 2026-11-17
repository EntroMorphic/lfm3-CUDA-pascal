#!/usr/bin/env python3
"""Scores tests/data/normalizer_audit.json against tools/speech_normalize.py (validation/audio-assurance-v1/
normalizer-audit.md rules 2-3): fine, left as written, ambiguous, or differs (each "differs" entry is then judged wrong
or not by hand and recorded in the audit's Results). Prints a summary and every entry that is not fine; writes OUT_JSON
when given.
Usage: normalizer_audit.py [OUT_JSON]"""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("speech_normalize", ROOT / "tools/speech_normalize.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def light(s):
    s = s.lower().replace("-", " ").replace("–", " ")
    s = re.sub(r"[^a-z0-9' ]", "", s)
    return " ".join(s.split())


def classify(e, got):
    if light(got) in {light(x) for x in [e["expect"], *e["also"]]}:
        return "fine"
    if got == e["text"]:
        return "left as written"
    if e["ambiguous"]:
        return "ambiguous"
    return "differs"


def main(out=None):
    corpus = json.loads((ROOT / "tests/data/normalizer_audit.json").read_text(encoding="utf-8"))["entries"]
    rows = []
    for e in corpus:
        got = ref.normalize(e["text"])
        rows.append(dict(e, got=got, cls=classify(e, got)))
    counts = {}
    for r in rows:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
    for r in rows:
        if r["cls"] != "fine":
            print(f"[{r['cls']}] ({r['category']}) {r['text']!r}\n    got:    {r['got']!r}\n    expect: {r['expect']!r}")
    print(json.dumps(dict(total=len(rows), **counts)))
    if out:
        Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
