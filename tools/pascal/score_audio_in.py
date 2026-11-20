#!/usr/bin/env python3
"""speech-v1 rules 4 and 10: audio in over LibriSpeech test-clean, scored as tools/check_lfm3_audio_in.py scores it.

Each native run (NATIVE_DIR/CASE.json from lfm3-speech) is decoded with the checkpoint's tokenizer (special tokens
skipped) and scored against the LibriSpeech reference with Whisper's English normalizer and word-level Levenshtein
distance (tools/lfm3_asr.py's word_errors), exactly as the reference's check does.

  parity   MODEL_DIR WHISPER_DIR CASES.json NATIVE_DIR THOR_CHECK.json OUT.json
    Rule 4: the corpus's word errors equal the Thor engine's (THOR_CHECK.json: audio-in-v1's round2-full-check.json,
    whose summary gives its WER and words), and each committed utterance's native transcript and word errors equal
    the Thor engine's. Writes every utterance's errors (the HF table of rule 10 is built from them).
  shipping MODEL_DIR WHISPER_DIR CASES.json SHIP_DIR PARITY_OUT.json THOR_CHECK.json OUT.json
    Rule 10: HF's per-utterance word errors are the parity build's (PARITY_OUT.json, from the parity mode, which must
    have passed), except the committed utterances, where HF's own; the shipping build's corpus WER minus HF's, with
    the reference check's paired bootstrap (10,000 resamples, numpy default_rng(0)); pass if the 95% upper bound is at
    most +0.25 points.
Exits 1 on a failed rule.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfm3_asr import word_errors  # noqa: E402


def score(model_dir, whisper_dir, cases_json, native_dir):
    from transformers import AutoTokenizer, WhisperTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    normalize = WhisperTokenizer.from_pretrained(whisper_dir).normalize
    rows = []
    for c in json.loads(Path(cases_json).read_text())["cases"]:
        nat = json.loads((Path(native_dir) / f"{c['case']}.json").read_text())
        text = tok.decode(nat["slots"], skip_special_tokens=True).strip()
        ref = normalize(c["reference"]).split()
        rows.append(dict(case=c["case"], reference_words=len(ref), native=text,
                         errors=word_errors(ref, normalize(text).split())))
    return rows


def main(mode, *a):
    if mode == "parity":
        model_dir, whisper_dir, cases_json, native_dir, thor_json, out_json = a
        rows = score(model_dir, whisper_dir, cases_json, native_dir)
        thor = json.loads(Path(thor_json).read_text())
        words = sum(r["reference_words"] for r in rows)
        errors = sum(r["errors"] for r in rows)
        thor_words = thor["summary"]["reference_words"]
        thor_errors = round(thor["summary"]["native_wer"] * thor_words / 100)
        by_case = {r["case"]: r for r in rows}
        committed = []
        for t in thor["cases"]:
            r = by_case[t["case"]]
            committed.append(dict(case=t["case"], text_equal=r["native"] == t["native"],
                                  errors_equal=r["errors"] == t["native_errors"]))
        ok = (len(rows) == thor["summary"]["cases"] and words == thor_words and errors == thor_errors
              and all(x["text_equal"] and x["errors_equal"] for x in committed))
        summary = dict(utterances=len(rows), words=words, errors=errors, wer=100 * errors / words,
                       thor_words=thor_words, thor_errors=thor_errors, committed=len(committed),
                       committed_equal=sum(x["text_equal"] and x["errors_equal"] for x in committed), pass_=ok)
        Path(out_json).write_text(json.dumps(dict(summary=summary, committed=committed, cases=rows), indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        print(f"rule 4 (audio in, parity): {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if mode == "shipping":
        model_dir, whisper_dir, cases_json, ship_dir, parity_json, thor_json, out_json = a
        parity = json.loads(Path(parity_json).read_text())
        if not parity["summary"]["pass_"]:
            sys.exit("rule 10 needs rule 4: the parity run did not reproduce the Thor engine")
        hf = {r["case"]: r["errors"] for r in parity["cases"]}
        for t in json.loads(Path(thor_json).read_text())["cases"]:
            hf[t["case"]] = t["hf_errors"]
        rows = score(model_dir, whisper_dir, cases_json, ship_dir)
        for r in rows:
            r["hf_errors"] = hf[r["case"]]
        w = np.array([r["reference_words"] for r in rows], dtype=float)
        s = np.array([r["errors"] for r in rows], dtype=float)
        h = np.array([r["hf_errors"] for r in rows], dtype=float)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(10_000):
            i = rng.integers(0, len(rows), len(rows))
            diffs.append(100 * (s[i].sum() - h[i].sum()) / w[i].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        ok = hi <= 0.25
        summary = dict(utterances=len(rows), words=int(w.sum()), shipping_wer=100 * s.sum() / w.sum(),
                       hf_wer=100 * h.sum() / w.sum(), difference=100 * (s.sum() - h.sum()) / w.sum(),
                       difference_ci95=[float(lo), float(hi)], shipping_better=int((s < h).sum()),
                       shipping_worse=int((s > h).sum()), pass_=bool(ok))
        Path(out_json).write_text(json.dumps(dict(summary=summary, cases=rows), indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        print(f"rule 10 (audio in, shipping against HF): {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    sys.exit(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
