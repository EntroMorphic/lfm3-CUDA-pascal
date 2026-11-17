#!/usr/bin/env python3
"""validation/multilingual-speech-v1 rule 4: the native engine against HF on FLEURS speech recognition, 18 languages.

For each case of CASES.json (tools/fleurs_cases.py asr-cases.json): HF's run (HF_DIR/hf/CASE/summary.json, from
tools/lfm3_audio_in_hf.py) and the native one (NATIVE_DIR/CASE.json, lfm3-speech on HF's request: the same prompt ids).
- prompt equal, generated ids equal (first difference), text equal;
- both texts scored against FLEURS's normalized transcription, both sides through Whisper's BasicTextNormalizer
  (English: Whisper's English normalizer, audio-in-v1's scoring); CER (characters, whitespace removed) for Chinese,
  Japanese and Korean as the model card scores them, WER for the rest;
- per language: corpus error rate (all errors / all reference units) for each engine and native minus HF in points,
  with a paired bootstrap 95% interval over that language's utterances;
- overall: the mean over languages of the per-language difference, with a stratified paired bootstrap 95% interval
  (each resample draws every language's utterances with replacement, within the language; 10,000 resamples,
  numpy default_rng(0)). Units differ between languages (characters, words), so the languages are averaged, not pooled.
The rule (README rule 4): overall upper bound <= +0.25 points; a language is flagged when its upper bound is over
+1.0 point (escalation: its full FLEURS test split). This tool reports; the verdict is the README's.
Usage (lfm3-omni-dev): check_lfm3_multilingual_asr.py MODEL_DIR WHISPER_DIR CASES.json HF_DIR NATIVE_DIR OUT.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_asr import word_errors  # noqa: E402
from lfm3_asr_multilingual import CER_LANGUAGES, basic_normalizer  # noqa: E402

OVERALL_BOUND, LANGUAGE_FLAG = 0.25, 1.0


def units(language, text, english, basic):
    if language == "en":
        return english(text).split()
    t = basic(text)
    return list(re.sub(r"\s+", "", t)) if language in CER_LANGUAGES else t.split()


def summarize(rows, resamples=10_000, seed=0):
    """rows: dicts with language, n (reference units), hf_errors, native_errors. Per-language and overall figures."""
    by = {}
    for r in rows:
        by.setdefault(r["language"], []).append(r)
    rng = np.random.default_rng(seed)
    langs, per = sorted(by), {}
    boot = np.zeros((resamples, len(langs)))
    for j, lang in enumerate(langs):
        n = np.array([r["n"] for r in by[lang]], float)
        h = np.array([r["hf_errors"] for r in by[lang]], float)
        a = np.array([r["native_errors"] for r in by[lang]], float)
        idx = rng.integers(0, len(n), (resamples, len(n)))
        boot[:, j] = 100 * (a[idx].sum(1) - h[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        per[lang] = dict(utterances=len(n), reference_units=int(n.sum()), metric=by[lang][0].get("metric"),
                         hf_rate=100 * h.sum() / max(n.sum(), 1), native_rate=100 * a.sum() / max(n.sum(), 1),
                         difference=100 * (a.sum() - h.sum()) / max(n.sum(), 1), ci95=[float(lo), float(hi)],
                         flagged=bool(hi > LANGUAGE_FLAG),
                         ids_equal=sum(r.get("ids_equal", False) for r in by[lang]),
                         native_better=int((a < h).sum()), native_worse=int((a > h).sum()))
    mean = boot.mean(1)
    lo, hi = np.percentile(mean, [2.5, 97.5])
    overall = dict(languages=len(langs), mean_difference=float(np.mean([per[l]["difference"] for l in langs])),
                   ci95=[float(lo), float(hi)], pass_=bool(hi <= OVERALL_BOUND),
                   flagged=[l for l in langs if per[l]["flagged"]])
    return per, overall


def main(model_dir, whisper_dir, cases_json, hf_dir, native_dir, out_json):
    from transformers import AutoTokenizer, WhisperTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    english, basic = WhisperTokenizer.from_pretrained(whisper_dir).normalize, basic_normalizer()
    spec = json.load(open(cases_json, encoding="utf-8"))
    rows, missing = [], []
    for c in spec["cases"]:
        hf_path, nat_path = Path(hf_dir) / "hf" / c["case"] / "summary.json", Path(native_dir) / f"{c['case']}.json"
        if not hf_path.exists() or not nat_path.exists():
            missing.append(c["case"])
            continue
        hf, nat = json.loads(hf_path.read_text()), json.loads(nat_path.read_text())
        hf_ids, nat_ids = hf["generated_ids"], nat["slots"]
        first = next((i for i, (a, b) in enumerate(zip(hf_ids, nat_ids)) if a != b),
                     None if len(hf_ids) == len(nat_ids) else min(len(hf_ids), len(nat_ids)))
        hf_text, nat_text = hf["text"].strip(), tok.decode(nat_ids, skip_special_tokens=True).strip()
        ref = units(c["language"], c["reference"], english, basic)
        rows.append(dict(case=c["case"], language=c["language"], metric=c["metric"], seconds=c.get("seconds"),
                         prompt_equal=nat["prompt"] == hf["prompt_ids"], ids_equal=first is None,
                         first_difference=first, text_equal=hf_text == nat_text, reference=c["reference"], hf=hf_text,
                         native=nat_text, n=len(ref),
                         hf_errors=word_errors(ref, units(c["language"], hf_text, english, basic)),
                         native_errors=word_errors(ref, units(c["language"], nat_text, english, basic)),
                         hf_generate_seconds=hf["generate_seconds"],
                         native_seconds=nat["prefill_seconds"] + nat["generate_seconds"]))
    per, overall = summarize(rows)
    summary = dict(cases=len(spec["cases"]), scored=len(rows), missing=missing,
                   prompts_equal=sum(r["prompt_equal"] for r in rows), ids_equal=sum(r["ids_equal"] for r in rows),
                   text_equal=sum(r["text_equal"] for r in rows), overall=overall, languages=per,
                   hf_generate_seconds=sum(r["hf_generate_seconds"] for r in rows),
                   native_seconds_end_to_end=sum(r["native_seconds"] for r in rows))
    Path(out_json).write_text(json.dumps(dict(summary=summary, cases=rows), ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("MULTILINGUAL ASR PASS" if overall["pass_"] and not missing else "MULTILINGUAL ASR FAIL")
    return 0 if overall["pass_"] and not missing else 1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
