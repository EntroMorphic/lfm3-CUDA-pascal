#!/usr/bin/env python3
"""validation/multilingual-speech-v1: per-locale summary of a fidelity round (rules 2, 3, 5).

Reads PROMPTS.json (tools/fleurs_cases.py; each case has "locale" and "language"), tools/lfm3_speech_report.py's
REPORT.json for the same round (per case: native and HF mean error, C1 flag, frames ratio, EOS counts), and both
engines' tools/lfm3_asr_multilingual.py results. Prints, per locale (and voice when the round has two): mean error
rate of each engine, flagged cases, the native/HF ratio of mean frames, utterances ended by EOS, and the share of
clips Whisper heard in the case's own language. Writes OUT_JSON. Reports; the README's rules give the verdicts.
Usage: lfm3_multilingual_report.py PROMPTS.json REPORT.json NATIVE_ASR.json HF_ASR.json OUT_JSON
"""
import json
import re
import statistics as st
import sys
from pathlib import Path


def group_key(case):
    """'tts-es-female-k3' -> 'es female'; 'long-es' -> 'es'; 'il-es-k1' -> 'es'."""
    m = re.fullmatch(r"tts-(.+)-(female|male)-k\d+", case["case"])
    return f"{case['locale']} {m.group(2)}" if m else case["locale"]


def heard_rate(asr, cases):
    """Share of clips Whisper detected as the case's own language, by group."""
    lang = {c["case"]: c["language"] for c in cases}
    out = {}
    for r in asr["results"]:
        base = re.fullmatch(r"(.+)-s\d+", r["case"]).group(1)
        out.setdefault(base, []).append(r.get("detected_language") == lang[base])
    return out


def summarize(spec, report, native_asr, hf_asr):
    heard = {"native": heard_rate(native_asr, spec["cases"]), "hf": heard_rate(hf_asr, spec["cases"])}
    groups = {}
    for c in spec["cases"]:
        groups.setdefault(group_key(c), []).append(c)
    out = {}
    for g, cs in groups.items():
        rows = [report["cases"][c["case"]] for c in cs]
        seeds = len(spec["seeds"])
        out[g] = dict(
            cases=len(cs), metric="cer" if cs[0]["language"] in ("zh", "ja", "ko") else "wer",
            native_error=st.mean(r["native"]["wer"] for r in rows), hf_error=st.mean(r["hf"]["wer"] for r in rows),
            flagged=[c["case"] for c, r in zip(cs, rows) if r["c1_flag"]],
            frames_ratio=st.mean(r["frames_ratio"] for r in rows if r["frames_ratio"]),
            native_eos=sum(r["native"]["eos"] for r in rows), hf_eos=sum(r["hf"]["eos"] for r in rows),
            runs=len(cs) * seeds,
            native_heard_own_language=st.mean(x for c in cs for x in heard["native"].get(c["case"], [])),
            hf_heard_own_language=st.mean(x for c in cs for x in heard["hf"].get(c["case"], [])))
    return out


def main(prompts, report_json, native_asr, hf_asr, out_json):
    spec = json.load(open(prompts, encoding="utf-8"))
    load = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
    out = summarize(spec, load(report_json), load(native_asr), load(hf_asr))
    print("| locale | metric | native | HF | flagged | frames native/HF | EOS native, HF | heard as own language (native, HF) |")
    print("|---|---|---|---|---|---|---|---|")
    for g, r in sorted(out.items()):
        print(f"| {g} | {r['metric']} | {r['native_error']:.3f} | {r['hf_error']:.3f} | {len(r['flagged'])} | "
              f"{r['frames_ratio']:.2f} | {r['native_eos']}/{r['runs']}, {r['hf_eos']}/{r['runs']} | "
              f"{r['native_heard_own_language']:.2f}, {r['hf_heard_own_language']:.2f} |")
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
