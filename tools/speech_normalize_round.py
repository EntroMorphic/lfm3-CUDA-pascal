#!/usr/bin/env python3
"""validation/speech-normalize-v1: numbers and abbreviations, four arms (none, rules, model, hybrid), by test.

  heldout-request OUT.req.json
      the request with which the model writes the held-out sentences (lfm3-speech, text mode, greedy)
  heldout-cases HELDOUT.json OUT_CASES.json
      the model's sentences (lfm3-speech output OUT.json "text", one per line) as a case file; "spoken" left empty for
      the references, written by hand before any arm runs
  rewrite-requests CASES.json OUT_DIR
      per case, the model's rewrite request (text mode, greedy) and OUT_DIR/list.txt for lfm3-speech --batch
  arms CASES.json REWRITE_DIR OUT_ARMS.json
      each arm's text per case (none, rules, model, hybrid), the guard's verdict and reasons, the rewrite's latency, and
      text accuracy per arm
  tts-requests ARMS.json OUT_DIR SEEDS
      one TTS request per distinct text and seed (lfm3-speech, audio mode, the default sentence split), and
      OUT_DIR/list.txt; identical texts share their runs (the engine is deterministic for a text and seed)
  score WHISPER_DIR ARMS.json TTS_DIR OUT.json
      Whisper large-v3 transcripts of every run, WER against the spoken reference (tools/lfm3_asr.py's scoring), case
      means over seeds, per-arm means per set, and the rules' one-sided paired Wilcoxon tests
  interleaved-requests OUT_DIR SEEDS / interleaved-score WHISPER_DIR DIR OUT.json
      phase 2: ten questions answered interleaved, with and without the instruction to write numbers in words
"""
import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_normalize import ABBREVIATIONS, TITLES, normalize  # noqa: E402

SYSTEM_TTS = "Perform TTS in American English using a female voice."
REWRITE_SYSTEM = ("You rewrite text for a text-to-speech voice. Write out every number, time, date, year, amount of money, "
                  "percentage, measurement, symbol and abbreviation in words, exactly as a person would read it aloud. "
                  "Keep every other word and all punctuation exactly as written. Reply with the rewritten text only.")
HELDOUT_PROMPT = ("Write 12 different everyday sentences, one per line. Each sentence should contain several of these: "
                  "numbers, times, dates, years, prices, percentages, measurements, phone numbers, street addresses, or "
                  "abbreviations such as Dr., Mr., Mrs., St., e.g., etc., or vs. Do not number the lines or add anything "
                  "else.")
ABBR = {k.lower() + "." for k in list(TITLES) + list(ABBREVIATIONS) + ["St", "No", "Jan", "Feb", "Mar", "Apr", "Jun",
                                                                       "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"]}
SYMBOLS = set("$£€%#&+=°/×−")
# Phase 2: interleaved replies, where no preprocessor can reach (the model speaks as it writes)
INTERLEAVED_SYSTEM = "Respond with interleaved text and audio."
INTERLEAVED_INSTRUCTION = " Write every number and abbreviation out in words."
INTERLEAVED_QUESTIONS = [
    "How tall is Mount Everest?", "What year did the first person walk on the moon?",
    "How many days are in a leap year, and why?", "What is the boiling point of water in Fahrenheit and Celsius?",
    "How far is the Moon from Earth?", "What's 15 percent of 240?",
    "When did World War Two end?", "How many people live in Tokyo, roughly?",
    "What time zone is New York in, and how many hours behind London is it?",
    "How much should I tip on a 64 dollar dinner?"]


def light(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", "", s.lower().replace("-", " ").replace("—", " "))).strip()


def word(w):
    return w.strip("\"'“”‘’()[],;:!?").lower()


def target(w):
    t = word(w)
    return bool(re.search(r"\d", t)) or any(c in SYMBOLS for c in t) or t in ABBR or t.rstrip(".") + "." in ABBR


def guard(original, rewrite):
    """(passes, reasons): every changed region of the original touches a target word and keeps its other words, in
    order; no digit in the rewrite."""
    reasons = []
    if re.search(r"\d", rewrite):
        reasons.append("digit left in the rewrite")
    a = [word(w).rstrip(".") for w in original.split()]
    b = [word(w).rstrip(".") for w in rewrite.split()]
    raw = original.split()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        region = raw[i1:i2] if i2 > i1 else raw[max(0, i1 - 1):i1 + 1]
        # the region must hold a target word, and its other words must all survive in the rewrite's region, in order
        # (difflib merges a changed neighbour into the target's region: "paid $3.50" -> "spent three dollars ...")
        kept = iter(b[j1:j2])
        others_kept = all(any(x == w for x in kept) for w, r in zip(a[i1:i2], raw[i1:i2]) if not target(r))
        if not any(target(w) for w in region) or not others_kept:
            reasons.append(f"{op} {' '.join(raw[i1:i2]) or '-'} -> {' '.join(rewrite.split()[j1:j2]) or '-'}")
    return not reasons, reasons


def tts_request(text, seed):
    return dict(messages=[{"role": "system", "content": SYSTEM_TTS}, {"role": "user", "content": text}], mode="audio",
                max_new_tokens=512, seed=seed, split=True)


def main(cmd, *a):
    if cmd == "heldout-request":
        Path(a[0]).write_text(json.dumps(dict(messages=[{"role": "user", "content": HELDOUT_PROMPT}], mode="text",
                                              max_new_tokens=1024, seed=0)))
    elif cmd == "heldout-cases":
        text = json.load(open(a[0]))["text"]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        json.dump(dict(set="heldout", system=SYSTEM_TTS, source=a[0],
                       cases=[dict(case=f"h{i + 1:02d}", text=line, spoken="") for i, line in enumerate(lines)]),
                  open(a[1], "w"), indent=1, ensure_ascii=False)
        print(len(lines), "sentences")
    elif cmd == "rewrite-requests":
        spec, out = json.load(open(a[0])), Path(a[1])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "list.txt", "w") as lst:
            for c in spec["cases"]:
                (out / f"{c['case']}.req.json").write_text(json.dumps(dict(
                    messages=[{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": c["text"]}],
                    mode="text", max_new_tokens=512, seed=0), ensure_ascii=False))
                lst.write(f"{out}/{c['case']}.req.json {out}/{c['case']}\n")
    elif cmd == "arms":
        spec, rw = json.load(open(a[0])), Path(a[1])
        cases = []
        for c in spec["cases"]:
            r = json.load(open(rw / f"{c['case']}.json"))
            model = r["text"].strip()
            ok, reasons = guard(c["text"], model)
            rules = normalize(c["text"])
            arms = dict(none=c["text"], rules=rules, model=model, hybrid=model if ok else rules)
            cases.append(dict(case=c["case"], spoken=c["spoken"], arms=arms, guard=ok, guard_reasons=reasons,
                              rewrite_seconds=r["prefill_seconds"] + r["generate_seconds"],
                              accurate={k: light(v) == light(c["spoken"]) for k, v in arms.items()}))
        n = len(cases)
        summary = dict(cases=n, guard_passes=sum(c["guard"] for c in cases),
                       text_accuracy={k: sum(c["accurate"][k] for c in cases) for k in ("none", "rules", "model", "hybrid")},
                       rewrite_seconds_mean=sum(c["rewrite_seconds"] for c in cases) / n,
                       rewrite_seconds_max=max(c["rewrite_seconds"] for c in cases))
        json.dump(dict(set=spec["set"], summary=summary, cases=cases), open(a[2], "w"), indent=1, ensure_ascii=False)
        print(json.dumps(summary))
    elif cmd == "tts-requests":
        arms, out, seeds = json.load(open(a[0])), Path(a[1]), range(1, int(a[2]) + 1)
        out.mkdir(parents=True, exist_ok=True)
        seen = {}
        with open(out / "list.txt", "w") as lst:
            for c in arms["cases"]:
                for text in c["arms"].values():
                    key = hashlib.sha256(text.encode()).hexdigest()[:16]
                    if key in seen:
                        continue
                    seen[key] = text
                    for s in seeds:
                        name = f"{key}-s{s}"
                        (out / f"{name}.req.json").write_text(json.dumps(tts_request(text, s), ensure_ascii=False))
                        lst.write(f"{out}/{name}.req.json {out}/{name}\n")
        (out / "texts.json").write_text(json.dumps(seen, indent=1, ensure_ascii=False) + "\n")
        print(len(seen), "distinct texts,", len(seen) * len(seeds), "runs")
    elif cmd == "score":
        score(*a)
    elif cmd == "interleaved-requests":  # OUT_DIR SEEDS
        out = Path(a[0])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "list.txt", "w") as lst:
            for arm, system in (("plain", INTERLEAVED_SYSTEM), ("instructed", INTERLEAVED_SYSTEM + INTERLEAVED_INSTRUCTION)):
                for i, q in enumerate(INTERLEAVED_QUESTIONS):
                    for seed in range(1, int(a[1]) + 1):
                        name = f"{arm}-q{i:02d}-s{seed}"
                        (out / f"{name}.req.json").write_text(json.dumps(dict(
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": q}],
                            mode="interleaved", max_new_tokens=1024, seed=seed)))
                        lst.write(f"{out}/{name}.req.json {out}/{name}\n")
    elif cmd == "interleaved-score":  # WHISPER_DIR DIR OUT.json
        interleaved_score(*a)
    else:
        sys.exit(__doc__)


def score(whisper_dir, arms_json, tts_dir, out_json):
    import librosa
    import numpy as np
    import torch
    from scipy.stats import wilcoxon
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from lfm3_asr import ascii_punctuation, fold_ordinals
    from lfm3_asr import score as wer_score
    arms, tts = json.load(open(arms_json)), Path(tts_dir)
    processor = WhisperProcessor.from_pretrained(whisper_dir)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_dir, dtype=torch.float16).to("cuda").eval()
    normalize_w = processor.tokenizer.normalize
    texts = json.load(open(tts / "texts.json"))
    key_of = {v: k for k, v in texts.items()}
    seeds = sorted({int(p.stem.split("-s")[1]) for p in tts.glob("*-s*.wav")})
    transcripts = {}
    t0 = time.time()
    for key in texts:
        for s in seeds:
            name = f"{key}-s{s}"
            audio, _ = librosa.load(tts / f"{name}.wav", sr=16_000)
            f = processor(audio, sampling_rate=16_000, return_tensors="pt", truncation=False, padding="longest",
                          return_attention_mask=True)
            if f.input_features.shape[-1] < 3000:
                f = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
            with torch.inference_mode():
                ids = model.generate(f.input_features.to("cuda", torch.float16), attention_mask=f.attention_mask.to("cuda"),
                                     language="en", task="transcribe", return_timestamps=True, num_beams=1,
                                     do_sample=False)
            meta = json.load(open(tts / f"{name}.json"))
            transcripts[name] = dict(text=processor.batch_decode(ids, skip_special_tokens=True)[0].strip(),
                                     chunks_ended_by_eos=all(c["ended_by_eos"] for c in meta.get("chunks", [])))
    rows = []
    for c in arms["cases"]:
        ref = fold_ordinals(normalize_w(ascii_punctuation(c["spoken"])).split())
        for arm, text in c["arms"].items():
            wers = []
            for s in seeds:
                hyp = fold_ordinals(normalize_w(ascii_punctuation(transcripts[f"{key_of[text]}-s{s}"]["text"])).split())
                errors, n = wer_score(ref, hyp)
                wers.append(errors / max(n, 1))
            rows.append(dict(case=c["case"], arm=arm, mean_wer=float(np.mean(wers)), max_wer=float(np.max(wers)),
                             wers=wers, example=transcripts[f"{key_of[text]}-s1"]["text"]))
    by = {(r["case"], r["arm"]): r["mean_wer"] for r in rows}
    cases = [c["case"] for c in arms["cases"]]
    summary = dict(set=arms["set"], seeds=len(seeds), mean_wer={a: float(np.mean([by[(c, a)] for c in cases]))
                                                                 for a in ("none", "rules", "model", "hybrid")})
    for other in ("none", "model", "hybrid"):  # one-sided: the rules' case WERs lower than the other arm's
        x, y = [by[(c, "rules")] for c in cases], [by[(c, other)] for c in cases]
        try:
            summary[f"wilcoxon_rules_lt_{other}_p"] = float(wilcoxon(x, y, alternative="less", zero_method="wilcox").pvalue)
        except ValueError as e:  # all differences zero
            summary[f"wilcoxon_rules_lt_{other}_p"] = str(e)
    summary["all_chunks_ended_by_eos"] = all(t["chunks_ended_by_eos"] for t in transcripts.values())
    summary["asr_seconds"] = time.time() - t0
    json.dump(dict(summary=summary, rows=rows, transcripts=transcripts), open(out_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


def interleaved_score(whisper_dir, d, out_json):
    """Per reply: whether its text holds a digit, and the speech's WER against the reply's own text (tools/lfm3_asr.py's
    scoring, as the interleaved runs of the TTS quality bar)."""
    import librosa
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from lfm3_asr import ascii_punctuation, fold_ordinals
    from lfm3_asr import score as wer_score
    processor = WhisperProcessor.from_pretrained(whisper_dir)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_dir, dtype=torch.float16).to("cuda").eval()
    norm = processor.tokenizer.normalize
    rows = []
    for meta_path in sorted(p for p in Path(d).glob("*-q*-s*.json") if not p.name.endswith(".req.json")):
        name = meta_path.stem
        meta = json.load(open(meta_path))
        audio, _ = librosa.load(Path(d) / f"{name}.wav", sr=16_000)
        f = processor(audio, sampling_rate=16_000, return_tensors="pt", truncation=False, padding="longest",
                      return_attention_mask=True)
        if f.input_features.shape[-1] < 3000:
            f = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(f.input_features.to("cuda", torch.float16), attention_mask=f.attention_mask.to("cuda"),
                                 language="en", task="transcribe", return_timestamps=True, num_beams=1, do_sample=False)
        heard = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        ref = fold_ordinals(norm(ascii_punctuation(meta["text"])).split())
        errors, n = wer_score(ref, fold_ordinals(norm(ascii_punctuation(heard)).split()))
        rows.append(dict(name=name, arm=name.split("-")[0], text=meta["text"], heard=heard,
                         has_digit=bool(re.search(r"\d", meta["text"])), wer=errors / max(n, 1)))
    summary = {arm: dict(replies=len(r), with_digits=sum(x["has_digit"] for x in r),
                         mean_wer=float(np.mean([x["wer"] for x in r])),
                         mean_wer_with_digits=float(np.mean([x["wer"] for x in r if x["has_digit"]] or [float("nan")])),
                         mean_wer_without_digits=float(np.mean([x["wer"] for x in r if not x["has_digit"]] or [float("nan")])))
               for arm in ("plain", "instructed") for r in [[x for x in rows if x["arm"] == arm]]}
    json.dump(dict(summary=summary, rows=rows), open(out_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
