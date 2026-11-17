#!/usr/bin/env python3
"""Audio input (#3): the native engine against HF on speech recognition (validation/audio-in-v1).

For each case of CASES.json (tools/librispeech_cases.py): HF's run (HF_DIR/hf/CASE/summary.json, from
tools/lfm3_audio_in_hf.py) and the native one (NATIVE_DIR/CASE.json from lfm3-speech, CASE.audio0.*.f32 when dumped).
- prompt: the native prompt (the mode machine adds no marker in text mode) equals HF's prompt ids;
- tokens: the native emitted slots against HF's generated ids (identical or not; first difference);
- text: both decoded with the checkpoint's tokenizer, special tokens skipped; scored against the LibriSpeech reference
  with Whisper's English normalizer (WhisperTokenizer.normalize, the standard) and word-level Levenshtein distance;
  corpus WER = all errors / all reference words; the native minus HF difference with a paired bootstrap 95% interval
  over utterances (10,000 resamples, numpy default_rng(0));
- stages (dumped cases): the features max |native - HF|; downsampler, encoder and projected outputs as relative L2
  distance to HF BF16; the projected output's relative L2 error to the FP32 reference for both engines, and their ratio.
Usage (lfm3-omni-dev): check_lfm3_audio_in.py MODEL_DIR WHISPER_DIR CASES.json HF_DIR NATIVE_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_asr import word_errors  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main(model_dir, whisper_dir, cases_json, hf_dir, native_dir, out_json):
    from transformers import AutoTokenizer, WhisperTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    normalize = WhisperTokenizer.from_pretrained(whisper_dir).normalize
    spec = json.load(open(cases_json))
    rows, stages = [], []
    for c in spec["cases"]:
        hf_path = Path(hf_dir) / "hf" / c["case"] / "summary.json"
        nat_path = Path(native_dir) / f"{c['case']}.json"
        if not hf_path.exists() or not nat_path.exists():
            rows.append(dict(case=c["case"], missing=[str(p) for p in (hf_path, nat_path) if not p.exists()]))
            continue
        hf, nat = json.loads(hf_path.read_text()), json.loads(nat_path.read_text())
        hf_ids, nat_ids = hf["generated_ids"], nat["slots"]
        first = next((i for i, (a, b) in enumerate(zip(hf_ids, nat_ids)) if a != b),
                     None if len(hf_ids) == len(nat_ids) else min(len(hf_ids), len(nat_ids)))
        hf_text = hf["text"].strip()
        nat_text = tok.decode(nat_ids, skip_special_tokens=True).strip()
        ref = normalize(c["reference"]).split() if c.get("reference") else []
        r = dict(case=c["case"], seconds=c.get("seconds"), prompt_equal=nat["prompt"] == hf["prompt_ids"],
                 audio_tokens=hf["audio_tokens"], ids_equal=first is None, first_difference=first,
                 reference=c.get("reference"), hf=hf_text, native=nat_text, reference_words=len(ref),
                 hf_errors=word_errors(ref, normalize(hf_text).split()),
                 native_errors=word_errors(ref, normalize(nat_text).split()),
                 hf_new_tokens=len(hf_ids), native_new_tokens=len(nat_ids),
                 hf_generate_seconds=hf["generate_seconds"], native_encode_seconds=nat.get("encode_seconds"),
                 native_prefill_seconds=nat["prefill_seconds"], native_generate_seconds=nat["generate_seconds"])
        rows.append(r)
        if c.get("dump"):
            d = Path(hf_dir) / "hf" / c["case"]
            load = lambda p: np.fromfile(p, dtype=np.float32)  # noqa: E731
            clips = len(c["audio"])
            stems = [Path(native_dir) / f"{c['case']}.audio{k}" for k in range(clips)]
            if not all((st.parent / (st.name + ".features.f32")).exists() for st in stems):
                stages.append(dict(case=c["case"], missing="native dumps"))
                continue

            def native(part):  # every clip's stage, in order (HF's dumps concatenate the clips the same way)
                if part == "mel":  # [128][T_k] per clip -> [128][sum T_k], as HF's concatenation along time
                    return np.concatenate([load(f"{st}.mel.f32").reshape(128, -1) for st in stems], axis=1).ravel()
                return np.concatenate([load(f"{st}.{part}.f32") for st in stems])

            st = dict(case=c["case"], clips=clips)
            mel_hf, mel_nat = load(d / "mel.f32"), native("mel")
            st["mel_shape_equal"] = mel_hf.shape == mel_nat.shape
            st["mel_max_abs"] = float(np.abs(mel_hf - mel_nat).max()) if st["mel_shape_equal"] else None
            if (d / "mel_fp64.f32").exists():  # the true features (FP64 arithmetic)
                mel64 = load(d / "mel_fp64.f32")
                st["mel_native_max_abs_to_fp64"] = float(np.abs(mel_nat - mel64).max())
                st["mel_hf_max_abs_to_fp64"] = float(np.abs(mel_hf - mel64).max())
            for name in ("down", "encoded", "features"):
                a, b = native(name), load(d / f"{name}.f32")
                st[f"{name}_rel_to_hf"] = rel(a, b) if a.shape == b.shape else None
                if (d / f"{name}_fp32.f32").exists():
                    t = load(d / f"{name}_fp32.f32")
                    st[f"{name}_native_rel_to_fp32"], st[f"{name}_hf_rel_to_fp32"] = rel(a, t), rel(b, t)
            truth = load(d / "features_fp32.f32")
            st["hf_rel_to_fp32"] = rel(load(d / "features.f32"), truth)
            st["native_rel_to_fp32"] = rel(native("features"), truth)
            st["ratio"] = st["native_rel_to_fp32"] / st["hf_rel_to_fp32"]
            stages.append(st)
    done = [r for r in rows if "missing" not in r]
    words = sum(r["reference_words"] for r in done)
    hf_e = np.array([r["hf_errors"] for r in done], dtype=float)
    nat_e = np.array([r["native_errors"] for r in done], dtype=float)
    w = np.array([r["reference_words"] for r in done], dtype=float)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(10_000):
        i = rng.integers(0, len(done), len(done))
        diffs.append(100 * (nat_e[i].sum() - hf_e[i].sum()) / w[i].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    summary = dict(
        cases=len(rows), scored=len(done), missing=len(rows) - len(done), reference_words=words,
        hf_wer=100 * hf_e.sum() / words, native_wer=100 * nat_e.sum() / words,
        difference=100 * (nat_e.sum() - hf_e.sum()) / words, difference_ci95=[float(lo), float(hi)],
        prompts_equal=sum(r["prompt_equal"] for r in done), ids_equal=sum(r["ids_equal"] for r in done),
        text_equal=sum(r["hf"] == r["native"] for r in done),
        native_better=int((nat_e < hf_e).sum()), native_worse=int((nat_e > hf_e).sum()),
        stages=dict(clips=len(stages),
                    mel_max_abs=max((s.get("mel_max_abs") or 0) for s in stages) if stages else None,
                    mel_native_closer_to_fp64=sum(s["mel_native_max_abs_to_fp64"] <= s["mel_hf_max_abs_to_fp64"]
                                                  for s in stages if "mel_hf_max_abs_to_fp64" in s),
                    mel_native_max_abs_to_fp64=max((s.get("mel_native_max_abs_to_fp64") or 0) for s in stages)
                    if stages else None,
                    mel_hf_max_abs_to_fp64=max((s.get("mel_hf_max_abs_to_fp64") or 0) for s in stages)
                    if stages else None,
                    ratio_max=max((s.get("ratio") or 0) for s in stages) if stages else None,
                    ratio_median=float(np.median([s["ratio"] for s in stages if "ratio" in s])) if stages else None),
        input_hours=sum(r["seconds"] or 0 for r in done) / 3600,
        hf_generate_seconds=sum(r["hf_generate_seconds"] for r in done),
        native_seconds=sum(r["native_prefill_seconds"] + r["native_generate_seconds"] for r in done),
        native_encode_seconds=sum(r["native_encode_seconds"] or 0 for r in done))
    Path(out_json).write_text(json.dumps(dict(summary=summary, stages=stages, cases=rows), indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
