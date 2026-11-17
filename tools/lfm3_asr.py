#!/usr/bin/env python3
"""ASR round-trip for the audio quality bar (journal/lfm3-tts-quality-bar.md, C1).

Transcribes each case's response.wav with Whisper large-v3 (English, greedy; sequential long-form for > 30 s), normalizes the
transcript and the intended words with Whisper's English text normalizer (after folding typographic punctuation to
ASCII; ordinal suffixes on numbers are then dropped on both sides), and reports word error rate (word-level
Levenshtein distance / reference words). The intended words are the user prompt in TTS mode ("audio") and the
model's own emitted text in interleaved mode (the speech should say what the text says).
Compounds (fold_compounds, since the step-4 red-team): 2 or 3 adjacent words in one text whose concatenation is a word
of the other text count as that word, both ways ("now high pressure" against the text's "now-high-pressure", which
the hyphen rule makes "nowhighpressure"). Before it, every such split counted as errors (il-fridge's WER was mostly
this).

Usage: lfm3_asr.py WHISPER_DIR OUT_JSON CASE_DIR [CASE_DIR ...]   (each CASE_DIR from lfm3_audio_trace.py)
       lfm3_asr.py rescore ASR_JSON [ASR_JSON ...]   recompute word_errors and wer in place from the stored normalized
       texts with the current rules (the earlier values are kept as word_errors_before / wer_before)
"""
import json
import re
import sys
from pathlib import Path


def word_errors(reference, hypothesis):
    """(substitutions + deletions + insertions) between word lists, by dynamic programming."""
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, h in enumerate(hypothesis, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


# Typographic punctuation the model's text uses and Whisper's transcripts and normalizer do not (found in the first
# run: curly apostrophes and U+2011 counted identical words as errors).
_ASCII = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2010": "-", "\u2011": "-",
                        "\u2012": "-", "\u2013": "-", "\u2014": " - ", "\u00a0": " ", "\u202f": " "})


def ascii_punctuation(text):
    """Typographic punctuation to ASCII, then hyphens inside words removed (on both sides): the model writes
    "meta-cognition" (U+2011) where Whisper writes "metacognition", and the normalizer would split the first into two
    words (found in #2 step 4's first native run)."""
    return re.sub(r"(?<=\w)-(?=\w)", "", text.translate(_ASCII))


def fold_ordinals(words):
    """'3rd' -> '3' on both sides: speech says "third", Whisper writes "3", the normalizer keeps "3rd"."""
    return [re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", w) for w in words]


def fold_compounds(reference, hypothesis):
    """Join runs of 2-3 adjacent words in each list whose concatenation is a word of the other list (longest first,
    left to right). Returns the two folded lists."""
    def fold(words, other):
        vocab, out, i = set(other), [], 0
        while i < len(words):
            for k in (3, 2):
                if i + k <= len(words) and "".join(words[i:i + k]) in vocab:
                    out.append("".join(words[i:i + k])); i += k
                    break
            else:
                out.append(words[i]); i += 1
        return out
    return fold(reference, hypothesis), fold(hypothesis, reference)


def score(reference, hypothesis):
    """(word errors, reference word count) after folding compounds."""
    r, h = fold_compounds(reference, hypothesis)
    return word_errors(r, h), len(r)


def rescore(paths):
    for path in paths:
        data = json.loads(Path(path).read_text())
        for r in data["results"]:
            if "wer_before" not in r:
                r["word_errors_before"], r["wer_before"] = r["word_errors"], r["wer"]
            errors, n = score(r["normalized_intended"].split(), r["normalized_transcript"].split())
            r["word_errors"], r["wer"], r["reference_words_folded"] = errors, errors / max(n, 1), n
            if abs(r["wer"] - r["wer_before"]) > 1e-12:
                print(f"{path}: {r['case']}: WER {r['wer_before']:.3f} -> {r['wer']:.3f}")
        data["scoring"] = "fold_compounds"
        Path(path).write_text(json.dumps(data, indent=1) + "\n")


def intended_text(summary):
    if summary["mode"] == "audio":
        return next(m["content"] for m in summary["conversation"] if m["role"] == "user")
    return summary["text"]


def main(whisper_dir, out_json, case_dirs):
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(whisper_dir)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_dir, dtype=torch.float16).to("cuda").eval()
    normalize = processor.tokenizer.normalize  # Whisper's English text normalizer (normalizer.json)
    results = []
    for case_dir in map(Path, case_dirs):
        summary = json.loads((case_dir / "summary.json").read_text())
        audio, _ = librosa.load(case_dir / "response.wav", sr=16_000)
        features = processor(audio, sampling_rate=16_000, return_tensors="pt", truncation=False,
                             padding="longest", return_attention_mask=True)
        if features.input_features.shape[-1] < 3000:  # short-form needs exactly 30 s of features
            features = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(features.input_features.to("cuda", torch.float16),
                                 attention_mask=features.attention_mask.to("cuda"), language="en",
                                 task="transcribe", return_timestamps=True, num_beams=1, do_sample=False)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        reference = fold_ordinals(normalize(ascii_punctuation(intended_text(summary))).split())
        hypothesis = fold_ordinals(normalize(ascii_punctuation(transcript)).split())
        errors, folded = score(reference, hypothesis)
        results.append(dict(case=summary["case"], mode=summary["mode"], audio_seconds=summary["audio_seconds"],
                            intended=intended_text(summary), transcript=transcript,
                            normalized_intended=" ".join(reference), normalized_transcript=" ".join(hypothesis),
                            reference_words=len(reference),
                            reference_words_folded=folded, word_errors=errors, wer=errors / max(folded, 1)))
        print(f"{summary['case']}: WER {results[-1]['wer']:.3f} ({errors}/{len(reference)})")
    Path(out_json).write_text(json.dumps(dict(asr=str(whisper_dir), scoring="fold_compounds", results=results), indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "rescore":
        rescore(sys.argv[2:])
    elif len(sys.argv) >= 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
