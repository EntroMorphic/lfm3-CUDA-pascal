#!/usr/bin/env python3
"""ASR round-trip in every speech locale (validation/multilingual-speech-v1; the C1/C4 measure of
journal/lfm3-tts-quality-bar.md, beyond English).

Like tools/lfm3_asr.py (whose English scoring it keeps unchanged for English cases), with the case's language from
PROMPTS.json ("language", Whisper's code, from tools/fleurs_cases.py):
- Whisper large-v3 transcribes response.wav with language=<the case's language>, task transcribe, greedy (sequential
  long-form over 30 s), and separately detects the clip's language (detect_language, the first 30 s): recorded, so a
  reply spoken in another language is seen, not scored as noise.
- English: Whisper's English normalizer and lfm3_asr's folds (ordinals, compounds), exactly as before.
- Other languages: Whisper's BasicTextNormalizer (lower case, punctuation and symbols to spaces; diacritics kept).
- Chinese, Japanese and Korean are scored by characters (CER: the normalized text without whitespace), as the model
  card scores them; the rest by words (WER).
- The intended text is the user's text in TTS mode and the model's own text in interleaved mode (lfm3_asr.intended_text).
OUT_JSON has lfm3_asr's layout ("results": case, wer, word_errors, ...; "wer" is the error rate in the case's metric),
so tools/lfm3_speech_report.py reads it unchanged; each result adds language, metric, detected_language.
Usage (lfm3-omni-dev): lfm3_asr_multilingual.py PROMPTS.json WHISPER_DIR OUT_JSON CASE_DIR [CASE_DIR ...]
  (each CASE_DIR is CASE-sSEED, from lfm3_speech_hf.py, lfm3_speech_chunked_hf.py or lfm3_speech_cases.py)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_asr import ascii_punctuation, fold_ordinals, intended_text, score, word_errors  # noqa: E402

CER_LANGUAGES = {"zh", "ja", "ko"}


def basic_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    except ImportError:  # moved in some transformers versions
        from transformers.models.whisper.tokenization_whisper import BasicTextNormalizer
    return BasicTextNormalizer()


def case_of(case_dir_name):
    """'tts-es-female-k0-s3' -> 'tts-es-female-k0'."""
    m = re.fullmatch(r"(.+)-s\d+", case_dir_name)
    if not m:
        raise ValueError(f"{case_dir_name}: not CASE-sSEED")
    return m.group(1)


def errors(language, intended, transcript, english_normalize=None, basic=None):
    """(errors, reference units, metric, normalized intended, normalized transcript)."""
    if language == "en":
        ref = fold_ordinals(english_normalize(ascii_punctuation(intended)).split())
        hyp = fold_ordinals(english_normalize(ascii_punctuation(transcript)).split())
        e, n = score(ref, hyp)
        return e, n, "wer", " ".join(ref), " ".join(hyp)
    ref, hyp = basic(intended), basic(transcript)
    if language in CER_LANGUAGES:
        r, h = list(re.sub(r"\s+", "", ref)), list(re.sub(r"\s+", "", hyp))
        return word_errors(r, h), len(r), "cer", "".join(r), "".join(h)
    r, h = ref.split(), hyp.split()
    return word_errors(r, h), len(r), "wer", " ".join(r), " ".join(h)


def main(prompts, whisper_dir, out_json, case_dirs):
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    spec = json.load(open(prompts, encoding="utf-8"))
    language_of = {c["case"]: c["language"] for c in spec["cases"]}
    processor = WhisperProcessor.from_pretrained(whisper_dir)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_dir, dtype=torch.float16).to("cuda").eval()
    english, basic = processor.tokenizer.normalize, basic_normalizer()
    id_to_token = processor.tokenizer.convert_ids_to_tokens
    results = []
    for case_dir in map(Path, case_dirs):
        summary = json.loads((case_dir / "summary.json").read_text())
        language = language_of[case_of(case_dir.name)]
        audio, _ = librosa.load(case_dir / "response.wav", sr=16_000)
        features = processor(audio, sampling_rate=16_000, return_tensors="pt", truncation=False,
                             padding="longest", return_attention_mask=True)
        if features.input_features.shape[-1] < 3000:  # short-form needs exactly 30 s of features
            features = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
        x = features.input_features.to("cuda", torch.float16)
        with torch.inference_mode():
            detected = id_to_token(int(model.detect_language(x[..., :3000]).flatten()[0])).strip("<|>")
            ids = model.generate(x, attention_mask=features.attention_mask.to("cuda"), language=language,
                                 task="transcribe", return_timestamps=True, num_beams=1, do_sample=False)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        intended = intended_text(summary)
        e, n, metric, ni, nt = errors(language, intended, transcript, english, basic)
        results.append(dict(case=summary.get("case", case_dir.name), mode=summary["mode"], language=language, detected_language=detected,
                            metric=metric, audio_seconds=summary["audio_seconds"], intended=intended,
                            transcript=transcript, normalized_intended=ni, normalized_transcript=nt,
                            reference_words=n, reference_words_folded=n, word_errors=e, wer=e / max(n, 1)))
        print(f"{case_dir.name}: {metric.upper()} {results[-1]['wer']:.3f} ({e}/{n}), heard as {detected}")
    Path(out_json).write_text(json.dumps(dict(asr=str(whisper_dir), scoring="multilingual-v1", results=results),
                                         ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
