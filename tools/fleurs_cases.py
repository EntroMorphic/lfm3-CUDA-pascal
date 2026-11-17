#!/usr/bin/env python3
"""validation/multilingual-speech-v1: the fixed case lists, drawn from FLEURS test before any run.

FLEURS (google/fleurs, CC-BY-4.0) is n-way parallel: each FLoRes sentence id is the same sentence, professionally
translated, in every language, read by native speakers. So one draw of ids gives every locale the same content.
Reads FLEURS_DIR/LANG/test.tsv (no header; id, file_name, raw_transcription, transcription, phonemes, num_samples,
gender, tab-separated) and, for ASR, FLEURS_DIR/LANG/audio/test/FILE (the extracted test.tar.gz; 16 kHz mono WAV,
checked: any other clip is refused, not resampled). Writes to OUT_DIR:
- tts.json (tools/lfm3_speech_hf.py's prompts format): every locale (19) x voice (female, male) x 6 sentences, seeds 1-3,
  mode audio, max_new_tokens 768, system "Perform TTS in <locale> using a <voice> voice." (the model card's prompt);
  American and British English read the English text. Each case carries "locale" and "language" (Whisper's code).
- tts-escalate.json: the same cases at seeds 4-15 (rule 2's escalation; only flagged cases are run).
- long.json (tools/lfm3_speech_chunked_hf.py's format): per locale, one paragraph of the next 5 drawn sentences joined
  by single spaces, female voice, seeds 1-3, max_new_tokens 1536 per chunk.
- interleaved.json: per language (18; English once), the next 2 drawn sentences as the user turn, system "Respond with
  interleaved text and audio.", seeds 1-3, max_new_tokens 1024.
- asr-cases.json (tools/lfm3_audio_in_hf.py's format): per language the first N_ASR utterances of test.tsv sorted by
  (id, file_name), system "Perform ASR in <language>.", mode text (greedy), max_new_tokens 512, reference = FLEURS's
  normalized transcription. Unless --no-audio, each clip's WAV is checked.
- selection.json: the eligible ids, the draw, and the per-language rows used.
The draw: ids present in all 18 languages' test.tsv whose raw transcription has no digit (Unicode \\d) in any language
(digits are the model's known weak spot, journal/lfm3-tts-quality-bar.md C3; tested separately by the numbers cases)
and whose English raw transcription is 8-30 words; sorted by int(id); random.Random(SEED).sample(eligible, 13): the
first 6 TTS, the next 5 long-form, the last 2 interleaved. Each id's text in a language is its first row by file_name.
Usage: fleurs_cases.py FLEURS_DIR OUT_DIR [--n-asr N (100)] [--seed S (2026)] [--audio-root DIR] [--no-audio]
  --audio-root: the FLEURS_DIR path as the GPU container sees it, written into asr-cases.json (default FLEURS_DIR).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

# The model card's 19 speech locales (README "Speech languages and locales"): locale key -> (name in the TTS prompt,
# name in the ASR prompt, FLEURS config, Whisper language code). English is one FLEURS language read in two locales.
LOCALES = {
    "ar": ("Arabic", "Arabic", "ar_eg", "ar"),
    "da": ("Danish", "Danish", "da_dk", "da"),
    "de": ("German", "German", "de_de", "de"),
    "el": ("Greek", "Greek", "el_gr", "el"),
    "en-gb": ("British English", "English", "en_us", "en"),
    "en-us": ("American English", "English", "en_us", "en"),
    "fi": ("Finnish", "Finnish", "fi_fi", "fi"),
    "fr": ("French", "French", "fr_fr", "fr"),
    "he": ("Hebrew", "Hebrew", "he_il", "he"),
    "it": ("Italian", "Italian", "it_it", "it"),
    "ja": ("Japanese", "Japanese", "ja_jp", "ja"),
    "ko": ("Korean", "Korean", "ko_kr", "ko"),
    "nl": ("Dutch", "Dutch", "nl_nl", "nl"),
    "pl": ("Polish", "Polish", "pl_pl", "pl"),
    "pt": ("Portuguese", "Portuguese", "pt_br", "pt"),
    "es": ("Spanish", "Spanish", "es_419", "es"),
    "sv": ("Swedish", "Swedish", "sv_se", "sv"),
    "tr": ("Turkish", "Turkish", "tr_tr", "tr"),
    "zh": ("Chinese", "Chinese", "cmn_hans_cn", "zh"),
}
VOICES = ("female", "male")
CER_LANGUAGES = {"zh", "ja", "ko"}  # the model card's multilingual ASR uses CER for these (and WER for the rest)
N_TTS, N_LONG, N_INTERLEAVED = 6, 5, 2
DIGIT = re.compile(r"\d")


def configs():
    """FLEURS config -> (ASR language name, Whisper code), one per language (18)."""
    out = {}
    for _, asr_name, config, code in LOCALES.values():
        out[config] = (asr_name, code)
    return out


def read_tsv(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != 7:
            raise SystemExit(f"{path}:{n}: expected 7 tab-separated fields, got {len(f)}")
        rows.append(dict(id=int(f[0]), file_name=f[1], raw=f[2], transcription=f[3], num_samples=int(f[5]), gender=f[6]))
    return rows


def first_rows(rows):
    """id -> its first row by file_name."""
    out = {}
    for r in sorted(rows, key=lambda r: (r["id"], r["file_name"])):
        out.setdefault(r["id"], r)
    return out


def draw(by_config, seed):
    """The eligible ids and the drawn 13 (see the module docstring)."""
    common = set.intersection(*(set(rows) for rows in by_config.values()))
    eligible = sorted(i for i in common
                      if not any(DIGIT.search(rows[i]["raw"]) for rows in by_config.values())
                      and 8 <= len(by_config["en_us"][i]["raw"].split()) <= 30)
    need = N_TTS + N_LONG + N_INTERLEAVED
    if len(eligible) < need:
        raise SystemExit(f"only {len(eligible)} eligible sentence ids; need {need}")
    picked = random.Random(seed).sample(eligible, need)
    return eligible, picked[:N_TTS], picked[N_TTS:N_TTS + N_LONG], picked[N_TTS + N_LONG:]


def wav_ok(path):
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() > 0
    except (wave.Error, EOFError, FileNotFoundError):
        return False


def build(fleurs, n_asr=100, seed=2026, audio_root=None, check_audio=True):
    fleurs = Path(fleurs)
    audio_root = Path(audio_root or fleurs)
    rows = {c: read_tsv(fleurs / c / "test.tsv") for c in configs()}
    by_config = {c: first_rows(r) for c, r in rows.items()}
    eligible, tts_ids, long_ids, il_ids = draw(by_config, seed)
    text = lambda locale, i: by_config[LOCALES[locale][2]][i]["raw"].strip()  # noqa: E731

    tts = dict(systems={}, cases=[], seeds=[1, 2, 3])
    for locale, (tts_name, _, config, code) in LOCALES.items():
        for voice in VOICES:
            key = f"tts_{locale}_{voice}"
            tts["systems"][key] = f"Perform TTS in {tts_name} using a {voice} voice."
            for k, i in enumerate(tts_ids):
                tts["cases"].append(dict(case=f"tts-{locale}-{voice}-k{k}", mode="audio", system=key, max_new_tokens=768,
                                         user=text(locale, i), locale=locale, language=code, fleurs_id=i))
    escalate = dict(tts, seeds=list(range(4, 16)))

    long = dict(systems={}, cases=[], seeds=[1, 2, 3])
    for locale, (tts_name, _, config, code) in LOCALES.items():
        key = f"tts_{locale}_female"
        long["systems"][key] = f"Perform TTS in {tts_name} using a female voice."
        long["cases"].append(dict(case=f"long-{locale}", mode="audio", system=key, max_new_tokens=1536,
                                  user=" ".join(text(locale, i) for i in long_ids), locale=locale, language=code,
                                  fleurs_ids=long_ids))

    interleaved = dict(systems={"interleaved": "Respond with interleaved text and audio."}, cases=[], seeds=[1, 2, 3])
    asr = dict(systems={}, cases=[])
    for config, (asr_name, code) in configs().items():
        locale = next(k for k, v in LOCALES.items() if v[2] == config and k != "en-gb")
        for k, i in enumerate(il_ids):
            interleaved["cases"].append(dict(case=f"il-{code}-k{k}", mode="interleaved", system="interleaved",
                                             max_new_tokens=1024, user=text(locale, i), locale=locale, language=code,
                                             fleurs_id=i))
        key = f"asr_{code}"
        asr["systems"][key] = f"Perform ASR in {asr_name}."
        chosen = sorted(rows[config], key=lambda r: (r["id"], r["file_name"]))[:n_asr]
        for r in chosen:
            wav = fleurs / config / "audio" / "test" / r["file_name"]
            if check_audio and not wav_ok(wav):
                raise SystemExit(f"{wav}: missing, or not 16 kHz mono 16-bit PCM (refused, not resampled)")
            asr["cases"].append(dict(case=f"asr-{code}-{Path(r['file_name']).stem}", audio=[str(audio_root / config / "audio" / "test" / r["file_name"])],
                                     system=key, mode="text", max_new_tokens=512, seed=0, reference=r["transcription"],
                                     raw_reference=r["raw"], language=code, metric="cer" if code in CER_LANGUAGES else "wer",
                                     seconds=round(r["num_samples"] / 16_000, 2)))
    selection = dict(seed=seed, n_asr=n_asr, eligible=eligible, tts_ids=tts_ids, long_ids=long_ids, interleaved_ids=il_ids,
                     texts={c: {i: by_config[c][i]["raw"] for i in tts_ids + long_ids + il_ids} for c in by_config})
    return dict(tts=tts, escalate=escalate, long=long, interleaved=interleaved, asr=asr, selection=selection)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("fleurs")
    p.add_argument("out")
    p.add_argument("--n-asr", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--audio-root")
    p.add_argument("--no-audio", action="store_true")
    a = p.parse_args(argv)
    built = build(a.fleurs, a.n_asr, a.seed, a.audio_root, not a.no_audio)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = dict(tts="tts.json", escalate="tts-escalate.json", long="long.json", interleaved="interleaved.json",
                 asr="asr-cases.json", selection="selection.json")
    for k, name in names.items():
        (out / name).write_text(json.dumps(built[k], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"tts {len(built['tts']['cases'])} cases, long {len(built['long']['cases'])}, interleaved "
          f"{len(built['interleaved']['cases'])}, asr {len(built['asr']['cases'])}; ids {built['selection']['tts_ids']} "
          f"{built['selection']['long_ids']} {built['selection']['interleaved_ids']}")


if __name__ == "__main__":
    main(sys.argv[1:])
