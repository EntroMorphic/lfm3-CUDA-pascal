"""validation/multilingual-speech-v1: the host parts of the round's tools, on synthetic data (no GPU, no FLEURS).

- tools/fleurs_cases.py: the draw is fixed by the seed, excludes digits and out-of-range English lengths, covers the
  model card's 19 locales x 2 voices, reads English for both English locales, and refuses a clip that is not 16 kHz
  mono 16-bit PCM.
- tools/lfm3_asr_multilingual.py: CER for Chinese, Japanese and Korean, WER otherwise, English scored exactly as
  tools/lfm3_asr.py scores it.
- tools/check_lfm3_multilingual_asr.py: the per-language and overall bootstrap reading.
- tools/check_lfm3_multilingual_server.py and tools/lfm3_multilingual_report.py: case lists and helpers.
- The sentence splitter on non-English text: native equals Python, and the known gap (no split on CJK or Arabic
  sentence marks) is pinned, so a fix shows up as a deliberate change.
"""
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import fleurs_cases  # noqa: E402
import check_lfm3_multilingual_asr as mlasr  # noqa: E402
import check_lfm3_multilingual_server as mlserver  # noqa: E402
import lfm3_asr  # noqa: E402
import lfm3_asr_multilingual as mlscore  # noqa: E402
import lfm3_multilingual_report as mlreport  # noqa: E402


def write_wav(path, rate=16_000, channels=1, width=2, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes(b"\0" * frames * channels * width)


def fake_fleurs(root, ids=range(1, 41), digit_ids=(3, 7), short_ids=(5,), missing=None, rows_per_id=2):
    """Every config's test.tsv with each id read by rows_per_id speakers; digits in some ids' text in one language."""
    for config in fleurs_cases.configs():
        lines = []
        for i in ids:
            if missing and missing == (config, i):
                continue
            words = 4 if (i in short_ids and config == "en_us") else 12
            text = " ".join(f"{config}-w{k}-{i}" for k in range(words)).replace("0", "o").replace("1", "l")
            text = text.translate(str.maketrans("23456789", "zeasbtgq"))
            if i in digit_ids and config == "de_de":
                text += " 42"
            for s in range(rows_per_id):
                name = f"{i:05d}{s}.wav"
                lines.append("\t".join([str(i), name, text, text.lower(), "p h", "16000", "FEMALE"]))
                write_wav(root / config / "audio" / "test" / name)
        (root / config / "test.tsv").parent.mkdir(parents=True, exist_ok=True)
        (root / config / "test.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cases_cover_every_locale_and_voice(tmp_path):
    fake_fleurs(tmp_path)
    b = fleurs_cases.build(tmp_path, n_asr=3)
    assert len(fleurs_cases.LOCALES) == 19 and len(fleurs_cases.configs()) == 18
    assert len(b["tts"]["cases"]) == 19 * 2 * 6 and b["tts"]["seeds"] == [1, 2, 3]
    assert b["escalate"]["seeds"] == list(range(4, 16)) and b["escalate"]["cases"] == b["tts"]["cases"]
    assert len(b["long"]["cases"]) == 19 and len(b["interleaved"]["cases"]) == 18 * 2
    assert len(b["asr"]["cases"]) == 18 * 3
    systems = set(b["tts"]["systems"].values())
    assert "Perform TTS in British English using a male voice." in systems
    assert "Perform TTS in Chinese using a female voice." in systems
    assert b["asr"]["systems"]["asr_zh"] == "Perform ASR in Chinese." and b["asr"]["systems"]["asr_en"] == "Perform ASR in English."
    by = {c["case"]: c for c in b["tts"]["cases"]}
    for k in range(6):  # both English locales read the same English text
        assert by[f"tts-en-gb-female-k{k}"]["user"] == by[f"tts-en-us-male-k{k}"]["user"]
        assert by[f"tts-en-gb-female-k{k}"]["user"].startswith("en_us-")
        assert by[f"tts-de-male-k{k}"]["user"].startswith("de_de-")
    assert {c["metric"] for c in b["asr"]["cases"] if c["language"] in ("zh", "ja", "ko")} == {"cer"}
    assert {c["metric"] for c in b["asr"]["cases"] if c["language"] == "fi"} == {"wer"}


def test_the_draw_is_fixed_and_excludes_digits_and_lengths(tmp_path):
    fake_fleurs(tmp_path)
    a, b = fleurs_cases.build(tmp_path, n_asr=1), fleurs_cases.build(tmp_path, n_asr=1)
    assert a["selection"] == b["selection"]
    s = a["selection"]
    assert 3 not in s["eligible"] and 7 not in s["eligible"] and 5 not in s["eligible"]
    drawn = s["tts_ids"] + s["long_ids"] + s["interleaved_ids"]
    assert len(set(drawn)) == 13 and set(drawn) <= set(s["eligible"])
    assert fleurs_cases.build(tmp_path, n_asr=1, seed=7)["selection"]["tts_ids"] != s["tts_ids"]


def test_an_id_missing_in_one_language_is_not_eligible(tmp_path):
    fake_fleurs(tmp_path, missing=("ja_jp", 10))
    assert 10 not in fleurs_cases.build(tmp_path, n_asr=1)["selection"]["eligible"]


def test_asr_takes_the_first_rows_by_id_then_file(tmp_path):
    fake_fleurs(tmp_path)
    b = fleurs_cases.build(tmp_path, n_asr=3, audio_root="/workspace/data/fleurs")
    es = [c for c in b["asr"]["cases"] if c["language"] == "es"]
    assert [c["case"] for c in es] == ["asr-es-000010", "asr-es-000011", "asr-es-000020"]
    assert es[0]["audio"] == ["/workspace/data/fleurs/es_419/audio/test/000010.wav"]
    assert es[0]["mode"] == "text" and es[0]["seed"] == 0


@pytest.mark.parametrize("bad", [dict(rate=24_000), dict(channels=2), dict(width=4)])
def test_a_clip_that_is_not_16k_mono_pcm16_is_refused(tmp_path, bad):
    fake_fleurs(tmp_path)
    write_wav(tmp_path / "it_it" / "audio" / "test" / "000010.wav", **bad)
    with pytest.raises(SystemExit, match="refused"):
        fleurs_cases.build(tmp_path, n_asr=2)
    fleurs_cases.build(tmp_path, n_asr=2, check_audio=False)  # --no-audio skips the clips


def test_too_few_eligible_ids_is_an_error(tmp_path):
    fake_fleurs(tmp_path, ids=range(1, 10))
    with pytest.raises(SystemExit, match="eligible"):
        fleurs_cases.build(tmp_path, n_asr=1)


def test_scoring_metrics():
    BasicTextNormalizer = pytest.importorskip("transformers.models.whisper.english_normalizer").BasicTextNormalizer
    basic = BasicTextNormalizer()
    e, n, metric, _, _ = mlscore.errors("zh", "我有三只猫。", "我有三只狗", basic=basic)
    assert (e, n, metric) == (1, 5, "cer")
    e, n, metric, ni, nt = mlscore.errors("es", "¿Dónde está la estación?", "donde esta la estación", basic=basic)
    assert metric == "wer" and n == 4 and e == 2  # diacritics are kept: dónde/donde and está/esta differ
    e, n, metric, _, _ = mlscore.errors("ja", "猫が三匹います。", "猫が三匹います", basic=basic)
    assert (e, metric) == (0, "cer")
    assert mlscore.case_of("tts-es-female-k0-s12") == "tts-es-female-k0"
    with pytest.raises(ValueError):
        mlscore.case_of("no-seed")


def test_english_is_scored_as_before():
    english = lambda t: t.lower().replace(",", "").replace(".", "")  # noqa: E731  (stand-in for Whisper's normalizer)
    intended, heard = "The 3rd now-high-pressure ridge.", "the 3 now high pressure ridge"
    e, n, metric, _, _ = mlscore.errors("en", intended, heard, english_normalize=english)
    ref = lfm3_asr.fold_ordinals(english(lfm3_asr.ascii_punctuation(intended)).split())
    hyp = lfm3_asr.fold_ordinals(english(lfm3_asr.ascii_punctuation(heard)).split())
    assert (e, n) == lfm3_asr.score(ref, hyp) and metric == "wer" and e == 0


def rows(lang, n, hf, native, units=10):
    return [dict(language=lang, n=units, hf_errors=hf, native_errors=native, metric="wer") for _ in range(n)]


def test_asr_summary_equal_engines_pass():
    per, overall = mlasr.summarize(rows("es", 50, 1, 1) + rows("zh", 50, 2, 2), resamples=2000)
    assert overall["pass_"] and overall["mean_difference"] == 0 and not overall["flagged"]
    assert per["es"]["hf_rate"] == pytest.approx(10.0) and per["zh"]["difference"] == 0


def test_asr_summary_flags_a_worse_language_and_fails_overall():
    per, overall = mlasr.summarize(rows("es", 50, 1, 1) + rows("fi", 50, 1, 2), resamples=2000)
    assert per["fi"]["difference"] == pytest.approx(10.0) and per["fi"]["flagged"]
    assert overall["flagged"] == ["fi"] and not overall["pass_"]
    assert overall["mean_difference"] == pytest.approx(5.0)  # languages averaged, not pooled


def test_server_case_lists():
    tts = dict(systems={f"tts_{l}_{v}": f"Perform TTS in {l} using a {v} voice." for l in fleurs_cases.LOCALES
                        for v in fleurs_cases.VOICES},
               cases=[dict(case=f"tts-{l}-{v}-k{k}", system=f"tts_{l}_{v}", user="x", locale=l, max_new_tokens=768)
                      for l in fleurs_cases.LOCALES for v in fleurs_cases.VOICES for k in range(6)])
    il = dict(systems={"interleaved": "Respond with interleaved text and audio."},
              cases=[dict(case=f"il-{c}-k{k}", system="interleaved", user="y", max_new_tokens=1024)
                     for _, (_, c) in fleurs_cases.configs().items() for k in range(2)])
    asr = dict(systems={"asr_es": "Perform ASR in Spanish."},
               cases=[dict(case=f"asr-es-{i}", language="es", system="asr_es", audio=["a.wav"], max_new_tokens=512)
                      for i in range(5)])
    got = mlserver.exact_cases(tts, il, asr)
    assert len([g for g in got if g[2] == "audio"]) == 38 and len([g for g in got if g[2] == "interleaved"]) == 18
    assert len([g for g in got if g[2] == "text"]) == 2
    norm = {g[0]: g[5]["normalize"] for g in got if g[2] == "audio"}
    assert norm["tts-en-gb-male-k0-s1"] and norm["tts-en-us-female-k0-s1"] and not norm["tts-es-female-k0-s1"]
    gates = mlserver.gate_cases(tts)
    assert len(gates) == 19 and sum(g[3] for g in gates) == 2
    assert set(mlserver.GATE_TEXT) == set(fleurs_cases.LOCALES)
    assert all(any(ch.isdigit() for ch in t) for t in mlserver.GATE_TEXT.values())


def test_digits_kept():
    assert mlserver.digits_kept("Tengo 3 gatos y 12 libros.", ["Tengo 3 gatos y 12 libros."])
    assert not mlserver.digits_kept("Tengo 3 gatos y 12 libros.", ["Tengo tres gatos y 12 libros."])
    assert mlserver.digits_kept("我有3只猫和12本书。", ["我有3只猫和12本书。"])


def test_report_groups():
    assert mlreport.group_key(dict(case="tts-en-gb-female-k3", locale="en-gb")) == "en-gb female"
    assert mlreport.group_key(dict(case="long-ja", locale="ja")) == "ja"
    assert mlreport.group_key(dict(case="il-ko-k1", locale="ko")) == "ko"


# the splitter on non-English text (native lfm/speech_text.hpp against tools/lfm3_speech_chunked_hf.py sentences())
SPEC = importlib.util.spec_from_file_location("chunked", ROOT / "tools/lfm3_speech_chunked_hf.py")
chunked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunked)
MULTILINGUAL = {  # written for this test: sentence structure, not content, matters
    "es": "El faro avisa a los barcos durante toda la noche. La luz gira despacio sobre el mar oscuro y frío. "
          "Los marineros la ven desde muy lejos y saben dónde está la costa.",
    "de": "Der Leuchtturm warnt die Schiffe in der ganzen Nacht. Das Licht dreht sich langsam über dem dunklen Meer. "
          "Die Seeleute sehen es von weitem und wissen, wo die Küste liegt.",
    "el": "Ο φάρος προειδοποιεί τα πλοία όλη τη νύχτα. Το φως γυρίζει αργά πάνω από τη σκοτεινή θάλασσα. "
          "Οι ναυτικοί το βλέπουν από μακριά και ξέρουν πού είναι η ακτή.",
    "zh": "灯塔整夜提醒过往的船只。灯光在黑暗的海面上缓慢旋转。水手们从很远的地方就能看到它，知道海岸在哪里。",
    "ja": "灯台は一晩中、船に注意を促します。光は暗い海の上をゆっくりと回ります。船乗りたちは遠くからそれを見て、海岸の位置を知ります。",
    "ar": "تحذر المنارة السفن طوال الليل. يدور الضوء ببطء فوق البحر المظلم البارد. "
          "يراه البحارة من بعيد جدا ويعرفون أين يقع الشاطئ؟ نعم يعرفون ذلك جيدا في كل ليلة هادئة.",
}


@pytest.fixture(scope="module")
def tool(tmp_path_factory):  # as tests/test_speech_text.py builds it
    import shutil
    import subprocess
    import test_speech_text
    if not test_speech_text.TOKENIZER.exists() or shutil.which("g++") is None:
        pytest.skip("checkpoint tokenizer and g++ required")
    binary = tmp_path_factory.mktemp("speech-text-ml") / "lfm3-text-tool"
    subprocess.run(["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)], check=True)
    return binary


def test_splitter_native_equals_python_on_multilingual_text(tool, tmp_path):
    import test_speech_text
    got = test_speech_text.run(tool, tmp_path, "split", [{"text": t} for t in MULTILINGUAL.values()])
    for (lang, t), g in zip(MULTILINGUAL.items(), got):
        assert g["sentences"] == chunked.sentences(t), lang


def test_splitter_gap_on_cjk_and_arabic_marks_is_pinned():
    """Known before any run (README "found before writing these rules"): the rule splits only after . ! ? followed by
    whitespace, and counts words by whitespace. A Chinese or Japanese paragraph is one chunk of one 'word' (never
    split, never capped); Arabic splits at '.', not at its question mark U+061F. Latin and Greek split by sentence."""
    s = {lang: chunked.sentences(t) for lang, t in MULTILINGUAL.items()}
    assert len(s["zh"]) == 1 and len(s["ja"]) == 1
    assert len(s["es"]) == 3 and len(s["de"]) == 3 and len(s["el"]) == 3
    assert len(s["ar"]) == 2 and "؟ نعم" in s["ar"][1]
