import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_asr as asr  # noqa: E402


def test_word_errors():
    ref = "the quick brown fox".split()
    assert asr.word_errors(ref, ref) == 0
    assert asr.word_errors(ref, "the quick fox".split()) == 1            # deletion
    assert asr.word_errors(ref, "the quick brown red fox".split()) == 1  # insertion
    assert asr.word_errors(ref, "a quick brown box".split()) == 2         # substitutions
    assert asr.word_errors(ref, []) == 4 and asr.word_errors([], ["x"]) == 1


def test_intended_text():
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "say this"}]
    assert asr.intended_text(dict(mode="audio", conversation=conv, text="")) == "say this"
    assert asr.intended_text(dict(mode="interleaved", conversation=conv, text="spoken reply")) == "spoken reply"


def test_ascii_punctuation():
    # intra-word hyphens are removed too (both sides), so "self‑reflection" and "selfreflection" match
    assert asr.ascii_punctuation("they’ve self‑reflection “yes”") == "they've selfreflection \"yes\""
    assert asr.ascii_punctuation("plain ascii") == "plain ascii"


def test_fold_ordinals():
    assert asr.fold_ordinals(["march", "3rd", "21st", "2024", "4th", "north", "3"]) == \
        ["march", "3", "21", "2024", "4", "north", "3"]


def test_intra_word_hyphens_are_removed_on_both_sides():
    assert asr.ascii_punctuation("a kind of meta\u2011cognition") == "a kind of metacognition"
    assert asr.ascii_punctuation("self-modeling - and more") == "selfmodeling - and more"


def test_compounds_count_as_one_word_either_way():
    ref = asr.ascii_punctuation("the now\u2011high\u2011pressure refrigerant").split()
    for spoken in ("the now high pressure refrigerant", "the now high-pressure refrigerant", "the nowhighpressure refrigerant"):
        assert asr.score(ref, asr.ascii_punctuation(spoken).split()) == (0, 3), spoken
    # the text's two words, Whisper's one: also no error
    assert asr.score("a kind of meta cognition".split(), "a kind of metacognition".split()) == (0, 4)
    # real errors still count: a missing word, a wrong word, a compound only partly spoken
    assert asr.score("the nowhighpressure refrigerant".split(), "the refrigerant".split())[0] == 1
    assert asr.score("the nowhighpressure refrigerant".split(), "the now pressure refrigerant".split())[0] >= 1
    assert asr.score("the quick brown fox".split(), "the quick brown box".split()) == (1, 4)


def test_rescore_keeps_the_earlier_values(tmp_path):
    import json
    f = tmp_path / "asr.json"
    f.write_text(json.dumps(dict(results=[dict(case="c-s1", normalized_intended="the nowhighpressure gas",
                                              normalized_transcript="the now high pressure gas", word_errors=3, wer=1.0)])))
    asr.rescore([f]); asr.rescore([f])  # idempotent: the first values stay the "before"
    r = json.loads(f.read_text())["results"][0]
    assert (r["word_errors"], r["wer"], r["word_errors_before"], r["wer_before"]) == (0, 0.0, 3, 1.0)
