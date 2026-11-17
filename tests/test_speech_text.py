"""Long-form TTS by sentences, natively (native/include/lfm/speech_text.hpp) through lfm3-text-tool.

The reference is the validated Python prototype: tools/lfm3_speech_chunked_hf.py sentences() for the split, and the
HF processor's chat template for each sentence's prompt ids (validation/speech-long-form-v1: WER 0.000 on 30/30
native runs built from those ids). Equal ids mean an identical generation.
"""
import importlib.util
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"
TOKENIZER = CHECKPOINT / "tokenizer.json"
SPEC = importlib.util.spec_from_file_location("chunked", ROOT / "tools/lfm3_speech_chunked_hf.py")
chunked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunked)

pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="checkpoint tokenizer required")


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("speech-text") / "lfm3-text-tool"
    subprocess.run(["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)], check=True)
    return binary


def run(tool, tmp_path, mode, rows):
    src, dst = tmp_path / f"{mode}.in", tmp_path / f"{mode}.out"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([str(tool), str(TOKENIZER), mode, str(src), str(dst)], check=True, capture_output=True)
    return [json.loads(line) for line in dst.read_text(encoding="utf-8").split("\n") if line]  # not splitlines(): U+2028


def test_whitespace_is_pythons_on_every_code_point(tool, tmp_path):
    native = run(tool, tmp_path, "spaces", [dict({"from": 0, "to": 0x110000})])[0]["spaces"]
    isspace = [c for c in range(0x110000) if chr(c).isspace()]
    regex = [c for c in range(0x110000) if re.fullmatch(r"\s", chr(c))]
    assert isspace == regex  # str.split()/strip() and the split regex agree on whitespace
    assert native == isspace


HAND = [
    "", "   ", "One.", "Hello world.", "a. b. c. d. e. f. g.",
    "This is a sentence of eight words in total. And here is another one that is long enough.",
    "Short one. This second sentence is comfortably long enough to stand. Tiny end.",
    "Wait... what? Really!   Yes.\tThat is how it works, every single time we try it out.",
    "Mr. Smith went to Washington on a very rainy day. He stayed for a week and then flew home.",
    "No terminal punctuation at all here but plenty of words to count",
    "Trailing spaces after the end of the sentence here.   \n\n",
    "　Ideographic space before. Not a break because of the nbsp? Actually it is one.",
    "Unicode — dashes, café, naïve, 漢字 and emoji 🙂 in a sentence. Then a second sentence with more words.",
    "Ends with a question? Ends with a bang! Ends with a period. And then one more full sentence here.",
    "x.y.z is not split. Neither is 3.14 or e.g. this, because no space follows those dots at all.",
    "Line one ends here.\nLine two starts here and keeps going for a while.\r\nLine three.",
    "A. B is separated by a line separator and has many words after it to count.",
    # punctuation and whitespace at the very start (the scan's first possible cut is after the first character)
    "!\n\nOne two three four five six seven.", "? Then a long sentence follows here with words. And another one here too.",
    # titles never end a sentence; abbreviations only before an ASCII capital (case-insensitive, ASCII only)
    "Dr. Alvarez met Mr. Chen at 9 a.m. on Jan. 5 to discuss the new U.S. trade rules. By 11 a.m. they had a plan ready.",
    "We met at 9 a.m. Then everyone went home for the rest of the long and quiet day.",
    "MR. SMITH and DR. JONES arrived. They sat down at the long table in the hall to begin.",
    "Trade with the U.\u212a. grew last year by a lot. The Kelvin sign is not an ASCII K at all here.",
    # the cap: a long sentence cut at its last clause mark at or before word 49, or evenly without one
    " ".join(["word"] * 30) + ", " + " ".join(["more"] * 40) + ".",
    " ".join(["plain"] * 120) + ".",
    " ".join(["dash"] * 20) + " \u2014 " + " ".join(["after"] * 40) + "; " + " ".join(["tail"] * 30),
    # CJK full stop followed by a space: not a break in the rule (adding it would change chunks)
    "This sentence has more than six words before its full stop。 Then another long sentence follows it here.",
]


def test_split_matches_python_on_edge_cases(tool, tmp_path):
    got = run(tool, tmp_path, "split", [{"text": t} for t in HAND])
    for t, g in zip(HAND, got):
        assert g["sentences"] == chunked.sentences(t), t


def test_the_cap_and_the_guard_as_chosen(tool, tmp_path):
    # validation/speech-split-v1: the default equals the tested cap-comma 49 chunks, and full 49 on the abbreviations
    spec = json.loads((ROOT / "validation/speech-e2e-v1/prompts-split-phase-b.json").read_text())
    cases = {c["case"]: c["user"] for c in spec["cases"]}
    for case in ("dialogue", "paragraphs", "long-comma", "long-nocomma"):
        assert chunked.sentences(cases[case]) == chunked.split_text(cases[case], "cap-comma", 49), case
    assert chunked.sentences(cases["abbrev"]) == chunked.split_text(cases["abbrev"], "full", 49)
    got = run(tool, tmp_path, "split", [{"text": t} for t in cases.values()])
    assert [g["sentences"] for g in got] == [chunked.sentences(t) for t in cases.values()]
    assert all(len(c.split()) <= chunked.CAP_WORDS for g in got for c in g["sentences"])
    # every validated prompt splits exactly as under the first rule, so its runs are unchanged
    for f in sorted((ROOT / "validation/speech-e2e-v1").glob("prompts*.json")):
        if "split-phase" in f.name:
            continue
        for c in json.loads(f.read_text())["cases"]:
            assert chunked.sentences(c["user"]) == chunked.sentences_v1(c["user"]), (f.name, c["case"])


def test_cap_boundaries_exhaustively(tool, tmp_path):
    # every chunk length 40-110 words, one clause mark at every position, each mark form; plus two marks at once
    forms = [lambda w: w + ",", lambda w: w + ";", lambda w: w + ":", lambda w: w + "\u2014", lambda w: w + "\u2013",
             lambda w: w + " -", lambda w: w + " \u2014", lambda w: w + " \u2013", lambda w: w + "--", lambda w: w + " --"]
    texts = [" ".join(["w"] * n) for n in range(40, 111)]
    for n in range(40, 111):
        for pos in range(1, n):
            for f in forms:
                words = ["w"] * n
                words[pos - 1] = f(words[pos - 1])
                texts.append(" ".join(words))
    for n in range(44, 100, 5):
        for a in range(1, n, 3):
            for b in range(a + 1, n, 7):
                words = ["w"] * n
                words[a - 1] += ","; words[b - 1] += ";"
                texts.append(" ".join(words))
    got = run(tool, tmp_path, "split", [{"text": t} for t in texts])
    bad = [t for t, g in zip(texts, got) if g.get("sentences") != chunked.sentences(t)]
    assert not bad, (len(bad), bad[:2])


def test_every_title_and_abbreviation(tool, tmp_path):
    # each list entry, in lower, title and upper case, before a lowercase word, a capital, and a quoted capital
    lead = "This sentence has enough words before the word"
    rows, want = [], []
    for entry in sorted(chunked.TITLES | chunked.ABBREVIATIONS) + ["it..."]:
        for form in {entry, entry[:1].upper() + entry[1:], entry.upper()}:
            for nxt, capital in (("then the rest follows on for many more words.", False),
                                 ("Then the rest follows on for many more words.", True),
                                 ("\u201cThen the rest follows on for many more words.\u201d", True),
                                 ("(Then the rest follows on for many more words.)", True),
                                 ("[Then the rest follows on for many more words.]", True),
                                 ("(A rest follows on here for many more words.)", True),
                                 ("'Then the rest follows on for many more words.'", True),
                                 ("\u2018Then the rest follows on for many more words.\u2019", True),
                                 ("\"(Then the rest follows on for many more words.)\"", True),
                                 ("\u201c\u2018[Then the rest follows on for many more words.]\u2019\u201d", True),
                                 ("\"(then the rest follows on for many more words.)\"", False)):
                rows.append(f"{lead} {form} {nxt}")
                title = entry in chunked.TITLES
                want.append(1 if title or not capital else 2)
    got = run(tool, tmp_path, "split", [{"text": t} for t in rows])
    for t, g, n in zip(rows, got, want):
        assert g["sentences"] == chunked.sentences(t), t
        assert len(g["sentences"]) == n, (t, g["sentences"])


def test_dashes_ellipses_and_whitespace(tool, tmp_path):
    cases = {  # text -> the chunks the rule gives
        "I was going to say something... but I forgot what it was all about.": 1,
        "I was going to say something important... Then I forgot what it was all about.": 2,
        "He asked me twice and I told him no. Then we left for the long drive home.": 2,
        "Line one of the text is here\n\nand line two of it continues right here.": 1,
    }
    long_dash = " ".join(["w"] * 30) + "-- " + " ".join(["v"] * 40) + "."
    lone_dash = " ".join(["w"] * 30) + " -- " + " ".join(["v"] * 40) + "."
    texts = list(cases) + [long_dash, lone_dash]
    got = run(tool, tmp_path, "split", [{"text": t} for t in texts])
    for t, g in zip(texts, got):
        assert g["sentences"] == chunked.sentences(t), t
        assert all("\n" not in c and "  " not in c for c in g["sentences"])  # every chunk: single spaces
    for t, g in zip(cases, got):
        assert len(g["sentences"]) == cases[t], (t, g["sentences"])
    assert [len(c.split()) for c in got[-2]["sentences"]] == [30, 40]  # "w--" ends the first piece
    assert [len(c.split()) for c in got[-1]["sentences"]] == [31, 40]  # a lone "--" is itself a mark: the cut falls after it


def test_size_is_linear(tool, tmp_path):
    import time
    texts = [" ".join(["word"] * 300000), "Short sentence number one here. " * 100000, "Mr. " * 200000 + "end."]
    t0 = time.time()
    got = run(tool, tmp_path, "split", [{"text": t} for t in texts])
    native = time.time() - t0
    t0 = time.time()
    want = [chunked.sentences(t) for t in texts]
    python = time.time() - t0
    assert [g["sentences"] for g in got] == want
    assert native < 20 and python < 20, (native, python)  # was quadratic: 200k words took 12 s and 9 GB natively


def test_clean_under_sanitizers(tmp_path):
    # memory errors and undefined behaviour (e.g. a whitespace scan running past the end) do not change outputs often
    # enough for the parity tests to see; AddressSanitizer and UBSan do
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path / "lfm3-text-tool-sanitized"
    subprocess.run(["g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    "-I", str(ROOT / "native/include"), str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)],
                   check=True)
    rng = random.Random(7)
    pieces = ["word", "Then", "Mr.", "no.", "a.m.", "etc.", "x...", "\"(So", ",", "--", "\u2014", " ", "  ", "\n", "\u3000",
              ".", "!", "?", "\u00e9", "\U0001F642"]
    texts = HAND + ["", " ", ".", "a.", " a. ", "a. ", "Mr. ", "\n\n."]
    texts += ["".join(rng.choice(pieces) + (" " if rng.random() < 0.7 else "") for _ in range(rng.randrange(0, 160)))
              for _ in range(3000)]
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    p = subprocess.run([str(binary), str(TOKENIZER), "split", str(src), str(dst)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    got = [json.loads(line) for line in dst.read_text(encoding="utf-8").split("\n") if line]
    assert [g["sentences"] for g in got] == [chunked.sentences(t) for t in texts]


def test_short_last_sentence_joins_the_one_before(tool, tmp_path):
    # the prototype's first version crashed on two parts and overwrote the wrong one on three (never hit by a
    # validation prompt); both now give the intended join
    texts = ["This first sentence has plenty of words in it. Short end.",
             "This first sentence has plenty of words in it. The second one is also long enough. Short end."]
    want = [["This first sentence has plenty of words in it. Short end."],
            ["This first sentence has plenty of words in it.", "The second one is also long enough. Short end."]]
    assert [chunked.sentences(t) for t in texts] == want
    assert [g["sentences"] for g in run(tool, tmp_path, "split", [{"text": t} for t in texts])] == want


def test_split_matches_python_fuzzed(tool, tmp_path):
    rng = random.Random(20261109)
    words = ["the", "a", "lighthouse", "keeper", "café", "漢字", "3.14", "e.g.", "Mr.", "🙂", "x", "Then", "Über",
             "dr.", "MRS.", "a.m.", "P.M.", "U.S.", "u.\u212a.", "Jan.", "etc.", "No.", "-", "\u2014", "\u2013", "word\u2014",
             "no.", "Jr.", "vs.", "word...", "\"Then", "(Next", "\u201cWhy", "--", "so--", "\u2018quiet",
             "'Then", "\u2018Then", "\"(Then", "\u201c'[So", "\u2026", "word....", "---"]
    puncts = [".", "!", "?", "...", "?!", ",", ";", ":", ""]
    spaces = [" ", "  ", "\t", "\n", " ", "　", " ", "\r\n", "\x1f", "\x85", "​"]
    texts = []
    for _ in range(5000):
        parts = []
        for _ in range(rng.randrange(0, 40) if rng.random() < 0.6 else rng.randrange(40, 160)):
            parts.append(rng.choice(words) + (rng.choice(puncts) if rng.random() < 0.3 else ""))
            parts.append(rng.choice(spaces) if rng.random() < 0.9 else "")
        if rng.random() < 0.3:
            parts.insert(0, rng.choice(spaces))
        texts.append("".join(parts))
    got = run(tool, tmp_path, "split", [{"text": t} for t in texts])
    mismatches = [t for t, g in zip(texts, got) if g.get("sentences") != chunked.sentences(t)]
    assert not mismatches, mismatches[:3]


def prompt_cases():
    seen = {}
    for f in sorted((ROOT / "validation/speech-e2e-v1").glob("prompts*.json")):
        spec = json.loads(f.read_text())
        for c in spec["cases"]:
            seen[(spec["systems"][c["system"]], c["user"])] = c["max_new_tokens"]
    return seen


def test_every_validation_prompt_matches_hf_ids(tool, tmp_path):
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    cases = prompt_cases()
    rows = [dict(messages=[{"role": "system", "content": s}, {"role": "user", "content": u}], capacity=4096,
                 max_new_tokens=n) for (s, u), n in cases.items()]
    got = run(tool, tmp_path, "speech", rows)
    split_any = 0
    for ((system, user), _), g in zip(cases.items(), got):
        texts = chunked.sentences(user)
        assert g["texts"] == texts
        split_any += len(texts) > 1
        want = [processor.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": t}],
                                              add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt")["input_ids"][0].tolist() for t in texts]
        assert g["ids"] == want, user
    assert len(rows) >= 10 and split_any >= 3


def test_each_sentence_is_rendered_as_its_own_single_request(tool, tmp_path):
    # without HF: every chunk's ids equal the chat mode's ids for [system, that sentence] (the system prompt kept)
    system = "Perform TTS in American English using a female voice."
    text = HAND[5] + " " + HAND[8]
    got = run(tool, tmp_path, "speech", [dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                                              capacity=4096, max_new_tokens=16)])[0]
    assert got["texts"] == chunked.sentences(text) and len(got["texts"]) >= 3
    single = run(tool, tmp_path, "chat", [dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": t}],
                                               capacity=4096, max_new_tokens=16) for t in got["texts"]])
    assert got["ids"] == [r["ids"] for r in single]
    bare = run(tool, tmp_path, "speech", [dict(messages=[{"role": "user", "content": text}], capacity=4096, max_new_tokens=16)])[0]
    assert bare["ids"] != got["ids"] and bare["texts"] == got["texts"]


@pytest.mark.parametrize("messages, why", [
    ([{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "x"}],
     "must be [system, user] or [user]"),
    ([{"role": "system", "content": "s"}], "must be [system, user] or [user]"),
    ([{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Read this sentence aloud please now."}],
     "must be [system, user] or [user]"),
    ([{"role": "user", "content": "First."}, {"role": "user", "content": "Read this sentence aloud please now."}],
     "must be [system, user] or [user]"),
    ([{"role": "user", "content": [{"type": "text", "text": "Hello there."}]}], "user content must be a string"),
    ([{"role": "user", "content": " \n "}], "user text is empty"),
])
def test_speech_split_refuses(tool, tmp_path, messages, why):
    got = run(tool, tmp_path, "speech", [dict(messages=messages, capacity=4096, max_new_tokens=16)])[0]
    assert why in got["error"]
