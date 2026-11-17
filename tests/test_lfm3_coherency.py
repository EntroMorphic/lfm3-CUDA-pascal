import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_coherency as coh  # noqa: E402


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids if not (skip_special_tokens and i == 99))


def write_case(tmp, engines, gens, texts, needle=None, limit=5):
    case = dict(capacity=512, max_new_tokens=limit, hf_prompt_tokens=3)
    if needle:
        case["needle"] = needle
    (tmp / "manifest.json").write_text(json.dumps({"c": case}))
    for engine, gen, text in zip(engines, gens, texts):
        (tmp / f"{engine}-c.json").write_text(json.dumps(dict(input_ids=[1, 2, 3], generated_ids=gen, text=text)))


def rows(out):
    return [line for line in out.splitlines() if line.startswith("| c |")]


def test_repeated_4grams():
    assert coh.repeated_4grams([1, 2, 3]) == 0.0
    assert coh.repeated_4grams([1, 2, 3, 4, 5]) == 0.0
    assert coh.repeated_4grams([1, 2, 3, 4] * 3) > 0.5


def test_report_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coh, "tokenizer", lambda _: FakeTokenizer())
    # a: stopped (stop token dropped from text); b: same first two ids, then diverges, hits the limit, text wrong.
    write_case(tmp_path, ["a", "b"], [[1, 2, 3, 99], [1, 2, 7, 8, 9]], ["t1 t2 t3", "wrong"])
    coh.report("ck", tmp_path, ["a", "b"])
    out = capsys.readouterr().out
    a, b = rows(out)
    assert "| 4 | yes | yes |" in a and a.endswith("| 4 |  |")
    assert "| 5 | **limit** | **no** |" in b and b.endswith("| 2 |  |")
    assert "b c: text differs from the HF decode" in out


def test_report_stop_exactly_at_limit_and_needle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coh, "tokenizer", lambda _: FakeTokenizer())
    write_case(tmp_path, ["a", "b"], [[1, 2, 3, 4, 98], [1, 2]], ["t1 t2 t3 t4", "violet harbor 42"],
               needle="violet harbor 42")
    coh.report("ck", tmp_path, ["a", "b"])
    out = capsys.readouterr().out
    a, b = rows(out)
    assert "| 5 | yes | yes |" in a and a.endswith("| **no** |")  # stop at the limit recognised; fact missing
    assert b.endswith("| 2 | yes |")
    assert "Every case" not in out  # b's text is not the decode of its ids


def test_report_missing_output_is_a_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coh, "tokenizer", lambda _: FakeTokenizer())
    write_case(tmp_path, ["a"], [[1, 2]], ["t1"])
    coh.report("ck", tmp_path, ["a", "b"])
    out = capsys.readouterr().out
    assert "| c | b | **failed: no output**" in out and "b c: no output" in out and "Every case" not in out
