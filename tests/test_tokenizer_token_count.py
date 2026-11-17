"""G6: token choice is bounded by the ids the tokenizer defines.

There is ONE implementation, lfm::tokenizer_token_count (native/include/lfm/model_config.hpp), used by
liquid-lfm3, liquid-lfm3-worker and the sampling oracle; gates read the count back from those binaries
rather than re-deriving it in Python, so no second implementation can disagree. These tests drive the
real function through native/probes/tokenizer_count_probe.cpp.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("tok") / "probe"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/probes/tokenizer_count_probe.cpp"), "-o", str(binary)], check=True)
    return binary


def count(probe, tmp_path, text, model_vocab=8):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return subprocess.run([str(probe), str(path), str(model_vocab)], capture_output=True, text=True).stdout.strip()


def tok(vocab, added=()):
    return json.dumps({"model": {"type": "BPE", "vocab": vocab}, "added_tokens": [{"id": i, "content": f"<{i}>"} for i in added]})


def test_contiguous_ids_give_the_count(probe, tmp_path):
    assert count(probe, tmp_path, tok({"a": 0, "b": 1, "c": 2}, [3, 4])) == "5"


def test_added_token_reusing_a_base_id_is_the_same_id(probe, tmp_path):
    assert count(probe, tmp_path, tok({"a": 0, "b": 1, "c": 2}, [2, 3])) == "4"


@pytest.mark.parametrize("name,text,message", [
    ("gap", tok({"a": 0, "b": 1, "c": 3}, [4]), "not contiguous"),
    ("beyond model rows", tok({"a": 0, "b": 1}, [2, 9]), "outside the model vocabulary"),
    ("negative id", tok({"a": 0, "b": -1}), "integer out of range"),
    ("empty vocabulary", tok({}, [0]), "nonempty"),
    ("two base tokens one id", tok({"a": 0, "b": 1, "c": 1}), "repeats an id"),
    ("fractional id", '{"model":{"vocab":{"a":0,"b":1,"c":2.5}},"added_tokens":[]}', "integer out of range"),
    ("boolean id", '{"model":{"vocab":{"a":0,"b":true}},"added_tokens":[]}', "must be a number"),
    ("string id", '{"model":{"vocab":{"a":0,"b":"1"}},"added_tokens":[]}', "must be a number"),
    ("NaN literal", '{"model":{"vocab":{"a":0,"b":NaN}},"added_tokens":[]}', "invalid JSON number"),
    ("duplicate key", '{"model":{"vocab":{"a":0,"a":1}},"added_tokens":[]}', "duplicate"),
    ("added_tokens not array", '{"model":{"vocab":{"a":0}},"added_tokens":""}', "must be an array"),
    ("added token without id", '{"model":{"vocab":{"a":0}},"added_tokens":[{"content":"x"}]}', "missing configuration field: id"),
    ("missing model", '{"added_tokens":[]}', "missing configuration field: model"),
    ("number overflow elsewhere", '{"x":1e400,"model":{"vocab":{"a":0,"b":1,"c":2}},"added_tokens":[]}', "outside the double range"),
    ("subnormal underflow elsewhere", '{"x":1e-310,"model":{"vocab":{"a":0,"b":1,"c":2}},"added_tokens":[]}', "outside the double range"),
    ("lone surrogate key", '{"model":{"vocab":{"\\ud800":0,"b":1}},"added_tokens":[]}', "low surrogate"),
    ("nesting deeper than 64", '{"x":' + "[" * 70 + "]" * 70 + ',"model":{"vocab":{"a":0}},"added_tokens":[]}', "too deep"),
])
def test_malformed_tokenizers_are_rejected(probe, tmp_path, name, text, message):
    out = count(probe, tmp_path, text)
    assert out.startswith("error") and message in out, (name, out)


def test_integral_float_id_is_that_id(probe, tmp_path):
    # The parser stores numbers as doubles, so 2.0 is the id 2.
    assert count(probe, tmp_path, '{"model":{"vocab":{"a":0,"b":1,"c":2.0}},"added_tokens":[]}') == "3"


@pytest.mark.skipif(not (CHECKPOINT / "tokenizer.json").exists(), reason="checkpoint not present")
def test_real_checkpoint_has_125033_tokens_of_128000_rows(probe, tmp_path):
    out = subprocess.run([str(probe), str(CHECKPOINT / "tokenizer.json"), "128000"], capture_output=True, text=True).stdout.strip()
    assert out == "125033"
