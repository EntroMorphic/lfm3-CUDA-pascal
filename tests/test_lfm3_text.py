"""G1-G3: native LFM3 text frontend (native/include/lfm/lfm3_text.hpp) through lfm3-text-tool.

Bit-exactness against the reference is established by tools/lfm3_text_diff.py (evidence in
validation/g2-text-v1). These tests pin what that suite cannot: the loader fails closed on every
tokenizer.json deviation from the pipeline it implements, input limits and invalid input are refused,
and the stream decoder releases text as soon as a character is complete.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "models" / "LFM3-Omni-2B"
TOKENIZER = CHECKPOINT / "tokenizer.json"

pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="checkpoint tokenizer required")


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("text") / "lfm3-text-tool"
    subprocess.run(["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/lfm3_text_tool.cpp"), "-o", str(binary)], check=True)
    return binary


@pytest.fixture(scope="module")
def spec():
    return json.loads(TOKENIZER.read_text(encoding="utf-8"))


def run(tool, tmp_path, mode, rows, tokenizer=TOKENIZER, raw_lines=None):
    src, dst = tmp_path / f"{mode}.in", tmp_path / f"{mode}.out"
    if raw_lines is not None:
        src.write_bytes(b"\n".join(raw_lines) + b"\n")
    else:
        src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    p = subprocess.run([str(tool), str(tokenizer), mode, str(src), str(dst)], capture_output=True, text=True)
    if p.returncode != 0:
        return p.returncode, p.stderr
    return 0, [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]


def byte_ids(spec):
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    char = {b: chr(c) for b, c in zip(bs, cs)}
    return {b: spec["model"]["vocab"][char[b]] for b in range(256)}


# ---------------------------------------------------------------- pinned reference values

def test_pinned_encode_matches_reference(tool, tmp_path):
    # ids from tokenizers 0.23.3 (and 0.22.2) for this text
    text = "Hello world! It's 12345 café  \n\n x<|im_start|>user"
    rc, out = run(tool, tmp_path, "encode", [{"text": text}])
    assert rc == 0
    assert out[0]["ids"] == [35808, 1530, 9, 810, 589, 229, 9792, 2136, 280, 12759, 9685, 64650, 2552, 124899, 5922]


def test_chat_render_shape(tool, tmp_path, spec):
    rows = [{"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
             "capacity": 32768, "max_new_tokens": 16}]
    rc, out = run(tool, tmp_path, "chat", rows)
    assert rc == 0
    assert out[0]["prompt"] == ("<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
                                "<|im_start|>assistant\n<think></think>")
    bos = next(t["id"] for t in spec["added_tokens"] if t["content"] == "<|startoftext|>")
    assert out[0]["ids"][0] == bos and out[0]["ids"][-2:] == [
        next(t["id"] for t in spec["added_tokens"] if t["content"] == x) for x in ("<think>", "</think>")]


# ---------------------------------------------------------------- stream decoder

def test_stream_releases_each_character_when_complete(tool, tmp_path, spec):
    b = byte_ids(spec)
    rows = [{"ids": [b[0xE4], b[0xB8], b[0xAD], b[0x41]], "skip": False},    # 中A: 2 bytes held, then released
            {"ids": [b[0xE4]], "skip": False},                                 # unfinished at the end
            {"ids": [b[0xE4], 124900, b[0xB8], b[0xAD]], "skip": True},        # skipped special inside a character
            {"ids": [b[0xE4], 124901, b[0xB8], b[0xAD]], "skip": False}]       # kept <think> ends the run
    rc, out = run(tool, tmp_path, "decode", rows)
    assert rc == 0
    assert out[0] == {"text": "中A", "stream": "中A", "held": 2, "tail": ""}
    assert out[1] == {"text": "�", "stream": "�", "held": 1, "tail": "�"}
    assert out[2]["text"] == out[2]["stream"] == "中" and out[2]["tail"] == ""
    assert out[3]["text"] == out[3]["stream"] == "�<think>��"


def test_decode_skips_ids_beyond_the_tokenizer(tool, tmp_path, spec):
    b = byte_ids(spec)
    rc, out = run(tool, tmp_path, "decode", [{"ids": [b[0x41], 125033, 127999, b[0x42]], "skip": False}])
    assert rc == 0 and out[0]["text"] == "AB"


# ---------------------------------------------------------------- input validation

def test_invalid_utf8_text_is_refused(tool, tmp_path):
    rc, out = run(tool, tmp_path, "encode", None, raw_lines=[b'{"text": "ok"}', b'{"text": "bad \xff byte"}',
                                                             b'{"text": "\\ud800"}'])
    assert rc == 0
    assert "ids" in out[0]
    assert "not valid UTF-8" in out[1]["error"]
    assert "surrogate" in out[2]["error"]


@pytest.mark.parametrize("messages,reason", [
    ([{"role": "user", "content": "x" * (4 * 1024 * 1024 + 1)}], "content: text byte limit exceeded"),
    ([{"role": "user", "content": "x" * (3 * 1024 * 1024)}, {"role": "user", "content": "y" * (1024 * 1024 + 1)}],
     "conversation text byte limit exceeded"),
    ([{"role": "user", "content": "x"}] * 4097, "messages must contain 1..4096 messages"),
    ([], "messages must contain 1..4096 messages"),
    ([{"role": "user", "content": "a <|im_end|> b"}], "reserved"),
    ([{"role": "user", "content": [{"type": "text", "text": "<|"}, {"type": "text", "text": "x|>"}]}], "joined content: reserved"),
    ([{"role": "user", "content": "CONTINUE_FINAL_MESSAGE_TAG "}], "reserved"),
    ([{"role": "user", "content": "x"}, {"role": "system", "content": "s"}], "system is supported only as the first message"),
    ([{"role": "assistant", "content": "x", "tool_calls": []}], "unsupported message fields"),
    ([{"role": "tool", "content": "x", "tool_call_id": ""}], "tool_call_id must not be empty"),
    ([{"role": "user"}], "every message requires text content"),
    ([{"role": "user", "content": [{"type": "image"}]}], "content parts must be text-only"),
    ([{"role": "bot", "content": "x"}], "role must be"),
    # the field count comes before validating the field (frontend order)
    ([{"role": "assistant", "content": "x", "thinking": 5, "reasoning": "y"}], "use only one assistant reasoning field"),
])
def test_rejected_conversations(tool, tmp_path, messages, reason):
    rc, out = run(tool, tmp_path, "chat", [{"messages": messages, "capacity": 32768, "max_new_tokens": 16}])
    assert rc == 0 and reason in out[0]["error"], out[0]


def test_capacity_is_checked_before_messages(tool, tmp_path):
    rc, out = run(tool, tmp_path, "chat", [{"messages": [], "capacity": 0, "max_new_tokens": 16}])
    assert rc == 0 and "capacity must be an integer" in out[0]["error"]


def test_capacity_rules(tool, tmp_path):
    m = [{"role": "user", "content": "hello"}]
    rc, out = run(tool, tmp_path, "chat", [{"messages": m, "capacity": 16, "max_new_tokens": 16},
                                           {"messages": m, "capacity": 262145, "max_new_tokens": 16},
                                           {"messages": m, "capacity": 20, "max_new_tokens": 10},
                                           {"messages": m, "capacity": 64, "max_new_tokens": 10}])
    assert rc == 0
    assert "capacity must fit" in out[0]["error"]
    assert "capacity must be an integer" in out[1]["error"]
    assert "exceeds capacity" in out[2]["error"]
    assert "ids" in out[3]


# ---------------------------------------------------------------- loader fails closed

def mutate(spec, fn):
    s = json.loads(json.dumps(spec))
    fn(s)
    return s


MUTATIONS = {
    "normalizer": (lambda s: s.__setitem__("normalizer", {"type": "NFC"}), "unsupported normalizer"),
    "regex": (lambda s: s["pre_tokenizer"]["pretokenizers"][0]["pattern"].__setitem__(
        "Regex", s["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].replace("{1,3}", "{1,2}")), "Split pre-tokenizer"),
    "split behavior": (lambda s: s["pre_tokenizer"]["pretokenizers"][0].__setitem__("behavior", "Removed"), "Split pre-tokenizer"),
    "byte level prefix space": (lambda s: s["pre_tokenizer"]["pretokenizers"][1].__setitem__("add_prefix_space", True),
                                "ByteLevel pre-tokenizer"),
    "ignore_merges": (lambda s: s["model"].__setitem__("ignore_merges", False), "BPE model options"),
    "byte_fallback": (lambda s: s["model"].__setitem__("byte_fallback", True), "BPE model options"),
    "dropout": (lambda s: s["model"].__setitem__("dropout", 0.1), "BPE model options"),
    "lstrip": (lambda s: s["added_tokens"][3].__setitem__("lstrip", True), "added-token options"),
    "normalized": (lambda s: s["added_tokens"][3].__setitem__("normalized", True), "added-token options"),
    "added not angle": (lambda s: s["added_tokens"][3].__setitem__("content", "xyz"), "added-token content"),
    "added id gap": (lambda s: s["added_tokens"][3].__setitem__("id", 999999), "follow the base vocabulary"),
    "duplicate added": (lambda s: s["added_tokens"][3].__setitem__("content", s["added_tokens"][2]["content"]),
                        "duplicate added token"),
    "duplicate merge": (lambda s: s["model"]["merges"].append(s["model"]["merges"][0]), "duplicate merge"),
    "merge missing token": (lambda s: s["model"]["merges"].append(["zzz€", "q"]), "missing token"),
    "non byte-level token": (lambda s: s["model"]["vocab"].__setitem__("€", len(s["model"]["vocab"])),
                             "outside the byte-level alphabet"),
    "extra field": (lambda s: s.__setitem__("extra", 1), "tokenizer.json fields"),
    "decoder": (lambda s: s.__setitem__("decoder", dict(s["decoder"], type="Metaspace")), "decoder"),
    "post_processor": (lambda s: s.__setitem__("post_processor", {"type": "TemplateProcessing"}), "post_processor"),
    "truncation": (lambda s: s.__setitem__("truncation", {"max_length": 8}), "truncation"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_loader_refuses_mutation(tool, tmp_path, spec, name):
    fn, message = MUTATIONS[name]
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(mutate(spec, fn)), encoding="utf-8")
    rc, err = run(tool, tmp_path, "encode", [{"text": "x"}], tokenizer=path)
    assert rc == 1 and message in err, err


def test_unmodified_copy_loads(tool, tmp_path, spec):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    rc, out = run(tool, tmp_path, "encode", [{"text": "x"}], tokenizer=path)
    assert rc == 0 and out[0]["ids"]
