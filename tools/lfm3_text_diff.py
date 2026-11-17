"""Differential test of the native LFM3 text frontend (native/include/lfm/lfm3_text.hpp) against the
reference stack: Rust `tokenizers` (encode/decode) and Transformers' chat-template render, through
tools/lfm3_text_frontend.py exactly as text serving uses them.

  cases     --out DIR                write encode/decode/chat inputs (deterministic, seeded)
  reference --dir DIR                reference outputs (run in the reference container)
  compare   --dir DIR [--report F]   native outputs (lfm3-text-tool) vs reference; exit 1 on any mismatch

Case families. encode: every Unicode scalar value in six regex contexts; every boundary of the measured
class tables; WikiText-2 test (each line, and the whole file); random and adversarial text (all
whitespace kinds, CR/LF runs, contractions in every case incl. U+017F and U+212A, digits of many
scripts, combining marks, emoji/ZWJ, CJK, added tokens and fragments of them, long runs).
decode: random ids incl. special and unknown ids, raw-byte tokens (lossy UTF-8), byte runs split
by added tokens, and decode(encode(x)) for the encode cases. chat: the frontend parity cases and
random conversations (all roles, reasoning fields, </think> with Python whitespace, list content,
tool-call-id merging, <tool_response> passthrough, rejected inputs).
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("encode", "decode", "chat")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- case generation

def scalar_values():
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


def pools(classes):
    letters = [cp for a, b in classes["letter"]["ranges"] for cp in range(a, b + 1)]
    numbers = [cp for a, b in classes["number"]["ranges"] for cp in range(a, b + 1)]
    ws = [0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F,
          0x205F, 0x3000, 0x1C, 0x1D, 0x1E, 0x1F, 0x200B, 0xFEFF]
    added = ["<|im_start|>", "<|im_end|>", "<|startoftext|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>",
             "</tool_call>", "<|tool_call_start|>", "<|pad|>", "<|image|>", "<|audio_out|>", "<function=",
             "<|im_", "<|", "|>", "<", ">", "<<think>>", "<|im_start|", "</think", "<think><think>"]
    words = ["the", "The", "THE", "hello", "world", "naïve", "café", "Straße", "日本語", "中文", "한국어", "русский",
             "Ελληνικά", "العربية", "हिन्दी", "ไทย", "ελπίδα", "İstanbul", "ǅemal", "ﬁne", "x", "a"]
    sfx = ["s", "t", "re", "ve", "m", "ll", "d"]
    marks = ["́", "̀", "̈", "‍", "️", "⃣", "ः", "ि"]
    emoji = ["😀", "👩🏽‍💻", "🇺🇸", "👍🏿", "🏳️‍🌈", "1️⃣", "🫠"]
    digits = ["0123456789", "٠١٢٣٤٥٦٧٨٩", "０１２３", "²³¹", "½¾", "ⅫⅣ", "१२३", "𝟘𝟙𝟚", "1234567", "007"]
    punct = list("!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~") + ["¡", "¿", "—", "–", "…", "。", "、", "«", "»", "’", "‘", "“", "”"]
    return dict(letters=letters, numbers=numbers, ws=ws, added=added, words=words, sfx=sfx, marks=marks, emoji=emoji,
                digits=digits, punct=punct)


def random_text(rng, P, n_items):
    out = []
    for _ in range(n_items):
        k = rng.random()
        if k < 0.16: out.append(rng.choice(P["words"]))
        elif k < 0.30: out.append("".join(chr(rng.choice(P["ws"])) for _ in range(rng.randint(1, 4))))
        elif k < 0.38: out.append(rng.choice(["\n", "\r\n", "\r", "\n\n", " \n", "\n ", "\t\n", "\n\n\n", "\r\r\n"]))
        elif k < 0.46:
            s = rng.choice(P["sfx"])
            s = "".join(c.upper() if rng.random() < 0.5 else c for c in s)
            if rng.random() < 0.15: s = s.replace("s", "ſ").replace("k", "K")
            out.append(rng.choice(["'", "’", "'"]) + s + rng.choice(["", "", "x", " ", "'", "s"]))
        elif k < 0.54: out.append(rng.choice(P["digits"])[: rng.randint(1, 10)])
        elif k < 0.60: out.append(rng.choice(P["marks"]))
        elif k < 0.65: out.append(rng.choice(P["emoji"]))
        elif k < 0.73: out.append("".join(rng.choice(P["punct"]) for _ in range(rng.randint(1, 4))))
        elif k < 0.79: out.append(rng.choice(P["added"]))
        elif k < 0.85: out.append("".join(chr(rng.choice(P["letters"])) for _ in range(rng.randint(1, 6))))
        elif k < 0.88: out.append(chr(rng.choice(P["numbers"])))
        elif k < 0.94:
            cp = rng.randrange(0x110000)
            out.append(chr(cp) if not 0xD800 <= cp <= 0xDFFF else "?")
        elif k < 0.97: out.append(rng.choice(["a", " ", "1", "!", "\n", "ab", "é"]) * rng.randint(20, 300))
        else: out.append(chr(rng.randrange(0x80)))
    return "".join(out)


def cmd_cases(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    classes = json.loads((REPO / "validation/g2-classes-v1/classes.json").read_text())["classes"]
    P = pools(classes)
    rng = random.Random(20261106)
    enc = []
    # every scalar value, six regex contexts (prefix-letter, number adjacency, runs, space, contraction)
    for cp in scalar_values():
        c = chr(cp)
        enc.append({"text": f"a{c}a {c}1{c}{c}\n'{c}!{c}"})
    # class-table boundaries
    edges = set()
    for name in ("letter", "number"):
        for lo, hi in classes[name]["ranges"]:
            edges.update((lo - 1, lo, hi, hi + 1))
    for cp in sorted(edges):
        if 0 <= cp < 0x110000 and not 0xD800 <= cp <= 0xDFFF:
            c = chr(cp)
            enc += [{"text": t} for t in (f" {c}{c}x", f"{c}{c}{c}{c}", f"1{c}2", f"'{c}d", f"\t{c}\n\n{c}")]
    # WikiText-2 test, each line and the whole file
    wt = (REPO.parent / "data/wikitext-2-raw-v1/test.txt").read_text(encoding="utf-8")
    enc += [{"text": line} for line in wt.split("\n")]
    enc.append({"text": wt})
    # random/adversarial
    for i in range(30000):
        enc.append({"text": random_text(rng, P, rng.randint(1, 40))})
    for i in range(200):
        enc.append({"text": random_text(rng, P, rng.randint(200, 2000))})
    enc += [{"text": t} for t in ["", " ", "\n", "  ", " \n ", "'", "''", "'s", "'S", "'ſ", "'K", "1234", "a" * 5000,
                                  " " * 1000 + "x", "\n" * 1000, "\r\n" * 500, "<|im_start|>" * 100, "<" * 300]]
    write_jsonl(out / "encode.jsonl", enc)

    dec = []
    live, model_rows = 125033, 128000
    for i in range(6000):
        n = rng.randint(1, 64)
        ids = [rng.randrange(live) if rng.random() < 0.9 else rng.randrange(124893, live) for _ in range(n)]
        dec.append({"ids": ids, "skip": i % 2 == 0})
    # raw-byte tokens: ids of single bytes are found by the reference; listed by byte here
    for i in range(6000):
        n = rng.randint(1, 24)
        seq = [{"byte": rng.choice([rng.randrange(256), rng.randrange(0x80, 0x100), rng.choice([0xE4, 0xF0, 0xED, 0xE0, 0xF4, 0xC2, 0xB8, 0x80, 0xBF, 0x9F, 0x8F])])}
               if rng.random() < 0.8 else {"id": rng.choice([124893, 124899, 124900, 124901, 124902, 125014, 125017, 125032])}
               for _ in range(n)]
        dec.append({"seq": seq, "skip": i % 2 == 0})
    # ids outside the tokenizer (model rows 125033..127999): the reference skips or rejects them
    for i in range(200):
        dec.append({"ids": [rng.randrange(live), rng.randrange(live, model_rows), rng.randrange(live)], "skip": i % 2 == 0})
    for e in rng.sample(range(len(enc)), 4000):
        dec.append({"roundtrip": e, "skip": False})
    write_jsonl(out / "decode.jsonl", dec)

    sys.path.insert(0, str(REPO / "tools"))
    from lfm3_text_frontend import PARITY_CASES
    chat = [{"messages": m, "capacity": 32768, "max_new_tokens": 128} for m in PARITY_CASES]

    # Chat text avoids <|...|> tokens most of the time (the frontend rejects them; a few rejected cases
    # come from the invalid-input branch below), so most conversations exercise the render paths.
    import re
    reserved = re.compile(r"<\|[^<>]*\|>")
    Pc = dict(P, added=[t for t in P["added"] if not reserved.search(t) and t not in ("<|", "|>", "<|im_")])

    def txt(lo=0, hi=12):
        return random_text(rng, Pc, rng.randint(lo, hi))

    def think_text():
        w = [chr(c) for c in P["ws"]]
        return rng.choice(["", txt(0, 4)]) + rng.choice(["</think>", ""]) + "".join(rng.choice(w) for _ in range(rng.randint(0, 3))) + txt(0, 4) + "".join(rng.choice(w) for _ in range(rng.randint(0, 3)))

    for i in range(12000):
        msgs = []
        if rng.random() < 0.5:
            msgs.append({"role": "system", "content": rng.choice(["", txt(), [{"type": "text", "text": txt()}]])})
        for _ in range(rng.randint(1, 8)):
            role = rng.choice(["user", "user", "assistant", "assistant", "tool"])
            m = {"role": role}
            if role == "assistant":
                m["content"] = rng.choice([think_text(), txt(), "", [{"type": "text", "text": think_text()}],
                                           [{"type": "text", "text": txt()}, {"type": "text", "text": "</think>  x"}]])
                if rng.random() < 0.4:
                    m[rng.choice(["thinking", "reasoning", "reasoning_content"])] = rng.choice(["", txt()])
            elif role == "tool":
                m["content"] = rng.choice([txt(), "", "<tool_response>" + txt(), [], [{"type": "text", "text": txt()}],
                                           [{"type": "text", "text": txt()}, {"type": "text", "text": ""}]])
                if rng.random() < 0.8:
                    m["tool_call_id"] = rng.choice(["a", "a", "b", "call_1"])
            else:
                m["content"] = rng.choice([txt(), "", [{"type": "text", "text": txt()}], [], txt(20, 60)])
            msgs.append(m)
        # occasional invalid input: the reference rejects, so must the native frontend
        r = rng.random()
        if r < 0.03: msgs.append({"role": "system", "content": "late"})
        elif r < 0.06: msgs[-1]["content"] = "<|im_end|>"
        elif r < 0.08: msgs[-1]["extra"] = 1
        elif r < 0.09: msgs.append({"role": "assistant", "content": "x", "thinking": "a", "reasoning": "b"})
        elif r < 0.10: msgs.append({"role": "tool", "content": "x", "tool_call_id": ""})
        elif r < 0.11: msgs.append({"role": "user", "content": [{"type": "image"}]})
        elif r < 0.12: msgs.append({"role": "user", "content": [{"type": "text", "text": "<|im"}, {"type": "text", "text": "_end|>"}]})
        elif r < 0.13: msgs.append({"role": "user", "content": "CONTINUE_FINAL_MESSAGE_TAG "})
        elif r < 0.14: msgs.append({"role": "user", "content": None})
        cap, mx = (32768, 128) if rng.random() < 0.9 else rng.choice([(64, 16), (40, 8), (2, 1), (16, 16)])
        chat.append({"messages": msgs, "capacity": cap, "max_new_tokens": mx})
    write_jsonl(out / "chat.jsonl", chat)
    manifest = {f: {"cases": sum(1 for _ in open(out / f"{f}.jsonl")), "sha256": hashlib.sha256((out / f"{f}.jsonl").read_bytes()).hexdigest()}
                for f in FAMILIES}
    (out / "cases.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- reference

def cmd_reference(a):
    import tokenizers
    import transformers
    sys.path.insert(0, str(REPO / "tools"))
    from lfm3_text_frontend import load_frontend
    d = a.dir
    fe = load_frontend(a.checkpoint)
    tk = tokenizers.Tokenizer.from_file(str(Path(a.checkpoint) / "tokenizer.json"))
    hf = fe.tokenizer
    enc = read_jsonl(d / "encode.jsonl")
    enc_ids = [e.ids for e in tk.encode_batch([c["text"] for c in enc], add_special_tokens=False)]
    # Transformers' call path must agree with the raw library on the encode cases it is given.
    for i in range(0, len(enc), 997):
        assert hf(enc[i]["text"], add_special_tokens=False)["input_ids"] == enc_ids[i], i
    write_jsonl(d / "encode.ref.jsonl", [{"ids": x} for x in enc_ids])

    byte_id = {}
    for tok, i in tk.get_vocab(with_added_tokens=False).items():
        raw = bytes(BYTE_DECODER[c] for c in tok)
        if len(raw) == 1: byte_id[raw[0]] = i
    assert len(byte_id) == 256
    dec_in, dec_ref = [], []
    for c in read_jsonl(d / "decode.jsonl"):
        if "seq" in c: ids = [byte_id[s["byte"]] if "byte" in s else s["id"] for s in c["seq"]]
        elif "roundtrip" in c: ids = enc_ids[c["roundtrip"]]
        else: ids = c["ids"]
        dec_in.append({"ids": ids, "skip": c["skip"]})
        try:
            text = hf.decode(ids, skip_special_tokens=c["skip"], clean_up_tokenization_spaces=False)
            raw = tk.decode(ids, skip_special_tokens=c["skip"])
            r = {"text": text}
            if raw != text: r["raw_tokenizers_differs"] = raw
            if "roundtrip" in c: r["roundtrip_equal"] = text == enc[c["roundtrip"]]["text"]
        except Exception as e:  # noqa: BLE001 - the reference's own rejection is the expected behaviour
            r = {"error": f"{type(e).__name__}: {e}"}
        dec_ref.append(r)
    write_jsonl(d / "decode.in.jsonl", dec_in)
    write_jsonl(d / "decode.ref.jsonl", dec_ref)

    chat_ref = []
    for c in read_jsonl(d / "chat.jsonl"):
        try:
            # encode_messages first: its capacity checks precede message validation, as in lfm3-text-tool
            ids = fe.encode_messages(c["messages"], c["capacity"], c["max_new_tokens"])
            prompt = fe.render_messages(c["messages"])
            r = {"prompt": prompt, "ids": ids}
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"}
        chat_ref.append(r)
    write_jsonl(d / "chat.ref.jsonl", chat_ref)
    versions = {"tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
                "python": sys.version.split()[0]}
    import jinja2
    versions["jinja2"] = jinja2.__version__
    (d / "reference.json").write_text(json.dumps({"versions": versions, "files": {
        f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in
        ("encode.ref.jsonl", "decode.in.jsonl", "decode.ref.jsonl", "chat.ref.jsonl")}}, indent=1) + "\n")
    print(json.dumps(versions))


def _byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


# ---------------------------------------------------------------- compare

PY_ONLY_SPACE = "\x1c\x1d\x1e\x1f"  # stripped by Python's str.strip (Jinja |trim), not \\s for the tokenizer


def chat_coverage(cases, ref):
    """How often the accepted chat cases reach each template path (a zero means a path is untested)."""
    c = dict(accepted=0, system=0, empty_system=0, list_content=0, prior_assistant_think_split=0,
             think_split_strip_py_only_space=0, reasoning_kept=0, reasoning_dropped=0, tool_merge=0,
             tool_passthrough=0, tool_list=0, tool_empty=0, rejected_capacity=0)
    for case, r in zip(cases, ref):
        if "error" in r:
            c["rejected_capacity"] += "capacity" in r["error"]
            continue
        c["accepted"] += 1
        ms = case["messages"]
        if ms[0]["role"] == "system":
            c["system"] += 1
            c["empty_system"] += not ms[0]["content"]
            ms = ms[1:]
        last_user = max((i for i, m in enumerate(ms) if m["role"] == "user"), default=-1)
        for i, m in enumerate(ms):
            content = m["content"]
            text = content if isinstance(content, str) else "".join(p["text"] for p in content)
            c["list_content"] += isinstance(content, list)
            if m["role"] == "assistant":
                has_r = any(k in m for k in ("thinking", "reasoning", "reasoning_content"))
                if i < last_user:
                    c["reasoning_dropped"] += has_r
                    if "</think>" in text:
                        c["prior_assistant_think_split"] += 1
                        tail = text.split("</think>")[-1]
                        c["think_split_strip_py_only_space"] += any(ch in PY_ONLY_SPACE for ch in tail[:1] + tail[-1:])
                else:
                    c["reasoning_kept"] += has_r
            if m["role"] == "tool":
                nxt = ms[i + 1] if i + 1 < len(ms) else None
                c["tool_merge"] += bool(nxt and nxt["role"] == "tool" and m.get("tool_call_id") and nxt.get("tool_call_id") == m.get("tool_call_id"))
                c["tool_passthrough"] += isinstance(content, str) and "<tool_response>" in content
                c["tool_list"] += isinstance(content, list) and bool(content)
                c["tool_empty"] += not content
    return c


def cmd_compare(a):
    d = a.dir
    report = {"families": {}, "pass": True}
    for fam, native_name, ref_name in (("encode", "encode.native.jsonl", "encode.ref.jsonl"),
                                       ("decode", "decode.native.jsonl", "decode.ref.jsonl"),
                                       ("chat", "chat.native.jsonl", "chat.ref.jsonl")):
        nat, ref = read_jsonl(d / native_name), read_jsonl(d / ref_name)
        if len(nat) != len(ref):
            raise SystemExit(f"{fam}: {len(nat)} native rows vs {len(ref)} reference rows")
        bad, both_err, notes, stream_held = [], 0, {}, 0
        for i, (n, r) in enumerate(zip(nat, ref)):
            if "error" in r or "error" in n:
                if "error" in r and "error" in n:
                    both_err += 1
                    # same reason: native "lfm3 chat: why" vs reference "ValueError: why"
                    if fam == "chat" and n["error"].removeprefix("lfm3 chat: ") != r["error"].split(": ", 1)[-1]:
                        bad.append({"case": i, "native": n["error"], "reference": r["error"]})
                else: bad.append({"case": i, "native": n, "reference": r})
                continue
            if fam == "encode" and n["ids"] != r["ids"]: bad.append({"case": i, "native": n["ids"][:50], "reference": r["ids"][:50]})
            if fam == "decode":
                if n["text"] != r["text"]: bad.append({"case": i, "native": n["text"][:200], "reference": r["text"][:200]})
                if n["stream"] != n["text"]: bad.append({"case": i, "stream_differs": True})
                # streaming must be prompt: never more than an unfinished character held back, and
                # finish() may release only that character's replacement
                if n["held"] > 3 or n["tail"].strip("\ufffd"): bad.append({"case": i, "stream_held": n["held"], "tail": n["tail"]})
                stream_held = max(stream_held, n["held"])
                if "raw_tokenizers_differs" in r: notes["raw_tokenizers_differs"] = notes.get("raw_tokenizers_differs", 0) + 1
                if r.get("roundtrip_equal") is False: notes["roundtrip_not_identity"] = notes.get("roundtrip_not_identity", 0) + 1
            if fam == "chat" and (n["prompt"] != r["prompt"] or n["ids"] != r["ids"]):
                bad.append({"case": i, "native": n["prompt"][-300:], "reference": r["prompt"][-300:]})
        if fam == "decode": notes["stream_max_held_bytes"] = stream_held
        if fam == "chat": notes["coverage"] = chat_coverage(read_jsonl(d / "chat.jsonl"), ref)
        report["families"][fam] = {"cases": len(ref), "mismatches": len(bad), "both_rejected": both_err,
                                   "notes": notes, "first_mismatches": bad[:8]}
        report["pass"] &= not bad
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.report: Path(a.report).write_text(text + "\n")
    print(json.dumps({k: {x: v[x] for x in ("cases", "mismatches", "both_rejected", "notes")} for k, v in report["families"].items()}, indent=1))
    for fam, v in report["families"].items():
        for m in v["first_mismatches"][:3]:
            print(fam, json.dumps(m, ensure_ascii=False)[:600])
    print("PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases"); c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("reference"); r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--checkpoint", default="/workspace/models/LFM3-Omni-2B")
    m = sub.add_parser("compare"); m.add_argument("--dir", type=Path, required=True); m.add_argument("--report")
    a = ap.parse_args()
    sys.exit({"cases": cmd_cases, "reference": cmd_reference, "compare": cmd_compare}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
