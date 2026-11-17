"""Measure the character classes LFM3's pre-tokenizer regex actually uses, through the real tokenizer.

The Split pre-tokenizer runs on Oniguruma (tokenizers links onig_sys), so \\p{L}, \\p{N}, \\s and the case
folding inside (?i:'s|'t|'re|'ve|'m|'ll|'d) follow Oniguruma's Unicode tables, which need not match
Python's unicodedata. Instead of assuming a Unicode version, every code point is classified by how the
real pre-tokenizer splits template strings around it; the resulting signature groups are named by
known exemplars and must cover every code point.

Run in the reference container:  python probe_pretokenizer_classes.py --output classes.json
Output: per-class code-point ranges, contraction-fold characters, exemplar checks, and versions.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split
import tokenizers


def ranges(points):
    out, start, prev = [], None, None
    for p in sorted(points):
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            out.append([start, prev]); start = prev = p
    if start is not None:
        out.append([start, prev])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="/workspace/models/LFM3-Omni-2B/tokenizer.json")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        sys.exit(f"exists: {a.output}")
    spec = json.loads(Path(a.tokenizer).read_text())["pre_tokenizer"]
    first = spec["pretokenizers"][0]
    assert first["type"] == "Split" and first["behavior"] == "Isolated" and not first["invert"], first
    pattern = first["pattern"]["Regex"]
    # The regex stage alone, through the same library (and so the same Oniguruma engine). ByteLevel would
    # expand non-ASCII characters into 2-4 byte characters and make piece lengths depend on UTF-8 length.
    split = Split(Regex(pattern), behavior="isolated", invert=False)
    Tokenizer.from_file(a.tokenizer)  # the file loads; its pre_tokenizer is Sequence[this Split, ByteLevel]

    def pieces(s):
        return [p for p, _ in split.pre_tokenize_str(s)]

    # Template strings: letters/numbers/whitespace/other each produce a distinct split pattern; piece
    # lengths are in characters here.
    templates = ["a{c}a", "1{c}1", "{c}{c}a", " {c}", "{c}\n", "!{c}!"]

    def signature(ch):
        sig = []
        for t in templates:
            ps = pieces(t.replace("{c}", ch))
            sig.append(tuple(len(p) for p in ps))
        return tuple(sig)

    exemplars = {"letter": "a", "number": "1", "space": " ", "newline": "\n", "carriage_return": "\r", "other": "!"}
    exemplar_sig = {name: signature(ch) for name, ch in exemplars.items()}
    groups = collections.defaultdict(list)
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue  # surrogates are not valid in UTF-8 text
        groups[signature(chr(cp))].append(cp)
    named, unnamed = {}, {}
    for sig, cps in groups.items():
        names = [n for n, s in exemplar_sig.items() if s == sig]
        if names:
            named["/".join(names)] = cps
        else:
            unnamed[repr(sig)] = cps

    # Contractions: which characters, in place of a suffix letter, make "'"+suffix its own piece. A letter
    # follows, so a mere punctuation+letters run ("'sb" as one piece) is distinguished from a contraction.
    fold = {}
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"]:
        hits = []
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            c = chr(cp)
            # replace each letter of the suffix in turn by c; "x" + "'" + candidate + "b": a contraction stays
            # separate from the following letter (non-letter hits here are the punctuation-run alternative)
            for i in range(len(suffix)):
                cand = suffix[:i] + c + suffix[i + 1:]
                ps = pieces("x'" + cand + "b")
                if ps == ["x", "'" + cand, "b"]:
                    hits.append([i, cp])
        fold[suffix] = hits
    result = {
        "versions": {"tokenizers": tokenizers.__version__},
        "pattern": pattern,
        "templates": templates,
        "exemplar_signatures": {k: [list(x) for x in v] for k, v in exemplar_sig.items()},
        "classes": {k: {"count": len(v), "ranges": ranges(v)} for k, v in named.items()},
        "unclassified_signatures": {k: {"count": len(v), "ranges": ranges(v)[:50]} for k, v in unnamed.items()},
        "contraction_characters": fold,
        "total_code_points": sum(len(v) for v in groups.values()),
    }
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"classes": {k: len(v) for k, v in named.items()},
                      "unclassified_groups": {k: len(v) for k, v in unnamed.items()},
                      "total": result["total_code_points"],
                      "contraction_hits": {k: len(v) for k, v in fold.items()}}, indent=1))


if __name__ == "__main__":
    main()
