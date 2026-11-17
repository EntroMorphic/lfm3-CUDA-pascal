#!/usr/bin/env python3
"""Real-text regression gate for the speech normalizer (validation/audio-assurance-v1/normalizer-audit.md, red-team
round 2). The corpora:
- every clean line of the local wikitext-2 and PG-19 test data that tests/test_speech_normalize.py reads;
- every sentence holding a digit in the local ultrachat test split (data/ultrachat/test_sft-*.parquet), split at
  ". X";
- red-team round 3's 304k sentences (when the Hugging Face cache holds the datasets): ag_news, gsm8k, amazon_polarity,
  imdb, one wikipedia shard, scifact, each split at sentence ends, kept when they hold a digit or a symbol the rules
  read, deduplicated, shuffled with a fixed seed and capped (the auditor's build.py, reproduced in hf_corpora()).

Each sentence is read by tools/speech_normalize.py and by the baseline reference at BASELINE (the commit before the
audit, read with git show). The reviewed record (validation/audio-assurance-v1/normalizer-gate.jsonl.gz) holds every
sentence whose reading changed from the baseline, with the reading that was reviewed and its verdict.

The gate fails, listing the sentences, when:
- a sentence changed from the baseline but is not in the record (a new change: review it);
- a recorded sentence now reads differently from its reviewed reading.

Usage:
  normalizer_gate.py            check against the record
  normalizer_gate.py --write    rewrite the record from the current reference (after a review; verdicts are kept for
                                sentences whose reading did not change, and new ones are marked "unreviewed")
"""
import gzip
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "fd349b5b7c9692c25d69418641810a017c3fbc94"
RECORD = ROOT / "validation/audio-assurance-v1/normalizer-gate.jsonl.gz"


def load_current():
    spec = importlib.util.spec_from_file_location("speech_normalize", ROOT / "tools/speech_normalize.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_baseline():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE}:tools/speech_normalize.py"], check=True,
                         capture_output=True, text=True).stdout
    m = types.ModuleType("speech_normalize_baseline")
    exec(compile(src, "speech_normalize_baseline", "exec"), m.__dict__)
    return m


def corpora():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_speech_normalize as t
    out = [("wiki-pg19", line) for line in t.corpus_lines() if not re.search(r"[\x00-\x08\x0e-\x1f�]", line)]
    files = sorted((ROOT.parent / "data/ultrachat").glob("test_sft-*.parquet"))
    if files:
        import pyarrow.parquet as pq
        sents = []
        for f in files:
            for msgs in pq.read_table(f, columns=["messages"]).column("messages").to_pylist():
                for m in msgs:
                    for line in m["content"].splitlines():
                        for s in re.split(r"(?<=[.!?]) (?=[A-Z])", line):
                            if re.search(r"\d", s) and len(s) < 1500:
                                sents.append(s.strip())
        out += [("ultrachat", s) for s in dict.fromkeys(sents)]
    return out


def hf_corpora():
    """Red-team round 3's corpus, as its build.py made it (same splitting, filtering, seed and caps)."""
    import glob
    import random
    import pyarrow.parquet as pq
    hub = Path.home() / ".cache/huggingface/hub"

    def snap(d):
        found = sorted(glob.glob(str(hub / d / "snapshots") + "/*/"))
        return found[0] if found else None
    out = []

    def add(name, texts, cap):
        got = []
        for t in texts:
            if not t:
                continue
            t = t.replace("\\$", "$")
            for line in t.splitlines():
                for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', line):
                    s = s.strip()
                    if 3 < len(s) < 1200 and re.search(r"[\d$%+^×£€§#@&]", s):
                        got.append(s)
        got = list(dict.fromkeys(got))
        random.Random(1).shuffle(got)
        out.extend((name, s) for s in got[:cap])
    ag, g, a = snap("datasets--fancyzhx--ag_news"), snap("datasets--openai--gsm8k"), snap("datasets--fancyzhx--amazon_polarity")
    i, w, sc = snap("datasets--stanfordnlp--imdb"), snap("datasets--wikimedia--wikipedia"), snap("datasets--BeIR--scifact")
    if not all((ag, g, a, i, w, sc)):
        return []
    add("agnews", [r for f in glob.glob(ag + "data/*.parquet") for r in pq.read_table(f, columns=["text"]).column("text").to_pylist()], 120000)
    add("gsm8k", [r for f in glob.glob(g + "main/*.parquet") for c in ("question", "answer")
                  for r in pq.read_table(f, columns=[c]).column(c).to_pylist()], 40000)
    add("amazon", pq.read_table(a + "amazon_polarity/test-00000-of-00001.parquet", columns=["content"]).column("content").to_pylist(), 60000)
    add("imdb", pq.read_table(i + "plain_text/test-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist(), 20000)
    f = sorted(glob.glob(w + "20231101.en/*.parquet"))[5]
    texts = []
    for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text"]):
        texts += b.column("text").to_pylist()
        if len(texts) >= 12000:
            break
    add("wikipedia", texts, 80000)
    add("scifact", pq.read_table(sc + "corpus/corpus-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist(), 20000)
    return out


def main(write=False):
    current, baseline = load_current(), load_baseline()
    record = {}
    if RECORD.exists():
        with gzip.open(RECORD, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                record[r["text"]] = r
    rows, new_change, differs, ok = [], [], [], 0
    sources = {}
    seen = set()
    for src, s in corpora() + hf_corpora():
        if s in seen:
            continue
        seen.add(s)
        sources[src] = sources.get(src, 0) + 1
        b, c = baseline.normalize(s), current.normalize(s)
        if b == c:
            continue
        r = record.get(s)
        if r is None:
            new_change.append((s, b, c))
            rows.append(dict(source=src, text=s, baseline=b, reviewed=c, verdict="unreviewed"))
        elif r["reviewed"] != c:
            differs.append((s, r["reviewed"], c))
            rows.append(dict(source=src, text=s, baseline=b, reviewed=c, verdict="unreviewed"))
        else:
            ok += 1
            rows.append(r)
    print(json.dumps(dict(sentences=sources, changed_from_baseline=len(rows), matching_review=ok,
                          new_changes=len(new_change), differ_from_review=len(differs))))
    if write:
        with gzip.open(RECORD, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", RECORD)
        return 0
    for s, b, c in new_change[:50]:
        print(f"NEW CHANGE  {s!r}\n  baseline: {b!r}\n  current:  {c!r}")
    for s, w, c in differs[:50]:
        print(f"DIFFERS     {s!r}\n  reviewed: {w!r}\n  current:  {c!r}")
    return 1 if new_change or differs else 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv[1:]))
