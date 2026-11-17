"""Rebuild the perplexity windows on the workstation and accept them only if they are Thor's.

The rule is tools/ppl_reference.py's ([BOS] + next L-1 corpus tokens, non-overlapping), tokenized with the
checkpoint's tokenizer.json through the `tokenizers` library. Each windows-L.i32 is written only if its SHA-256 equals
the one in the Thor round's summary.json (which also binds the corpus hash, token count and BOS id).

Usage: python3 tools/pascal/make_windows.py CHECKPOINT CORPUS.txt THOR_SUMMARY.json OUTPUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    checkpoint, corpus, summary_path, out = (Path(x) for x in sys.argv[1:5])
    summary = json.loads(summary_path.read_text())
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != summary["corpus_sha256"]:
        raise SystemExit("corpus sha256 differs from the Thor summary")
    tok = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    bos = tok.token_to_id("<|startoftext|>")
    tokens = tok.encode(corpus.read_text(encoding="utf-8"), add_special_tokens=False).ids
    if bos != summary["bos_id"] or len(tokens) != summary["corpus_tokens"] or bos in tokens:
        raise SystemExit(f"tokenization differs: bos {bos}, {len(tokens)} tokens")
    out.mkdir(parents=True, exist_ok=True)
    for length, info in summary["windows"].items():
        step = int(length) - 1
        rows = np.asarray([[bos] + tokens[i * step:(i + 1) * step] for i in range(len(tokens) // step)], dtype=np.int32)
        digest = hashlib.sha256(rows.tobytes()).hexdigest()
        if digest != info["sha256"] or len(rows) != info["count"]:
            raise SystemExit(f"windows-{length}: {len(rows)} windows sha256 {digest}, Thor has {info['count']} {info['sha256']}")
        (out / info["file"]).write_bytes(rows.tobytes())
        print(f"{info['file']}: {len(rows)} windows, sha256 {digest} = Thor's")
    (out / "windows.json").write_text(json.dumps(dict(
        source=str(summary_path), corpus_sha256=summary["corpus_sha256"], corpus_tokens=len(tokens), bos_id=bos,
        window_rule=summary["window_rule"], windows=summary["windows"]), indent=1) + "\n")


if __name__ == "__main__":
    main()
