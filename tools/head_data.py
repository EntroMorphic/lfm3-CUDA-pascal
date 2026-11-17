#!/usr/bin/env python3
"""Draft-head data, step 1: chat prompts for self-distillation (journal/lfm3-single-stream_synth.md, decision 4).

From UltraChat 200k (MIT; data/ultrachat, pinned; *.first-user.jsonl from the parquet): the first user turn of each conversation, shuffled with a fixed
seed, 20-2,000 characters, at most 512 tokens after the model's chat template. Writes lfm3-batch requests:
  train-greedy.json, train-sampled.json (T 0.528): TRAIN prompts each, from train_sft (the training data);
  test.json: TEST prompts from test_sft (held out, for acceptance measurements).
Usage (lfm3-omni-dev): head_data.py ULTRACHAT_DIR MODEL_DIR OUT_DIR [TRAIN TEST]
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def prompts(path, tok, count, seed):
    rows = [json.loads(line)["user"] for line in open(path)]  # *.first-user.jsonl, converted from the parquet
    random.Random(seed).shuffle(rows)
    out = []
    for text in rows:
        if not text or not 20 <= len(text) <= 2000:
            continue
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) > 512:
            continue
        out.append(list(map(int, ids)))
        if len(out) == count:
            break
    return out


def main(data, model, out, train=12000, test=256):
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    d = Path(data)
    tr = prompts(next(d.glob("train_sft-*.first-user.jsonl")), tok, int(train), 0)
    te = prompts(next(d.glob("test_sft-*.first-user.jsonl")), tok, int(test), 1)
    half = len(tr) // 2
    common = dict(capacity=2048, batch=64, max_new_tokens=768, all_prompts=True)
    json.dump(dict(common, prompts=tr[:half], temperature=0, seed=0), open(out / "train-greedy.json", "w"))
    json.dump(dict(common, prompts=tr[half:], temperature=0.528, seed=1), open(out / "train-sampled.json", "w"))
    json.dump(dict(common, prompts=te, temperature=0, seed=2), open(out / "test.json", "w"))
    print(f"train {len(tr)} prompts, test {len(te)}; mean prompt tokens {sum(map(len, tr)) / len(tr):.0f}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        sys.exit(__doc__)
    main(*sys.argv[1:])
