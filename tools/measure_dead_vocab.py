"""Measure how LFM3-Omni-2B treats output ids that have no token (G6), with the HF SDPA reference.

The tokenizer defines ids 0..N-1 (N = 125033, contiguous); the tied output layer scores 128000. On the
perplexity windows (teacher forced), per position: probability mass on dead ids, whether the
argmax is dead, the best dead id's rank, and the NLL change if dead ids were masked
(log-softmax over live ids only). Read-only; writes one JSON summary.

Usage (reference container, Thor ptxas env):
  python measure_dead_vocab.py --windows /workspace/liquid-CUDA/build/ppl-v1/windows-4096.i32 --length 4096 \\
      --output /workspace/liquid-CUDA/build/dead-vocab-v1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.cache_utils import DynamicCache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--model", default="/workspace/models/LFM3-Omni-2B")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"exists: {a.output}")
    tok = AutoProcessor.from_pretrained(a.model, trust_remote_code=True).tokenizer
    live = len(tok)  # base vocab + added tokens
    windows = np.fromfile(a.windows, dtype=np.int32).reshape(-1, a.length)
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    lm, cfg = model.model.language_model, model.config.text_config
    vocab = cfg.vocab_size
    positions = argmax_dead = 0
    dead_mass_sum = dead_mass_max = 0.0
    nll_delta_sum = 0.0
    best_dead_rank_min = vocab
    rank_hist = {"top1": 0, "top10": 0, "top100": 0, "top1000": 0}
    with torch.inference_mode():
        for row in windows:
            ids = torch.from_numpy(row.astype(np.int64))[None].cuda()
            n = ids.shape[1]
            h = lm(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(n, device="cuda")[None],
                   past_key_values=DynamicCache(config=cfg), use_cache=True).last_hidden_state[0]
            targets = ids[0, 1:]
            for r0 in range(0, n - 1, 256):
                r1 = min(n - 1, r0 + 256)
                logits = model.lm_head(h[r0:r1]).float()
                lse_all = torch.logsumexp(logits, -1)
                lse_live = torch.logsumexp(logits[:, :live], -1)
                dead_mass = 1.0 - torch.exp(lse_live - lse_all)
                best_dead = logits[:, live:].max(-1).values
                rank = (logits > best_dead[:, None]).sum(-1)  # 0 = a dead id is the argmax
                positions += r1 - r0
                argmax_dead += int((rank == 0).sum())
                dead_mass_sum += float(dead_mass.sum())
                dead_mass_max = max(dead_mass_max, float(dead_mass.max()))
                best_dead_rank_min = min(best_dead_rank_min, int(rank.min()))
                for k, lim in (("top1", 1), ("top10", 10), ("top100", 100), ("top1000", 1000)):
                    rank_hist[k] += int((rank < lim).sum())
                # Masking dead ids changes NLL by (lse_live - lse_all) <= 0 at every position (targets are live).
                nll_delta_sum += float((lse_live - lse_all).sum())
                assert int(targets[r0:r1].max()) < live
    result = {"windows": str(a.windows), "length": a.length, "positions": positions, "live_ids": live, "model_vocab": vocab,
              "dead_ids": vocab - live, "argmax_is_dead": argmax_dead, "best_dead_rank_min": best_dead_rank_min,
              "positions_with_dead_in_topk": rank_hist, "mean_dead_probability_mass": dead_mass_sum / positions,
              "max_dead_probability_mass": dead_mass_max,
              "mean_nll_change_if_masked": nll_delta_sum / positions,
              "note": "NLL change is exact per position: -log softmax_live(target) - (-log softmax_all(target)) = lse_live - lse_all"}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
