#!/usr/bin/env python3
"""Draft head for tree speculation (journal/lfm3-single-stream_synth.md, decision 4): an EAGLE-style head trained on
the frozen LFM3-Omni-2B's own features and outputs.

Head, at position t: x = fc([embed(token[t+1]); f_t]), where f_t is the target's final normalized hidden state (what
its LM head reads), then one decoder layer (RMSNorm, causal GQA attention 16/4 x 128 with RoPE, RMSNorm, SwiGLU of
width 4,096), a final RMSNorm, and a head over a draft vocabulary (the 32,768 most frequent output tokens),
initialised from the target's LM head rows. It predicts f_{t+1} and token[t+2].
Loss over response positions: smooth-L1(g_t, f_{t+1}) + 0.1 x CE(head(g_t), token[t+2]), plus the same for a second,
unrolled pass whose feature inputs are the first pass's predictions (shifted by one): the head learns from its own
errors as it will see them while drafting.
Data: lfm3-batch outputs of the self-distillation run (tools/head_data.py): prompt ids + generated ids.
Usage (lfm3-omni-dev): head_train.py MODEL_DIR DATA_DIR OUT_DIR [--epochs 2] [--tokens-per-batch 16384]
Writes OUT_DIR/head.pt (state dict, config, draft vocabulary), OUT_DIR/train.jsonl (losses).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

H, HEADS, KV, D, MLP, DRAFT_VOCAB = 2048, 16, 4, 128, 4096, 32768


def rms(x, w, eps=1e-5):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rotary(x, pos, theta=1e6):  # x [b, heads, t, d], full rotary, rotate-half pairs
    inv = 1.0 / theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos[:, None, :, None].float() * inv
    c, s = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


class Head(nn.Module):
    def __init__(self, lm_head_rows):
        super().__init__()
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.n1, self.n2, self.nf = (nn.Parameter(torch.ones(H)) for _ in range(3))
        self.q = nn.Linear(H, HEADS * D, bias=False)
        self.k = nn.Linear(H, KV * D, bias=False)
        self.v = nn.Linear(H, KV * D, bias=False)
        self.o = nn.Linear(HEADS * D, H, bias=False)
        self.w1 = nn.Linear(H, MLP, bias=False)
        self.w3 = nn.Linear(H, MLP, bias=False)
        self.w2 = nn.Linear(MLP, H, bias=False)
        self.head = nn.Linear(H, DRAFT_VOCAB, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(lm_head_rows)

    def forward(self, f, e, pos, mask):
        """f, e [b, t, H]; pos [b, t]; mask [b, t] (True = real). Returns the predicted features g [b, t, H]."""
        x = self.fc(torch.cat([e, f], -1))
        h = rms(x, self.n1)
        b, t, _ = h.shape
        q = rotary(self.q(h).view(b, t, HEADS, D).transpose(1, 2), pos)
        k = rotary(self.k(h).view(b, t, KV, D).transpose(1, 2), pos)
        v = self.v(h).view(b, t, KV, D).transpose(1, 2)
        k, v = (z.repeat_interleave(HEADS // KV, 1) for z in (k, v))
        causal = torch.ones(t, t, dtype=torch.bool, device=h.device).tril()
        attn_mask = causal[None, None] & mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, HEADS * D))
        h = rms(x, self.n2)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x

    def logits(self, g):
        return self.head(rms(g, self.nf))


def load_data(data_dir, names=("train-greedy", "train-sampled")):
    seqs = []
    for name in names:
        req = json.load(open(Path(data_dir) / f"{name}.json"))
        out = json.load(open(Path(data_dir) / f"{name}.out.json"))
        for prompt, row in zip(req["prompts"], out["rows"]):
            seqs.append((prompt, row["generated_ids"]))
    return seqs


def batches(seqs, tokens_per_batch, seed):
    order = list(range(len(seqs)))
    random.Random(seed).shuffle(order)
    batch, longest = [], 0
    for i in order:
        n = len(seqs[i][0]) + len(seqs[i][1])
        if batch and max(longest, n) * (len(batch) + 1) > tokens_per_batch:
            yield batch
            batch, longest = [], 0
        batch.append(i); longest = max(longest, n)
    if batch:
        yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("data"); p.add_argument("out")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--tokens-per-batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seqs", type=int, default=0)
    p.add_argument("--data-names", default="train-greedy,train-sampled", help="lfm3-batch request/output stems")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForMultimodalLM
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    for q in model.parameters():
        q.requires_grad_(False)
    lm, text_config = model.model.language_model, model.config.text_config
    embed = model.lm_head.weight  # tied: the embedding table
    seqs = load_data(a.data, a.data_names.split(","))
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    counts = torch.zeros(text_config.vocab_size, dtype=torch.long)
    for prompt, gen in seqs:
        counts += torch.bincount(torch.tensor(gen), minlength=text_config.vocab_size)
    d2t = counts.argsort(descending=True)[:DRAFT_VOCAB].sort().values  # draft index -> token id
    t2d = torch.full((text_config.vocab_size,), -1, dtype=torch.long)
    t2d[d2t] = torch.arange(DRAFT_VOCAB)
    coverage = float(counts[d2t].sum() / counts.sum())
    head = Head(embed[d2t.cuda()].float()).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_per_epoch = sum(1 for _ in batches(seqs, a.tokens_per_batch, 0))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    t2d_gpu = t2d.cuda()
    log = open(out / "train.jsonl", "w")
    print(f"{len(seqs)} sequences, draft vocabulary covers {coverage:.4f} of output tokens, {total} steps", flush=True)
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        for idx in batches(seqs, a.tokens_per_batch, epoch):
            L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
            ids = torch.zeros(len(idx), L, dtype=torch.long)
            mask = torch.zeros(len(idx), L, dtype=torch.bool)
            resp = torch.zeros(len(idx), L, dtype=torch.bool)  # positions t whose target token[t+2] is a response token
            for r, i in enumerate(idx):
                pr, gen = seqs[i]
                s = pr + gen
                ids[r, : len(s)] = torch.tensor(s)
                mask[r, : len(s)] = True
                resp[r, max(0, len(pr) - 2): len(s) - 2] = True
            ids, mask, resp = ids.cuda(), mask.cuda(), resp.cuda()
            pos = torch.arange(L, device="cuda")[None].expand(len(idx), L)
            with torch.no_grad():
                f = torch.empty(len(idx), L, H, dtype=torch.bfloat16, device="cuda")
                for r in range(len(idx)):  # unpadded, one sequence at a time, as the engine prefills
                    n = int(mask[r].sum())
                    hs = lm(input_ids=ids[r: r + 1, :n], attention_mask=torch.ones(1, n, dtype=torch.long, device="cuda"),
                            position_ids=pos[r: r + 1, :n], past_key_values=DynamicCache(config=text_config),
                            use_cache=True).last_hidden_state[0]
                    f[r, :n] = hs
                    f[r, n:] = 0
                e_next = embed[torch.roll(ids, -1, 1)]  # embedding of token[t+1]
            target_f = torch.roll(f, -1, 1).float()           # f_{t+1}
            target_tok = t2d_gpu[torch.roll(ids, -2, 1)]       # draft index of token[t+2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g1 = head(f, e_next, pos, mask)
                # unrolled pass: the feature input at t is the first pass's prediction for f_t (made at t - 1)
                f2 = torch.cat([f[:, :1], g1[:, :-1].to(f.dtype)], 1)
                g2 = head(f2, e_next, pos, mask)
            loss = 0.0
            stats = {}
            for k_, g in (("1", g1), ("2", g2)):
                feat = F.smooth_l1_loss(g.float()[resp], target_f[resp])
                tok_ok = resp & (target_tok >= 0)
                lg = head.logits(g[tok_ok].float())
                ce = F.cross_entropy(lg, target_tok[tok_ok])
                acc = float((lg.argmax(-1) == target_tok[tok_ok]).float().mean())
                loss = loss + feat + 0.1 * ce
                stats.update({f"feat{k_}": float(feat), f"ce{k_}": float(ce), f"acc{k_}": acc})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            stats.update(step=step, epoch=epoch, lr=sched.get_last_lr()[0], loss=float(loss), seconds=time.time() - t0)
            log.write(json.dumps(stats) + "\n"); log.flush()
            if step % 50 == 0 or step == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}), flush=True)
    torch.save(dict(state=head.state_dict(), d2t=d2t, coverage=coverage,
                    config=dict(H=H, HEADS=HEADS, KV=KV, D=D, MLP=MLP, DRAFT_VOCAB=DRAFT_VOCAB, rope_theta=1e6)),
               out / "head.pt")
    print("saved", out / "head.pt", flush=True)


if __name__ == "__main__":
    main()
