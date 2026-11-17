#!/usr/bin/env python3
"""Ternary drafts of LFM3-Omni-2B for exact tree/chain speculation: a small, CPU-runnable demonstration.

Draft: every projection of the language model, and a copy of its tied LM head, is replaced by residual ternary planes per
64-weight group along K: plane j is s_j * T_j, T_j = sign(r) * [|r| > 0.7 mean|r|], s_j = mean |r| over T_j != 0,
r the residual after the earlier planes (1, 2 or 3 planes). Norms and conv kernels are kept exact. The verifier is the
BF16 model itself, so speculation with this draft leaves the output unchanged; only acceptance and speed are at stake.

Measured (teacher forced on held-out LFM3 outputs, so one forward per model per sequence is enough):
- agreement: the fraction of positions where the draft's argmax equals the BF16 model's argmax for the same prefix;
- chain acceptance: for greedy chain speculation, the draft tokens a pass accepts from position i are exactly the run
  of consecutive agreeing positions from i (capped at the draft length k), so tokens per pass = accepted + 1.
Estimated (not measured): single-request tok/s on Thor = tokens per pass / (target pass + k draft steps). Target pass
time is the batched decoder's measured 16.2 ms (<= 8 rows) or 18.3 ms (<= 32 rows); a draft step reads the draft's
bytes (2 bits per weight per plane + a BF16 scale per plane per group, the exact parts in BF16) at 240 GB/s, plus
0.5 ms per step of overhead.
Usage: ternary_draft_demo.py MODEL_DIR TEST_REQUEST.json TEST_OUT.json OUT_JSON [--samples 6] [--max-new 256]
"""
import argparse
import json
import time

import torch

GROUP = 64


@torch.no_grad()
def ternary_planes(w, planes):
    """The residual ternary approximation of w [n, k] (k % 64 == 0) with `planes` planes, as a dense tensor."""
    n, k = w.shape
    r = w.float().reshape(n, k // GROUP, GROUP)
    approx = torch.zeros_like(r)
    for _ in range(planes):
        delta = 0.7 * r.abs().mean(-1, keepdim=True)
        t = torch.sign(r) * (r.abs() > delta)
        cnt = (t != 0).sum(-1, keepdim=True).clamp(min=1)
        s = (r.abs() * (t != 0)).sum(-1, keepdim=True) / cnt
        approx += s * t
        r = r - s * t
    return approx.reshape(n, k).to(w.dtype)


@torch.no_grad()
def int_group(w, bits):
    """Symmetric absmax integer quantization per 64-weight group (the comparison point for the ternary planes)."""
    n, k = w.shape
    r = w.float().reshape(n, k // GROUP, GROUP)
    q = 2 ** (bits - 1) - 1
    s = r.abs().amax(-1, keepdim=True).clamp(min=1e-12) / q
    return (torch.round(r / s).clamp(-q, q) * s).reshape(n, k).to(w.dtype)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3(x):
    """Round positive FP32 scales to FP8 E4M3 (NVFP4's block-scale format)."""
    return x.clamp(min=2.0 ** -9, max=448.0).to(torch.float8_e4m3fn).float()


@torch.no_grad()
def nvfp4(w, ternary=False):
    """NVFP4: blocks of 16 along K, an E4M3 scale per block, E2M1 values (ternary=True: values limited to {-1,0,1},
    the scale then rounded to E4M3: ternary carried in NVFP4)."""
    n, k = w.shape
    r = w.float().reshape(n, k // 16, 16)
    if ternary:
        delta = 0.7 * r.abs().mean(-1, keepdim=True)
        t = torch.sign(r) * (r.abs() > delta)
        cnt = (t != 0).sum(-1, keepdim=True).clamp(min=1)
        s = e4m3((r.abs() * (t != 0)).sum(-1, keepdim=True) / cnt)
        return (s * t).reshape(n, k).to(w.dtype)
    s = e4m3(r.abs().amax(-1, keepdim=True) / 6.0)
    x = (r / s).abs()
    grid = E2M1.to(x.device)
    q = grid[(x[..., None] - grid).abs().argmin(-1)] * torch.sign(r)
    return (q * s).reshape(n, k).to(w.dtype)


@torch.no_grad()
def mxfp8(w):
    """MXFP8: blocks of 32 along K, a power-of-two (E8M0) scale per block, E4M3 values."""
    n, k = w.shape
    r = w.float().reshape(n, k // 32, 32)
    amax = r.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))  # the smallest power of two that keeps the block in range
    return ((r / s).to(torch.float8_e4m3fn).float() * s).reshape(n, k).to(w.dtype)


def projection_params(lm):
    out = []
    for name, mod in lm.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[1] % GROUP == 0:
            out.append((name, mod))
    return out


@torch.no_grad()
def argmax_tf(model, lm, ids, head=None):
    """Teacher-forced argmax at every position (the next-token prediction for each prefix)."""
    hs = lm(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None],
            position_ids=torch.arange(len(ids))[None], use_cache=False).last_hidden_state[0]
    out = []
    for r0 in range(0, len(ids), 128):
        out.append((hs[r0:r0 + 128].float() @ (model.lm_head.weight if head is None else head).float().T).argmax(-1))
    return torch.cat(out)


def chain(agree, k):
    """Tokens per pass for chain speculation with draft length k over agreement flags."""
    i, passes, n = 0, 0, len(agree)
    while i < n:
        a = 0
        while a < k and i + a < n and agree[i + a]:
            a += 1
        i += a + 1
        passes += 1
    return n / passes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("req"); p.add_argument("out_json"); p.add_argument("out")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--variants", default="", help="comma-separated subset of the draft variants")
    a = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    from transformers import AutoModelForMultimodalLM
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True,
                                                     attn_implementation="sdpa").eval()
    lm = model.model.language_model
    req, res = json.load(open(a.req)), json.load(open(a.out_json))
    seqs = []
    for prompt, row in list(zip(req["prompts"], res["rows"]))[: a.samples]:
        gen = row["generated_ids"][: a.max_new]
        seqs.append((torch.tensor(prompt + gen), len(prompt)))
    print(f"loaded in {time.time() - t0:.0f}s; {len(seqs)} sequences, {sum(len(s) - p for s, p in seqs)} scored tokens", flush=True)
    target = []
    for ids, pl in seqs:
        target.append(argmax_tf(model, lm, ids)[pl - 1:-1])  # predictions for the generated tokens
    native_match = sum(int((t == ids[pl:]).sum()) for t, (ids, pl) in zip(target, seqs)) / sum(len(t) for t in target)
    print(f"BF16 HF argmax equals the native engine's tokens at {native_match:.4f} of positions", flush=True)
    projs = projection_params(lm)
    proj_params = sum(m.weight.numel() for _, m in projs)
    head_params = model.lm_head.weight.numel()
    embed = lm.get_input_embeddings().weight
    # bytes a draft step reads beyond the ternary matrices: norms, conv kernels (the embedding lookup reads one row)
    exact_params = sum(q.numel() for q in lm.parameters()) - proj_params - embed.numel()
    originals = {name: m.weight.detach().clone() for name, m in projs}
    report = dict(samples=len(seqs), scored_tokens=sum(len(t) for t in target), hf_vs_native=native_match, drafts={})
    variants = [(f"ternary-{n}", lambda w, n=n: ternary_planes(w, n), n * (2 + 16 / GROUP)) for n in (1, 2, 3)]
    variants += [(f"int{b}", lambda w, b=b: int_group(w, b), b + 16 / GROUP) for b in (4, 8)]
    variants += [("nvfp4", lambda w: nvfp4(w), 4.5), ("ternary-in-nvfp4", lambda w: nvfp4(w, True), 4.5)]
    variants += [("mxfp8", lambda w: mxfp8(w), 8.25)]
    only = [v for v in a.variants.split(",") if v] if a.variants else None
    if only:
        variants = [v for v in variants if v[0] in only]
    for planes, quant, bits in variants:
        for name, m in projs:
            m.weight.data.copy_(quant(originals[name]))
        head_t = quant(model.lm_head.weight.detach())  # the draft's own head (the tied one stays exact)
        agree = []
        for (ids, pl), tgt in zip(seqs, target):
            d = argmax_tf(model, lm, ids, head_t)[pl - 1:-1]
            agree.append((d == tgt).tolist())
        flat = [x for s in agree for x in s]
        rate = sum(flat) / len(flat)
        draft_bytes = (proj_params + head_params) * bits / 8 + exact_params * 2
        draft_ms = draft_bytes / 240e9 * 1e3 + 0.5
        rows = {}
        for k in (1, 2, 3, 4, 6, 8, 12, 16):
            tpp = sum(chain(s, k) * len(s) for s in agree) / len(flat)
            target_ms = 16.2 if k + 1 <= 8 else 18.3
            rows[k] = dict(tokens_per_pass=tpp, est_tok_s=tpp / (target_ms + k * draft_ms) * 1e3)
        best = max(rows, key=lambda k: rows[k]["est_tok_s"])
        report["drafts"][planes] = dict(agreement=rate, bits_per_weight=bits, draft_gb=draft_bytes / 1e9,
                                        draft_step_ms=draft_ms, chain=rows, best_k=best)
        print(f"{planes}: {bits:.2f} bits/weight, draft {draft_bytes/1e9:.2f} GB ({draft_ms:.1f} ms/step), "
              f"agreement {rate:.4f}; best k={best}: {rows[best]['tokens_per_pass']:.2f} tokens/pass, "
              f"~{rows[best]['est_tok_s']:.0f} tok/s (est.)", flush=True)
    for name, m in projs:
        m.weight.data.copy_(originals[name])
    report["baseline_tok_s"] = 1e3 / 16.2
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
