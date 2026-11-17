#!/usr/bin/env python3
"""Offline acceptance of the trained draft head (tools/head_train.py) for tree speculation, teacher forced on held-out
greedy LFM3 outputs: the go/no-go before the native head (journal/lfm3-single-stream_synth.md, decision 4).

The state before a pass: tokens s[0..i] known, s[i] pending (not yet run through the target). The head's committed
context is positions t < i with true inputs (embed(s[t+1]), f_t), f_t the target's final normalized hidden state. Its
output at t = i - 1 drafts s[i+1]. Drafting a node of depth d (token guess for s[i+d]) runs the head at position
i + d - 1 with input (embed(guess), the parent's predicted feature), attending to the committed context and the
node's ancestors. Tree (EAGLE-2 style): at each depth, the `width` most probable frontier nodes (by cumulative draft
log-probability) are expanded into their top-`width` children, until R - 1 nodes or depth `max_depth`. The pass
accepts the longest tree path equal to s[i+1..], plus the target's own next token. Reported: tokens per pass for each
row budget R, and per-depth acceptance.
Usage (lfm3-omni-dev): head_eval.py MODEL_DIR HEAD.pt DATA_DIR OUT_JSON [--samples 64] [--names test]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import head_train as ht


@torch.no_grad()
def head_kv(head, x, pos):
    """The layer's K/V (after rotary) and outputs for input rows x [n, H] at positions pos [n] (no attention mixing
    beyond what the caller supplies): returns h (normed), q, k, v."""
    h = ht.rms(x, head.n1)
    n = h.shape[0]
    q = ht.rotary(head.q(h).view(1, n, ht.HEADS, ht.D).transpose(1, 2), pos[None])[0]
    k = ht.rotary(head.k(h).view(1, n, ht.KV, ht.D).transpose(1, 2), pos[None])[0]
    v = head.v(h).view(n, ht.KV, ht.D).transpose(0, 1)
    return q, k, v


@torch.no_grad()
def finish(head, x, attn):
    """Residual, MLP: the head's predicted feature g from its input x and attention output attn [n, HEADS*D]."""
    x = x + head.o(attn)
    h = ht.rms(x, head.n2)
    return x + head.w2(F.silu(head.w1(h)) * head.w3(h))


@torch.no_grad()
def attend(q, keys, values):
    """q [HEADS, n, D] each query with its own key list: keys/values [HEADS, n, m, D] (padded with -inf mask) ."""
    s = torch.einsum("hnd,hnmd->hnm", q.float(), keys.float()) / ht.D ** 0.5
    return s


@torch.no_grad()
def simulate(head, d2t, f, s, prompt_len, budgets, width, max_depth, embed):
    """Tokens per pass for one sequence s (prompt + generation) with target features f [L, H]."""
    L = len(s)
    st = torch.tensor(s, device="cuda")
    # committed context with true inputs, every position t < L - 1
    x_all = head.fc(torch.cat([embed[st[1:]], f[:-1]], -1).float())
    pos_all = torch.arange(L - 1, device="cuda")
    q_all, k_all, v_all = head_kv(head, x_all, pos_all)
    rep = ht.HEADS // ht.KV
    kr, vr = k_all.repeat_interleave(rep, 0), v_all.repeat_interleave(rep, 0)  # [HEADS, L-1, D]
    # the head's output at each committed position with causal attention over the committed context
    causal = torch.ones(L - 1, L - 1, dtype=torch.bool, device="cuda").tril()
    a = F.scaled_dot_product_attention(q_all[None].float(), kr[None].float(), vr[None].float(), attn_mask=causal[None, None])[0]
    g_all = finish(head, x_all, a.transpose(0, 1).reshape(L - 1, -1).to(x_all.dtype))
    lp_all = F.log_softmax(head.logits(g_all).float(), -1)
    results = {}
    for R in budgets:
        i, passes, depth_hits = prompt_len, 0, [0] * (max_depth + 1)
        while i < L - 1:
            # depth-1 candidates from the committed position i - 1
            nodes = []  # (token, parent index or -1, depth, cumulative logprob, g, k, v)
            frontier = []
            lp = lp_all[i - 1]
            top = lp.topk(width)
            for v_, t_ in zip(top.values.tolist(), top.indices.tolist()):
                if len(nodes) >= R - 1:
                    break
                nodes.append([int(d2t[t_]), -1, 1, v_, g_all[i - 1], None, None])
            frontier = list(range(len(nodes)))
            depth = 1
            while depth < max_depth and len(nodes) < R - 1 and frontier:
                # expand the `width` most probable frontier nodes
                frontier.sort(key=lambda j: -nodes[j][3])
                expand = frontier[:width]
                new = []
                for j in expand:
                    tok, parent, d, cum, gpar, _, _ = nodes[j]
                    # path of ancestors (their K/V) from depth 1 to this node
                    path = []
                    a_ = j
                    while a_ >= 0:
                        path.append(a_); a_ = nodes[a_][1]
                    path.reverse()
                    p_ = i + d - 1
                    x = head.fc(torch.cat([embed[tok], gpar.to(embed.dtype)], -1).float()[None])
                    q, k, v = head_kv(head, x, torch.tensor([p_], device="cuda"))
                    nodes[j][5], nodes[j][6] = k, v
                    keys = torch.cat([k_all[:, :i]] + [nodes[a_][5] for a_ in path], 1).repeat_interleave(rep, 0)
                    vals = torch.cat([v_all[:, :i]] + [nodes[a_][6] for a_ in path], 1).repeat_interleave(rep, 0)
                    att = F.scaled_dot_product_attention(q[None].float(), keys[None].float(), vals[None].float())[0]
                    g = finish(head, x, att.transpose(0, 1).reshape(1, -1).to(x.dtype))[0]
                    lpc = F.log_softmax(head.logits(g[None]).float(), -1)[0]
                    topc = lpc.topk(width)
                    for vv, tt in zip(topc.values.tolist(), topc.indices.tolist()):
                        if len(nodes) + len(new) >= R - 1:
                            break
                        new.append([int(d2t[tt]), j, d + 1, cum + vv, g, None, None])
                frontier = [j for j in frontier if j not in expand]
                base = len(nodes)
                nodes.extend(new)
                frontier += list(range(base, len(nodes)))
                depth += 1
            # accept: walk the tree along s[i+1], s[i+2], ...
            parent, a_ = -1, 0
            while i + 1 + a_ < L:
                nxt = next((j for j, n in enumerate(nodes) if n[1] == parent and n[0] == s[i + 1 + a_]), None)
                if nxt is None:
                    break
                parent, a_ = nxt, a_ + 1
            depth_hits[min(a_, max_depth)] += 1
            i += a_ + 1
            passes += 1
        results[R] = dict(passes=passes, tokens=L - prompt_len, tokens_per_pass=(L - prompt_len) / passes,
                          accepted_depth_histogram=depth_hits)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("head"); p.add_argument("data"); p.add_argument("out")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--names", default="test")
    p.add_argument("--budgets", default="8,16,32,64")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=8)
    a = p.parse_args()
    from transformers import AutoModelForMultimodalLM
    from transformers.cache_utils import DynamicCache
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    lm, cfg = model.model.language_model, model.config.text_config
    embed = model.lm_head.weight
    ck = torch.load(a.head, map_location="cuda")
    head = ht.Head(torch.zeros(ht.DRAFT_VOCAB, ht.H)).cuda()
    head.load_state_dict(ck["state"])
    head.eval()
    d2t = ck["d2t"].tolist()
    seqs = ht.load_data(a.data, a.names.split(","))[: a.samples]
    budgets = [int(b) for b in a.budgets.split(",")]
    totals = {R: [0, 0] for R in budgets}
    per = []
    for n, (prompt, gen) in enumerate(seqs):
        s = prompt + gen
        ids = torch.tensor([s], device="cuda")
        with torch.no_grad():
            f = lm(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(len(s), device="cuda")[None],
                   past_key_values=DynamicCache(config=cfg), use_cache=True).last_hidden_state[0]
        r = simulate(head, d2t, f, s, len(prompt), budgets, a.width, a.max_depth, embed)
        per.append(r)
        for R in budgets:
            totals[R][0] += r[R]["tokens"]; totals[R][1] += r[R]["passes"]
        if n % 8 == 0:
            print(n, {R: round(totals[R][0] / totals[R][1], 3) for R in budgets}, flush=True)
    summary = {R: dict(tokens=totals[R][0], passes=totals[R][1], tokens_per_pass=totals[R][0] / totals[R][1]) for R in budgets}
    json.dump(dict(summary=summary, samples=len(seqs), width=a.width, max_depth=a.max_depth, per_sample=per), open(a.out, "w"), indent=1)
    print("tokens per pass:", {R: round(v["tokens_per_pass"], 3) for R, v in summary.items()})


if __name__ == "__main__":
    main()
