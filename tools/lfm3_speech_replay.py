#!/usr/bin/env python3
"""#2 step 4, the deterministic gate: the native speech loop with forced inputs against HF (no RNG anywhere).

For each traced case (validation/tts-trace-v1), lfm3-speech replays the trace's text tokens and raw frames
("forced") and records, per decoded step, the input embedding it fed and, per frame step, the final-normed hidden
state the depth decoder read. This tool rebuilds the same input sequence (the prompt with its mode marker, then every
decoded step's fed token or frame, in order) and runs it through HF as one prefill:
- A3: native frame embeddings against HF's get_audio_out_embeds (BF16), bit for bit; text steps' embeddings too;
- the hidden state at each frame step (the output at the position before it) in exact FP32 (eager attention,
  TF32 off) as the anchor, and in BF16 SDPA and BF16 eager (one-pass replays); the trace's own HF hidden states are
  HF's real BF16 SDPA decode.
PASS iff A3 is bitwise and, per case, native's median relative L2 from FP32 over frame steps is at most the worst of
HF's BF16 references (SDPA decode, SDPA one pass, eager one pass): text tier 2a's rule (the larger of eager's and
SDPA's). The SDPA-only reading (the first version of this tool) is reported too.
Usage (lfm3-omni-dev): lfm3_speech_replay.py MODEL_DIR TRACE_DIR NATIVE_DIR OUT_JSON CASE [CASE ...]
  NATIVE_DIR/CASE.{json,hidden.bf16,embeds.bf16}: lfm3-speech forced outputs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def bf16_file(path, width=2048):
    raw = np.fromfile(path, np.uint16).reshape(-1, width)
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32))  # BF16 bits -> exact FP32


def rel(x, ref):
    return ((x - ref).norm(dim=-1) / ref.norm(dim=-1)).numpy()


@torch.no_grad()
def hidden_states(model, embeds, n):
    lm = model.model.language_model
    from transformers.cache_utils import DynamicCache
    out = lm(inputs_embeds=embeds[None], attention_mask=torch.ones(1, n, dtype=torch.long, device="cuda"),
             position_ids=torch.arange(n, device="cuda")[None], past_key_values=DynamicCache(config=model.config.text_config),
             use_cache=True).last_hidden_state[0]
    return out.float().cpu()


def main(model_dir, trace_dir, native_dir, out_json, *cases):
    from transformers import AutoModelForMultimodalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report, ok = {}, True
    models = {}
    for key, dtype, attn in (("bf16", torch.bfloat16, "sdpa"), ("bf16_eager", torch.bfloat16, "eager"),
                             ("fp32", torch.float32, "eager")):
        models[key] = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=dtype, device_map="cuda",
                                                               trust_remote_code=True, attn_implementation=attn).eval()
    for case in cases:
        nat = json.load(open(Path(native_dir) / f"{case}.json"))
        n_hidden = bf16_file(Path(native_dir) / f"{case}.hidden.bf16")
        n_embeds = np.fromfile(Path(native_dir) / f"{case}.embeds.bf16", np.uint16).reshape(-1, 2048)
        trace_file = Path(trace_dir) / case / "frames.pt"  # HF's real decode, when this case was traced
        trace_hidden = torch.load(trace_file)["hidden"].float() if trace_file.exists() else None
        prompt, steps = nat["prompt"], nat["steps"]
        decoded = [s for s in steps if s[1]]
        frame_steps = [j for j, s in enumerate(steps) if s[0]]
        res = {}
        for key, m in models.items():
            emb_tokens = m.get_input_embeddings()
            pieces = [emb_tokens(torch.tensor(prompt, device="cuda"))]
            step_embeds = []
            for s in decoded:
                if s[0]:
                    e = m.model.get_audio_out_embeds(torch.tensor([s[3:15]], device="cuda"))[0]
                else:
                    e = emb_tokens(torch.tensor([s[2]], device="cuda"))[0]
                step_embeds.append(e)
            if key == "bf16":  # A3 and the text embeddings: native vs HF, bit for bit
                hf_bits = torch.stack(step_embeds).view(torch.int16).cpu().numpy().astype(np.uint16)
                same = (hf_bits == n_embeds[: len(step_embeds)])
                frame_rows = [i for i, s in enumerate(decoded) if s[0]]
                res["a3_frames_bitwise"] = bool(same[frame_rows].all()) if frame_rows else None
                res["a3_frame_rows"] = len(frame_rows)
                res["text_embeddings_bitwise"] = bool(same[[i for i, s in enumerate(decoded) if not s[0]]].all())
                res["embedding_rows_checked"] = int(same.shape[0])
            seq = torch.cat([pieces[0]] + [e[None] for e in step_embeds])
            h = hidden_states(m, seq, len(seq))
            P = len(prompt)
            res[{"fp32": "fp32", "bf16": "bf16_replay", "bf16_eager": "eager_replay"}[key]] = h[[P - 1 + j for j in frame_steps]]
        anchor = res.pop("fp32")
        bf16_replay = res.pop("bf16_replay")
        eager_replay = res.pop("eager_replay")
        assert len(n_hidden) == len(frame_steps), (case, len(n_hidden), len(frame_steps))
        refs = [("native", n_hidden), ("hf_replay", bf16_replay), ("hf_eager_replay", eager_replay)]
        if trace_hidden is not None:
            assert len(trace_hidden) == len(frame_steps), case
            refs.insert(1, ("hf_decode", trace_hidden))
        r = {k: rel(v, anchor) for k, v in refs}
        med = {k: float(np.median(v)) for k, v in r.items()}
        bar = max(v for k, v in med.items() if k != "native")
        res["sdpa_only_bar"] = max(v for k, v in med.items() if k in ("hf_decode", "hf_replay"))
        # by position: the median over frame steps in each quarter of the sequence (long-context drift)
        q = np.array_split(np.arange(len(frame_steps)), 4)
        res["median_by_quarter"] = {k: [float(np.median(v[i])) for i in q] for k, v in r.items()}
        res["sdpa_only_pass"] = med["native"] <= res["sdpa_only_bar"]
        res.update(frames=len(frame_steps), positions=len(prompt) + len(decoded), median_rel_l2_vs_fp32=med,
                   p99_vs_fp32={k: float(np.percentile(v, 99)) for k, v in r.items()},
                   native_vs_hf_decode_median=float(np.median(rel(n_hidden, trace_hidden))) if trace_hidden is not None else None,
                   bar=bar, pass_=med["native"] <= bar)
        ok &= res["pass_"] and res["a3_frames_bitwise"] is not False and res["text_embeddings_bitwise"]
        report[case] = res
        print(case, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}),
              {k: round(v, 6) for k, v in med.items()}, flush=True)
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    sys.exit(0 if main(*sys.argv[1:]) else 1)
