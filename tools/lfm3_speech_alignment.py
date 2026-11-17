#!/usr/bin/env python3
"""#2 step 4, the long-form TTS failure's mechanism: where in the prompt's text the backbone attends while it speaks.

For each native case (lfm3-speech's CASE.json: prompt ids and every decoded step's fed token or frame), the exact
input sequence is rebuilt (as tools/lfm3_speech_replay.py does) and run once through HF in FP32 eager (TF32 off) with
attention weights. At each frame step (the position whose final hidden state the depth decoder reads), for every
attention layer and head: the attention mass on the user text's tokens, and its centroid there, as a word index
of the text. Alignment heads are chosen without the failing runs: on the clean runs listed first, the heads whose
centroid rises most monotonically with the frame index (Spearman). Writes OUT_DIR/CASE.npz (mass, centroid, argmax
word, per layer and head) and OUT_DIR/alignment.json (head ranking, and per case the alignment heads' centroid
track, downsampled).
Usage (lfm3-omni-dev): lfm3_speech_alignment.py MODEL_DIR OUT_DIR N_CLEAN NATIVE_JSON...  (clean runs first)
       lfm3_speech_alignment.py summary OUT_DIR   OUT_DIR/mass_by_layer_type.json from the npz files: per case, the
       attention mass on the text averaged over the heads of the sliding-window layers and of the full-attention layers,
       in 25-frame bins, and the frame at which the window's trailing edge passes the text's first and last token
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def main(model_dir, out_dir, n_clean, *cases):
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cuda",
                                                 trust_remote_code=True, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    lm = m.model.language_model
    types = m.config.text_config.layer_types
    att_layers = [i for i, t in enumerate(types) if t != "conv"]  # full_attention and sliding_attention (window 256)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    report = dict(attention_layers=att_layers, layer_types=[types[i] for i in att_layers],
                  sliding_window=m.config.text_config.sliding_window, cases={})
    per_case = {}
    for path in cases:
        nat = json.load(open(path))
        name = Path(path).stem
        prompt, steps = nat["prompt"], nat["steps"]
        # the user text's token span: the processor's ids hold the text's own tokenization between the user header
        # and <|im_end|>; locate it by encoding the request's text with offsets
        req = json.load(open(Path(path).with_suffix("").as_posix().replace("/native/", "/requests/") + ".req.json"))
        text = req["conversation"][-1]["content"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        start = next(i for i in range(len(prompt) - len(ids) + 1) if prompt[i:i + len(ids)] == ids)
        span = np.arange(start, start + len(ids))
        word_of_char = np.cumsum([0] + [1 if c == " " else 0 for c in text])  # word index at each character
        tok_word = np.array([word_of_char[max(b - 1, a)] for a, b in enc["offset_mapping"]], np.float64)  # its last char
        emb = m.get_input_embeddings()
        pieces = [emb(torch.tensor(prompt, device="cuda"))]
        decoded = [s for s in steps if s[1]]
        for s in decoded:
            pieces.append(m.model.get_audio_out_embeds(torch.tensor([s[3:15]], device="cuda")) if s[0]
                          else emb(torch.tensor([s[2]], device="cuda")))
        seq = torch.cat(pieces)
        n, P = len(seq), len(prompt)
        frame_rows = np.array([P - 1 + j for j, s in enumerate(decoded) if s[0]], np.int64)
        o = lm(inputs_embeds=seq[None], attention_mask=torch.ones(1, n, dtype=torch.long, device="cuda"),
               position_ids=torch.arange(n, device="cuda")[None], past_key_values=DynamicCache(config=m.config.text_config),
               use_cache=True, output_attentions=True)
        atts = [a for a in o.attentions if a is not None]
        assert len(atts) == len(att_layers), len(atts)
        mass, cen, amax = [], [], []
        tw = torch.tensor(tok_word, device="cuda", dtype=torch.float32)
        for a in atts:  # [1, heads, n, n]
            w = a[0][:, torch.tensor(frame_rows, device="cuda")][:, :, torch.tensor(span, device="cuda")]  # heads, F, T
            ms = w.sum(-1)
            mass.append(ms.cpu().numpy())
            cen.append(((w * tw).sum(-1) / ms.clamp_min(1e-30)).cpu().numpy())
            amax.append(tw[w.argmax(-1)].cpu().numpy())
        mass, cen, amax = np.stack(mass), np.stack(cen), np.stack(amax)  # layers, heads, frames
        # the frame at which a 256-key window's trailing edge passes each text token (keys i > q - window are seen)
        win = m.config.text_config.sliding_window
        edge_frame = [int(np.searchsorted(frame_rows - win + 1, t, side="right")) for t in span]
        np.savez(out / f"{name}.npz", mass=mass, centroid=cen, argmax_word=amax, words=int(tok_word.max()) + 1,
                 span=span, tok_word=tok_word, frame_rows=frame_rows, edge_frame=np.array(edge_frame))
        per_case[name] = dict(mass=mass, cen=cen, amax=amax)
        print(name, "frames", len(frame_rows), "text tokens", len(span), "words", int(tok_word.max()) + 1, flush=True)
        del o, atts
        torch.cuda.empty_cache()
    names = list(per_case)
    clean = names[: int(n_clean)]
    L, H = per_case[names[0]]["cen"].shape[:2]
    score = np.zeros((L, H))
    for c in clean:
        for l in range(L):
            for h in range(H):
                score[l, h] += spearmanr(np.arange(per_case[c]["cen"].shape[2]), per_case[c]["cen"][l, h]).statistic / len(clean)
    rank = sorted(((float(score[l, h]), att_layers[l], h) for l in range(L) for h in range(H)), reverse=True)
    report["head_ranking_on_clean"] = rank[:16]
    top = [(att_layers.index(layer), h) for _, layer, h in rank[:4]]
    for c in names:
        cen = np.stack([per_case[c]["cen"][l, h] for l, h in top])
        mass = np.stack([per_case[c]["mass"][l, h] for l, h in top])
        report["cases"][c] = dict(frames=int(cen.shape[1]), clean=c in clean,
                                  centroid_top4_every4=[[round(float(v), 1) for v in row[::4]] for row in cen],
                                  mass_top4_every4=[[round(float(v), 3) for v in row[::4]] for row in mass])
    (out / "alignment.json").write_text(json.dumps(report) + "\n")
    print("top heads (spearman on clean, layer, head):", rank[:8])


def summary(out_dir):
    out = Path(out_dir)
    rep = json.loads((out / "alignment.json").read_text())
    sliding = [k for k, t in enumerate(rep["layer_types"]) if t == "sliding_attention"]
    full = [k for k, t in enumerate(rep["layer_types"]) if t == "full_attention"]
    res = dict(bin_frames=25, sliding_layers=[rep["attention_layers"][k] for k in sliding],
               full_layers=[rep["attention_layers"][k] for k in full], cases={})
    for f in sorted(out.glob("*.npz")):
        z = np.load(f)
        mass, edge = z["mass"], z["edge_frame"]  # layers, heads, frames
        bins = lambda m: [round(float(m[:, :, i:i + 25].mean()), 4) for i in range(0, m.shape[2], 25)]
        res["cases"][f.stem] = dict(frames=int(mass.shape[2]), clean=rep["cases"][f.stem]["clean"],
                                    edge_passes_first_token=int(edge.min()), edge_passes_last_token=int(edge.max()),
                                    sliding_mass=bins(mass[sliding]), full_mass=bins(mass[full]))
    (out / "mass_by_layer_type.json").write_text(json.dumps(res, indent=1) + "\n")
    for c, v in res["cases"].items():
        print(c, "edge passes text at frames", v["edge_passes_first_token"], "-", v["edge_passes_last_token"],
              "| sliding mass by 25-frame bin:", v["sliding_mass"][:14], "| full:", v["full_mass"][:14])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "summary":
        summary(sys.argv[2])
    elif len(sys.argv) >= 5:
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
