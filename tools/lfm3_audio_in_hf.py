#!/usr/bin/env python3
"""Audio input (#3 audio in): the HF reference for speech recognition and spoken questions on LFM3-Omni-2B.

For each case of CASES.json, the conversation as the model card writes it (system text, then a user turn whose content
is the audio clip(s) and optional text), rendered by the processor's chat template, the clips read with soundfile as
float32 16 kHz mono (the same samples lfm3-speech reads from the same WAV file), and generate() in BF16 with SDPA:
generation_mode as the case says, text greedy (generation_config), speech at audio_temperature 0.4 and
audio_repetition_penalty 1.2 with torch.manual_seed(seed).
Writes OUT_DIR/ENGINE/CASE/{summary.json, response.txt[, response.wav]} (tools/lfm3_asr.py's layout) and
OUT_DIR/requests/CASE.req.json for lfm3-speech: the same prompt ids, the clip paths, mode, max_new_tokens, seed.
With "dump": true on a case, also OUT_DIR/ENGINE/CASE/{mel.f32 [clip][128][frames], down.f32 (the downsampler's
output, [token][768]), encoded.f32 (the encoder's, [token][768]), features.f32 (projected, [token][2048]),
{down,encoded,features}_fp32.f32 (the encoder and projector in FP32 with TF32 off, from the same FP32 mel: the true
stages) and mel_fp64.f32 (the extractor's arithmetic in FP64 throughout, window and mel filters included: the true
features, rounded to float32 only to be stored)}.
CASES.json: {"systems": {key: text}, "cases": [{"case", "audio": [WAV, ...] (paths as seen in this container),
  "system": key or null, "text": optional text after the clips, "mode": "text"|"audio"|"interleaved"|null,
  "max_new_tokens", "seed", "reference": optional transcript, "dump": optional}]}
Usage (lfm3-omni-dev): lfm3_audio_in_hf.py MODEL_DIR CASES.json OUT_DIR [requests]
  requests: write OUT_DIR/requests only (no model is loaded, only the processor).
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ENGINE = "hf"


def clip(path):
    x, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != 16_000 or x.ndim != 1:
        raise SystemExit(f"{path}: expected 16 kHz mono, got {rate} Hz, shape {x.shape}")
    return x


def mel_fp64(fe, x):
    """Lfm3OmniFeatureExtractor's features for one clip with every operation in FP64 (its float64 mel filters as
    mel_filter_bank returns them, a float64 periodic Hann window, a float64 FFT)."""
    w = torch.as_tensor(x, dtype=torch.float64)
    w = torch.nn.functional.pad(w, (0, fe.pad_length(w.shape[0]) - w.shape[0]))
    spec = torch.stft(w, n_fft=fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                      window=torch.hann_window(fe.win_length, dtype=torch.float64), center=False, return_complex=True)
    power = spec.abs().square()
    log_mel = (torch.from_numpy(fe.mel_filters).T @ power).log10().clamp(min=fe.log_floor)
    return ((log_mel + fe.normalization_offset) / fe.normalization_divisor).cuda()


def conversation(spec, c, clips):
    content = [{"type": "audio", "audio": a} for a in clips]
    if c.get("text"):
        content.append({"type": "text", "text": c["text"]})
    conv = [{"role": "user", "content": content}]
    if c.get("system"):
        conv.insert(0, {"role": "system", "content": spec["systems"][c["system"]]})
    return conv


def request(c, ids):
    return dict(prompt_ids=ids, audio=c["audio"], mode=c.get("mode") or "none", max_new_tokens=c["max_new_tokens"],
                seed=c.get("seed", 0), record_audio=bool(c.get("dump")))


def main(model_dir, cases, out_dir, only_requests=False):
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    spec = json.load(open(cases))
    out = Path(out_dir)
    (out / "requests").mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = None
    if not only_requests:
        torch.set_grad_enabled(False)
        model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                         trust_remote_code=True, attn_implementation="sdpa").eval()
        codec = processor.audio_tokenizer.to(device="cuda").eval()
        encoder32 = projector32 = None
    for c in spec["cases"]:
        clips = [clip(p) for p in c["audio"]]
        conv = conversation(spec, c, clips)
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        (out / "requests" / f"{c['case']}.req.json").write_text(json.dumps(request(c, ids)) + "\n")
        if only_requests:
            continue
        inputs = inputs.to("cuda")
        d = out / ENGINE / c["case"]
        d.mkdir(parents=True, exist_ok=True)
        if c.get("dump"):
            m = model.model
            feats, mask = inputs["input_features"], inputs["feature_attention_mask"]
            caught = {}
            hooks = [m.audio_encoder.downsample.register_forward_hook(lambda mod, a, o: caught.__setitem__("down", o)),
                     m.audio_encoder.register_forward_hook(lambda mod, a, o: caught.__setitem__("enc", o))]
            features = m.get_audio_features(feats, mask)
            for h in hooks:
                h.remove()
            lengths = mask.sum(-1)
            tokens = m.audio_encoder.downsample.output_lengths(lengths)
            valid = torch.arange(caught["enc"][0].shape[1], device="cuda")[None] < tokens[:, None]
            down = caught["down"].transpose(1, 2)[valid]
            enc = caught["enc"][0][valid]
            if encoder32 is None:
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                encoder32 = copy.deepcopy(m.audio_encoder).float().eval()
                projector32 = copy.deepcopy(m.audio_projector).float().eval()
            caught32 = {}
            hook = encoder32.downsample.register_forward_hook(lambda mod, a, o: caught32.__setitem__("down", o))
            e32, l32 = encoder32(feats.float(), lengths)
            hook.remove()
            v32 = torch.arange(e32.shape[1], device="cuda")[None] < l32[:, None]
            f32 = projector32(e32[v32])
            down32 = caught32["down"].transpose(1, 2)[v32]
            mel = torch.cat([feats[i, :, :int(lengths[i])] for i in range(feats.shape[0])], dim=-1)
            mel64 = torch.cat([mel_fp64(processor.feature_extractor, x) for x in clips], dim=-1)
            for name, t in (("mel", mel), ("down", down), ("encoded", enc), ("features", features),
                            ("down_fp32", down32), ("encoded_fp32", e32[v32]), ("features_fp32", f32),
                            ("mel_fp64", mel64)):
                t.float().contiguous().cpu().numpy().tofile(d / f"{name}.f32")
        torch.manual_seed(c.get("seed", 0))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kw = dict(max_new_tokens=c["max_new_tokens"], return_dict_in_generate=True)
        if c.get("mode"):
            kw["generation_mode"] = c["mode"]
        if c.get("mode") in ("audio", "interleaved"):
            kw.update(audio_temperature=0.4, audio_repetition_penalty=1.2)
        o = model.generate(**inputs, **kw)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        prompt_len = len(ids)
        end = int(o.sequence_lengths[0]) if getattr(o, "sequence_lengths", None) is not None else o.sequences.shape[1]
        gen = o.sequences[0, prompt_len:end].tolist()
        text = processor.decode(gen, skip_special_tokens=True)
        audio_seconds = 0.0
        if getattr(o, "audio_lengths", None) is not None and int(o.audio_lengths[0]):
            n = int(o.audio_lengths[0])
            wav = codec.decode(o.audio_codes[:1, :, :n]).audio_values[0].squeeze(0).float().cpu().numpy()
            sf.write(d / "response.wav", wav, 24_000)
            audio_seconds = len(wav) / 24_000
        (d / "response.txt").write_text(text + "\n")
        (d / "summary.json").write_text(json.dumps(dict(
            case=c["case"], mode=c.get("mode") or "none", seed=c.get("seed", 0), engine=ENGINE, audio=c["audio"],
            conversation=[{"role": x["role"], "content": x["content"] if isinstance(x["content"], str) else
                           [p if p["type"] == "text" else {"type": "audio"} for p in x["content"]]} for x in conv],
            prompt_ids=ids, audio_tokens=ids.count(125031), generated_ids=gen, text=text,
            reference=c.get("reference"), audio_seconds=audio_seconds,
            input_seconds=sum(len(x) for x in clips) / 16_000, generate_seconds=seconds), indent=1) + "\n")
        print(c["case"], f"{ids.count(125031)} audio tokens, {len(gen)} new in {seconds:.2f}s:", text[:100], flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or (len(sys.argv) == 5 and sys.argv[4] != "requests"):
        sys.exit(__doc__)
    main(*sys.argv[1:4], only_requests=len(sys.argv) == 5)
