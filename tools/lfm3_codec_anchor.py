#!/usr/bin/env python3
"""Codec anchors for A2 (journal/lfm3-tts-quality-bar.md): the HF codec decoder from forced codes, in four precisions.

For each case from lfm3_audio_trace.py (its sequences.pt holds the reference's codes), decodes:
  bf16      HF as the reference runs it (BF16, SDPA, Triton conv, GPU), by `codec.decode`. It must equal the trace's
            waveform.f32.npy bit for bit.
  bf16-eager  the same with eager attention.
  bf16-stream the same streamed in 12-frame chunks with LiquidCodecStreamingCache (documented equal to whole).
  fp32      every parameter upcast to FP32, TF32 off, GPU.
  fp64      every parameter in FP64 on the CPU. The model's FP32 islands (RMSNorm, rotary tables and their
            application, the ISTFT) are widened to FP64 here, so this is the real math, not FP32 in places.
The fp32 and fp64 runs go through `decode_steps`, a step-by-step copy of `LiquidCodecModel.decode` (whole sequence,
no streaming). Run on BF16 it must equal `codec.decode` bit for bit, which checks the copy.

Reports SNR (dB) and max abs error of each against fp64 and fp32. For --dump CASE, saves that case's intermediates
(latents, embeddings, upsampled input, every layer's output, final norm, lin output, waveform) in bf16, fp32 and fp64,
for debugging a native port layer by layer.

Usage (Thor, lfm3-omni-dev): lfm3_codec_anchor.py MODEL_DIR OUT_DIR --dump CASE CASE_DIR [CASE_DIR ...]
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def snr_db(x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    noise = ((x - ref) ** 2).sum()
    return float("inf") if noise == 0 else float(10 * np.log10((ref ** 2).sum() / noise))


def compare(x, ref):
    if len(x) != len(ref):
        return dict(length=[len(x), len(ref)])
    return dict(snr_db=snr_db(x, ref), max_abs=float(np.abs(np.asarray(x, np.float64) - ref).max()))


def widen_fp32_islands(lfm3_model=None):
    """Patch the model's hard FP32 casts to compute in max(FP32, input dtype): unchanged for BF16/FP32 inputs.
    `lfm3_model` is any Lfm3Model of the checkpoint (default: the codec's trunk)."""
    import importlib
    lfm3 = importlib.import_module(type(_CODEC.decoder.lfm if lfm3_model is None else lfm3_model).__module__)

    def wide(t):
        return torch.float64 if t.dtype == torch.float64 else torch.float32

    def rms_forward(self, h):
        dtype, w = h.dtype, wide(h)
        h = h.to(w)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dtype)
    lfm3.Lfm3RMSNorm.forward = rms_forward

    def apply_rotary(q, k, cos, sin, unsqueeze_dim=1):
        w = wide(q)
        cos, sin = cos.unsqueeze(unsqueeze_dim).to(w), sin.unsqueeze(unsqueeze_dim).to(w)
        q_embed = q.to(w) * cos + lfm3.rotate_half(q.to(w)) * sin
        k_embed = k.to(w) * cos + lfm3.rotate_half(k.to(w)) * sin
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    lfm3.apply_rotary_pos_emb = apply_rotary

    real_rotary = lfm3.Lfm3RotaryEmbedding.forward

    def rotary_forward(self, x, position_ids, layer_type):
        # the codec's pre-hook casts x to FP32, so the FP64 model is recognized by its (converted) buffers
        if getattr(self, f"{layer_type}_inv_freq").dtype != torch.float64:
            return real_rotary(self, x, position_ids, layer_type)
        cfg = self.config
        dim = cfg.head_dim
        rope = cfg.rope_parameters[layer_type]
        angles = int(rope.get("partial_rotary_factor", 1.0) * dim) // 2
        inv = 1.0 / (rope["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float64) / dim))
        inv = torch.cat((inv, torch.zeros(dim // 2 - angles, dtype=torch.float64)))
        freqs = position_ids[..., None].to(torch.float64) * inv.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        scale = getattr(self, f"{layer_type}_attention_scaling")
        return emb.cos() * scale, emb.sin() * scale
    lfm3.Lfm3RotaryEmbedding.forward = rotary_forward


def decode_steps(codec, codes, record=None):
    """LiquidCodecModel.decode for a whole sequence (no streaming cache), step by step, in the codec's dtype; the
    ISTFT in FP32 (as HF) unless the codec is FP64."""
    def keep(name, t):
        if record is not None:
            record[name] = t.detach().to("cpu").clone()
    wide = torch.float64 if codec.from_quantizer.weight.dtype == torch.float64 else torch.float32
    latents = codec.quantizer.decode(codes).mT.to(codec.from_quantizer.weight.dtype)
    keep("latents", latents)
    embeddings = codec.from_quantizer(latents)
    keep("embeddings", embeddings)
    decoder = codec.decoder
    x = F.interpolate(embeddings.mT, embeddings.shape[1] * decoder.frame_stack, mode="nearest-exact").mT
    keep("upsampled", x)
    hooks = [layer.register_forward_hook(lambda m, a, o, i=i: keep(f"layer{i}", o if torch.is_tensor(o) else o[0]))
             for i, layer in enumerate(decoder.lfm.layers)]
    try:
        hidden = decoder.lfm(inputs_embeds=x, use_cache=False).last_hidden_state
    finally:
        for h in hooks:
            h.remove()
    keep("final_norm", hidden)
    lin = decoder.lin(hidden)
    keep("lin", lin)
    log_magnitude, angle = torch.chunk(lin.mT.contiguous(), 2, 1)
    spec = torch.polar(log_magnitude.to(wide).exp(), angle.to(wide))
    istft = decoder.istft
    window = istft.window.to(wide)
    num_frames = spec.shape[-1]
    frames = torch.fft.irfft(spec, istft.n_fft, dim=1, norm="backward") * window[None, :, None]
    output_size = (num_frames - 1) * istft.hop_length + istft.win_length
    fold = {"output_size": (1, output_size), "kernel_size": (1, istft.win_length), "stride": (1, istft.hop_length)}
    audio = F.fold(frames, **fold)[:, 0, 0]
    envelope = F.fold(window.square().expand(1, num_frames, -1).transpose(1, 2), **fold)[0, 0, 0]
    trim = (istft.win_length - istft.hop_length) // 2
    audio, envelope = audio[:, trim:-trim], envelope[trim:-trim]
    out = audio / envelope
    if wide == torch.float32:
        out = out.float()
    keep("waveform", out)
    return out[0]


_CODEC = None


def main(model_dir, out_dir, dump_case, case_dirs):
    global _CODEC
    from transformers import AutoProcessor
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    bf16 = processor.audio_tokenizer.to(device="cuda").eval()
    _CODEC = bf16
    module = sys.modules[type(bf16).__module__]
    cases = []
    for case_dir in map(Path, case_dirs):
        codes = torch.load(case_dir / "sequences.pt")["audio_codes"]
        reference = np.load(case_dir / "waveform.f32.npy")
        cases.append((case_dir.name, codes, reference))

    results = {}
    waves = {name: {} for name, _, _ in cases}
    records = {}
    with torch.inference_mode():
        for name, codes, reference in cases:
            g = codes.cuda()
            w = waves[name]
            w["bf16"] = bf16.decode(g).audio_values[0].cpu().numpy()
            rec = {} if name == dump_case else None
            w["bf16-steps"] = decode_steps(bf16, g, rec).cpu().numpy()
            if rec is not None:
                records["bf16"] = rec
            stream = module.LiquidCodecStreamingCache(bf16.config)
            chunks = [bf16.decode(g[..., i:i + 12], streaming_cache=stream, finalize=i + 12 >= g.shape[-1]).audio_values
                      for i in range(0, g.shape[-1], 12)]
            w["bf16-stream"] = torch.cat(chunks, dim=-1)[0].cpu().numpy()
            results[name] = dict(frames=int(codes.shape[-1]),
                                 bf16_equals_trace=bool(np.array_equal(w["bf16"], reference)),
                                 steps_equal_decode=bool(np.array_equal(w["bf16-steps"], w["bf16"])),
                                 stream_vs_whole=compare(w["bf16-stream"], w["bf16"]))
        bf16.config._attn_implementation = "eager"
        for layer in bf16.decoder.lfm.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.config._attn_implementation = "eager"
        bf16.decoder.lfm.config._attn_implementation = "eager"
        for name, codes, _ in cases:
            waves[name]["bf16-eager"] = bf16.decode(codes.cuda()).audio_values[0].cpu().numpy()
            results[name]["eager_differs_from_sdpa"] = not np.array_equal(waves[name]["bf16-eager"], waves[name]["bf16"])
        for layer in bf16.decoder.lfm.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.config._attn_implementation = "sdpa"
        bf16.decoder.lfm.config._attn_implementation = "sdpa"
        bf16.config._attn_implementation = "sdpa"

        widen_fp32_islands()
        for name, codes, _ in cases:  # the patches must not change BF16
            again = decode_steps(bf16, codes.cuda()).cpu().numpy()
            results[name]["bf16_unchanged_by_widening"] = bool(np.array_equal(again, waves[name]["bf16"]))
        fp32 = copy.deepcopy(bf16).float()
        for name, codes, _ in cases:
            rec = {} if name == dump_case else None
            waves[name]["fp32"] = decode_steps(fp32, codes.cuda(), rec).cpu().numpy()
            if rec is not None:
                records["fp32"] = rec
        del fp32
        fp64 = copy.deepcopy(bf16).to("cpu").double()
        for name, codes, _ in cases:
            rec = {} if name == dump_case else None
            waves[name]["fp64"] = decode_steps(fp64, codes, rec).numpy()
            if rec is not None:
                records["fp64"] = rec

    for name, _, _ in cases:
        w = waves[name]
        results[name]["vs_fp64"] = {k: compare(w[k], w["fp64"]) for k in ("fp32", "bf16", "bf16-eager", "bf16-stream")}
        results[name]["vs_fp32"] = {k: compare(w[k], w["fp32"]) for k in ("bf16", "bf16-eager", "bf16-stream")}
        np.save(out / f"{name}.fp64.npy", w["fp64"])
        np.save(out / f"{name}.fp32.npy", w["fp32"])
        print(name, json.dumps(results[name]))
    if records:
        torch.save(records, out / f"{dump_case}.intermediates.pt")
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    (out / "anchor.json").write_text(json.dumps(dict(environment=env, results=results), indent=1) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 5 or args[2] != "--dump":
        sys.exit(__doc__)
    main(args[0], args[1], args[3], args[4:])
