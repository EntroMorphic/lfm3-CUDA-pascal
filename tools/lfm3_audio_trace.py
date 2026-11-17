#!/usr/bin/env python3
"""Measured runtime trace of the HF text -> text + audio path (journal/lfm3-tts-source-map.md, open questions 1-6).

Runs the reference model exactly as test_interleaved.py does (BF16, SDPA, CUDA) for one case, with instrumentation
that does not change any computation:
  - every scaled_dot_product_attention call: head dim (128 backbone, 32 depth decoder, 64 codec), dtypes, is_causal,
    mask kind, query/key lengths, GQA; for masked calls, the number of keys each query row may attend (window);
  - parameter dtypes per component;
  - per audio frame: the CUDA generator's Philox offset before/after, the LM hidden state, and for each of the three
    depth-decoder passes the revealed mask, input codes and logits (exact dtype), and the raw sampled codes
    (before the whole-frame EOS rewrite);
  - kernel names of one codec decode and one depth-decoder pass (torch.profiler, after generation).
Writes OUT/CASE/: response.wav, response.txt, frames.pt, sequences.pt, summary.json.

Usage (Thor, in the lfm3-omni-dev container):
  python3 lfm3_audio_trace.py MODEL_DIR OUT_DIR CASE MODE SEED MAX_NEW SYSTEM_PROMPT USER_PROMPT [REFERENCE_DIR]
MODE is interleaved or audio. REFERENCE_DIR (optional) holds a previous response.wav/response.txt to compare with.
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def main(model_dir, out_dir, case, mode, seed, max_new, system, user, reference=None):
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    out = Path(out_dir) / case
    out.mkdir(parents=True, exist_ok=False)
    seed, max_new = int(seed), int(max_new)
    torch.manual_seed(seed)  # as test_interleaved.py: before loading
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    depth = model.model.depth_decoder

    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
               bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    import transformers
    env["transformers"] = transformers.__version__

    def dtypes(module):
        c = collections.Counter()
        for p in module.parameters():
            c[str(p.dtype)] += p.numel()
        for name, b in module.named_buffers():
            c[f"buffer {b.dtype}"] += b.numel()
        return dict(c)
    components = dict(depth_decoder=depth, codebook_embedding=model.model.codebook_embedding, codec=codec)
    param_dtypes = {k: dtypes(v) for k, v in components.items()}
    attn_impl = dict(model=getattr(model.config, "_attn_implementation", None),
                     depth=getattr(depth.model.config, "_attn_implementation", None),
                     codec=getattr(getattr(codec, "config", None), "_attn_implementation", None))

    # --- attention calls -------------------------------------------------------------------------------------
    sdpa_calls = collections.Counter()
    windows = {}
    real_sdpa = F.scaled_dot_product_attention

    def logged_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
        component = {128: "backbone", 32: "depth", 64: "codec"}.get(q.shape[-1], f"head_dim {q.shape[-1]}")
        mask = "none" if attn_mask is None else f"{attn_mask.dtype} {tuple(attn_mask.shape)}"
        lq, lk = q.shape[-2], k.shape[-2]
        key = (component, str(q.dtype), bool(is_causal), "none" if attn_mask is None else str(attn_mask.dtype),
               q.shape[1], k.shape[1], "1" if lq == 1 else "many", bool(enable_gqa), scale)
        sdpa_calls[key] += 1
        if attn_mask is not None and component not in windows and lq > 1:
            allowed = attn_mask if attn_mask.dtype == torch.bool else attn_mask > -1e30
            per_row = allowed.reshape(-1, allowed.shape[-2], allowed.shape[-1])[0].sum(-1).tolist()
            windows[component] = dict(mask=mask, query_len=lq, key_len=lk, allowed_keys_first_rows=per_row[:40],
                                      allowed_keys_max=max(per_row))
        return real_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                         enable_gqa=enable_gqa, **kw)
    F.scaled_dot_product_attention = logged_sdpa
    torch.nn.functional.scaled_dot_product_attention = logged_sdpa

    # --- per-frame records -------------------------------------------------------------------------------------
    frames = []
    passes = None

    def offset():
        state = torch.cuda.get_rng_state()
        return int(np.frombuffer(state.numpy().tobytes()[8:16], np.int64)[0])

    real_frame = model.sample_audio_frame
    real_depth_forward = depth.forward

    def depth_forward(hidden_states, codes, masked):
        logits = real_depth_forward(hidden_states, codes, masked)
        if passes is not None:
            passes.append(dict(masked=masked.detach().cpu().clone(), codes=codes.detach().cpu().clone(),
                               logits=logits.detach().cpu().clone()))
        return logits

    def sample_frame(hidden_state, *args, **kwargs):
        nonlocal passes
        passes = []
        before = offset()
        codes = real_frame(hidden_state, *args, **kwargs)
        frames.append(dict(hidden=hidden_state.detach().cpu().clone(), raw_codes=codes.detach().cpu().clone(),
                           offset_before=before, offset_after=offset(), passes=passes))
        passes = None
        return codes
    depth.forward = depth_forward
    model.sample_audio_frame = sample_frame

    conversation = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new, generation_mode=mode, do_sample=False,
                                 audio_temperature=0.4, audio_repetition_penalty=1.2)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        prompt_len = inputs["input_ids"].shape[1]
        slots = outputs.sequences[0, prompt_len:outputs.sequence_lengths[0]].cpu()
        text = processor.decode(slots, skip_special_tokens=True)
        n_frames = int(outputs.audio_lengths[0])
        codes = outputs.audio_codes[:1, :, :n_frames]
        waveform = codec.decode(codes).audio_values[0].squeeze(0).float().cpu().numpy() if n_frames else np.zeros(0)
    model.sample_audio_frame = real_frame
    depth.forward = real_depth_forward
    sdpa_generation = dict(sdpa_calls)

    # --- kernels of one codec decode and one depth pass (after generation; no effect on outputs) --------------
    kernels = {}
    if n_frames:
        from torch.profiler import ProfilerActivity, profile
        with torch.inference_mode():
            for name, fn in (("codec_decode", lambda: codec.decode(codes)),
                             ("depth_pass", lambda: depth(frames[0]["hidden"][None].cuda(),
                                                          frames[0]["passes"][0]["codes"].cuda(),
                                                          frames[0]["passes"][0]["masked"].cuda()))):
                fn(); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn(); torch.cuda.synchronize()
                kernels[name] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    F.scaled_dot_product_attention = real_sdpa
    torch.nn.functional.scaled_dot_product_attention = real_sdpa

    sf.write(out / "response.wav", waveform, 24_000)  # soundfile's default for .wav: 16-bit PCM, as the reference
    np.save(out / "waveform.f32.npy", waveform)  # the codec's exact output
    (out / "response.txt").write_text(text + "\n")
    torch.save(dict(
        hidden=torch.stack([f["hidden"] for f in frames]) if frames else None,
        raw_codes=torch.stack([f["raw_codes"] for f in frames]) if frames else None,
        offsets=torch.tensor([[f["offset_before"], f["offset_after"]] for f in frames]),
        pass_masked=torch.stack([torch.stack([p["masked"][0] for p in f["passes"]]) for f in frames]) if frames else None,
        pass_codes=torch.stack([torch.stack([p["codes"][0] for p in f["passes"]]) for f in frames]) if frames else None,
        pass_logits=torch.stack([torch.stack([p["logits"][0] for p in f["passes"]]) for f in frames]) if frames else None,
    ), out / "frames.pt")
    torch.save(dict(prompt_ids=inputs["input_ids"][0].cpu(), slots=slots, audio_codes=codes.cpu()),
               out / "sequences.pt")

    # --- summary ----------------------------------------------------------------------------------------------
    cfg = model.config
    eos = cfg.audio_eos_code
    raw = [f["raw_codes"] for f in frames]
    eos_counts = collections.Counter(int((c == eos).sum()) for c in raw)
    runs, kind = [], None
    for t in slots.tolist():
        k = "A" if t == cfg.audio_out_token_id else "T"
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    deltas = collections.Counter(f["offset_after"] - f["offset_before"] for f in frames)
    summary = dict(
        case=case, mode=mode, seed=seed, max_new_tokens=max_new, conversation=conversation, environment=env,
        generation_seconds=seconds, prompt_tokens=prompt_len, generated_slots=len(slots), audio_frames=n_frames,
        sampled_frames=len(frames), audio_seconds=len(waveform) / 24_000,
        text=text, attention_implementation=attn_impl, parameter_dtypes=param_dtypes,
        sdpa_calls=[dict(component=k[0], dtype=k[1], is_causal=k[2], mask=k[3], query_heads=k[4], key_heads=k[5],
                         query_len=k[6], enable_gqa=k[7], scale=k[8], calls=n) for k, n in sdpa_generation.items()],
        attention_windows=windows, kernels=kernels,
        frame_depth_passes=sorted({len(f["passes"]) for f in frames}),
        pass_logits_dtype=str(frames[0]["passes"][0]["logits"].dtype) if frames else None,
        hidden_dtype=str(frames[0]["hidden"].dtype) if frames else None,
        raw_eos_codes_per_frame=dict(sorted(eos_counts.items())),
        philox_offset_delta_per_frame=dict(sorted(deltas.items())),
        slot_runs=[f"{k}{n}" for k, n in runs],
        waveform=dict(samples=len(waveform), peak=float(np.abs(waveform).max()) if len(waveform) else 0.0,
                      rms=float(np.sqrt((waveform.astype(np.float64) ** 2).mean())) if len(waveform) else 0.0,
                      codec_parameter_dtype=str(codec.dtype) if hasattr(codec, "dtype") else None),
    )
    if reference:
        ref = Path(reference)
        # Both files are 16-bit PCM: compare the samples as stored.
        rw, _ = sf.read(ref / "response.wav", dtype="int16")
        mw, _ = sf.read(out / "response.wav", dtype="int16")
        same_len = len(rw) == len(mw)
        summary["reference"] = dict(
            directory=str(ref), text_equal=(ref / "response.txt").read_text() == text + "\n",
            samples=[len(rw), len(mw)], wav_pcm16_equal=bool(same_len and np.array_equal(rw, mw)),
            wav_pcm16_max_abs_diff=int(np.abs(rw.astype(np.int32) - mw).max()) if same_len else None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("case", "generated_slots", "audio_frames", "audio_seconds",
                                              "generation_seconds")}))


if __name__ == "__main__":
    if len(sys.argv) not in (9, 10):
        sys.exit(__doc__)
    main(*sys.argv[1:])
