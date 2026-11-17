#!/usr/bin/env python3
"""#2 step 4 (C1-C4): the HF reference over the fixed prompt set, every seed, and the native engine's requests.

For each case of PROMPTS.json (validation/speech-e2e-v1/prompts.json): the conversation (system + user), its prompt
ids from the processor's chat template (exactly what HF generates from, written to OUT_DIR/requests/CASE-sSEED.req.json
for lfm3-speech, the same ids), and for each seed HF's own generation as tools/lfm3_audio_trace.py runs it (BF16,
SDPA; torch.manual_seed(seed) before generate; generate(generation_mode, do_sample=False, audio_temperature=0.4,
audio_repetition_penalty=1.2)), uninstrumented: OUT_DIR/hf/CASE-sSEED/{summary.json, response.wav, response.txt}, the
layout tools/lfm3_asr.py reads.
Usage (lfm3-omni-dev): lfm3_speech_hf.py MODEL_DIR PROMPTS.json OUT_DIR [float32|depth32|depth16|requests]
  float32: the whole model (backbone, depth decoder) in FP32 with TF32 off, the codec as shipped: the true model's
  behaviour, as a referee between BF16 engines. Cases then go to OUT_DIR/hf-fp32/.
  depth32: the backbone in BF16, the depth decoder in FP32 (its input hidden state cast up): OUT_DIR/hf-depth32/.
  depth16: the backbone in FP32, the depth decoder in BF16 (its input cast down): OUT_DIR/hf-depth16/.
  requests: write OUT_DIR/requests only (for native-only rounds); no model is loaded.
  LFM3_SLIDING_WINDOW=N (environment): override the backbone's sliding window (256); the engine dir gets "-wN".
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main(model_dir, prompts, out_dir, dtype="bfloat16"):
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    spec = json.load(open(prompts))
    out = Path(out_dir)
    (out / "requests").mkdir(parents=True, exist_ok=True)
    variant = dtype
    if variant == "requests":
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        for c in spec["cases"]:
            conversation = [{"role": "system", "content": spec["systems"][c["system"]]}, {"role": "user", "content": c["user"]}]
            ids = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt")["input_ids"][0].tolist()
            for seed in spec["seeds"]:
                json.dump(dict(prompt_ids=ids, mode=c["mode"], max_new_tokens=c["max_new_tokens"], seed=seed,
                               conversation=conversation), open(out / "requests" / f"{c['case']}-s{seed}.req.json", "w"))
        return
    backbone = {"bfloat16": "bfloat16", "float32": "float32", "depth32": "bfloat16", "depth16": "float32"}[variant]
    if "float32" in (backbone, variant) or variant == "depth32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    window = os.environ.get("LFM3_SLIDING_WINDOW")
    if window:
        config.text_config.sliding_window = int(window)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, config=config, dtype=getattr(torch, backbone), device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    assert model.config.text_config.sliding_window == config.text_config.sliding_window
    if variant in ("depth32", "depth16"):
        depth = model.model.depth_decoder.to(torch.float32 if variant == "depth32" else torch.bfloat16)
        cast = next(depth.parameters()).dtype
        depth.register_forward_pre_hook(lambda m, args: (args[0].to(cast), *args[1:]))
    engine = {"bfloat16": "hf", "float32": "hf-fp32", "depth32": "hf-depth32", "depth16": "hf-depth16"}[variant]
    engine += f"-w{window}" if window else ""
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    for c in spec["cases"]:
        conversation = [{"role": "system", "content": spec["systems"][c["system"]]}, {"role": "user", "content": c["user"]}]
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to("cuda")
        ids = inputs["input_ids"][0].tolist()
        for seed in spec["seeds"]:
            name = f"{c['case']}-s{seed}"
            json.dump(dict(prompt_ids=ids, mode=c["mode"], max_new_tokens=c["max_new_tokens"], seed=seed,
                           conversation=conversation), open(out / "requests" / f"{name}.req.json", "w"))
            d = out / engine / name
            d.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=c["max_new_tokens"], generation_mode=c["mode"], do_sample=False,
                                   audio_temperature=0.4, audio_repetition_penalty=1.2)
                slots = o.sequences[0, len(ids):o.sequence_lengths[0]].cpu()
                text = processor.decode(slots, skip_special_tokens=True)
                n = int(o.audio_lengths[0])
                wav = codec.decode(o.audio_codes[:1, :, :n]).audio_values[0].squeeze(0).float().cpu().numpy() if n else np.zeros(0, np.float32)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            sf.write(d / "response.wav", wav, 24_000)
            (d / "response.txt").write_text(text + "\n")
            (d / "summary.json").write_text(json.dumps(dict(
                case=name, mode=c["mode"], seed=seed, engine=engine, conversation=conversation, text=text,
                audio_seconds=len(wav) / 24_000, frames=n, slots=len(slots), generate_seconds=seconds,
                real_time_factor=(len(wav) / 24_000) / seconds if seconds else 0), indent=1) + "\n")
            print(name, c["mode"], f"{len(wav) / 24_000:.1f}s audio in {seconds:.1f}s:", text[:100].replace("\n", " "), flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
