#!/usr/bin/env python3
"""HF's replies to lfm3-speech image requests (validation/images-v1 rules 3-4): each request's "messages" (image parts
{"type": "image", "image": path}, as tools/thor/vision_e2e_round.sh writes them) through the checkpoint's chat
template, LFM3-Omni-2B in BF16 on the GPU (attn_implementation "sdpa"), greedy, generation_mode "text", the request's
max_new_tokens; the same settings as tools/run_lfm3_vision_bench_hf.py (rule 3's reference).
Writes one JSONL row per request: name (the request file's stem), prompt_ids, generated_ids, reply, seconds; the
first row records the environment.
Usage: run_lfm3_vision_replies_hf.py MODEL_DIR OUT_JSONL REQUEST.json...
"""
import json
import platform
import sys
import time
from pathlib import Path


def main(model_dir, out_jsonl, *requests):
    import PIL
    import torch
    import torchvision
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    t0 = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation="sdpa", trust_remote_code=True).eval()
    env = dict(machine=platform.machine(), transformers=transformers.__version__, torch=torch.__version__,
               torchvision=torchvision.__version__, pillow=PIL.__version__, cuda=torch.version.cuda,
               device=torch.cuda.get_device_name(), dtype="bfloat16", attn="sdpa",
               load_seconds=round(time.monotonic() - t0, 1))
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(dict(env=env)) + "\n")
        for path in requests:
            req = json.loads(Path(path).read_text())
            t = time.monotonic()
            inputs = processor.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=req["max_new_tokens"], generation_mode="text",
                                     do_sample=False)
            n = inputs["input_ids"].shape[1]
            ids = gen.sequences[0, n:gen.sequence_lengths[0]].tolist()
            rec = dict(name=Path(path).stem, prompt_ids=inputs["input_ids"][0].tolist(), generated_ids=ids,
                       reply=processor.decode(ids, skip_special_tokens=True), seconds=round(time.monotonic() - t, 3))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{rec['name']}] {rec['reply']!r}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
