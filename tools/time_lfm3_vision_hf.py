#!/usr/bin/env python3
"""validation/images-v1, speed against HF: HF's time per stage on lfm3-speech image requests (the "messages" of
tools/thor/vision_e2e_round.sh's requests), LFM3-Omni-2B in BF16 on the GPU (SDPA, greedy, generation_mode "text",
each request's max_new_tokens): rule 3's reference configuration.
Per request and pass: front end (the processor call: image loading, resize, patchify, tokenizing; CPU), image encode
alone (get_image_features, a separate call), time to first token (generate with a streamer: from the call to the
first new token), decode (first new token to the last), and the tokens generated. Every stage ends with a CUDA
synchronize. One untimed pass over all requests, then PASSES timed passes; one JSONL row per request and timed pass.
Also, with --max-image, the encoder alone on a 2048 x 1024 noise image (8,192 patches), 1 untimed + PASSES timed.
Usage: time_lfm3_vision_hf.py MODEL_DIR OUT_JSONL PASSES [--max-image] REQUEST.json...
"""
import json
import platform
import sys
import time
from pathlib import Path


def main(argv):
    max_image = "--max-image" in argv
    argv = [a for a in argv if a != "--max-image"]
    if len(argv) < 5:
        sys.exit(__doc__)
    model_dir, out_jsonl, passes, requests = argv[1], argv[2], int(argv[3]), argv[4:]
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from transformers.generation.streamers import BaseStreamer

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation="sdpa", trust_remote_code=True).eval()
    sync = torch.cuda.synchronize

    class FirstToken(BaseStreamer):
        def __init__(self):
            self.calls, self.first = 0, None

        def put(self, value):
            self.calls += 1
            if self.calls == 2 and self.first is None:  # the first put is the prompt; the second, the first new token
                sync()
                self.first = time.perf_counter()

        def end(self):
            pass

    def run(req):
        t0 = time.perf_counter()
        inputs = processor.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            sync()
            t2 = time.perf_counter()
            streamer = FirstToken()
            t3 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=req["max_new_tokens"], generation_mode="text",
                                 do_sample=False, streamer=streamer)
            sync()
            t4 = time.perf_counter()
        n = inputs["input_ids"].shape[1]
        tokens = int(out.sequence_lengths[0]) - n
        first = streamer.first if streamer.first is not None else t4
        return dict(front_end_seconds=t1 - t0, encode_seconds=t2 - t1, first_token_seconds=first - t3,
                    decode_seconds=t4 - first, generate_seconds=t4 - t3, end_to_end_seconds=(t1 - t0) + (t4 - t3),
                    prompt_tokens=n, new_tokens=tokens)

    env = dict(machine=platform.machine(), transformers=transformers.__version__, torch=torch.__version__,
               device=torch.cuda.get_device_name(), dtype="bfloat16", attn="sdpa", passes=passes)
    reqs = {Path(p).stem: json.loads(Path(p).read_text()) for p in requests}
    with open(out_jsonl, "w") as f:
        f.write(json.dumps(dict(env=env)) + "\n")
        for name, req in reqs.items():  # untimed pass
            run(req)
        for k in range(passes):
            for name, req in reqs.items():
                row = dict(name=name, pass_=k, **run(req))
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)
        if max_image:
            rng = np.random.default_rng(2048)
            im = Image.fromarray(rng.integers(0, 256, (1024, 2048, 3), dtype=np.uint8))
            msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "Describe."}]}]
            inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                   return_tensors="pt").to("cuda")
            pv = inputs["pixel_values"].to(torch.bfloat16)
            grid = inputs["image_grid_thw"]
            for k in range(passes + 1):
                sync()
                t = time.perf_counter()
                with torch.no_grad():
                    model.model.get_image_features(pv, grid)
                sync()
                if k:
                    row = dict(name="max-image-encode", pass_=k - 1, patches=int(pv.shape[0]),
                               encode_seconds=time.perf_counter() - t)
                    f.write(json.dumps(row) + "\n")
                    print(row, flush=True)


if __name__ == "__main__":
    main(sys.argv)
