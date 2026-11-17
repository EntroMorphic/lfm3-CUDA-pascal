#!/usr/bin/env python3
"""Stage 0 of the images round: Hugging Face reference dumps for the LFM3-Omni-2B vision path (CPU, no GPU needed).

For each case, in OUT_DIR/<dtype>/<case>/: the processor's outputs (input_ids, pixel_values, image_grid_thw), the
vision helpers' outputs (position ids, attention seqlens, frame shapes), the rotary cos/sin, the patchifier, the
embedding norm, every one of the 14 encoder layers, and the projector's four stages (layer norm, linear 1, GELU,
linear 2). Every tensor is saved as float32 .npy (BF16 runs are upcast exactly) or int64 .npy, with a manifest of
shapes and sha256 sums.

The float32 run also does the end to end: the last-position logits of the prefill, and a greedy text reply
(GEN_TOKENS tokens), the coherency reference.

Runs once in float32 (the gold reference) and once in bfloat16 (what HF runs on a GPU: the port's error budget is
measured against the distance between the two). Needs transformers 5.20.2 (the model card asks for >= 5.20.2) and
torchvision.
Usage: dump_lfm3_vision_reference.py MODEL_DIR OUT_DIR [GEN_TOKENS]
       dump_lfm3_vision_reference.py MODEL_DIR OUT_DIR --images NAME=PATH...
         (held-out images, validation/images-v1 rule 2: the vision stages only, no generation, FP32 then BF16; each
         image loaded as HF's load_image loads it, one image per case, the DESCRIBE prompt)
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SKLEARN_IMAGES = Path(__import__("sklearn").__file__).parent / "datasets" / "images"
WALLPAPERS = Path("/usr/share/backgrounds")
DESCRIBE = "Describe the provided image."
GROUNDED = "Answer questions about the provided image. Base every answer only on what can be seen."


def make_images(dst):
    """The test images, decoded once and saved losslessly (PNG), so the port reads the same pixels."""
    dst.mkdir(parents=True, exist_ok=True)
    china = Image.open(SKLEARN_IMAGES / "china.jpg").convert("RGB")
    flower = Image.open(SKLEARN_IMAGES / "flower.jpg").convert("RGB")
    rgba = flower.copy()
    rgba.putalpha(Image.linear_gradient("L").resize(flower.size))
    rng = np.random.default_rng(0)
    imgs = {
        "china": china,                                                      # 640x427: rounding to 32 (427 -> 416)
        "flower": flower,
        "cherry": Image.open(WALLPAPERS / "Cherry_Tree_in_Lakones_by_elenastravoravdi.jpg").convert("RGB"),  # 4K
        "mirror": Image.open(WALLPAPERS / "Mirror_by_Uday_Nakade.jpg").convert("RGB"),  # 3866x2175, odd
        "small": china.resize((200, 133), Image.BICUBIC),                    # below min pixels: upscaled
        "wide": china.crop((0, 200, 640, 248)),                              # 640x48, aspect 13.3
        "tall": flower.crop((300, 0, 340, 427)),                             # 40x427
        "noise": Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)),
        "gray": china.convert("L"),                                          # convert_rgb path
        "rgba": rgba,                                                        # alpha path
    }
    out = {}
    for name, im in imgs.items():
        p = dst / f"{name}.png"
        im.save(p)
        out[name] = p
    return out


def cases(images, held_out=False):
    """(name, conversation, generate end to end?)"""
    if held_out:
        return [(name, [{"role": "system", "content": DESCRIBE},
                        {"role": "user", "content": [{"type": "image", "image": str(p)}]}], False)
                for name, p in images.items()]
    def one(img, system=DESCRIBE, text=None):
        content = [{"type": "image", "image": str(images[img])}]
        if text:
            content.append({"type": "text", "text": text})
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]
    two = [{"role": "system", "content": GROUNDED},
           {"role": "user", "content": [{"type": "image", "image": str(images["china"])},
                                        {"type": "image", "image": str(images["flower"])},
                                        {"type": "text", "text": "What is different between these two images?"}]}]
    return [
        ("china", one("china"), True),
        ("flower", one("flower", GROUNDED, "What kind of flower is this, and what color is it?"), True),
        ("cherry", one("cherry"), True),
        ("mirror", one("mirror"), False),
        ("small", one("small"), True),
        ("wide", one("wide"), False),
        ("tall", one("tall"), False),
        ("noise", one("noise"), False),
        ("gray", one("gray"), True),
        ("rgba", one("rgba"), False),
        ("two", two, True),
    ]


class Recorder:
    def __init__(self, model):
        self.t = {}
        self.handles = []
        vt = model.model.vision_tower
        proj = model.model.multi_modal_projector
        self.idx = 0  # images run through the tower one call at a time; the key carries the call index

        def hook(name):
            def f(_m, _inp, out):
                o = out[0] if isinstance(out, tuple) and name != "rotary" else out
                self.t[f"{self.idx}.{name}"] = o
            return f
        self.handles.append(vt.patchifier.register_forward_hook(hook("patchifier")))
        self.handles.append(vt.embedding_norm.register_forward_hook(hook("embedding_norm")))
        self.handles.append(vt.rotary_pos_emb.register_forward_hook(hook("rotary")))
        for i, layer in enumerate(vt.layers):
            self.handles.append(layer.register_forward_hook(hook(f"layer{i:02d}")))
        self.handles.append(proj.layer_norm.register_forward_hook(hook("proj.layer_norm")))
        self.handles.append(proj.linear_1.register_forward_hook(hook("proj.linear_1")))
        self.handles.append(proj.act.register_forward_hook(hook("proj.gelu")))
        self.handles.append(proj.linear_2.register_forward_hook(hook("proj.linear_2")))
        # the transformers.vision_utils helpers, as the vision module calls them
        mod = sys.modules[type(vt).__module__]
        for fn in ("get_vision_position_ids", "get_vision_attention_seqlens", "get_vision_frame_shapes"):
            orig = getattr(mod, fn)

            def wrapped(*a, _orig=orig, _fn=fn, **k):
                r = _orig(*a, **k)
                self.t[f"{self.idx}.{_fn}"] = r
                return r
            setattr(mod, fn, wrapped)


def save(arr_dir, name, value, manifest):
    if isinstance(value, (tuple, list)):
        if all(isinstance(v, int) for v in value):
            value = torch.tensor(value)
        else:
            for i, v in enumerate(value):
                save(arr_dir, f"{name}.{i}", v, manifest)
            return
    if isinstance(value, int):
        value = torch.tensor(value)
    if not torch.is_tensor(value):
        manifest[name] = dict(value=repr(value))
        return
    v = value.detach().cpu()
    a = v.float().numpy() if v.is_floating_point() else v.to(torch.int64).numpy()
    p = arr_dir / f"{name}.npy"
    np.save(p, a)
    manifest[name] = dict(shape=list(a.shape), dtype=str(a.dtype), source_dtype=str(v.dtype),
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def run(model_dir, out_dir, dtype, gen_tokens, images, processor, held_out=False):
    from transformers import AutoModelForMultimodalLM
    t0 = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=dtype, device_map="cpu",
                                                     trust_remote_code=True).eval()
    load_s = time.monotonic() - t0
    rec = Recorder(model)
    summary = dict(dtype=str(dtype), load_seconds=load_s, cases={})
    for name, conv, e2e in cases(images, held_out):
        d = out_dir / str(dtype).split(".")[-1] / name
        d.mkdir(parents=True, exist_ok=True)
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt")
        manifest = {}
        for k in ("input_ids", "pixel_values", "image_grid_thw"):
            save(d, k, inputs[k], manifest)
        rec.t.clear()
        rec.idx = 0
        t0 = time.monotonic()
        with torch.no_grad():
            feats = model.model.get_image_features(inputs["pixel_values"].to(dtype), inputs["image_grid_thw"])
        vision_s = time.monotonic() - t0
        for k, v in rec.t.items():
            save(d, k, v, manifest)
        save(d, "image_features", feats, manifest)
        info = dict(grid_thw=inputs["image_grid_thw"].tolist(), prompt_tokens=inputs["input_ids"].shape[1],
                    image_tokens=int((inputs["input_ids"] == 125030).sum()), image_features=list(feats.shape),
                    vision_seconds=vision_s,
                    prompt=processor.decode(inputs["input_ids"][0]).replace("<|image|>" * 4, "")[:600])
        if e2e and dtype == torch.float32:
            t0 = time.monotonic()
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
                out = model.generate(**inputs, max_new_tokens=gen_tokens, generation_mode="text", do_sample=False)
            n = inputs["input_ids"].shape[1]
            ids = out.sequences[0, n:out.sequence_lengths[0]]
            save(d, "prefill_last_logits", logits, manifest)
            save(d, "generated_ids", ids, manifest)
            info.update(generate_seconds=time.monotonic() - t0, top5=torch.topk(logits, 5).indices.tolist(),
                        reply=processor.decode(ids, skip_special_tokens=True))
            print(f"[{name}] {info['reply']!r}", flush=True)
        (d / "manifest.json").write_text(json.dumps(dict(info=info, tensors=manifest), indent=1))
        summary["cases"][name] = info
        print(f"{dtype} {name}: grid {info['grid_thw']} image tokens {info['image_tokens']} "
              f"vision {vision_s:.1f} s", flush=True)
    del model
    return summary


def main(model_dir, out_dir, gen_tokens=128, *named):
    import torchvision
    import transformers
    from transformers import AutoProcessor
    out = Path(out_dir)
    held_out = gen_tokens == "--images"
    if held_out:
        images = {n: Path(p) for n, p in (a.split("=", 1) for a in named)}
        gen_tokens = 0
    else:
        images = make_images(out / "images")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    torch.set_num_threads(torch.get_num_threads())
    report = dict(versions=dict(transformers=transformers.__version__, torch=torch.__version__,
                                torchvision=torchvision.__version__, python=platform.python_version(),
                                numpy=np.__version__),
                  images={n: dict(size=Image.open(p).size, mode=Image.open(p).mode,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n, p in images.items()},
                  runs=[])
    for dtype in (torch.float32, torch.bfloat16):
        report["runs"].append(run(model_dir, out, dtype, int(gen_tokens), images, processor, held_out))
        (out / "report.json").write_text(json.dumps(report, indent=1))
    print("DONE")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) and not (len(sys.argv) > 4 and sys.argv[3] == "--images"):
        sys.exit(__doc__)
    main(*sys.argv[1:])
