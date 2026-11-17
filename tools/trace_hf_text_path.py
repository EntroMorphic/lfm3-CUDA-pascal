"""Trace the Hugging Face text-generation path for LFM3-Omni-2B, end to end, as ground truth.

Run inside the reference container (lfm3-omni-dev), with Thor's ptxas for Triton:
  TRITON_PTXAS_PATH=/workspace/tools/ptxas TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \
  python /workspace/liquid-CUDA/tools/trace_hf_text_path.py --output /workspace/liquid-CUDA/build/hf-trace-v1

Records (to --output):
  prompt.json   rendered chat-template string and token ids by both paths (they must agree)
  calls.json    Python call sequence through transformers / model code / torch.nn.functional for
                the prefill forward and one decode forward (glue, in order)
  modules.json  every module call in prefill and decode: qualified name, class, input/output shapes+dtypes
  kernels.json  every GPU kernel, attributed to its enclosing module and ATen op, with library guess
  result.json   generated ids and text (must match a plain run) plus versions and settings
Read-only with respect to the model and the repository; writes only under --output.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROMPT = "Hypothetically, might reflective recursion be a function of cognition?"
CONVERSATION = [{"role": "system", "content": "Respond with text."}, {"role": "user", "content": PROMPT}]


def describe(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "device": x.device.type}
    if isinstance(x, (list, tuple)):
        return [describe(v) for v in x[:4]]
    if isinstance(x, dict):
        return {k: describe(v) for k, v in list(x.items())[:6]}
    if x is None or isinstance(x, (int, float, bool, str)):
        return x
    return type(x).__name__


def library_of(kernel_name):
    n = kernel_name
    if n.startswith("nvjet") or "cublas" in n.lower() or "gemv" in n or "gemmk" in n.lower() or "splitKreduce" in n:
        return "cuBLAS/cuBLASLt"
    if "flash" in n.lower() or "fmha" in n.lower() or "pytorch_flash" in n:
        return "flash/mem-efficient attention (ATen SDPA)"
    if "cudnn" in n.lower() or n.startswith("sm") and "fprop" in n:
        return "cuDNN"
    if n.startswith("triton_") or n.startswith("_prefill") or n.startswith("_decode"):
        return "Triton"
    if "at::native" in n or "at_cuda_detail" in n or "elementwise" in n or "reduce_kernel" in n or "vectorized" in n:
        return "ATen native CUDA"
    return "other"


class CallRecorder:
    """sys.setprofile recorder of Python calls into transformers, the model's own code, and torch.nn.functional."""

    def __init__(self, roots):
        self.roots = roots
        self.calls = []
        self.depth = 0

    def __call__(self, frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename
            if any(r in fn for r in self.roots):
                short = fn.split("site-packages/")[-1].split("LFM3-Omni-2B/")[-1].split("LFM3_hyphen_Omni_hyphen_2B/")[-1]
                self.calls.append({"depth": self.depth, "file": short, "line": frame.f_code.co_firstlineno,
                                   "function": frame.f_code.co_qualname if hasattr(frame.f_code, "co_qualname") else frame.f_code.co_name})
            self.depth += 1
        elif event == "return":
            self.depth = max(0, self.depth - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/workspace/models/LFM3-Omni-2B")
    p.add_argument("--max-new-tokens", type=int, default=48)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)

    # 1. Prompt: template rendering and tokenization, by both paths.
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    text = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=False)
    via_template = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=True,
                                                 return_dict=True, return_tensors="pt")
    tok = processor.tokenizer
    direct = tok(text, add_special_tokens=False)["input_ids"]
    direct_special = tok(text)["input_ids"]
    ids = via_template["input_ids"][0].tolist()
    prompt = {"rendered": text, "ids_via_apply_chat_template": ids,
              "ids_direct_add_special_tokens_false": direct, "ids_direct_default": direct_special,
              "agree_template_vs_direct_false": ids == direct, "agree_template_vs_direct_default": ids == direct_special,
              "tokens": tok.convert_ids_to_tokens(ids), "tokenizer_class": type(tok).__name__,
              "processor_class": type(processor).__name__, "input_keys": sorted(via_template.keys())}
    (a.output / "prompt.json").write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n")

    # 2. Model load (same as test_text.py).
    t0 = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    load_seconds = time.perf_counter() - t0
    lm = model.model.language_model
    dtypes = collections.Counter(str(p.dtype) for p in model.parameters())
    tied = model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()

    # Plain run first: the reference result the traced run must reproduce.
    inputs = via_template.to("cuda")
    with torch.inference_mode():
        plain = model.generate(**inputs, max_new_tokens=a.max_new_tokens, generation_mode="text", do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    plain_ids = plain.sequences[0, n_in:plain.sequence_lengths[0]].tolist()

    # 3. Module, op and kernel trace of one generate call; calls only for the first two forwards.
    module_log, stack = [], []
    names = {m: n for n, m in model.named_modules()}
    forward_counter = {"n": 0}

    def pre(mod, args, kwargs):
        name = names.get(mod, "?")
        if mod is lm:
            forward_counter["n"] += 1
        rf = torch.autograd.profiler.record_function(f"MOD::{name}")
        rf.__enter__()
        stack.append(rf)
        if forward_counter["n"] <= 2:
            module_log.append({"forward": forward_counter["n"], "module": name, "class": type(mod).__name__,
                               "inputs": describe(list(args)), "kwargs": {k: describe(v) for k, v in kwargs.items()
                                                                          if isinstance(v, torch.Tensor)}})

    def post(mod, args, kwargs, out):
        rf = stack.pop()
        rf.__exit__(None, None, None)
        if forward_counter["n"] <= 2 and module_log and module_log[-1]["module"] == names.get(mod) and "output" not in module_log[-1]:
            module_log[-1]["output"] = describe(out if isinstance(out, torch.Tensor) else getattr(out, "last_hidden_state", out))

    hooks = [m.register_forward_pre_hook(pre, with_kwargs=True) for m in model.modules()]
    hooks += [m.register_forward_hook(post, with_kwargs=True) for m in model.modules()]

    # trust_remote_code modules execute from transformers_modules/<model>/<hash>/, not the model folder
    recorder = CallRecorder(["transformers/", "transformers_modules/", "LFM3-Omni-2B/", "torch/nn/functional.py"])
    orig_forward = lm.forward

    def traced_forward(*args, **kwargs):
        if forward_counter["n"] <= 2:  # the pre-hook has already counted this forward: 1 = prefill, 2 = first decode
            sys.setprofile(recorder)
            try:
                return orig_forward(*args, **kwargs)
            finally:
                sys.setprofile(None)
        return orig_forward(*args, **kwargs)

    lm.forward = traced_forward
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        traced = model.generate(**inputs, max_new_tokens=4, generation_mode="text", do_sample=False)
        torch.cuda.synchronize()
    lm.forward = orig_forward
    for h in hooks:
        h.remove()
    traced_ids = traced.sequences[0, n_in:traced.sequence_lengths[0]].tolist()

    # Attribute kernels to the innermost module range and the ATen op that launched them.
    events = prof.events()
    kernels = []
    for ev in events:
        if not getattr(ev, "kernels", None):
            continue
        mod, op = None, ev.name
        parent = ev.cpu_parent
        while parent is not None:
            if parent.name.startswith("MOD::"):
                mod = parent.name[5:]
                break
            parent = parent.cpu_parent
        for k in ev.kernels:
            kernels.append({"module": mod, "aten_op": op, "input_shapes": ev.input_shapes, "kernel": k.name,
                            "library": library_of(k.name), "duration_us": k.duration})
    by_kind = collections.defaultdict(lambda: {"count": 0, "us": 0.0})
    for k in kernels:
        key = (k["library"], k["kernel"][:120])
        by_kind[key]["count"] += 1
        by_kind[key]["us"] += k["duration_us"]
    summary = sorted(({"library": l, "kernel": n, **v} for (l, n), v in by_kind.items()), key=lambda r: -r["us"])
    (a.output / "kernels.json").write_text(json.dumps({"kernels": kernels, "summary": summary}, indent=1) + "\n")
    (a.output / "modules.json").write_text(json.dumps(module_log, indent=1) + "\n")
    (a.output / "calls.json").write_text(json.dumps(recorder.calls, indent=0) + "\n")

    import hashlib, inspect
    sources = {}
    for cls in {type(model), type(lm), type(lm.layers[0]), type(lm.layers[0].conv), type(lm.layers[2].self_attn)}:
        f = inspect.getfile(cls)
        sources[cls.__name__] = {"file": f, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
    conv_mod = sys.modules[type(lm.layers[0].conv).__module__]
    import importlib
    lfm3_conv_mod = importlib.import_module(conv_mod.__package__ + ".lfm3_conv") if conv_mod.__package__ else None
    if lfm3_conv_mod is not None:
        f = lfm3_conv_mod.__file__
        sources["lfm3_conv"] = {"file": f, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
                                "can_use_lfm3_conv_now": bool(lfm3_conv_mod.can_use_lfm3_conv(
                                    torch.zeros(1, 2048, 2, dtype=torch.bfloat16, device="cuda"),
                                    lm.layers[0].conv.conv.weight.squeeze(1)))}
    result = {
        "runtime_sources": sources,
        "plain_generated_ids": plain_ids, "plain_text": processor.decode(plain_ids, skip_special_tokens=True),
        "traced_first4_ids": traced_ids, "traced_matches_plain_prefix": traced_ids == plain_ids[: len(traced_ids)],
        "load_seconds": load_seconds, "parameter_dtypes": dict(dtypes), "lm_head_tied_to_embeddings": tied,
        "attn_implementation": model.config._attn_implementation if hasattr(model.config, "_attn_implementation") else None,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "triton": __import__("triton").__version__, "gpu": torch.cuda.get_device_name(0)},
        "torch_flags": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
                        "sdp_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                        "sdp_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                        "sdp_math": torch.backends.cuda.math_sdp_enabled(),
                        "deterministic": torch.are_deterministic_algorithms_enabled()},
        "counts": {"module_calls_logged": len(module_log), "python_calls_logged": len(recorder.calls),
                   "kernel_launches": len(kernels)},
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("traced_matches_plain_prefix", "counts", "versions", "torch_flags")}, indent=1))
    return 0 if result["traced_matches_plain_prefix"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
