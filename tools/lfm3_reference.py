"""Pinned Transformers oracle and explicit token-ID requests for native validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32: a higher-precision anchor for measuring BF16 implementations (G4/G5); traces are still saved as BF16")
    p.add_argument("--prompt", default="Hypothetically, might reflective recursion be a function of cognition?")
    p.add_argument("--length", type=int, help="Repeat/truncate content tokens to this length (boundary tests)")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="Three warmed manual-greedy runs with traces disabled")
    p.add_argument("--input-ids-file", type=Path, help="Replay input_ids from a saved native request")
    p.add_argument("--trace-first-layers", type=int, default=24, help="Limit diagnostic activation capture, not model execution")
    args = p.parse_args()
    # Never mix runs: stale traces (e.g. an earlier run's .logits.f32 or later steps) would be read as this run's.
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory {args.output} is not empty")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dtype == "float32":
        # A precision anchor: no TF32 anywhere (cuDNN defaults to allowing it).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    model = AutoModelForMultimodalLM.from_pretrained(
        str(args.checkpoint), dtype=getattr(torch, args.dtype), device_map="cuda",
        trust_remote_code=True, attn_implementation=args.attention,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    backbone = model.model.language_model
    inputs = processor.apply_chat_template([
        {"role": "system", "content": "Respond with text."},
        {"role": "user", "content": args.prompt},
    ], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    if args.input_ids_file:
        if args.length:
            raise ValueError("input-ids-file and length are mutually exclusive")
        saved = json.loads(args.input_ids_file.read_text())["input_ids"]
        if not isinstance(saved, list) or not 1 <= len(saved) <= 4096 or any(
                type(x) is not int or not 0 <= x < backbone.config.vocab_size for x in saved):
            raise ValueError("invalid replay input_ids")
        input_ids = torch.tensor([saved], device="cuda")
    if not 1 <= args.trace_first_layers <= len(backbone.layers):
        raise ValueError("invalid trace-first-layers")
    if args.length:
        content = processor.tokenizer.encode(args.prompt, add_special_tokens=False)
        input_ids = torch.tensor([(content * (args.length // len(content) + 1))[:args.length]], device="cuda")
    trace = args.output / "reference"
    trace.mkdir(exist_ok=True)
    step = 0
    trace_enabled = not args.no_trace
    shapes = {}

    def save(name, tensor):
        if not trace_enabled:
            return
        tensor = tensor.detach().contiguous().cpu()
        key = f"{step}.{name}"
        if name == "logits" and tensor.dtype == torch.float32:
            # An FP32 anchor's logits are kept exactly too: BF16 rounding of logits (~1/16 at |z|~21)
            # would hide differences between implementations' actual outputs (G4).
            (trace / (key + ".f32")).write_bytes(tensor.numpy().tobytes())
        tensor = tensor.bfloat16()
        shapes[key] = list(tensor.shape)
        (trace / (key + ".bf16")).write_bytes(tensor.view(torch.uint16).numpy().tobytes())

    def hook(name):
        def capture(module, inputs, output):
            save(name, output[0] if isinstance(output, tuple) else output)
        return capture

    # Capture actual eager attention boundaries without replacing its arithmetic.
    attention_layer = next(layer for layer in backbone.layers if layer.is_attention_layer)
    implementation = sys.modules[type(attention_layer.self_attn).__module__]
    original_attention = implementation.eager_attention_forward

    def traced_attention(module, query, key, value, *positional, **keyword):
        result = original_attention(module, query, key, value, *positional, **keyword)
        if module.layer_idx < args.trace_first_layers:
            prefix = f"layers.{module.layer_idx}.self_attn."
            save(prefix + "q_rotary", query.transpose(1, 2))
            save(prefix + "k_rotary", key[:, :, -query.shape[-2]:].transpose(1, 2))
            save(prefix + "attended", result[0])
            save(prefix + "probabilities", result[1])
        return result

    implementation.eager_attention_forward = traced_attention

    handles = [backbone.embed_tokens.register_forward_hook(hook("embed")),
               backbone.embedding_norm.register_forward_hook(hook("final_norm")),
               model.lm_head.register_forward_hook(hook("logits"))]
    for i, layer in enumerate(backbone.layers):
        if i >= args.trace_first_layers:
            continue
        prefix = f"layers.{i}."
        for attr in ("operator_norm", "ffn_norm", "feed_forward"):
            handles.append(getattr(layer, attr).register_forward_hook(hook(prefix + attr)))
        handles.append(layer.register_forward_hook(hook(prefix + "output")))
        operator = layer.self_attn if layer.is_attention_layer else layer.conv
        handles.append(operator.register_forward_hook(hook(prefix + "operator")))
        for name in ("w1", "w3"):
            handles.append(getattr(layer.feed_forward, name).register_forward_hook(hook(prefix + "feed_forward." + name)))

        def before_w2(module, inputs, name=prefix + "feed_forward.product"):
            save(name, inputs[0])
        handles.append(layer.feed_forward.w2.register_forward_pre_hook(before_w2))
        if not layer.is_attention_layer:
            handles.append(operator.in_proj.register_forward_hook(hook(prefix + "conv.in_proj")))

            def before_projection(module, inputs, name=prefix + "conv.product"):
                save(name, inputs[0])
            handles.append(operator.out_proj.register_forward_pre_hook(before_projection))
    suppress = sorted(set(model._placeholder_token_ids + [model.config.mixed_start_token_id, model.config.audio_start_token_id]))
    eos = backbone.config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    eos = sorted(set(eos + [model.config.text_end_token_id]))

    def pick(logits):
        logits = logits[:, -1].clone()
        logits[:, suppress] = -float("inf")
        return int(logits.argmax(-1))

    generated = []
    times = []
    cache = None
    with torch.inference_mode():
        for step in range(args.tokens):
            current = input_ids if step == 0 else torch.tensor([[generated[-1]]], device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            cache = out.past_key_values
            generated.append(pick(out.logits))
            if generated[-1] in eos:
                break
    for h in handles:
        h.remove()
    implementation.eager_attention_forward = original_attention
    trace_enabled = False
    request = dict(input_ids=input_ids[0].tolist(), teacher_ids=generated[:-1],
                   suppress_tokens=suppress, stop_tokens=eos, capacity=min(4096, max(512, input_ids.shape[1] + args.tokens)),
                   check_reset=True, trace_directory=str(args.output / "native"))
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output / "shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
    metadata = dict(backend="Transformers", attention=args.attention, dtype=args.dtype,
                    precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                   float32_matmul_precision=torch.get_float32_matmul_precision()),
                    torch=torch.__version__,
                    transformers=transformers.__version__, generated_ids=generated,
                    text=processor.decode(generated, skip_special_tokens=True), input_ids=input_ids[0].tolist(),
                    suppress_tokens=suppress, stop_tokens=eos,
                    traced_call_seconds=times, timing_warning="trace copies included; not a throughput benchmark",
                    source_sha256={name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
                                   for name in ["config.json", "modeling_lfm3.py", "lfm3_conv.py"]})
    if args.benchmark:
        samples = []
        with torch.inference_mode():
            for run in range(4):
                tokens = []
                cache = None
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                prefill_end = time.perf_counter()
                cache = out.past_key_values
                for t in range(args.tokens):
                    token = pick(out.logits)
                    tokens.append(token)
                    if token in eos or t == args.tokens - 1:
                        break
                    out = model(input_ids=torch.tensor([[token]], device="cuda"),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                torch.cuda.synchronize()
                end = time.perf_counter()
                if run:
                    samples.append(dict(prefill_seconds=prefill_end-start, decode_seconds=end-prefill_end,
                                        decode_forward_calls=len(tokens)-1, generated_ids=tokens))
        metadata["benchmark"] = dict(warmup_runs=1, samples=samples,
                                      timing_scope="synchronized manual greedy; no traces; excludes loading, reset, decoding text")
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ["attention", "generated_ids", "text"]}, indent=2))


if __name__ == "__main__":
    main()
