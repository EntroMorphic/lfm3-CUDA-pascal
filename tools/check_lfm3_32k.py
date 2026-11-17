"""Long-context checkpoint validation against memory-efficient Transformers SDPA.

This is a distinct tolerance-based backend comparison, not eager bitwise parity.
Keep original short-context exact gates separate. No adaptive tolerances.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-logit-error", type=float, default=0.5)
    p.add_argument("--max-relative-rms", type=float, default=0.02)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM
    from lfm3_text_frontend import load_frontend

    if not (0 < args.max_logit_error <= 0.5 and 0 < args.max_relative_rms <= 0.02):
        raise ValueError("only the established or tighter tolerances are permitted")
    binary_hash = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    frontend = load_frontend(args.checkpoint)
    content = frontend.tokenizer.encode("Context boundary verification: count 17, 25, 42; remember the earlier numbers.", add_special_tokens=False)
    if not content:
        raise ValueError("empty content")
    cases = [("prefill8192", 8192, 0), ("decode_into32768", 32767, 1), ("prefill32768", 32768, 0)]
    metadata = dict(status="running", binary_sha256=binary_hash, reference_backend="Transformers SDPA",
                    native_backend="native tiled (attention label from the binary's results)", torch=torch.__version__, transformers=transformers.__version__,
                    max_logit_error=args.max_logit_error, max_relative_rms=args.max_relative_rms,
                    cases=[], limitations="Not eager bitwise parity; finite fixed fixtures; no performance claim")
    def save():
        (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    try:
        model = AutoModelForMultimodalLM.from_pretrained(str(args.checkpoint), dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True, attn_implementation="sdpa").eval()
        references = []
        with torch.inference_mode():
            for name, length, decodes in cases:
                metadata["location"] = name + ":reference"
                save()
                directory = args.output / name
                directory.mkdir()
                prompt = (content * ((length + len(content) - 1) // len(content)))[:length]
                teacher, cache, ids, outputs = [], None, prompt, []
                for step in range(decodes + 1):
                    out = model.model.language_model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    scores = model.lm_head(out.last_hidden_state[:, -1:, :])
                    outputs.append(scores.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
                    next_id = int(scores[0, 0].argmax())
                    if step < decodes:
                        teacher.append(next_id)
                        ids = [next_id]
                    del out, scores
                for step, raw in enumerate(outputs):
                    (directory / f"{step}.reference.bf16").write_bytes(raw)
                request = dict(input_ids=prompt, teacher_ids=teacher, capacity=length+decodes,
                               experimental_long_context=True, trace_logits_only=True, check_reset=True,
                               trace_directory=str(directory / "native"))
                (directory / "request.json").write_text(json.dumps(request))
                references.append((directory, outputs, request))
                del cache
                torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        for directory, outputs, request in references:
            metadata["location"] = directory.name + ":native"
            save()
            with (directory / "native.log").open("w") as stream:
                subprocess.run([str(args.binary), "--experimental", str(args.checkpoint), str(directory / "request.json"),
                                str(directory / "native.json")], env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            result = json.loads((directory / "native.json").read_text())
            if result.get("reset_exact_logits_ok") is not True:
                raise ValueError("reset logits mismatch")
            comparisons = []
            for step, raw in enumerate(outputs):
                actual_bytes = (directory / "native" / f"{step}.logits.bf16").read_bytes()
                if len(actual_bytes) != len(raw):
                    raise ValueError("logits length mismatch")
                expected = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
                actual = torch.frombuffer(bytearray(actual_bytes), dtype=torch.bfloat16).float()
                if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                    raise ValueError("nonfinite logits")
                difference = actual - expected
                maximum = difference.abs().max().item()
                relative = (difference.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8)).item()
                match = actual.argmax().item() == expected.argmax().item()
                passed = maximum <= args.max_logit_error and relative <= args.max_relative_rms and match
                comparisons.append(dict(step=step, max_abs=maximum, relative_rms=relative,
                                        argmax_equal=match, bitwise_equal=raw==actual_bytes, passed=passed))
            entry = dict(case=directory.name, logits=comparisons, reset_exact=True,
                         passed=all(x["passed"] for x in comparisons))
            metadata["cases"].append(entry)
            save()
            if not entry["passed"]:
                raise ValueError("long-context logits acceptance failed; see summary")
        if hashlib.sha256(args.binary.read_bytes()).hexdigest() != binary_hash:
            raise ValueError("binary changed during checks")
        metadata["status"] = "passed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        save()
        raise
    save()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
