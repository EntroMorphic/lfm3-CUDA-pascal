#!/usr/bin/env python3
"""Depth-decoder anchors for A1/B1 (journal/lfm3-tts-quality-bar.md): HF's depth decoder on the reference's own
passes (lfm3_audio_trace.py's frames.pt: LM hidden state, revealed mask and codes of each of a frame's 3 passes),
codes forced, no RNG. Four precisions:
  sdpa   BF16, SDPA, one pass per call as in generation; must equal the trace's logits bit for bit.
  eager  BF16, eager attention, one pass per call.
  fp32   FP32 parameters, TF32 off, GPU, all passes batched.
  fp64   FP64 parameters on the CPU, all passes batched, with the FP32 islands widened (both RMSNorm conventions,
         rotary tables and their application).
Writes OUT/CASE.{sdpa,eager,fp32}.f32.npy and OUT/CASE.fp64.npy (logits [frames, 3, 12, 2017]), and the native
tool's inputs OUT/CASE.hidden.bf16 ([frames, 2048] raw BF16), .codes.i32 and .masked.u8 ([frames, 3, 12]).
Usage (Thor, lfm3-omni-dev): lfm3_depth_anchor.py MODEL_DIR TRACE_DIR OUT_DIR CASE [CASE ...]
"""
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfm3_codec_anchor as codec_anchor  # noqa: E402


def widen_omni_norm(depth):
    omni = importlib.import_module(type(depth).__module__)

    def forward(self, h):
        dtype = h.dtype
        w = torch.float64 if dtype == torch.float64 else torch.float32
        h = h.to(w)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (h * self.weight).to(dtype)
    omni.Lfm3OmniRMSNorm.forward = forward


def set_attention(depth, implementation):
    depth.model.config._attn_implementation = implementation


def main(model_dir, trace_dir, out_dir, cases):
    from transformers import AutoModelForMultimodalLM
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    depth = model.model.depth_decoder
    summary = {}
    inputs = {}
    for case in cases:
        f = torch.load(Path(trace_dir) / case / "frames.pt")
        hidden = f["hidden"].reshape(f["hidden"].shape[0], -1)          # [F, 2048] BF16
        codes, masked, golden = f["pass_codes"], f["pass_masked"], f["pass_logits"]
        assert hidden.dtype == torch.bfloat16 and codes.shape[1:] == (3, 12) and golden.shape[1:] == (3, 12, 2017)
        inputs[case] = (hidden, codes, masked, golden)
        hidden.view(torch.int16).numpy().astype("<i2").tofile(out / f"{case}.hidden.bf16")
        codes.to(torch.int32).numpy().astype("<i4").tofile(out / f"{case}.codes.i32")
        masked.to(torch.uint8).numpy().tofile(out / f"{case}.masked.u8")

    def per_pass(hidden, codes, masked):
        F = hidden.shape[0]
        result = torch.empty(F, 3, 12, 2017, dtype=torch.bfloat16)
        for i in range(F):
            for p in range(3):
                result[i, p] = depth(hidden[i:i + 1].cuda(), codes[i, p][None].cuda(), masked[i, p][None].cuda())[0].cpu()
        return result

    def batched(module, hidden, codes, masked, device, dtype):
        F = hidden.shape[0]
        h = hidden[:, None].expand(F, 3, -1).reshape(3 * F, -1).to(device, dtype)
        logits = module(h, codes.reshape(3 * F, 12).to(device), masked.reshape(3 * F, 12).to(device))
        return logits.reshape(F, 3, 12, 2017).cpu()

    with torch.inference_mode():
        for case, (hidden, codes, masked, golden) in inputs.items():
            sdpa = per_pass(hidden, codes, masked)
            summary[case] = dict(frames=int(hidden.shape[0]), sdpa_equals_trace=bool(torch.equal(sdpa, golden)))
            np.save(out / f"{case}.sdpa.f32.npy", sdpa.float().numpy())
        set_attention(depth, "eager")
        for case, (hidden, codes, masked, _) in inputs.items():
            eager = per_pass(hidden, codes, masked)
            summary[case]["eager_differs_from_sdpa"] = not np.array_equal(eager.float().numpy(),
                                                                         np.load(out / f"{case}.sdpa.f32.npy"))
            np.save(out / f"{case}.eager.f32.npy", eager.float().numpy())
        set_attention(depth, "sdpa")
        codec_anchor.widen_fp32_islands(depth.model)
        widen_omni_norm(depth)
        for case, (hidden, codes, masked, golden) in inputs.items():  # the widening must not change BF16
            again = depth(hidden[:1].cuda(), codes[0, 0][None].cuda(), masked[0, 0][None].cuda())[0].cpu()
            summary[case]["bf16_unchanged_by_widening"] = bool(torch.equal(again, golden[0, 0]))
        fp32 = copy.deepcopy(depth).float()
        for case, (hidden, codes, masked, _) in inputs.items():
            np.save(out / f"{case}.fp32.f32.npy", batched(fp32, hidden, codes, masked, "cuda", torch.float32).numpy())
        del fp32
        fp64 = copy.deepcopy(depth).to("cpu").double()
        for case, (hidden, codes, masked, _) in inputs.items():
            np.save(out / f"{case}.fp64.npy", batched(fp64, hidden, codes, masked, "cpu", torch.float64).numpy())
            print(case, json.dumps(summary[case]), flush=True)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    (out / "anchor.json").write_text(json.dumps(dict(environment=env, results=summary), indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
