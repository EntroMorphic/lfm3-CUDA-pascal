#!/usr/bin/env python3
"""A2 check of the native codec (liquid-codec) against the HF anchors of lfm3_codec_anchor.py.

  codes CASE_DIR OUT.i32
      The reference's codes (lfm3_audio_trace.py's sequences.pt, [1, 12, T]) as raw int32 [T][12] for liquid-codec.
  stages INTERMEDIATES.pt DUMP_DIR
      Per-stage SNR of the native dump (LFM_CODEC_DUMP) against the FP64 and FP32 intermediates, and HF BF16's.
  report ANCHOR_DIR NATIVE_DIR OUT_JSON CASE [CASE ...]
      For each case: native (whole; streamed in 12-frame and 1-frame chunks) against FP64 and FP32, with HF BF16
      (SDPA and eager) as the bar. A2 holds for a case iff native's SNR against FP32 is at least each HF BF16
      variant's (the bar: native >= HF BF16), and both streamed decodes equal the whole one bit for bit.
"""
import json
import sys
from pathlib import Path

import numpy as np


def snr_db(x, ref):
    x, ref = np.asarray(x, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    if x.shape != ref.shape:
        raise ValueError(f"shape {x.shape} != {ref.shape}")
    noise = ((x - ref) ** 2).sum()
    return float("inf") if noise == 0 else float(10 * np.log10((ref ** 2).sum() / noise))


def codes(case_dir, out):
    import torch
    c = torch.load(Path(case_dir) / "sequences.pt")["audio_codes"]
    assert c.ndim == 3 and c.shape[:2] == (1, 12), c.shape
    c = c[0].T.contiguous().to(torch.int32).numpy()
    assert c.min() >= 0 and c.max() < 2016
    c.astype("<i4").tofile(out)
    print(f"{case_dir}: {c.shape[0]} frames -> {out}")


def stages(intermediates, dump_dir):
    import torch
    rec = torch.load(intermediates)
    rows = []
    for name in ["latents", "embeddings"] + [f"layer{i}" for i in range(8)] + ["final_norm", "lin"]:
        native = np.fromfile(Path(dump_dir) / f"{name}.f32", np.float32)
        fp64 = rec["fp64"][name].double().numpy().ravel()
        fp32 = rec["fp32"][name].double().numpy().ravel()
        bf16 = rec["bf16"][name].double().numpy().ravel()
        rows.append(dict(stage=name, native_vs_fp64=snr_db(native, fp64), fp32_vs_fp64=snr_db(fp32, fp64),
                         hf_bf16_vs_fp64=snr_db(bf16, fp64)))
        print("%-11s native %7.1f  fp32 %7.1f  hf-bf16 %6.1f  (dB vs fp64)" % (
            name, rows[-1]["native_vs_fp64"], rows[-1]["fp32_vs_fp64"], rows[-1]["hf_bf16_vs_fp64"]))
    return rows


def report(anchor_dir, native_dir, out_json, cases):
    anchor_dir, native_dir = Path(anchor_dir), Path(native_dir)
    anchor = json.loads((anchor_dir / "anchor.json").read_text())["results"]
    results, ok = {}, True
    for case in cases:
        fp64 = np.load(anchor_dir / f"{case}.fp64.npy")
        fp32 = np.load(anchor_dir / f"{case}.fp32.npy")
        whole = np.fromfile(native_dir / f"{case}.whole.f32", np.float32)
        hf = anchor[case]
        r = dict(native_vs_fp64=snr_db(whole, fp64), native_vs_fp32=snr_db(whole, fp32),
                 fp32_vs_fp64=hf["vs_fp64"]["fp32"]["snr_db"],
                 hf_bf16_vs_fp32=hf["vs_fp32"]["bf16"]["snr_db"], hf_eager_vs_fp32=hf["vs_fp32"]["bf16-eager"]["snr_db"],
                 hf_stream_vs_fp32=hf["vs_fp32"]["bf16-stream"]["snr_db"],
                 native_max_abs_vs_fp64=float(np.abs(whole.astype(np.float64) - fp64).max()))
        for chunk in (12, 1):
            streamed = np.fromfile(native_dir / f"{case}.chunk{chunk}.f32", np.float32)
            r[f"stream{chunk}_bitwise_equal_whole"] = bool(streamed.shape == whole.shape and
                                                           np.array_equal(streamed.view(np.uint32), whole.view(np.uint32)))
        bar = max(r["hf_bf16_vs_fp32"], r["hf_eager_vs_fp32"])
        r["a2_bar_db"] = bar
        r["a2_pass"] = bool(r["native_vs_fp32"] >= bar and r["stream12_bitwise_equal_whole"]
                            and r["stream1_bitwise_equal_whole"])
        ok &= r["a2_pass"]
        results[case] = r
        print("%-18s native/fp64 %6.1f  native/fp32 %6.1f  (fp32/fp64 %6.1f)  bar %5.1f  stream12 %s stream1 %s  %s" % (
            case, r["native_vs_fp64"], r["native_vs_fp32"], r["fp32_vs_fp64"], bar,
            r["stream12_bitwise_equal_whole"], r["stream1_bitwise_equal_whole"], "PASS" if r["a2_pass"] else "FAIL"))
    Path(out_json).write_text(json.dumps(dict(results=results, all_pass=ok), indent=1) + "\n")
    print("A2", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else (None, [])
    if cmd == "codes" and len(args) == 2:
        codes(*args)
    elif cmd == "stages" and len(args) == 2:
        stages(*args)
    elif cmd == "report" and len(args) >= 4:
        sys.exit(0 if report(args[0], args[1], args[2], args[3:]) else 1)
    else:
        sys.exit(__doc__)
