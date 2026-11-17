#!/usr/bin/env python3
"""Red-team of the FP64 codec anchor: do the FP32-island widening patches take effect in the FP64 run?

Decodes one case on the CPU in FP64 without and with lfm3_codec_anchor.widen_fp32_islands(). The patches are
effective iff the two differ. Also reports each against the native decode, and checks that the patched
decode reproduces the anchor's saved fp64 waveform bit for bit.
Usage: lfm3_codec_widening_check.py MODEL_DIR CASE_DIR ANCHOR_DIR NATIVE_WHOLE_F32
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfm3_codec_anchor as anchor  # noqa: E402


def main(model_dir, case_dir, anchor_dir, native_path):
    from transformers import AutoProcessor
    codec = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True).audio_tokenizer.eval()
    anchor._CODEC = codec
    codes = torch.load(Path(case_dir) / "sequences.pt")["audio_codes"]
    fp64 = copy.deepcopy(codec).to("cpu").double()
    with torch.inference_mode():
        plain = anchor.decode_steps(fp64, codes).numpy()
        anchor.widen_fp32_islands()
        widened = anchor.decode_steps(fp64, codes).numpy()
    saved = np.load(Path(anchor_dir) / f"{Path(case_dir).name}.fp64.npy")
    native = np.fromfile(native_path, np.float32)
    r = dict(case=Path(case_dir).name, widening_changes_fp64=not np.array_equal(plain, widened),
             unwidened_vs_widened_db=anchor.snr_db(plain, widened),
             widened_equals_saved_anchor=bool(np.array_equal(widened, saved)),
             native_vs_widened_db=anchor.snr_db(native, widened), native_vs_unwidened_db=anchor.snr_db(native, plain))
    print(json.dumps(r))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
