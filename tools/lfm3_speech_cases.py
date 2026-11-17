#!/usr/bin/env python3
"""Package lfm3-speech outputs as case directories for tools/lfm3_asr.py and listening: CASE/summary.json (case,
mode, seed, conversation, the emitted text decoded, audio seconds, frame and slot counts, timings), CASE/response.wav,
CASE/response.txt. The conversation comes from the request (tools/lfm3_speech_hf.py wrote it with the prompt ids).
Usage (lfm3-omni-dev): lfm3_speech_cases.py MODEL_DIR REQUEST_DIR NATIVE_DIR OUT_DIR NAME [NAME ...]
  REQUEST_DIR/NAME.req.json and NATIVE_DIR/NAME.{json,wav} (lfm3-speech's output stem NATIVE_DIR/NAME)."""
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer

AUDIO_OUT = 125032


def main(model, req_dir, native_dir, out, *names):
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    for name in names:
        req = json.load(open(Path(req_dir) / f"{name}.req.json"))
        nat = json.load(open(Path(native_dir) / f"{name}.json"))
        text = tok.decode([t for t in nat["slots"] if t != AUDIO_OUT], skip_special_tokens=True)
        d = Path(out) / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(native_dir) / f"{name}.wav", d / "response.wav")
        (d / "response.txt").write_text(text + "\n")
        keep = ("frames", "raw_frames", "decode_steps", "prefill_seconds", "generate_seconds", "real_time_factor",
                "lm_step_seconds", "depth_seconds", "sample_host_seconds", "codec_seconds")
        (d / "summary.json").write_text(json.dumps(dict(
            case=name, mode=req["mode"], seed=req["seed"], engine="native", conversation=req["conversation"], text=text,
            audio_seconds=nat["audio_seconds"], slots=len(nat["slots"]), **{k: nat[k] for k in keep}), indent=1) + "\n")
        print(name, req["mode"], f"{nat['audio_seconds']:.1f}s audio in {nat['generate_seconds']:.1f}s:", text[:100].replace("\n", " "), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
