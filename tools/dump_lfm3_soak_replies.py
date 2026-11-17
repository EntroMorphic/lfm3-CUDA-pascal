#!/usr/bin/env python3
"""The soak's and fuzz run's reply contents, for reading (validation/audio-assurance-v1 rule 6). Every valid reply in
those runs was byte-identical to its case's reference, so the 42 cases' replies (check_lfm3_speech_stream.cases) and
the soak's 3 text prompts are every distinct reply they produced. One fresh server; per case: the request's text,
the reply's mode, text, audio seconds and WAV sha256; the WAVs are written to OUT_DIR for listening.
Usage: dump_lfm3_soak_replies.py SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_DIR"""
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("soak", HERE / "check_lfm3_audio_soak.py")
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


def main(binary, model, build, libri, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = soak.check.cases(build, libri)
    server = soak.start(binary, model, out)
    rows = []
    try:
        for name, _, body in cases:
            status, reply = server.post(body)
            asked = [m["content"] if isinstance(m["content"], str) else "[audio]" for m in body.get("messages", [])
                     if m.get("role") == "user"]
            wav = base64.b64decode(reply.get("audio_wav_base64", "")) if status == 200 else b""
            if wav:
                (out / f"{name}.wav").write_bytes(wav)
            rows.append(dict(case=name, status=status, mode=body.get("mode"), asked=asked, text=reply.get("text"),
                             audio_seconds=reply.get("audio_seconds"), wav_sha256=hashlib.sha256(wav).hexdigest()[:16]))
        for i, body in enumerate(soak.TEXTS):
            status, reply = soak.generate(server, body)
            rows.append(dict(case=f"text-{i}", status=status, asked=[body["messages"][0]["content"]],
                             text=reply.get("text")))
    finally:
        code, _ = server.stop()
    (out / "replies.json").write_text(json.dumps(dict(rows=rows, exit=code), indent=1))
    print(json.dumps(dict(cases=len(rows), ok=sum(r["status"] == 200 for r in rows), exit=code)))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
