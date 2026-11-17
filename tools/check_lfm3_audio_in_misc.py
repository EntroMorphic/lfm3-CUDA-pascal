#!/usr/bin/env python3
"""Audio input (#3), validation/audio-in-v1 rules 4-5 and the native renderer: refusals, and native chat rendering of
audio parts against HF's processor.

prepare OUT_DIR CLIP.wav MODEL_DIR: writes OUT_DIR/{*.wav, *.req.json, expect.json}:
  - refusals: each request must fail with a message containing the expected words (and leave no outputs);
  - render: conversations with audio parts sent as "messages" (rendered and expanded natively); the native prompt must
    equal HF's prompt ids for the same conversation (processor.apply_chat_template), and the run must succeed.
check OUT_DIR: reads OUT_DIR/CASE.stderr / CASE.json (written by running lfm3-speech per request) and expect.json;
  prints each verdict and writes OUT_DIR/misc-check.json; exit status 1 on any failure.
Run prepare in lfm3-omni-dev; then lfm3-speech MODEL OUT_DIR/CASE.req.json OUT_DIR/CASE 2> OUT_DIR/CASE.stderr for each
request; then check.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np


def write_wav(path, samples, rate=16_000, channels=1, width=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        if width == 2:
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        else:
            w.writeframes((np.clip(samples, -1, 1) * 127 + 128).astype("u1").tobytes())


def prepare(out, clip, model_dir):
    import soundfile as sf
    from transformers import AutoProcessor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x, rate = sf.read(clip, dtype="float32")
    assert rate == 16_000
    good = out / "in-good.wav"
    write_wav(good, x)
    x = sf.read(good, dtype="float32")[0]  # exactly the samples the engine reads
    short = x[: 16_000 * 2]
    write_wav(out / "in-short.wav", short)
    write_wav(out / "in-rate24k.wav", x[:24_000], rate=24_000)
    write_wav(out / "in-stereo.wav", np.repeat(x[:16_000], 2), channels=2)
    write_wav(out / "in-empty.wav", np.zeros(0))
    write_wav(out / "in-pcm8.wav", x[:16_000], width=1)
    (out / "in-garbage.wav").write_bytes(b"not a wav file at all")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tokens = lambda n: -(-(1 + (n + (-(n - 400) % 160) - 400) // 160) // 8)  # noqa: E731

    def hf_ids(conv, clips):
        hconv = [dict(m, content=[dict(p, audio=clips.pop(0)) if p.get("type") == "audio" else p for p in m["content"]])
                 if isinstance(m["content"], list) else m for m in conv]
        return processor.apply_chat_template(hconv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                             return_tensors="pt")["input_ids"][0].tolist()

    expect, A, S = {}, str(good), str(out / "in-short.wav")
    asr = [{"role": "system", "content": "Perform ASR in English."},
           {"role": "user", "content": [{"type": "audio", "audio": A}]}]
    ids = hf_ids(json.loads(json.dumps(asr)), [x])
    n = ids.count(125031)
    assert n == tokens(len(x)), (n, tokens(len(x)))
    # render: native messages vs HF ids
    renders = {
        "render-asr": (asr, [x]),
        "render-question": ([{"role": "system", "content": "Answer questions about the provided audio clip."},
                             {"role": "user", "content": [{"type": "audio", "audio": A},
                                                          {"type": "text", "text": "What is the speaker describing?"}]}],
                            [x]),
        "render-two-clips": ([{"role": "user", "content": [{"type": "text", "text": "Compare these:"},
                                                           {"type": "audio", "audio": S},
                                                           {"type": "audio", "audio": A}]}], [short, x]),
        "render-multiturn": ([{"role": "user", "content": [{"type": "audio", "audio": S}]},
                              {"role": "assistant", "content": "I heard a short sentence."},
                              {"role": "user", "content": "Say it again, in text."}], [short]),
    }
    for name, (conv, clips) in renders.items():
        (out / f"{name}.req.json").write_text(json.dumps(dict(messages=conv, mode="text", max_new_tokens=64)))
        expect[name] = dict(kind="render", prompt=hf_ids(json.loads(json.dumps(conv)), list(clips)))
    # refusals
    span = ids.index(125031)
    refusals = {
        "rate": (dict(prompt_ids=ids, audio=[str(out / "in-rate24k.wav")]), "expected 16000 Hz"),
        "stereo": (dict(prompt_ids=ids, audio=[str(out / "in-stereo.wav")]), "expected mono"),
        "empty": (dict(prompt_ids=ids, audio=[str(out / "in-empty.wav")]), "no samples"),
        "pcm8": (dict(prompt_ids=ids, audio=[str(out / "in-pcm8.wav")]), "16-bit PCM or 32-bit float"),
        "garbage": (dict(prompt_ids=ids, audio=[str(out / "in-garbage.wav")]), "not a RIFF/WAVE"),
        "missing-file": (dict(prompt_ids=ids, audio=[str(out / "in-nonexistent.wav")]), "cannot open"),
        "too-few-placeholders": (dict(prompt_ids=ids[:span] + ids[span + 1:], audio=[A]), "placeholders, the prompt has"),
        "too-many-placeholders": (dict(prompt_ids=ids[:span] + [125031] + ids[span:], audio=[A]),
                                  "placeholders, the prompt has"),
        "no-markers": (dict(prompt_ids=ids[:span - 1] + ids[span:span + n] + ids[span + n + 1:], audio=[A]),
                       "wrapped in <|vision_start|>"),
        "clip-without-span": (dict(prompt_ids=ids, audio=[A, A]), "fewer <|audio|> spans than audio clips"),
        "span-without-clip": (dict(prompt_ids=ids), "more <|audio|> spans than audio clips"),
        "image-placeholder": (dict(prompt_ids=ids[:span] + [125030] + ids[span:], audio=[A]), "image, video"),
        "chunks": (dict(chunks=[ids, ids], audio=[A]), "chunks cannot take audio"),
        "split": (dict(messages=asr, split=True, mode="audio"), "content parts must be text-only"),
        "messages-and-audio": (dict(messages=asr, audio=[A]), "messages' audio parts"),
        "assistant-audio": (dict(messages=[{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": [{"type": "audio", "audio": A}]},
                                           {"role": "user", "content": "again"}]), "user messages only"),
        "capacity": (dict(messages=asr, capacity=n + 8), "exceeds capacity"),
        # the stem's .wav output is this request's input clip: refused before anything is removed
        "in-good": (dict(prompt_ids=ids, audio=[A]), "is one of OUT_STEM's outputs"),
    }
    for name, (req, words) in refusals.items():
        req = dict(dict(mode="text", max_new_tokens=16), **req)
        (out / f"{name}.req.json").write_text(json.dumps(req))
        expect[name] = dict(kind="refusal", words=words)
    (out / "expect.json").write_text(json.dumps(expect, indent=1) + "\n")
    print(" ".join(expect))


def check(out):
    out = Path(out)
    expect = json.loads((out / "expect.json").read_text())
    results, failed = {}, 0
    for name, e in expect.items():
        err = (out / f"{name}.stderr").read_text() if (out / f"{name}.stderr").exists() else ""
        produced = (out / f"{name}.json").exists()
        if e["kind"] == "refusal":
            ok = e["words"] in err and not produced and (
                (out / "in-good.wav").exists() if name == "in-good" else not (out / f"{name}.wav").exists())
        else:
            ok = produced and json.loads((out / f"{name}.json").read_text())["prompt"] == e["prompt"]
        failed += not ok
        results[name] = dict(ok=ok, stderr=err.strip()[:300])
        print("PASS" if ok else "FAIL", name, "|", err.strip()[:160])
    (out / "misc-check.json").write_text(json.dumps(dict(failed=failed, results=results), indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "prepare":
        prepare(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        sys.exit(__doc__)
