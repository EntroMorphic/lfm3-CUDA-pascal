#!/usr/bin/env python3
"""GPU check for audio input on POST /speech (#3; Thor, lfm3-profile-v1; validation/audio-in-v1). Black box: the
server is started like a deployment (token file, --port 0) and driven only over HTTP. Python standard library only.

Each case is a conversation with audio parts, sent twice: to the server, each clip as base64 of the WAV file's bytes,
and to lfm3-speech, each clip as the file's path (rendered and expanded by the same native code). The server's reply
must equal lfm3-speech's output: the text byte for byte, and the WAV byte for byte (in text mode both are the empty
WAV). Cases: speech recognition of three LibriSpeech test-clean utterances (text, greedy); a spoken clip with a
question about it (text); a spoken question answered in interleaved text and speech, and in speech alone (seed 1);
two clips in one turn; a 90-second recording (several utterances joined) transcribed, which must also finish
within 240 s (80% of the default engine timeout); and text with numbers read as speech with the server's defaults
(split, normalize), against lfm3-speech asked for both, the chunk texts (normalized sentences) equal too. Then: the same request twice gives the same reply; a pinned greedy
/generate before and after gives the same ids; the server exits cleanly. Writes OUT_JSON; exits 1 on any failure.
Usage: check_lfm3_audio_in_server.py SERVER_BINARY SPEECH_BINARY MODEL_DIR WAV_DIR WORK_DIR OUT_JSON
"""
import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

ASR = "Perform ASR in English."
NUMBERS = ("The meeting starts at 7:45 and ends by 9:00. Order number 389 cost $3.50, about 2.75% more than in 1984. "
           "Dr. Patel met Mr. Lee at St. Mary's on Nov. 9, 2026.")


def join_wavs(paths, out, seconds):
    frames = b""
    for p in paths:
        with wave.open(str(p), "rb") as w:
            assert w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            frames += w.readframes(w.getnframes()) + b"\0\0" * 8000  # half a second of silence between
        if len(frames) >= seconds * 32_000:
            break
    frames = frames[: seconds * 32_000]
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16_000)
        w.writeframes(frames)


def main(server, speech, model, wav_dir, work, out_json):
    wav_dir, work = Path(wav_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    W = lambda u: str(wav_dir / f"{u}.wav")  # noqa: E731
    long_wav = work / "joined-90s.wav"
    join_wavs(sorted(wav_dir.glob("1089-134686-*.wav")) + sorted(wav_dir.glob("1188-133604-*.wav")), long_wav, 90)

    def conv(clips, system=None, text=None):
        content = [{"type": "audio", "audio": c} for c in clips] + ([{"type": "text", "text": text}] if text else [])
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]

    cases = [  # name, messages (clip paths), mode, max_new_tokens, seed
        ("asr-1089-134686-0000", conv([W("1089-134686-0000")], ASR), "text", 512, 0),
        ("asr-4446-2275-0004", conv([W("4446-2275-0004")], ASR), "text", 512, 0),
        ("asr-121-123859-0002", conv([W("121-123859-0002")], ASR), "text", 512, 0),
        ("question-text", conv([W("1089-134686-0000")], "Answer questions about the provided audio clip.",
                               "What is the speaker describing?"), "text", 256, 0),
        ("two-clips-text", conv([W("4446-2275-0004"), W("1580-141084-0005")], None,
                                "Who are the two speakers talking about? Answer briefly."), "text", 256, 0),
        ("spoken-interleaved-s1", conv([W("1580-141084-0005")], "Respond with interleaved text and audio."),
         "interleaved", 1024, 1),
        ("spoken-audio-s1", conv([W("4446-2275-0004")], "Respond with audio."), "audio", 1024, 1),
        ("asr-joined-90s", conv([str(long_wav)], ASR), "text", 2048, 0),
        # text in, speech out with the server's defaults (split and normalize on), against lfm3-speech asked for both
        ("tts-numbers-normalized-s1", [{"role": "system", "content": "Perform TTS in American English using a female voice."},
                                       {"role": "user", "content": NUMBERS}], "audio", 512, 1),
    ]
    token = secrets.token_urlsafe(32)
    token_path = Path(out_json).with_suffix(".token")
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    proc = subprocess.Popen([server, "--experimental", model, "--token-file", str(token_path), "--port", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    port = json.loads(proc.stdout.readline())["port"]
    token_path.unlink()

    def post(path, body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    report = dict(cases={}, server_sha256=hashlib.sha256(Path(server).read_bytes()).hexdigest(),
                  speech_sha256=hashlib.sha256(Path(speech).read_bytes()).hexdigest())
    ok = True
    sid = post("/sessions", {"capacity": 1024})[1]["session_id"]
    control = dict(session_id=sid, max_new_tokens=128,
                   messages=[{"role": "user", "content": "Explain in one paragraph how a lighthouse warns ships at night."}])
    s0, before = post("/generate", control)
    post(f"/sessions/{sid}/reset", {})
    first = None
    for name, messages, mode, count, seed in cases:
        # the reference: lfm3-speech with the clips as paths
        req_path, stem = work / f"{name}.req.json", work / name
        driver = dict(messages=messages, mode=mode, max_new_tokens=count, seed=seed)
        if mode == "audio" and all(isinstance(m["content"], str) for m in messages):
            driver.update(split=True, normalize=True)  # the server's defaults for text in, speech out
        req_path.write_text(json.dumps(driver))
        r = subprocess.run([speech, model, str(req_path), str(stem)], capture_output=True, text=True)
        ref = json.loads((work / f"{name}.json").read_text()) if r.returncode == 0 else {}
        ref_wav = (work / f"{name}.wav").read_bytes() if r.returncode == 0 else b""
        # the server: the clips as base64 WAV bytes
        body_messages = [dict(m, content=[dict(p, audio=base64.b64encode(Path(p["audio"]).read_bytes()).decode())
                                          if p.get("type") == "audio" else p for p in m["content"]])
                         if isinstance(m["content"], list) else m for m in messages]
        body = dict(messages=body_messages, mode=mode, max_new_tokens=count, seed=seed)
        t0 = time.time()
        status, reply = post("/speech", body)
        wall = time.time() - t0
        got_wav = base64.b64decode(reply.get("audio_wav_base64", "")) if status == 200 else b""
        res = dict(status=status, reference_exit=r.returncode, reference_stderr=r.stderr[-300:],
                   text_identical=status == 200 and reply.get("text") == ref.get("text"),
                   wav_identical=status == 200 and got_wav == ref_wav, wav_bytes=len(got_wav), text=reply.get("text"),
                   audio_seconds=reply.get("audio_seconds"), prefill_seconds=reply.get("prefill_seconds"),
                   generate_seconds=reply.get("generate_seconds"), wall_seconds=round(wall, 2),
                   reference_encode_seconds=ref.get("encode_seconds"), prompt_tokens=len(ref.get("prompt", [])))
        if name.startswith("tts-"):  # the chunks the server split are the normalized sentences
            res["chunk_texts"] = [c.get("text") for c in reply.get("chunks", [])]
            res["chunks_identical"] = res["chunk_texts"] == [c.get("text") for c in ref.get("chunks", [])]
        res["pass"] = status == 200 and r.returncode == 0 and res["text_identical"] and res["wav_identical"] and \
            res.get("chunks_identical", True) and \
            bool(reply.get("text") or mode == "audio") and (name != "asr-joined-90s" or wall < 240)
        ok &= res["pass"]
        report["cases"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "reference_stderr"}), flush=True)
        if first is None:
            first = (body, reply)
    status, again = post("/speech", first[0])
    same = status == 200 and all(again.get(k) == first[1].get(k) for k in ("text", "audio_wav_base64", "frames", "chunks"))
    report["repeat_identical"] = same
    ok &= same
    s1, after = post("/generate", control)
    text_same = s0 == s1 == 200 and before["generated_ids"] == after["generated_ids"]
    report["text_control_identical"] = text_same
    ok &= text_same
    proc.terminate()
    report["exit_code"] = proc.wait(timeout=120)
    report["stderr"] = proc.stderr.read()[-2000:]
    ok &= report["exit_code"] == 0
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print("repeat identical:", same, "| text control identical:", text_same, "| exit:", report["exit_code"],
          "| PASS" if ok else "| FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
