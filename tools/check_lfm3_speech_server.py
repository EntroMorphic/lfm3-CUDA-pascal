#!/usr/bin/env python3
"""GPU check for POST /speech on liquid-lfm3-server (#2; Thor, lfm3-profile-v1). Black box: the server is started like a
deployment (token file, --port 0) and driven only over HTTP.

  1. the WAV bytes /speech returns equal lfm3-speech's for the same request, on validated runs (sent with
     "normalize": false, as they were made; normalization is checked by check_lfm3_audio_in_server.py): long-form TTS split by
     sentences (lighthouse and bread, seed 1), one-sentence TTS unsplit (fox, seed 1), interleaved (il-study, seed 1;
     its text also equal to the validated native text), and the Phase B abbreviations under the default rule;
  2. the same request twice gives the same bytes;
  3. text is unaffected: a pinned greedy /generate before and after the speech requests gives the same ids;
  4. worst cases the Service admits finish within 240 s (80% of the default engine timeout), the server still serving:
     512 chunks of ~235-token prompts at 12 steps (the step budget and most of the prefill cap), and 512 sentences at
     4000 steps each (cut early by the budget: truncated). The report records the server binary's sha256.
Writes OUT_JSON (per case: pass, sizes, timings) and exits 1 on any failure.
Usage: check_lfm3_speech_server.py SERVER_BINARY MODEL_DIR BUILD_DIR OUT_JSON   (BUILD_DIR: the validated runs)
"""
import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main(binary, model, build, out_json):
    b = Path(build)
    token = secrets.token_urlsafe(32)
    token_path = Path(out_json).with_suffix(".token")
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    proc = subprocess.Popen([binary, "--experimental", model, "--token-file", str(token_path), "--port", "0"],
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

    import hashlib
    report, ok = dict(cases={}, server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest()), True
    sid = post("/sessions", {"capacity": 1024})[1]["session_id"]
    control = dict(session_id=sid, max_new_tokens=128,
                   messages=[{"role": "user", "content": "Explain in one paragraph how a lighthouse warns ships at night."}])
    s0, before = post("/generate", control)
    post(f"/sessions/{sid}/reset", {})

    def req(path):
        return json.load(open(b / path))

    cases = [  # name, request body, reference WAV, reference text (interleaved)
        ("lighthouse-split-s1", dict(messages=req("speech-chunk-native-v1/requests/tts-long-male-s1.req.json")["conversation"],
                                     mode="audio", normalize=False, max_new_tokens=1536, seed=1), "speech-chunk-native-v1/native/tts-long-male-s1.wav", None),
        ("bread-split-s1", dict(messages=req("speech-chunk-native-v1/requests/probe-other-long-s1.req.json")["conversation"],
                                mode="audio", normalize=False, max_new_tokens=1536, seed=1), "speech-chunk-native-v1/native/probe-other-long-s1.wav", None),
        ("fox-female-unsplit-s1", dict(messages=req("speech-e2e-v1/requests/tts-fox-female-s1.req.json")["conversation"],
                                       mode="audio", normalize=False, max_new_tokens=req("speech-e2e-v1/requests/tts-fox-female-s1.req.json")["max_new_tokens"],
                                       seed=1, split=False), "speech-e2e-v1/native/tts-fox-female-s1.wav", None),
        ("il-study-s1", dict(messages=req("speech-e2e-v1/requests/il-study-s1.req.json")["conversation"], mode="interleaved",
                             max_new_tokens=req("speech-e2e-v1/requests/il-study-s1.req.json")["max_new_tokens"], seed=1),
         "speech-e2e-v1/native/il-study-s1.wav", json.load(open(b / "speech-e2e-v1/native-cases/il-study-s1/summary.json"))["text"]),
        ("abbrev-default-s1", dict(messages=req("speech-split-b-full-49/requests/abbrev-s1.req.json")["conversation"],
                                   mode="audio", normalize=False, max_new_tokens=1536, seed=1), "speech-split-b-full-49/native/abbrev-s1.wav", None),
    ]
    first = None
    for name, body, wav, text in cases:
        t0 = time.time()
        status, r = post("/speech", body)
        wall = time.time() - t0
        got = base64.b64decode(r.get("audio_wav_base64", "")) if status == 200 else b""
        want = (b / wav).read_bytes()
        res = dict(status=status, wav_identical=got == want, wav_bytes=len(got), reference_bytes=len(want),
                   chunks=len(r.get("chunks", [])), all_chunks_ended_by_eos=all(c["ended_by_eos"] for c in r.get("chunks", [])),
                   audio_seconds=r.get("audio_seconds"), generate_seconds=r.get("generate_seconds"), wall_seconds=round(wall, 2))
        if text is not None:
            res["text_identical"] = r.get("text") == text
        res["pass"] = status == 200 and res["wav_identical"] and res["all_chunks_ended_by_eos"] and res.get("text_identical", True)
        ok &= res["pass"]
        report["cases"][name] = res
        print(name, json.dumps(res), flush=True)
        if first is None:
            first = (body, got)
    # Worst cases the Service admits. "budget": 512 chunks of ~235-token prompts at 12 steps each, which uses about the
    # whole step budget (6144) and ~120k prompt tokens of prefill; "large cap": 512 sentences each allowed 4000 steps,
    # which the budget cuts early. Each must finish within 240 s (80% of the default engine timeout), the server still
    # serving; "large cap" must come back truncated.
    system = "Perform TTS in American English using a male voice. " + " ".join(["lighthouse keeper"] * 70)
    sentence = "The small boat drifted slowly past the quiet harbor wall."
    for name, body in (("budget", dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": " ".join([sentence] * 512)}],
                                         mode="audio", normalize=False, max_new_tokens=12, seed=1)),
                       ("large-cap", dict(messages=[{"role": "system", "content": "Perform TTS in American English using a male voice."},
                                                    {"role": "user", "content": " ".join([sentence] * 512)}],
                                          mode="audio", normalize=False, max_new_tokens=4000, seed=1))):
        t0 = time.time()
        status, r = post("/speech", body, timeout=900)
        wall = time.time() - t0
        res = dict(status=status, wall_seconds=round(wall, 1), truncated=r.get("truncated"), chunks_run=len(r.get("chunks", [])),
                   frames=r.get("frames"), prefill_seconds=r.get("prefill_seconds"), generate_seconds=r.get("generate_seconds"))
        res["pass"] = status == 200 and wall < 240 and (name != "large-cap" or r.get("truncated") is True)
        report[f"worst_case_{name}"] = res
        ok &= res["pass"]
        print("worst case", name, json.dumps(res), flush=True)
    status, again = post("/speech", first[0])
    same = status == 200 and base64.b64decode(again["audio_wav_base64"]) == first[1]
    report["repeat_identical"] = same
    ok &= same
    s1, after = post("/generate", control)
    text_same = s0 == s1 == 200 and before["generated_ids"] == after["generated_ids"]
    report["text_control_identical"] = text_same
    report["text_control_statuses"] = [s0, s1]
    report["text_control_tokens"] = len(before.get("generated_ids", []))
    ok &= text_same
    proc.terminate()
    report["exit_code"] = proc.wait(timeout=120)
    report["stderr"] = proc.stderr.read()[-2000:]
    ok &= report["exit_code"] == 0
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print("repeat identical:", same, "| text control identical:", text_same, "| exit:", report["exit_code"], "| PASS" if ok else "| FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
