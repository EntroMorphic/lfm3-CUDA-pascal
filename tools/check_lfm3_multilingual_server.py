#!/usr/bin/env python3
"""validation/multilingual-speech-v1 rules 6 and 7 (exact; may share Thor): POST /speech in every locale, black box.

Rule 6, the server equals the engine. Every case is sent to the server (a deployment start: token file, --port 0)
and run by lfm3-speech from the same messages (rendered, split and normalized by the same native code); the reply's
text, WAV and chunk texts must equal lfm3-speech's byte for byte:
- TTS, sentence k0 of every locale x voice (38), seed 1, with the server's defaults (split on; normalize on only for
  American and British English), lfm3-speech asked for the same;
- interleaved, sentence k0 of every language (18), seed 1;
- ASR, the first 2 clips of every language (36), the clip as base64 on the server and as a path for lfm3-speech.
Rule 7, the language gate (audio-assurance-v1 rule 10), live:
- every non-English TTS locale (17) with "normalize": true gets 400 and no audio;
- every non-English TTS locale with digits in its text and the defaults ("Tengo 3 gatos." style: GATE_TEXT) is not
  normalized: each chunk's text is the input's own words, digits kept;
- American and British English with the same digits are normalized by default: no ASCII digit is left in the chunk
  texts, which equal lfm3-speech's with normalize on.
Then the same request twice gives the same reply, and the server exits 0 on SIGTERM.
Writes OUT_JSON; exits 1 on any failure.
Usage: check_lfm3_multilingual_server.py SERVER SPEECH MODEL_DIR CASES_DIR WORK_DIR OUT_JSON
  CASES_DIR: tools/fleurs_cases.py's output (tts.json, interleaved.json, asr-cases.json), paths as this process sees them.
"""
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENGLISH = {"en-us", "en-gb"}
# a short sentence with digits in each language's own words (written for this check; rule 7 tests only that the
# server leaves non-English text alone, so the wording need not be idiomatic)
GATE_TEXT = {
    "ar": "لدي 3 قطط و 12 كتابا.", "da": "Jeg har 3 katte og 12 bøger.", "de": "Ich habe 3 Katzen und 12 Bücher.",
    "el": "Έχω 3 γάτες και 12 βιβλία.", "fi": "Minulla on 3 kissaa ja 12 kirjaa.", "fr": "J'ai 3 chats et 12 livres.",
    "he": "יש לי 3 חתולים ו-12 ספרים.", "it": "Ho 3 gatti e 12 libri.", "ja": "猫が3匹と本が12冊あります。",
    "ko": "고양이 3마리와 책 12권이 있어요.", "nl": "Ik heb 3 katten en 12 boeken.", "pl": "Mam 3 koty i 12 książek.",
    "pt": "Tenho 3 gatos e 12 livros.", "es": "Tengo 3 gatos y 12 libros.", "sv": "Jag har 3 katter och 12 böcker.",
    "tr": "3 kedim ve 12 kitabım var.", "zh": "我有3只猫和12本书。",
    "en-us": "I have 3 cats and 12 books.", "en-gb": "I have 3 cats and 12 books.",
}


def gate_cases(tts):
    """(name, locale, messages, expect_normalized) for rule 7, one per locale (female voice)."""
    out = []
    for locale, text in GATE_TEXT.items():
        system = tts["systems"][f"tts_{locale}_female"]
        out.append((f"gate-{locale}", locale, [{"role": "system", "content": system}, {"role": "user", "content": text}],
                    locale in ENGLISH))
    return out


def exact_cases(tts, interleaved, asr):
    """(name, messages, mode, max_new_tokens, seed, driver extras) for rule 6."""
    out = []
    for c in tts["cases"]:
        if c["case"].endswith("-k0"):
            m = [{"role": "system", "content": tts["systems"][c["system"]]}, {"role": "user", "content": c["user"]}]
            out.append((f"{c['case']}-s1", m, "audio", c["max_new_tokens"], 1,
                        dict(split=True, normalize=c["locale"] in ENGLISH)))
    for c in interleaved["cases"]:
        if c["case"].endswith("-k0"):
            m = [{"role": "system", "content": interleaved["systems"][c["system"]]}, {"role": "user", "content": c["user"]}]
            out.append((f"{c['case']}-s1", m, "interleaved", c["max_new_tokens"], 1, {}))
    seen = {}
    for c in asr["cases"]:
        if seen.get(c["language"], 0) >= 2:
            continue
        seen[c["language"]] = seen.get(c["language"], 0) + 1
        m = [{"role": "system", "content": asr["systems"][c["system"]]},
             {"role": "user", "content": [{"type": "audio", "audio": a} for a in c["audio"]]}]
        out.append((c["case"], m, "text", c["max_new_tokens"], 0, {}))
    return out


def digits_kept(text, chunks):
    """Every digit run of the input is still in the chunk texts, in order (rule 7, non-English)."""
    return re.findall(r"\d+", text) == re.findall(r"\d+", " ".join(chunks))


def main(server, speech, model, cases_dir, work, out_json):
    cases_dir, work = Path(cases_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    load = lambda n: json.load(open(cases_dir / n, encoding="utf-8"))  # noqa: E731
    tts, interleaved, asr = load("tts.json"), load("interleaved.json"), load("asr-cases.json")
    token = secrets.token_urlsafe(32)
    token_path = Path(out_json).with_suffix(".token")
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    proc = subprocess.Popen([server, "--experimental", model, "--token-file", str(token_path), "--port", "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    port = json.loads(proc.stdout.readline())["port"]
    token_path.unlink()

    def post(body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/speech", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def engine(name, messages, mode, count, seed, extra):
        req_path, stem = work / f"{name}.req.json", work / name
        req_path.write_text(json.dumps(dict(messages=messages, mode=mode, max_new_tokens=count, seed=seed, **extra),
                                       ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([speech, model, str(req_path), str(stem)], capture_output=True, text=True)
        if r.returncode:
            return r, {}, b""
        return r, json.loads((work / f"{name}.json").read_text()), (work / f"{name}.wav").read_bytes()

    def b64(messages):
        return [dict(m, content=[dict(p, audio=base64.b64encode(Path(p["audio"]).read_bytes()).decode())
                                 if p.get("type") == "audio" else p for p in m["content"]])
                if isinstance(m["content"], list) else m for m in messages]

    report = dict(rule6={}, rule7={}, server_sha256=hashlib.sha256(Path(server).read_bytes()).hexdigest(),
                  speech_sha256=hashlib.sha256(Path(speech).read_bytes()).hexdigest())
    ok, first = True, None
    for name, messages, mode, count, seed, extra in exact_cases(tts, interleaved, asr):
        r, ref, ref_wav = engine(name, messages, mode, count, seed, extra)
        body = dict(messages=b64(messages), mode=mode, max_new_tokens=count, seed=seed)
        status, reply = post(body)
        got = base64.b64decode(reply.get("audio_wav_base64", "")) if status == 200 else b""
        res = dict(status=status, engine_exit=r.returncode, engine_stderr=r.stderr[-300:],
                   text_identical=status == 200 and reply.get("text") == ref.get("text"),
                   wav_identical=status == 200 and got == ref_wav,
                   chunks_identical=[c.get("text") for c in reply.get("chunks", [])] == [c.get("text") for c in ref.get("chunks", [])],
                   chunks=len(reply.get("chunks", [])), audio_seconds=reply.get("audio_seconds"))
        res["pass"] = status == 200 and r.returncode == 0 and res["text_identical"] and res["wav_identical"] and \
            res["chunks_identical"]
        ok &= res["pass"]
        report["rule6"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "engine_stderr"}, ensure_ascii=False), flush=True)
        first = first or (body, reply)
    for name, locale, messages, english in gate_cases(tts):
        res = dict(locale=locale)
        status, reply = post(dict(messages=messages, mode="audio", max_new_tokens=512, seed=1, normalize=True))
        res["normalize_true_status"] = status
        res["normalize_true_ok"] = status == 200 if english else (status == 400 and "audio_wav_base64" not in reply)
        status, reply = post(dict(messages=messages, mode="audio", max_new_tokens=512, seed=1))
        chunks = [c.get("text", "") for c in reply.get("chunks", [])]
        res.update(default_status=status, chunk_texts=chunks)
        r, ref, ref_wav = engine(name, messages, "audio", 512, 1, dict(split=True, normalize=english))
        res["equals_engine"] = status == 200 and chunks == [c.get("text") for c in ref.get("chunks", [])] and \
            base64.b64decode(reply.get("audio_wav_base64", "")) == ref_wav
        text = messages[1]["content"]
        res["default_ok"] = status == 200 and (not re.search(r"[0-9]", " ".join(chunks)) if english
                                               else digits_kept(text, chunks))
        res["pass"] = res["normalize_true_ok"] and res["default_ok"] and res["equals_engine"]
        ok &= res["pass"]
        report["rule7"][name] = res
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
    status, again = post(first[0])
    same = status == 200 and all(again.get(k) == first[1].get(k) for k in ("text", "audio_wav_base64", "frames", "chunks"))
    report["repeat_identical"] = same
    ok &= same
    proc.terminate()
    report["exit_code"] = proc.wait(timeout=120)
    report["stderr"] = proc.stderr.read()[-2000:]
    ok &= report["exit_code"] == 0
    report["pass"] = bool(ok)
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("MULTILINGUAL SERVER PASS" if ok else "MULTILINGUAL SERVER FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
