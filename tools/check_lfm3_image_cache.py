#!/usr/bin/env python3
"""validation/image-cache-v1 rules 1 and 4 (and image-queue-v1 part 2's exactness): liquid-lfm3-server with its image
caches on (the default) against the same server with --image-cache-mb 0, black box (Thor, lfm3-profile-v1).
Per server: the 11 stage-0 requests each sent twice in a row unstreamed then twice streamed (with the cache on, every
send after the first is a hit), each reply against lfm3-speech's text for the request (ROUND_DIR/out); the 11 sent at
once, twice; three-turn conversations re-sending cherry's and two's images, twice; the listening set's interleaved
requests (LISTENING_DIR: the req/ of tools/vision_listening_set.py, when given), text and WAV. With the cache on and
off, every reply must be the same. Rule 4's "one byte changed is a miss" and "refusals refused again" are host tests
(tests/test_lfm3_http_image_in.py).
Usage: check_lfm3_image_cache.py SERVER MODEL_DIR ROUND_DIR OUT_JSON [LISTENING_DIR]
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client, streamed_text  # noqa: E402
from check_lfm3_speech_stream import Server  # noqa: E402
from measure_lfm3_image_turns import QUESTIONS  # noqa: E402


def turns(s, first):
    body = json.loads(json.dumps(first))
    body["max_new_tokens"], body["mode"] = 64, "text"
    user = next(m for m in body["messages"] if m["role"] == "user")
    user["content"] = [p for p in user["content"] if p.get("type") == "image"] + [{"type": "text", "text": QUESTIONS[0]}]
    texts = []
    for k in range(3):
        st, reply = s.post(body)
        assert st == 200, (st, reply)
        texts.append(reply["text"])
        if k < 2:
            body["messages"] += [{"role": "assistant", "content": reply["text"]},
                                 {"role": "user", "content": QUESTIONS[k + 1]}]
    return texts


def run(server_bin, model, cases, listening, extra):
    out = {"stage0": {}, "concurrent": [], "turns": {}, "listening": {}}
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch, extra=["--queue-size", "16", *extra])
        try:
            for name, (body, _) in cases.items():
                sends = [s.post(body) for _ in range(2)]
                streams = [s.stream(body) for _ in range(2)]
                out["stage0"][name] = [r.get("text") if st == 200 else f"HTTP {st} {r}" for st, r in sends] + \
                    [streamed_text(ev)[0] if st == 200 else f"HTTP {st}" for st, ev, _, _ in streams]
            for _ in range(2):
                got = {}
                ts = [threading.Thread(target=lambda n=n, b=b: got.__setitem__(n, s.post(b))) for n, (b, _) in cases.items()]
                [t.start() for t in ts]
                [t.join() for t in ts]
                out["concurrent"].append({n: (r.get("text") if st == 200 else f"HTTP {st} {r}") for n, (st, r) in got.items()})
            for name in ("cherry", "two"):
                out["turns"][name] = [turns(s, cases[name][0]) for _ in range(2)]
            for name, body in listening.items():
                st, r = s.post(body)
                out["listening"][name] = [r.get("text"), r.get("audio_wav_base64")] if st == 200 else f"HTTP {st} {r}"
        finally:
            s.stop()
    return out


def main(server_bin, model, round_dir, out_json, listening_dir=None):
    rd = Path(round_dir)
    cases = {p.stem: (as_client(json.loads(p.read_text())), json.loads((rd / "out" / f"{p.stem}.json").read_text())["text"])
             for p in sorted((rd / "req").glob("*.json"))}
    listening = {}
    if listening_dir:
        for p in sorted(Path(listening_dir).glob("*.json")):
            listening[p.name] = as_client(json.loads(p.read_text()))
    on = run(server_bin, model, cases, listening, [])
    off = run(server_bin, model, cases, listening, ["--image-cache-mb", "0"])
    checks = {}
    for name, (_, want) in cases.items():
        checks[f"stage0/{name}"] = all(t == want for t in on["stage0"][name] + off["stage0"][name])
        checks[f"concurrent/{name}"] = all(c[name] == want for c in on["concurrent"] + off["concurrent"])
    for name in on["turns"]:
        checks[f"turns/{name}"] = on["turns"][name][0] == on["turns"][name][1] == off["turns"][name][0] == off["turns"][name][1]
    for name in on["listening"]:
        checks[f"listening/{name}"] = on["listening"][name] == off["listening"][name]
    ok = all(checks.values())
    Path(out_json).write_text(json.dumps(dict(passed=ok, checks=checks, on=on, off=off), indent=1))
    for k, v in checks.items():
        if not v:
            print("DIFFER", k)
    print(f"{sum(checks.values())} of {len(checks)} identical; IMAGE CACHE CHECK {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
