#!/usr/bin/env python3
"""Rule 5 of validation/text-wait-v1: nothing slower alone (end to end, through the server; Thor quiet).

PAIRS pairs, each a fresh baseline server and a fresh server under test, their order alternating between pairs. On each:
  text   3 arrivals of rule 2's prompt (128 tokens: create, generate, close), alone; the median seconds;
  one    tts-long-male-s1 unstreamed, alone, twice: the faster's audio seconds per wall second;
  crowd  8 TTS requests at once (the long-form and museum cases): total audio seconds per wall second.
Ratios per pair (under test against baseline, higher is better): text = baseline seconds / test seconds; one and crowd
= test rate / baseline rate. Pass: the median ratio of each is at least 0.99. Replies must also be identical between
the two servers (rule 1).
Usage: check_lfm3_text_wait_speed.py SERVER BASELINE_SERVER MODEL_DIR BUILD_DIR LIBRI_WAV_DIR OUT_JSON [PAIRS]
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wait", HERE / "check_lfm3_text_wait.py")
wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wait)
check, batch = wait.check, wait.batch
CROWD = ["tts-long-male-s1", "tts-long-male-s2", "tts-long-male-s3", "lighthouse-split-s1", "tts-museum-female-s1",
         "tts-museum-male-s1", "tts-museum-female-s2", "tts-museum-male-s2"]


def measure(binary, model, scratch, cases):
    server = batch.start(binary, model, scratch)
    try:
        server.post(cases["tts-fox-female-s1"])  # warm
        wait.text_arrival(server, wait.RULE2)
        texts = [wait.text_arrival(server, wait.RULE2) for _ in range(3)]
        assert all(t[1] == 200 for t in texts), texts
        ones = []
        for _ in range(2):
            t0 = time.monotonic()
            status, reply = batch.post(server, cases["tts-long-male-s1"])
            ones.append((reply["audio_seconds"] / (time.monotonic() - t0), hashlib.sha256(
                reply["audio_wav_base64"].encode()).hexdigest()[:16]))
            assert status == 200
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda n: batch.post(server, cases[n]), CROWD))
        wall = time.monotonic() - t0
        assert all(s == 200 for s, _ in got)
        return dict(text_seconds=statistics.median(t[0] for t in texts), text_ids=texts[0][2]["ids_sha"],
                    one_rate=max(o[0] for o in ones), one_wav=ones[0][1],
                    crowd_rate=sum(r["audio_seconds"] for _, r in got) / wall,
                    crowd_wavs=[hashlib.sha256(r["audio_wav_base64"].encode()).hexdigest()[:16] for _, r in got])
    finally:
        server.stop()


def main(binary, baseline, model, build, libri, out_json, pairs=5):
    out = Path(out_json)
    cases = {n: b for n, _, b in check.cases(build, libri)}
    report = dict(server_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(), pairs=[])
    for i in range(int(pairs)):
        order = [("baseline", baseline), ("test", binary)] if i % 2 == 0 else [("test", binary), ("baseline", baseline)]
        got = {name: measure(b, model, out.parent, cases) for name, b in order}
        b, t = got["baseline"], got["test"]
        pair = dict(order=[o[0] for o in order], baseline=b, test=t,
                    text=b["text_seconds"] / t["text_seconds"], one=t["one_rate"] / b["one_rate"],
                    crowd=t["crowd_rate"] / b["crowd_rate"],
                    identical=b["text_ids"] == t["text_ids"] and b["one_wav"] == t["one_wav"]
                    and b["crowd_wavs"] == t["crowd_wavs"])
        report["pairs"].append(pair)
        print(f"pair {i}: text {pair['text']:.4f} one {pair['one']:.4f} crowd {pair['crowd']:.4f} "
              f"identical {pair['identical']}", flush=True)
        out.write_text(json.dumps(report, indent=1))
    med = {k: statistics.median(p[k] for p in report["pairs"]) for k in ("text", "one", "crowd")}
    report["median_ratio"] = med
    report["pass_"] = all(v >= 0.99 for v in med.values()) and all(p["identical"] for p in report["pairs"])
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(median_ratio=med, pass_=report["pass_"])))
    return 0 if report["pass_"] else 1


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
