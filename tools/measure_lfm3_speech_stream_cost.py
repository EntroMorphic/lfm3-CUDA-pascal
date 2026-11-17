#!/usr/bin/env python3
"""Rule 6 re-measurement (validation/speech-stream-v1): the lighthouse request, PAIRS pairs of one unstreamed and one
streamed run, the order alternating pair by pair; the cost is the median over pairs of streamed / unstreamed
generate_seconds (pass: <= 1.03).
Usage (lfm3-profile-v1): measure_lfm3_speech_stream_cost.py SERVER_BINARY MODEL_DIR BUILD_DIR OUT_JSON [PAIRS]
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("check", Path(__file__).with_name("check_lfm3_speech_stream.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def main(binary, model, build, out_json, pairs="15"):
    light = json.loads((Path(build) / "speech-chunk-native-v1/requests/tts-long-male-s1.req.json").read_text())
    body = dict(messages=light["conversation"], mode="audio", normalize=False, max_new_tokens=1536, seed=1)
    server = check.Server(binary, model, Path(out_json).parent)
    rows = []
    try:
        server.post(body)  # warm
        for i in range(int(pairs)):
            got = {}
            for kind in (("plain", "stream") if i % 2 == 0 else ("stream", "plain")):
                got[kind] = (server.post(body)[1] if kind == "plain" else server.stream(body)[1][-1]).get(
                    "generate_seconds", float("inf"))
            rows.append(dict(pair=i, first="plain" if i % 2 == 0 else "stream", **got, ratio=got["stream"] / got["plain"]))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        code, _ = server.stop()
    ratio = statistics.median(r["ratio"] for r in rows)
    report = dict(pairs=rows, median_ratio=ratio, median_plain=statistics.median(r["plain"] for r in rows),
                  median_stream=statistics.median(r["stream"] for r in rows), server_exit=code,
                  server_sha256=check.hashlib.sha256(Path(binary).read_bytes()).hexdigest(), pass_=ratio <= 1.03 and code == 0)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    print("COST", "PASS" if report["pass_"] else "FAIL", json.dumps({k: v for k, v in report.items() if k != "pairs"}))
    return 0 if report["pass_"] else 1


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
