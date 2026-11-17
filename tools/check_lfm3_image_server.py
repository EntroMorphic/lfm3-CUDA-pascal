#!/usr/bin/env python3
"""Image input through liquid-lfm3-server (#4; validation/images-v1; Thor, lfm3-profile-v1). Black box: the server is
started on the checkpoint and fed lfm3-speech's own requests (tools/thor/vision_e2e_round.sh: the 11 stage-0
conversations, text mode, 128 new tokens), each image part's path replaced by the file's bytes in base64, as a
client sends them. Per request, the server's reply text, unstreamed and streamed, must equal lfm3-speech's for the
same conversation exactly (the same engine path: the server decodes, preprocesses and encodes as lfm3-speech does).
Then the same requests at once (concurrent rows) must give the same texts again.
Usage: check_lfm3_image_server.py SERVER MODEL_DIR ROUND_DIR OUT_JSON   (ROUND_DIR: vision_e2e_round.sh's output)
"""
import base64
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_speech_stream import Server  # noqa: E402


def as_client(req):
    body = json.loads(json.dumps(req))
    body.pop("record_image", None)
    for m in body["messages"]:
        if isinstance(m["content"], list):
            for part in m["content"]:
                if part.get("type") == "image":
                    part["image"] = base64.b64encode(Path(part["image"]).read_bytes()).decode()
    return body


def streamed_text(events):
    text = "".join(e.get("text", "") for e in events if e.get("event") == "text")
    done = [e for e in events if e.get("event") == "done"]
    return text, done[0] if done else None


def main(server_bin, model, round_dir, out_json):
    rd = Path(round_dir)
    cases = {p.stem: (as_client(json.loads(p.read_text())), json.loads((rd / "out" / f"{p.stem}.json").read_text())["text"])
             for p in sorted((rd / "req").glob("*.json"))}
    rows, ok = {}, True
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch)
        try:
            for name, (body, want) in cases.items():
                status, reply = s.post(body)
                st, events, _, _ = s.stream(body)
                text, done = streamed_text(events)
                row = dict(status=status, unstreamed_equal=status == 200 and reply.get("text") == want,
                           stream_status=st, streamed_equal=st == 200 and text == want and done is not None
                           and done.get("text") == want)
                rows[name] = row
                ok &= row["unstreamed_equal"] and row["streamed_equal"]
                print(name, row, flush=True)
            # all at once: the concurrent speech batch's rows
            got, busy = {}, {}
            def one(name, body):
                # the front end takes one encode at a time (lfm3_http.hpp frontend_call): a client retries a 429
                for attempt in range(200):
                    got[name] = s.post(body)
                    if got[name][0] != 429:
                        break
                    busy[name] = attempt + 1
                    time.sleep(0.05 * (1 + attempt % 10))
            threads = [threading.Thread(target=one, args=(n, b)) for n, (b, _) in cases.items()]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for name, (_, want) in cases.items():
                status, reply = got[name]
                rows[name]["concurrent_equal"] = status == 200 and reply.get("text") == want
                ok &= rows[name]["concurrent_equal"]
            print("concurrent", {n: r["concurrent_equal"] for n, r in rows.items()}, "429 retries", busy, flush=True)
            rows["_concurrent_429_retries"] = busy
        finally:
            s.proc.terminate()
            s.proc.wait(timeout=120)
    Path(out_json).write_text(json.dumps(dict(passed=ok, cases=rows), indent=1))
    print("IMAGE SERVER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
