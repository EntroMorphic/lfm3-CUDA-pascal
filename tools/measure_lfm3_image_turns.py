#!/usr/bin/env python3
"""validation/image-cache-v1 rule 0 (and rule 2): what each turn of a conversation re-sending its images costs.
Black box on liquid-lfm3-server (Thor, lfm3-profile-v1, a quiet machine), one request at a time: a three-turn
conversation on a stage-0 request's images (each turn the previous reply and a new question appended, the images
re-sent, as every chat client does), text mode, 64 new tokens; per turn the client's end to end and the server's
prefill_seconds (the engine's decode, preprocessing, encode and prefill) and generate_seconds. REPEATS conversations,
the first discarded as warm-up; medians per turn, with the time to the first text event (each turn is streamed).
Usage: measure_lfm3_image_turns.py SERVER MODEL_DIR IMAGE_REQUEST_JSON OUT_JSON [REPEATS]
"""
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client  # noqa: E402
from check_lfm3_speech_stream import Server  # noqa: E402

QUESTIONS = ["What stands out most in this picture?", "What colors dominate it?", "Describe the light."]


def conversation(s, first):
    body = json.loads(json.dumps(first))
    body["max_new_tokens"], body["mode"] = 64, "text"
    user = next(m for m in body["messages"] if m["role"] == "user")
    user["content"] = [p for p in user["content"] if p.get("type") == "image"] + [{"type": "text", "text": QUESTIONS[0]}]
    turns = []
    for k in range(3):
        t0 = time.monotonic()
        st, events, times, _ = s.stream(body)
        end = time.monotonic() - t0
        assert st == 200, (st, events)
        done = [e for e in events if e["event"] == "done"][0]
        first_text = next((tm for e, tm in zip(events, times) if e["event"] == "text"), None)
        turns.append(dict(end_to_end=end, first_text=first_text, prefill=done["prefill_seconds"],
                          generate=done["generate_seconds"], text=done["text"]))
        if k < 2:
            body["messages"] += [{"role": "assistant", "content": done["text"]},
                                 {"role": "user", "content": QUESTIONS[k + 1]}]
    return turns


def main(server_bin, model, image_req, out_json, repeats=4):
    first = as_client(json.loads(Path(image_req).read_text()))
    runs = []
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch)
        try:
            for r in range(int(repeats)):
                runs.append(conversation(s, first))
                print(r, [(round(t["end_to_end"], 3), round(t["prefill"], 3)) for t in runs[-1]], flush=True)
        finally:
            s.stop()
    kept = runs[1:]
    med = [{k: statistics.median(run[i][k] for run in kept) for k in ("end_to_end", "first_text", "prefill", "generate")}
           for i in range(3)]
    same = all(run[i]["text"] == kept[0][i]["text"] for run in kept for i in range(3))
    Path(out_json).write_text(json.dumps(dict(runs=runs, medians=med, texts_repeatable=same), indent=1))
    for i, m in enumerate(med):
        print(f"turn {i + 1}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print("texts identical across conversations:", same)


if __name__ == "__main__":
    main(*sys.argv[1:])
