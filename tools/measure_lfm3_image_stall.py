#!/usr/bin/env python3
"""validation/image-queue-v1 rule 0 (and rule 2): how long a 4K image request stalls the other rows' audio. Black box
on liquid-lfm3-server (Thor, lfm3-profile-v1, a quiet machine): 7 rows stream long speech (audio mode, unsplit); 3 s
after they start, an 8th request brings a 4K photograph (stage-0 "cherry", text mode, 32 tokens). Per streaming row,
the longest gap between consecutive audio events from the image request's start to 4 s after it, against the same
window of a run without the image. Runs alternate (without, with) REPEATS times on one server; the result is each
run's worst row and the difference of the medians.
Usage: measure_lfm3_image_stall.py SERVER MODEL_DIR IMAGE_REQUEST_JSON OUT_JSON [REPEATS [SERVER_ARGS...]]
"""
import json
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client  # noqa: E402
from check_lfm3_speech_stream import Server  # noqa: E402

TEXT = ("The river bends around the old mill, where the water slows and darkens under the willows. In the morning "
        "the fishermen come down the path with their rods and their thermoses, and they speak quietly, as if the fish "
        "could hear them. By noon the light has moved across the stones and the children arrive, shouting, to swim in "
        "the shallow pool below the weir. In the evening the swallows skim the surface, and the mill wheel, which has "
        "not turned in fifty years, throws a long shadow over the reeds.")
ROWS, START_IMAGE, WINDOW = 7, 3.0, 4.0


def one_run(s, image_body, warm=False):
    body = dict(messages=[{"role": "user", "content": TEXT}], mode="audio", max_new_tokens=900, split=False)
    results = [None] * ROWS
    t_start = time.monotonic()
    def stream(i):
        # text admission refuses a second request meeting the busy front end (429 frontend_busy): retry, as a client
        for _ in range(200):
            results[i] = s.stream(dict(body, seed=i))
            if results[i][0] != 429:
                return
            time.sleep(0.05)
    threads = [threading.Thread(target=stream, args=(i,)) for i in range(ROWS)]
    for t in threads:
        t.start()
    time.sleep(START_IMAGE)
    t_image = time.monotonic()
    image_result = s.post(image_body) if image_body else None
    image_seconds = time.monotonic() - t_image if image_body else None  # the image request alone, not the streams
    for t in threads:
        t.join()
    if warm:  # a cold server's first audio can land after the window: the warm-up run measures nothing
        return None
    gaps = []
    for status, events, times, t0 in results:
        assert status == 200, status
        audio = [t0 + tm for e, tm in zip(events, times) if e["event"] == "audio"]
        inside = [a for a in audio if t_image <= a <= t_image + WINDOW]
        before = [a for a in audio if a < t_image][-1:]  # the gap that spans the window's start counts too
        seq = before + inside
        gaps.append(max((b - a for a, b in zip(seq, seq[1:])), default=None))
    if all(g is None for g in gaps):
        raise RuntimeError(f"no row had two audio events in the window: {[len(r[1]) for r in results]} events per row")
    return dict(worst_gap=max(g for g in gaps if g is not None), row_gaps=gaps,
                image_status=image_result[0] if image_result else None,
                image_seconds=image_seconds, started=t_image - t_start)


def main(server_bin, model, image_req, out_json, repeats=3, *server_args):
    image_body = as_client(json.loads(Path(image_req).read_text()))
    image_body["max_new_tokens"] = 32
    runs = []
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch, extra=server_args)
        try:
            one_run(s, None, warm=True)
            for r in range(repeats):
                for kind in ("without", "with"):
                    out = one_run(s, image_body if kind == "with" else None)
                    out["kind"] = kind
                    runs.append(out)
                    print(kind, round(out["worst_gap"], 4), [round(g, 3) for g in out["row_gaps"] if g], flush=True)
        finally:
            code, err = s.stop()
    med = {k: statistics.median(r["worst_gap"] for r in runs if r["kind"] == k) for k in ("without", "with")}
    result = dict(runs=runs, median_worst_gap=med, stall_seconds=med["with"] - med["without"])
    Path(out_json).write_text(json.dumps(result, indent=1))
    print("median worst gap without", round(med["without"], 4), "with", round(med["with"], 4),
          "stall", round(result["stall_seconds"], 4))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]), *sys.argv[6:])
