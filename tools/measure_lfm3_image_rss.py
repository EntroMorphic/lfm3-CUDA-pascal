#!/usr/bin/env python3
"""validation/image-queue-v1 rule 3: the server's peak resident memory with 16 simultaneous 4K image requests, against
its idle resident memory (after one warm-up image request, so the engine's lasting allocations are in the idle
figure). Black box on liquid-lfm3-server (Thor, lfm3-profile-v1): VmRSS sampled every 10 ms from /proc while the 16
requests (stage-0 "cherry" and "mirror", alternating, text mode, 16 new tokens) run; each request's status recorded
(200, or 429 queue_full / 504 by the server's admission: memory is the question here, not admission).
Usage: measure_lfm3_image_rss.py SERVER MODEL_DIR ROUND_DIR OUT_JSON   (ROUND_DIR: vision_e2e_round.sh's output)
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_image_server import as_client  # noqa: E402
from check_lfm3_speech_stream import Server  # noqa: E402


def rss_kb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("no VmRSS")


def main(server_bin, model, round_dir, out_json):
    bodies = []
    for name in ("cherry", "mirror"):
        b = as_client(json.loads((Path(round_dir) / "req" / f"{name}.json").read_text()))
        b["max_new_tokens"] = 16
        bodies.append(b)
    with tempfile.TemporaryDirectory() as scratch:
        s = Server(server_bin, model, scratch)
        try:
            pid = s.proc.pid
            assert s.post(bodies[0])[0] == 200  # warm: the encoder, its weights and scratch
            time.sleep(2)
            idle = rss_kb(pid)
            peak, stop = [idle], threading.Event()
            def sample():
                while not stop.is_set():
                    peak.append(rss_kb(pid))
                    time.sleep(0.01)
            sampler = threading.Thread(target=sample)
            sampler.start()
            statuses = [None] * 16
            def one(i):
                statuses[i] = s.post(bodies[i % 2])
            threads = [threading.Thread(target=one, args=(i,)) for i in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            stop.set()
            sampler.join()
            after = rss_kb(pid)
        finally:
            s.stop()
    codes = [(st, r.get("error")) for st, r in statuses]
    result = dict(idle_kb=idle, peak_kb=max(peak), after_kb=after, growth_mb=(max(peak) - idle) / 1024,
                  passed=(max(peak) - idle) <= 1024 * 1024, statuses=codes)
    Path(out_json).write_text(json.dumps(result, indent=1))
    print(f"idle {idle / 1024:.0f} MB, peak {max(peak) / 1024:.0f} MB (+{result['growth_mb']:.0f} MB), after "
          f"{after / 1024:.0f} MB; statuses {sorted(set(map(str, codes)))}; RULE 3 {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
