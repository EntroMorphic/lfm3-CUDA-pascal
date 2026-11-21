"""speech-v1 rule 12, through liquid-lfm3-server over HTTP: device memory with speech at the limits.

The server runs with --speech-rows ROWS. Each of serve-v1 rule 11's five text configurations (tools/pascal/serve_memory.py:
one session of 32,768; 4,096 + 28,672; 2,048 + 30,720; 16,384 + 16,384; /generate_batch on 4,096 + 28,672, every
session filled to capacity - 33 prompt tokens, 32 new) runs while every speech row generates: ROWS /speech requests
(the lighthouse paragraph three times, audio mode, split by sentence as the server does by default, max_new_tokens 1536
a sentence, seeds 1..ROWS; sent 0.5 s apart, each again after a 429 frontend_busy) are sent first, the text
configuration starts SETTLE (20) seconds after the first, and the configuration counts only if every speech request was still running when
its text requests ended. A sixth configuration is audio
in at the request limits beside text at its limits: the single text session (TOTAL) filled and resident, then one clip
as long as the 4 MiB body allows (LibriSpeech utterances joined, the last cut to fill it, 16 kHz 16-bit: 98.2 s;
transcribed, 64 new tokens) sent three
times in turn, while ROWS - 1 speech rows generate; it counts only if every audio request succeeded while the session
was open, and every speech row was still running when the last one ended. (Red-team review: the earlier version sent
the clip with the speech load, 20 s before the text session existed, so the two were never resident together.) Whole-card memory every 0.2 s (nvidia-smi);
every request must succeed (200) and every configuration's peak must be at most LIMIT MiB. First, for the record, the
speech rows alone (no text session). /generate_batch is an exclusive job: the server drains the speech rows before it
runs (lfm3_http.hpp, speech_first), so that configuration cannot overlap speech; it counts when every speech request
ended before the batch did (the speech batch, created at the first request and kept, is resident while the batch runs).
TOTAL (default 32,768) is the server's --total-capacity; the configurations are scaled to it (serve-v1 rule 11: each
pair's smaller session c * TOTAL // 32,768, the larger the rest; the single session TOTAL). A configuration whose call is refused (the worker gone: the service fails closed
on an engine error, 503 worker_unavailable) fails with the peak sampled until then, and the run stops there.
    python3 tools/pascal/speech_memory.py SERVER CHECKPOINT ROWS LIBRI_WAV_DIR OUTPUT [LIMIT=7740] [TOTAL=32768]
"""
import base64
import io
import json
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_lfm3_native_server import NativeServer, Client  # noqa: E402

server_path, checkpoint, rows, libri, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
limit = int(sys.argv[6]) if len(sys.argv) > 6 else 7740
total = int(sys.argv[7]) if len(sys.argv) > 7 else 32768
SETTLE = 20
out.mkdir(parents=True, exist_ok=False)
PARAGRAPH = ("The river leaves the mountains in early spring, carrying silt and snowmelt past the old mill, the orchards "
             "and the long fields of the valley floor, until it widens into the slow brown estuary where the herons wait. ")
_P = json.loads(Path("validation/speech-e2e-v1/prompts.json").read_text())
_C = next(c for c in _P["cases"] if c["case"] == "tts-long-male")
LIGHTHOUSE = [dict(role="system", content=_P["systems"][_C["system"]]), dict(role="user", content=" ".join([_C["user"]] * 3))]
records, report = [], dict(limit_mib=limit, speech_rows=rows, total_capacity=total, configurations=[], status="failed")


class Sampler:
    def __init__(self):
        self.peak, self.stop, self.rows, self.errors = 0, threading.Event(), [], 0
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                v = int(r.stdout.split()[0]) if r.returncode == 0 else None
            except (OSError, ValueError, IndexError): v = None
            if v is None: self.errors += 1
            else: self.rows.append(v); self.peak = max(self.peak, v)
            time.sleep(0.2)

    def done(self):
        self.stop.set(); self.thread.join(); return self.peak


CLIP_BUDGET = ((4 * 1024 * 1024 - 4096) * 3 // 4 - 44) // 2 * 2  # PCM bytes: the base64 body within 4 MiB, 4 KiB for the JSON


def long_clip():
    """LibriSpeech utterances joined (sorted names), the last one cut, to fill the 4 MiB /speech body: 16 kHz 16-bit mono
    is the densest format the server takes (lfm/wav.hpp; 16 kHz only), so this is the longest clip a request carries.
    (Red-team-2 D1: the earlier version stopped at the first utterance that did not fit, 88% of the budget.)"""
    pcm = bytearray()
    for w in sorted(libri.glob("*.wav")):
        with wave.open(str(w)) as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2), w
            pcm += f.readframes(f.getnframes())
        if len(pcm) >= CLIP_BUDGET: break
    assert len(pcm) >= CLIP_BUDGET, f"LibriSpeech holds {len(pcm)} bytes, fewer than the clip's {CLIP_BUDGET}"
    frames = [bytes(pcm[:CLIP_BUDGET])]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b"".join(frames))
    return buf.getvalue()


def text(repeats): return [dict(role="user", content=PARAGRAPH * repeats)]


server = NativeServer(server_path, checkpoint, out, "server", None, "--speech-rows", rows, "--total-capacity", total)
client = Client(server.port, server.token, records)
failed = False


def post_speech(body, result, delay):
    time.sleep(delay)
    for attempt in range(100):  # 429 frontend_busy: another request holds the front end (the reference's refusal); again
        post_once(body, result)
        if (result["status"], result.get("error")) != (429, "frontend_busy"): return
        result["frontend_busy"] = attempt + 1; time.sleep(0.25)


def post_once(body, result):
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}/speech", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + server.token})
    result["start"] = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            reply = json.loads(r.read()); result["status"] = r.status
    except urllib.error.HTTPError as e:
        result["status"], reply = e.code, json.loads(e.read() or b"{}")
    except Exception as e:  # noqa: BLE001
        result["status"], reply = None, dict(error=repr(e))
    result["end"] = time.monotonic()
    result.update({k: reply.get(k) for k in ("error", "frames", "audio_seconds", "truncated")})


try:
    sid = client.call("POST", "/sessions", dict(capacity=min(4096, total)))["session_id"]  # the calibration (TOTAL may be less)
    n = {}
    for r in ((1, 11) if total >= 4096 else (1, 3)):
        client.call("POST", f"/sessions/{sid}/reset", {})
        n[r] = client.call("POST", "/generate", dict(session_id=sid, messages=text(r), max_new_tokens=1))["consumed_tokens"]
    hi = max(n); b = (n[hi] - n[1]) / (hi - 1); a = n[1] - b
    client.call("POST", f"/sessions/{sid}/close", {})
    # Idle after the warm requests (the weights resident): every configuration's peak must be above it (red-team-2 D7:
    # sampled before the first request, the idle figure was 2 MiB and the check could not fail).
    idle = Sampler(); time.sleep(2); report["before_mib"] = idle.done(); report["before_samples"] = len(idle.rows)
    assert len(idle.rows) >= 3 and report["before_mib"] > 1024, f"idle sample {report['before_mib']} MiB ({len(idle.rows)} samples)"
    report["calibration"] = {f"tokens_at_{k}": v for k, v in n.items()}
    def fill(capacity): return max(1, int((capacity - 33 - a) // b))
    clip = long_clip()
    report["clip"] = dict(bytes=len(clip), seconds=(len(clip) - 44) / 32000, pcm_budget_bytes=CLIP_BUDGET)
    assert len(clip) - 44 == CLIP_BUDGET

    def run_configuration(name, capacities, batch=False, speech_rows=rows, audio_in=False):
        global failed
        speech = [dict(messages=LIGHTHOUSE, mode="audio", max_new_tokens=1536, seed=k + 1)
                  for k in range(speech_rows)]
        audio_body = dict(messages=[{"role": "system", "content": "Respond with text."},
                                    {"role": "user", "content": [{"type": "audio", "audio": base64.b64encode(clip).decode()},
                                                                 {"type": "text", "text": "Transcribe the audio."}]}],
                          mode="text", max_new_tokens=64, seed=1)
        audio = []
        sampler = Sampler(); entry = dict(name=name, capacities=capacities, speech=len(speech), prompts=[], statuses=[])
        results = [dict() for _ in speech]
        threads = [threading.Thread(target=post_speech, args=(body, res, 0.5 * k)) for k, (body, res) in enumerate(zip(speech, results))]
        try:
            [t.start() for t in threads]; time.sleep(SETTLE)
            text_start = time.monotonic()
            sids = [client.call("POST", "/sessions", dict(capacity=c))["session_id"] for c in capacities]
            bodies = [dict(session_id=s, messages=text(fill(c)), max_new_tokens=32) for s, c in zip(sids, capacities)]
            if batch:
                status, reply = client.call("POST", "/generate_batch", dict(requests=bodies), expected=(200, 400, 429, 500, 503))
                entry["statuses"].append(status)
                if status == 200:
                    entry["prompts"] = [x["consumed_tokens"] - len(x["generated_ids"]) + 1 for x in reply["results"]]
                    entry["gpu_batching"] = reply.get("gpu_batching")
                else: entry["error"] = reply
            else:
                for body in bodies:
                    status, reply = client.call("POST", "/generate", body, expected=(200, 400, 429, 500, 503))
                    entry["statuses"].append(status)
                    if status == 200: entry["prompts"].append(reply["consumed_tokens"] - len(reply["generated_ids"]) + 1)
                    else: entry["error"] = reply
            text_end = time.monotonic()
            if audio_in:  # the session still open and filled: the clip, three times in turn
                for k in range(3):
                    audio.append(dict()); post_speech(audio_body, audio[-1], 0)
            last_end = time.monotonic()
            for s in sids: client.call("POST", f"/sessions/{s}/close", {})
            [t.join() for t in threads]
        finally:
            entry["peak_mib"] = sampler.done(); entry["samples"] = len(sampler.rows); entry["sampler_errors"] = sampler.errors
            report["last_peak_mib"] = entry["peak_mib"]
        entry["text_seconds"] = round(text_end - text_start, 1)
        entry["speech"] = [dict(r, start=round(r["start"] - text_start, 1), end=round(r["end"] - text_start, 1)) for r in results]
        entry["audio_in"] = [dict(r, start=round(r["start"] - text_start, 1), end=round(r["end"] - text_start, 1)) for r in audio]
        overlapped = all(r.get("end", 0) >= last_end for r in results[:speech_rows])
        if batch: overlapped = all(r.get("end", text_end) < text_end for r in results[:speech_rows])  # drained, then the batch
        entry["speech_overlapped_text"] = overlapped
        ok = all(s == 200 for s in entry["statuses"]) and all(r["status"] == 200 for r in results) and overlapped and \
            (not audio_in or (len(audio) == 3 and all(r["status"] == 200 and text_end <= r["start"] and r["end"] <= last_end
                                                      for r in audio))) and \
            entry["peak_mib"] <= limit and entry["samples"] >= 5 and entry["peak_mib"] > report["before_mib"] and \
            len(entry["prompts"]) == len(capacities) and all(p >= c - 33 - 2 * b for p, c in zip(entry["prompts"], capacities))
        if batch: ok = ok and entry.get("gpu_batching") is True
        entry["ok"] = ok; failed |= not ok; report["configurations"].append(entry)
        print(f"rule 12 {name} with {len(speech)} speech{' and 3 audio in' if audio_in else ''}: text {entry['statuses']} in {entry['text_seconds']} s, speech "
              f"{[r['status'] for r in results]} (overlapped {overlapped}), peak {entry['peak_mib']} MiB "
              f"({entry['samples']} samples): {'ok' if ok else 'FAIL'}", flush=True)

    def pair(c): return [c * total // 32768, total - c * total // 32768]
    configurations = [(("speech-only", []), {}), ((f"one-{total}", [total]), {})]
    configurations += [(("+".join(map(str, pair(c))), pair(c)), {}) for c in (4096, 2048, 16384)]
    configurations += [(("batch-" + "+".join(map(str, pair(4096))), pair(4096)), dict(batch=True)),
                       ((f"audio-in-one-{total}", [total]), dict(speech_rows=rows - 1, audio_in=True))]
    for args, kw in configurations:
        try:
            run_configuration(*args, **kw)
        except AssertionError as e:  # a refused call (a worker gone: 503 worker_unavailable); later ones would not run
            failed = True
            report["configurations"].append(dict(name=args[0], ok=False, error=repr(e)[:2000], peak_mib=report.get("last_peak_mib")))
            print(f"rule 12 {args[0]}: {repr(e)[:300]}, peak {report.get('last_peak_mib')} MiB: FAIL (stopped)", flush=True)
            break
    report["status"] = "failed" if failed else "passed"
finally:
    report["server_exit"] = server.stop()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "records.json").write_text(json.dumps(records, indent=1)[:2_000_000] + "\n")
print(f"SPEECH MEMORY (rule 12, {rows} rows)", "PASS" if report["status"] == "passed" and report.get("server_exit") == 0 else "FAIL")
sys.exit(0 if report["status"] == "passed" and report.get("server_exit") == 0 else 1)
