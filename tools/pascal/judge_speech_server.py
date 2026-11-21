"""speech-v1 rule 11 as amended by decision 10 (2026-11-20): the five speech server checks judged on everything but
timing, every timing limit recorded with its figure. Reads the checks' OUT_JSONs from DIR (PREFIX-speech-server.json,
-speech-stream.json, -audio-in-server.json, -speech-concurrent.json, -batch-server.json) and writes DIR/PREFIX-rule11.json.

Judged (each check's own criteria, the timing terms left out):
  speech-server: every case (200, WAV equal to lfm3-speech's, every chunk ended by EOS, interleaved text equal); both
    worst cases 200, the large-cap one truncated; the repeat equal; the text control equal; exit 0.
  stream: every case exact (streamed PCM and done event equal the unstreamed reply), with audio events unless ASR, the
    budget case truncated; after the cancelled stream the next request exact; exit 0.
  audio-in-server: every case (200, reference exit 0, text, WAV and chunk texts equal, text present unless audio);
    repeat equal; text control equal; exit 0.
  concurrent: the check's verdict (byte equality only; it has no timing term).
  batch-server: every crowd identical; the text under load equal; every real-time run done; exit 0.
Recorded, not judged: speech-server's 240 s worst-case limit; stream's first audio (TTS median <= 1.0 s, max <= 1.5 s;
interleaved median <= 1.5 s), no gaps, cancel delay <= 1.0 s, stream cost <= 1.03; audio-in's 240 s on the 90 s clip;
batch-server's first audio under load (median <= 1.0 s) and real time at 8 (no gaps).
    python3 tools/pascal/judge_speech_server.py DIR PREFIX
"""
import json
import sys
from pathlib import Path

d, prefix = Path(sys.argv[1]), sys.argv[2]
load = lambda name: json.loads((d / f"{prefix}-{name}.json").read_text())
verdict, timing = {}, {}

s = load("speech-server")
verdict["speech-server"] = all(c["pass"] for c in s["cases"].values()) and \
    s["worst_case_budget"]["status"] == 200 and s["worst_case_large-cap"]["status"] == 200 and \
    s["worst_case_large-cap"]["truncated"] is True and s["repeat_identical"] and s["text_control_identical"] and \
    s["exit_code"] == 0
timing["speech-server"] = {f"worst case {n} (limit 240 s)": s[f"worst_case_{n}"]["wall_seconds"] for n in ("budget", "large-cap")}

t = load("speech-stream")
verdict["speech-stream"] = all(c["exact"] and (c["kind"] == "asr" or c["audio_events"] > 0) and
                               (n != "budget-truncated-s1" or c["truncated"] is True) for n, c in t["cases"].items()) and \
    t["cancel"]["next_exact"] and t["server_exit"] == 0
timing["speech-stream"] = dict(first_audio=t["first_audio"], first_audio_pass=t["rule2_pass"],
                               stalled_cases={n: [c["stalls"], c["stalled_seconds"]] for n, c in t["cases"].items() if c["stalls"]},
                               no_gaps_pass=t["rule3_pass"], cancel_delay=t["cancel"]["delay"],
                               cancel_pass=t["cancel"]["pass_"], cost_ratio=t["cost"]["ratio"], cost_pass=t["cost"]["pass_"])

a = load("audio-in-server")
verdict["audio-in-server"] = all(c["status"] == 200 and c["reference_exit"] == 0 and c["text_identical"] and
                                 c["wav_identical"] and c.get("chunks_identical", True) for c in a["cases"].values()) and \
    a["repeat_identical"] and a["text_control_identical"] and a["exit_code"] == 0
timing["audio-in-server"] = {"asr-joined-90s (limit 240 s)": a["cases"]["asr-joined-90s"]["wall_seconds"]}

verdict["speech-concurrent"] = load("speech-concurrent")["pass"] is True

b = load("batch-server")
verdict["batch-server"] = b["rule1_pass"] and b["text_under_load"]["identical"] and \
    all(r["all_done"] for r in b["real_time"].values()) and b["server_exit"] == 0
timing["batch-server"] = dict(first_audio_under_load_median=b["first_audio_under_load"]["median"],
                              first_audio_pass=b["first_audio_under_load"]["pass_"],
                              stalls={n: r["stalls"] for n, r in b["real_time"].items()}, real_time_8_pass=b["rule3_pass"],
                              largest_n_without_gaps=b["largest_n_without_gaps"],
                              text_under_load=b["text_under_load"])

report = dict(build=prefix, judged=verdict, recorded_timing=timing, rule11_pass=all(verdict.values()))
(d / f"{prefix}-rule11.json").write_text(json.dumps(report, indent=1) + "\n")
for k, v in verdict.items(): print(f"rule 11 {prefix} {k}: {'pass' if v else 'FAIL'}")
print(f"RULE 11 ({prefix}, decision 10)", "PASS" if report["rule11_pass"] else "FAIL")
sys.exit(0 if report["rule11_pass"] else 1)
