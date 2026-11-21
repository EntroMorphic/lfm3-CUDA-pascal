"""speech-v1 rule 11 as amended by decision 10 (2026-11-20): the five speech server checks judged on everything but
timing, every timing limit recorded with its figure. Reads the checks' OUT_JSONs from DIR (PREFIX-speech-server.json,
-speech-stream.json, -audio-in-server.json, -speech-concurrent.json, -batch-server.json) and writes DIR/PREFIX-rule11.json.

First, that the reports are of this build: every report's server and lfm3-speech SHA-256 are those of one of BUILDS
(comma-separated build directories: the final build and the development builds whose binaries the rule run shows
compute the same, speech-v1 decision 11), and the references (REFS_DIR, the checks' BUILD_DIR) were made by one of
their lfm3-speech binaries; and that every check ran all
its cases (speech-server 5 and both worst cases; stream 42: 25 TTS, 14 interleaved, 3 ASR; audio-in 9; batch 4
crowds of 42 and real time at 8, 12, 16 and 7 + 5).
Judged (each check's own criteria, the timing terms left out):
  speech-server: every case (200, WAV equal to lfm3-speech's, every chunk ended by EOS, interleaved text equal); both
    worst cases 200, the large-cap one truncated; the repeat equal; the text control equal; exit 0.
  stream: every case exact (streamed PCM and done event equal the unstreamed reply), with audio events unless ASR, the
    budget case truncated; after the cancelled stream the next request exact; the stream-cost loop's ten requests all
    completed (finite times); exit 0.
  audio-in-server: every case (200, reference exit 0, text, WAV and chunk texts equal, text present unless audio);
    repeat equal; text control equal; exit 0.
  concurrent: the check's verdict (byte equality only; it has no timing term), over its 41 requests and five runs.
  batch-server: every crowd identical; the text under load equal; every real-time run done; every server it started
    exited 0 (server_exits: alone, crowd, real time at 8, 12, 16, 7 + 5).
Recorded, not judged: speech-server's 240 s worst-case limit; stream's first audio (TTS median <= 1.0 s, max <= 1.5 s;
interleaved median <= 1.5 s), no gaps, cancel delay <= 1.0 s, stream cost <= 1.03; audio-in's 240 s on the 90 s clip;
batch-server's first audio under load (median <= 1.0 s) and real time at 8 (no gaps).
    python3 tools/pascal/judge_speech_server.py DIR PREFIX BUILDS REFS_DIR
"""
import hashlib
import json
import math
import sys
from pathlib import Path

if len(sys.argv) != 5: sys.exit(__doc__)
d, prefix, builds, refs = Path(sys.argv[1]), sys.argv[2], [Path(x) for x in sys.argv[3].split(",")], Path(sys.argv[4])
load = lambda name: json.loads((d / f"{prefix}-{name}.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
servers = {sha(b / "liquid-lfm3-server") for b in builds if (b / "liquid-lfm3-server").exists()}
speeches = {sha(b / "lfm3-speech") for b in builds}
verdict, timing = {}, {}
verdict["reports of these builds"] = all(load(n).get("server_sha256") in servers for n in
                                         ("speech-server", "speech-stream", "audio-in-server", "batch-server")) and \
    load("batch-server").get("reference_server_sha256") in servers and \
    load("audio-in-server").get("speech_sha256") in speeches and load("speech-concurrent").get("speech_sha256") in speeches and \
    (refs / "lfm3-speech.sha256").read_text().split()[0] in speeches

s = load("speech-server")
verdict["speech-server"] = len(s["cases"]) == 5 and all(c["pass"] for c in s["cases"].values()) and \
    s["worst_case_budget"]["status"] == 200 and s["worst_case_large-cap"]["status"] == 200 and \
    s["worst_case_large-cap"]["truncated"] is True and s["repeat_identical"] and s["text_control_identical"] and \
    s["exit_code"] == 0
timing["speech-server"] = {f"worst case {n} (limit 240 s)": s[f"worst_case_{n}"]["wall_seconds"] for n in ("budget", "large-cap")}

t = load("speech-stream")
kinds = [c["kind"] for c in t["cases"].values()]
verdict["speech-stream"] = (kinds.count("tts"), kinds.count("interleaved"), kinds.count("asr")) == (25, 14, 3) and \
    "budget-truncated-s1" in t["cases"] and \
    all(len(t["cost"][k]) == 5 and all(math.isfinite(x) for x in t["cost"][k]) for k in ("plain", "stream")) and \
    all(c["exact"] and (c["kind"] == "asr" or c["audio_events"] > 0) and
                               (n != "budget-truncated-s1" or c["truncated"] is True) for n, c in t["cases"].items()) and \
    t["cancel"]["next_exact"] and t["server_exit"] == 0
timing["speech-stream"] = dict(first_audio=t["first_audio"], first_audio_pass=t["rule2_pass"],
                               stalled_cases={n: [c["stalls"], c["stalled_seconds"]] for n, c in t["cases"].items() if c["stalls"]},
                               no_gaps_pass=t["rule3_pass"], cancel_delay=t["cancel"]["delay"],
                               cancel_pass=t["cancel"]["pass_"], cost_ratio=t["cost"]["ratio"], cost_pass=t["cost"]["pass_"])

a = load("audio-in-server")
verdict["audio-in-server"] = len(a["cases"]) == 9 and \
    all(c["status"] == 200 and c["reference_exit"] == 0 and c["text_identical"] and c["wav_identical"] and
        c.get("chunks_identical", True) and bool(c["text"] or c["mode"] == "audio") for c in a["cases"].values()) and \
    a["repeat_identical"] and a["text_control_identical"] and a["exit_code"] == 0
timing["audio-in-server"] = {"asr-joined-90s (limit 240 s)": a["cases"]["asr-joined-90s"]["wall_seconds"]}

cc = load("speech-concurrent")
verdict["speech-concurrent"] = cc["pass"] is True and cc["requests"] == 41 and sorted(cc["runs"]) == sorted(["alone", "conc4", "conc8", "conc8-staggered", "conc16-staggered"])

b = load("batch-server")
verdict["batch-server"] = len(b["crowds"]) == 4 and all(c["identical"] == 42 and not c["differ"] for c in b["crowds"].values()) and \
    b["rule1_pass"] and b["text_under_load"]["identical"] and \
    sorted(b["real_time"]) == sorted(["8", "12", "16", "7 tts + 5 spoken joining"]) and \
    all(r["all_done"] for r in b["real_time"].values()) and b["server_exit"] == 0 and \
    b.get("server_exits", {}) == {"alone": 0, "crowd": 0, "real time 8": 0, "real time 12": 0, "real time 16": 0, "7 tts + 5 spoken": 0}
timing["batch-server"] = dict(first_audio_under_load_median=b["first_audio_under_load"]["median"],
                              first_audio_pass=b["first_audio_under_load"]["pass_"],
                              stalls={n: r["stalls"] for n, r in b["real_time"].items()}, real_time_8_pass=b["rule3_pass"],
                              largest_n_without_gaps=b["largest_n_without_gaps"],
                              text_under_load=b["text_under_load"])

report = dict(build=prefix, builds=[str(b) for b in builds], refs=str(refs), judged=verdict, recorded_timing=timing, rule11_pass=all(verdict.values()))
(d / f"{prefix}-rule11.json").write_text(json.dumps(report, indent=1) + "\n")
for k, v in verdict.items(): print(f"rule 11 {prefix} {k}: {'pass' if v else 'FAIL'}")
print(f"RULE 11 ({prefix}, decision 10)", "PASS" if report["rule11_pass"] else "FAIL")
sys.exit(0 if report["rule11_pass"] else 1)
