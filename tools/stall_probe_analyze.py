#!/usr/bin/env python3
"""Analysis for tools/thor/stall_probe.sh: each request's prefill window (its output JSON's mtime, less its generate,
codec-free TTS so the rest is small) against the 100 ms counter samples. A stall: prefill more than 1 s above the
run's median. Per stall and for a matched set of normal requests, the counter deltas inside the window.
Usage: stall_probe_analyze.py OUT_DIR"""
import json, statistics, sys
from pathlib import Path

o = Path(sys.argv[1])
samples = [json.loads(l) for l in open(o / "sampler.jsonl") if l.strip()]
reqs = []
for line in open(o / "list.txt"):
    stem = Path(line.split()[1].replace("/workspace/liquid-CUDA/build", str(o.parent)))
    f = stem.with_suffix(".json") if stem.suffix else Path(str(stem) + ".json")
    if not f.exists():
        continue
    j = json.loads(f.read_text())
    end = f.stat().st_mtime - j.get("generate_seconds", 0)
    reqs.append(dict(name=stem.name, prefill=j["prefill_seconds"], start=end - j["prefill_seconds"], end=end))
med = statistics.median(r["prefill"] for r in reqs)
stalls = [r for r in reqs if r["prefill"] > med + 1.0]
print(f"{len(reqs)} requests, median prefill {med:.3f} s, stalls (> median + 1 s): {len(stalls)}")


def delta(r):
    inside = [s for s in samples if r["start"] - 0.15 <= s["t"] <= r["end"] + 0.15]
    if len(inside) < 2:
        return None
    a, b = inside[0], inside[-1]
    d = {k: b["vm"][k] - a["vm"][k] for k in b["vm"] if b["vm"][k] != a["vm"][k]}
    if a["proc"] and b["proc"]:
        d.update({"proc_" + k: b["proc"][k] - a["proc"][k] for k in ("minflt", "majflt", "utime", "stime")
                  if b["proc"][k] != a["proc"][k]})
    return d


for r in stalls:
    print(f"STALL {r['name']} prefill {r['prefill']:.3f}: {delta(r)}")
normal = [r for r in reqs if r not in stalls][:: max(1, len(reqs) // 20)]
for r in normal[:10]:
    print(f"normal {r['name']} prefill {r['prefill']:.3f}: {delta(r)}")
