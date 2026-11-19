#!/usr/bin/env bash
# Runs in the container: phase 2's speed cases with build TAG's liquid-lfm3, as shipped (no tracing, no switches).
#   tools/pascal/speed.sh TAG OUT_NAME [REPEATS=3]   -> /workspace/build/OUT_NAME/{CASE.N.json, summary.json, summary.txt}
# Prompts are the first tokens of window 0 of the 32,768-token perplexity windows. Prefill cases run the prompt alone
# (no decode step; the prompt may fill the capacity); decode cases teacher-force their steps, so every run does the
# same work. One warm-up run first.
# Each run records the engine's prefill and decode times, peak device memory (the whole card's) and the SM and memory
# clocks sampled every 0.2 s. summary.json holds the median of the REPEATS runs and their spread. CASES (lines of
# CASE CAPACITY PROMPT TEACHER) replaces the list below.
set -uo pipefail
tag=${1:?usage: speed.sh TAG OUT_NAME [REPEATS]}; name=${2:?}; repeats=${3:-3}
bin=/workspace/build/$tag/liquid-lfm3; out=/workspace/build/$name; M=/workspace/models/LFM3-Omni-2B
W=/workspace/data/ppl-windows/windows-32768.i32
test -x "$bin" || { echo "no binary: $bin"; exit 2; }
test ! -e "$out" || { echo "exists: $out"; exit 2; }
unset LFM3_COMPACT_WINDOWS LFM3_PREFILL_HEADS
idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${idle:-9999}" -le 64 ] || { echo "the card is not idle: ${idle:-?} MiB in use"; exit 2; }
mkdir -p "$out"; echo "binary $(sha256sum "$bin")" | tee "$out/binary.txt"; binary_sha=$(sha256sum "$bin" | cut -c1-64)

# CASE CAPACITY PROMPT TEACHER
cases=${CASES:-"prefill-32 4096 32 0
prefill-257 4096 257 0
prefill-1024 4096 1024 0
prefill-4096 4096 4096 0
prefill-32768 32768 32768 0
decode-1k 1024 300 700
decode-8k 8192 7000 1000
decode-32k 32768 31000 700"}

one() {  # CASE CAPACITY PROMPT TEACHER RUN -> $out/CASE.RUN.json
  local d=$out/$1.$5; mkdir -p "$d"
  python3 - "$W" "$2" "$3" "$4" "$d" <<'PY' || return 1
import json, sys
from array import array
path, capacity, prompt, teacher, d = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5]
a = array("i"); a.frombytes(open(path, "rb").read()); w = a[:32768]
assert prompt + teacher <= capacity
r = {"input_ids": list(w[:prompt]), "capacity": capacity}
if capacity > 4096: r["experimental_long_context"] = True
if teacher: r["teacher_ids"] = list(w[prompt:prompt + teacher])
json.dump(r, open(d + "/request.json", "w"))
PY
  ( while :; do nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,pstate --format=csv,noheader,nounits; sleep 0.2; done ) > "$d/samples.txt" 2>/dev/null &
  local sampler=$!
  "$bin" --experimental $M "$d/request.json" "$d/native.json" > "$d/log" 2>&1; local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  [ $rc = 0 ] || { echo "$1.$5: exit $rc: $(tail -c 300 "$d/log")"; return 1; }
  python3 - "$d" "$out/$1.$5.json" "$1" "$2" "$3" "$4" "$binary_sha" <<'PY'
import json, sys
d, dest, case = sys.argv[1:4]; capacity, prompt, teacher = map(int, sys.argv[4:7])
native = json.load(open(d + "/native.json"))
rows = [line.split(", ") for line in open(d + "/samples.txt") if line.count(",") == 3]
busy = [r for r in rows if r[3].strip() in ("P0", "P2")]
record = {"case": case, "capacity": capacity, "prompt_tokens": prompt, "teacher_steps": teacher,
          "binary_sha256": sys.argv[7], "prefill_seconds": native["prefill_seconds"],
          "decode_seconds": native["decode_seconds"], "teacher_argmax": native["teacher_argmax"],
          "generated_ids": native["generated_ids"],
          "peak_device_mib": max(int(r[0]) for r in rows) if rows else None,  # None fails the summary below
          "sm_mhz": sorted({int(r[1]) for r in busy}), "mem_mhz": sorted({int(r[2]) for r in busy})}
if teacher: record["ms_per_step"] = native["decode_seconds"] / teacher * 1e3
json.dump(record, open(dest, "w"))
PY
  local prc=$?; [ $prc = 0 ] && rm -rf "$d"; return $prc
}

fail=0
one warmup 1024 300 100 0 || fail=1
while read -r case capacity prompt teacher; do
  for r in $(seq 1 "$repeats"); do one "$case" "$capacity" "$prompt" "$teacher" "$r" || fail=1; done
done <<< "$cases"
python3 - "$out" "$repeats" <<'PY' | tee "$out/summary.txt"
import json, os, statistics, sys
out, repeats = sys.argv[1], int(sys.argv[2]); summary = {}
names = sorted({f.rsplit(".", 2)[0] for f in os.listdir(out) if f.endswith(".json") and f[0] != "w" and f != "summary.json"},
               key=lambda c: (c.split("-")[0], int(c.split("-")[1].rstrip("k")) * (1000 if c.endswith("k") else 1)))
for case in names:
    runs = [json.load(open(f"{out}/{case}.{r}.json")) for r in range(1, repeats + 1)]
    key = "ms_per_step" if "ms_per_step" in runs[0] else "prefill_seconds"
    values = [r[key] for r in runs]
    same = all(r["teacher_argmax"] == runs[0]["teacher_argmax"] and r["generated_ids"] == runs[0]["generated_ids"] for r in runs)
    summary[case] = {"metric": key, "median": statistics.median(values), "min": min(values), "max": max(values),
                     "prefill_seconds_median": statistics.median(r["prefill_seconds"] for r in runs),
                     "peak_device_mib": max(r["peak_device_mib"] for r in runs),  # TypeError if a run has none: no summary
                     "sm_mhz": sorted({m for r in runs for m in r["sm_mhz"]}), "mem_mhz": sorted({m for r in runs for m in r["mem_mhz"]}),
                     "runs_same_ids": same}
    s = summary[case]
    unit = "ms per step" if key == "ms_per_step" else "s prefill"
    print(f"{case:14s} {s['median']:9.3f} {unit} (min {s['min']:.3f}, max {s['max']:.3f}); peak {s['peak_device_mib']} MiB; "
          f"sm {s['sm_mhz'][0] if s['sm_mhz'] else '?'}-{s['sm_mhz'][-1] if s['sm_mhz'] else '?'} MHz, mem {s['mem_mhz']}; ids {'same' if same else 'DIFFER'}")
json.dump(summary, open(out + "/summary.json", "w"), indent=1)
PY
[ "${PIPESTATUS[0]}" = 0 ] || { echo "speed.sh: the summary failed"; fail=1; }
exit $fail
