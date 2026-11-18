#!/usr/bin/env bash
# Runs in the container: long-decode-v1's runs (validation/long-decode-v1/README.md). Decodes token by token at a
# capacity of 32,768 with build TAG's liquid-lfm3, with the windowed layers' compact buffers on (the default above
# 4,096; LFM3_COMPACT_WINDOWS unset), off (=0) and on again, and a control at capacity 4,096.
#   tools/pascal/long_decode.sh TAG OUT_NAME        -> /workspace/build/OUT_NAME/{CASE.RUN.json, summary.json}
# Each run's record holds the ids, the SHA-256 of every step's FP32 logits, the NLL of the next window token at each
# teacher-forced step, the times, and the peak device memory sampled every 0.2 s. Exit 0 if rules 2 and 3 hold.
set -uo pipefail
tag=${1:?usage: long_decode.sh TAG OUT_NAME}; name=${2:?}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bin=/workspace/build/$tag/liquid-lfm3; out=/workspace/build/$name; M=/workspace/models/LFM3-Omni-2B
W=/workspace/data/ppl-windows
test ! -e "$out" || { echo "exists: $out"; exit 2; }
mkdir -p "$out"
echo "binary $(sha256sum "$bin")"; sha256sum "$W/windows-4096.i32" "$W/windows-32768.i32"
binary_sha=$(sha256sum "$bin" | cut -c1-64)
unset LFM3_COMPACT_WINDOWS LFM3_PREFILL_HEADS  # nothing inherited from the caller
# The peaks are the whole card's memory in use, and rule 2 compares them: the card must be otherwise idle.
idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "${idle:-9999}" -le 64 ] || { echo "the card is not idle: ${idle:-?} MiB in use before the first run"; exit 2; }

run() {  # CASE RUN COMPACT(unset|0) CAPACITY LENGTH WINDOW PROMPT TEACHER NEW
  local case=$1 run=$2 compact=$3 d=$out/$1.$2; mkdir -p "$d"
  python3 - "$W/windows-$5.i32" "$5" "$6" "$7" "$8" "$9" "$4" "$d" <<'PY'
import json, sys
from array import array
path, length, window, prompt, teacher, new, capacity, d = sys.argv[1], *map(int, sys.argv[2:8]), sys.argv[8]
a = array("i"); a.frombytes(open(path, "rb").read()); w = a[window * length:(window + 1) * length]
assert len(w) == length and 0 < prompt <= length and (teacher == 0 or prompt + teacher < length)
r = {"input_ids": list(w[:prompt]), "capacity": capacity, "trace_directory": d + "/trace", "trace_logits_only": True}
if capacity > 4096: r["experimental_long_context"] = True
if teacher: r["teacher_ids"] = list(w[prompt:prompt + teacher])
if new: r["max_new_tokens"] = new
json.dump(r, open(d + "/request.json", "w"))
json.dump(list(w[prompt:prompt + teacher + 1]) if teacher else [], open(d + "/targets.json", "w"))
PY
  if [ "$compact" = unset ]; then unset LFM3_COMPACT_WINDOWS; else export LFM3_COMPACT_WINDOWS=$compact; fi
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) > "$d/memory.txt" 2>/dev/null &
  local sampler=$!
  "$bin" --experimental $M "$d/request.json" "$d/native.json" > "$d/log" 2>&1; local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null; unset LFM3_COMPACT_WINDOWS
  if [ $rc != 0 ]; then echo "$case.$run: exit $rc: $(tail -c 300 "$d/log")"; rm -rf "$d/trace"; return 1; fi
  python3 - "$d" "$out/$case.$run.json" "$case" "$run" "$compact" "$4" "$5" "$6" "$7" "$binary_sha" "$W/windows-$5.i32" <<'PY'
import hashlib, json, math, os, sys
from array import array
d, dest, case, run, compact = sys.argv[1:6]; capacity, length, window, prompt = map(int, sys.argv[6:10])
binary_sha, windows_sha = sys.argv[10], hashlib.sha256(open(sys.argv[11], "rb").read()).hexdigest()
native = json.load(open(d + "/native.json")); targets = json.load(open(d + "/targets.json"))
steps = sorted(int(f.split(".")[0]) for f in os.listdir(d + "/trace") if f.endswith(".logits.f32"))
assert steps == list(range(len(steps))), "trace steps are not 0..n-1"
shas, nll = [], []
for k in steps:
    raw = open(f"{d}/trace/{k}.logits.f32", "rb").read(); shas.append(hashlib.sha256(raw).hexdigest())
    if k < len(targets):  # the next window token's NLL under this step's logits, over the whole output layer
        x = array("f"); x.frombytes(raw); m = max(x)
        nll.append(m + math.log(math.fsum(math.exp(v - m) for v in x)) - x[targets[k]])
peak = max(int(v) for v in open(d + "/memory.txt").read().split())
json.dump({"case": case, "run": run, "compact_windows_env": compact, "capacity": capacity, "window_length": length,
           "window": window, "prompt_tokens": prompt, "steps": len(steps),
           "binary_sha256": binary_sha, "windows_sha256": windows_sha,
           "teacher_argmax": native["teacher_argmax"], "generated_ids": native["generated_ids"],
           "logits_sha256": shas, "logits_digest": hashlib.sha256("".join(shas).encode()).hexdigest(),
           "first_position": prompt, "nll": nll, "prefill_seconds": native["prefill_seconds"],
           "decode_seconds": native["decode_seconds"], "peak_device_mib": peak}, open(dest, "w"))
print(f"{case}.{run} (LFM3_COMPACT_WINDOWS {compact}): {len(steps)} steps, {len(native['teacher_argmax'])} argmax ids, "
      f"{len(native['generated_ids'])} generated, digest {hashlib.sha256(''.join(shas).encode()).hexdigest()[:16]}, "
      f"prefill {native['prefill_seconds']:.1f} s, decode {native['decode_seconds']:.1f} s, peak {peak} MiB")
PY
  local prc=$?; rm -rf "$d/trace"; return $prc
}

fail=0
#   case    run      compact cap   len   window prompt teacher new
run control default  unset   4096  4096  0      300    1100    0 || fail=1
for c in "short 0 300" "middle 0 4000" "end 4 31667"; do
  set -- $c
  run $1 default unset 32768 32768 $2 $3 1100 0 || fail=1
  run $1 full    0     32768 32768 $2 $3 1100 0 || fail=1
  run $1 again   unset 32768 32768 $2 $3 1100 0 || fail=1
done
# Free-running greedy generation: the first window from 1 on whose default run reaches 520 tokens.
free_window=
for w in 1 2 3 5 6 7 8; do
  rm -rf "$out/free.default" "$out/free.default.json"
  run free default unset 32768 32768 $w 6000 0 600 || { fail=1; break; }
  n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['generated_ids']))" "$out/free.default.json")
  echo "free: window $w generated $n tokens"
  if [ "$n" -ge 520 ]; then free_window=$w; break; fi
  mv "$out/free.default.json" "$out/free-short-window$w.default.json"
done
if [ -n "$free_window" ]; then
  run free full  0     32768 32768 $free_window 6000 0 600 || fail=1
  run free again unset 32768 32768 $free_window 6000 0 600 || fail=1
else echo "free: no window reached 520 generated tokens"; fail=1; fi

python3 - "$out" <<'PY' || fail=1
import json, os, sys
out = sys.argv[1]; ok = True; summary = {}
def load(case, run):
    p = f"{out}/{case}.{run}.json"; return json.load(open(p)) if os.path.exists(p) else None
for case in ("short", "middle", "end", "free"):
    d, f, a = load(case, "default"), load(case, "full"), load(case, "again")
    if not (d and f and a): print(f"{case}: a run is missing"); ok = False; continue
    want = 1101 if case != "free" else len(d["generated_ids"])
    same = lambda x, y: all(x[k] == y[k] for k in ("teacher_argmax", "generated_ids", "logits_sha256"))
    r2, r3, count = same(d, f), same(d, a), d["steps"] == want and f["steps"] == want
    less = d["peak_device_mib"] < f["peak_device_mib"]
    differing = sum(x != y for x, y in zip(d["logits_sha256"], f["logits_sha256"]))
    summary[case] = {"steps": d["steps"], "steps_expected": want, "compact_equals_full": r2, "steps_differing": differing,
                     "default_repeat_identical": r3, "peak_default_mib": d["peak_device_mib"],
                     "peak_full_mib": f["peak_device_mib"], "default_peak_below_full": less,
                     "decode_seconds": {"default": d["decode_seconds"], "full": f["decode_seconds"]},
                     "prefill_seconds": {"default": d["prefill_seconds"], "full": f["prefill_seconds"]}}
    print(f"{case}: {d['steps']} steps (want {want}); compact vs full {'IDENTICAL' if r2 else f'DIFFER at {differing} steps'}; "
          f"repeat {'IDENTICAL' if r3 else 'DIFFER'}; peak {d['peak_device_mib']} MiB compact, {f['peak_device_mib']} MiB full "
          f"({'compact lower' if less else 'COMPACT NOT LOWER'}); decode {d['decode_seconds']:.1f} s compact, {f['decode_seconds']:.1f} s full")
    ok &= r2 and r3 and count and less
c, s = load("control", "default"), load("short", "default")
if c and s:
    summary["control"] = {"steps": c["steps"], "peak_mib": c["peak_device_mib"], "decode_seconds": c["decode_seconds"],
                          "same_tokens_as_short": c["teacher_argmax"] == s["teacher_argmax"],
                          "logits_equal_short": c["logits_sha256"] == s["logits_sha256"]}
    print(f"control (capacity 4,096): {c['steps']} steps; its logits {'equal' if c['logits_sha256'] == s['logits_sha256'] else 'differ from'} "
          f"the short case's at capacity 32,768 (information)")
else: print("control: missing"); ok = False
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
sys.exit(0 if ok else 1)
PY
echo "LONG DECODE (rules 2 and 3) $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
