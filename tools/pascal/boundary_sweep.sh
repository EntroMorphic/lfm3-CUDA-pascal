#!/usr/bin/env bash
# Runs in the container: the compact window buffers against the full-length layout at the boundaries.
#   tools/pascal/boundary_sweep.sh TAG OUT_NAME [WRAPPER...]   -> /workspace/build/OUT_NAME/{sweep.txt, CASE.SIDE.json}
# For each prompt length around the window (256), the buffer (512) and their multiples, teacher-forces STEPS tokens
# with the compact buffers and without, at capacity 4,097 (the smallest where compact is the default; side A unset,
# side B LFM3_COMPACT_WINDOWS=0) and at 4,096 (side A =1, side B unset). Also: a request that fills the capacity
# exactly by teacher forcing, a generation of STEPS tokens up to the end of the capacity (the last generated token is
# never fed back, so it reaches position 4,096 of 4,097), and a prompt of the whole capacity. Every pair must give the
# same ids and byte-identical FP32 logits at every step, with the expected number of steps. Exit 0 only if all do;
# exit 2 if no pair under this rule ran.
# Then, for information (a difference is never a failure, a run that fails is): the same request at capacity 4,097 and
# at 4,096, both as shipped, for the prompt lengths in XPROMPTS. Above 4,096 prefill attention runs in tiles of 128
# rows, so this shows where the capacity changes the bits.
# WRAPPER (e.g. compute-sanitizer --tool memcheck --error-exitcode 9) wraps the engine; STEPS, PROMPTS and XPROMPTS
# via env (PROMPTS="" or XPROMPTS="" skips that part).
set -uo pipefail
tag=${1:?usage: boundary_sweep.sh TAG OUT_NAME [WRAPPER...]}; name=${2:?}; shift 2
bin=/workspace/build/$tag/liquid-lfm3; out=/workspace/build/$name; M=/workspace/models/LFM3-Omni-2B
W=/workspace/data/ppl-windows/windows-32768.i32
steps=${STEPS:-600}; prompts=${PROMPTS-"1 2 8 9 255 256 257 258 511 512 513 767 768 769 1025"}
xprompts=${XPROMPTS-"9 129 136 137 257 300 513 1025 2049 3329"}
test -x "$bin" || { echo "no binary: $bin"; exit 2; }
test ! -e "$out" || { echo "exists: $out"; exit 2; }
mkdir -p "$out"; echo "binary $(sha256sum "$bin")"; fail=0; pairs=0

one() {  # CASE SIDE COMPACT(unset|0|1) CAPACITY PROMPT TEACHER NEW WRAPPER... -> $out/CASE.SIDE.json; 0 if the run and its record are good
  local case=$1 side=$2 compact=$3 capacity=$4 prompt=$5 teacher=$6 new=$7; shift 7
  local d=$out/$case.$side; rm -rf "$d"; mkdir -p "$d"
  python3 - "$W" "$capacity" "$prompt" "$teacher" "$new" "$d" <<'PY' || return 1
import json, sys
from array import array
path, capacity, prompt, teacher, new, d = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
a = array("i"); a.frombytes(open(path, "rb").read()); w = a[:32768]
r = {"input_ids": list(w[:prompt]), "capacity": capacity, "trace_directory": d + "/trace", "trace_logits_only": True}
if capacity > 4096: r["experimental_long_context"] = True
if teacher: r["teacher_ids"] = list(w[prompt:prompt + teacher])
if new: r["max_new_tokens"] = new
json.dump(r, open(d + "/request.json", "w"))
PY
  if [ "$compact" = unset ]; then unset LFM3_COMPACT_WINDOWS; else export LFM3_COMPACT_WINDOWS=$compact; fi
  "$@" "$bin" --experimental $M "$d/request.json" "$d/native.json" > "$d/log" 2>&1; local rc=$?
  unset LFM3_COMPACT_WINDOWS
  if [ $rc != 0 ]; then echo "$case.$side: exit $rc: $(tail -c 300 "$d/log" | tr '\n' ' ')"; rm -rf "$d/trace"; return 1; fi
  # Under compute-sanitizer its own summary decides, whatever the exit status (red-team-2 C8); its log is kept.
  if [[ " $* " == *compute-sanitizer* ]]; then
    cp "$d/log" "$out/$case.$side.sanitizer.log"
    grep -q "ERROR SUMMARY: 0 errors" "$d/log" || { echo "$case.$side: the sanitizer's log has no 'ERROR SUMMARY: 0 errors'"; rm -rf "$d/trace"; return 1; }
  fi
  python3 - "$d" "$out/$case.$side.json" <<'PY'
import hashlib, json, os, sys
d, dest = sys.argv[1:3]; native = json.load(open(d + "/native.json"))
steps = sorted(int(f.split(".")[0]) for f in os.listdir(d + "/trace") if f.endswith(".logits.f32"))
assert steps and steps == list(range(len(steps))), "trace steps are not 0..n-1"
shas = [hashlib.sha256(open(f"{d}/trace/{k}.logits.f32", "rb").read()).hexdigest() for k in steps]
json.dump({"steps": len(steps), "teacher_argmax": native["teacher_argmax"], "generated_ids": native["generated_ids"],
           "logits_sha256": shas}, open(dest, "w"))
PY
  local prc=$?; rm -rf "$d/trace"; [ $prc = 0 ] && rm -rf "$d"; return $prc
}

pair() {  # CASE CAPACITY_A CAPACITY_B PROMPT TEACHER NEW WANT_STEPS(0: whatever side A ran) COMPACT_A COMPACT_B RULE(rule|info) WRAPPER...
  local case=$1 capacity=$2 capacity_b=$3 prompt=$4 teacher=$5 new=$6 want=$7 ca=$8 cb=$9 rule=${10}; shift 10
  [ "$rule" = rule ] && pairs=$((pairs + 1))
  one "$case" a "$ca" "$capacity" "$prompt" "$teacher" "$new" "$@" || { echo "$case: side a FAILED" | tee -a "$out/sweep.txt"; fail=1; return; }
  one "$case" b "$cb" "$capacity_b" "$prompt" "$teacher" "$new" "$@" || { echo "$case: side b FAILED" | tee -a "$out/sweep.txt"; fail=1; return; }
  python3 - "$out/$case.a.json" "$out/$case.b.json" "$case" "$want" "$rule" <<'PY' | tee -a "$out/sweep.txt"
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); case, want = sys.argv[3], int(sys.argv[4])
same = all(a[k] == b[k] for k in ("teacher_argmax", "generated_ids", "logits_sha256"))
count = a["steps"] == b["steps"] and (want == 0 or a["steps"] == want) and a["steps"] > 0
differ = sum(x != y for x, y in zip(a["logits_sha256"], b["logits_sha256"]))
note = "" if count else f" (WANT {want}, other side {b['steps']})"
verdict = "IDENTICAL" if same else f"DIFFER at {differ} steps"
if sys.argv[5] == "info":
    first = next((i for i, (x, y) in enumerate(zip(a["logits_sha256"], b["logits_sha256"])) if x != y), None)
    print(f"{case}: {a['steps']} steps{note}; {verdict}{'' if same else f', first at step {first}'} (information)")
    sys.exit(0 if count else 1)
print(f"{case}: {a['steps']} steps{note}; {verdict} {'ok' if same and count else 'FAIL'}")
sys.exit(0 if same and count else 1)
PY
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1
}

for p in $prompts; do
  pair "c4097-p$p" 4097 4097 "$p" "$steps" 0 $((steps + 1)) unset 0 rule "$@"
  pair "c4096-p$p" 4096 4096 "$p" "$steps" 0 $((steps + 1)) 1 unset rule "$@"
done
for p in $xprompts; do
  pair "x4097-4096-p$p" 4097 4096 "$p" "$steps" 0 $((steps + 1)) unset unset info "$@"
done
if [ -z "${PROMPTS+set}" ]; then
  # To the end of the capacity: prompt + teacher tokens = capacity; prompt + generated tokens = capacity; prompt alone.
  pair fill-teacher 4097 4097 $((4097 - steps)) "$steps" 0 $((steps + 1)) unset 0 rule "$@"
  pair fill-generate 4097 4097 $((4097 - steps)) 0 "$steps" "$steps" unset 0 rule "$@"
  pair fill-prompt 4097 4097 4097 0 0 1 unset 0 rule "$@"
fi
done_pairs=$(grep -c ' ok$' "$out/sweep.txt" 2>/dev/null || true)
echo "pairs run $pairs, ok ${done_pairs:-0}" | tee -a "$out/sweep.txt"
[ "${done_pairs:-0}" = "$pairs" ] || fail=1
if [ "$pairs" = 0 ]; then echo "BOUNDARY SWEEP NOT RUN: no pair under the rule"; exit 2; fi
echo "BOUNDARY SWEEP $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
