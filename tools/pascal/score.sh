#!/usr/bin/env bash
# Runs in the container: score perplexity windows with build TAG's lfm3-score, as tools/thor/run_ppl.sh does for the
# Thor engine. Writes OUT/nll-pascal-L.f32, its meta.json (the windows and scorer hashed where and when they are
# read), the scorer's JSON line, and the peak device memory sampled every 0.2 s while it ran.
#   tools/pascal/score.sh TAG OUT_NAME LENGTH... [WINDOWS_DIR=/workspace/data/ppl-windows via env]
# The peak is the whole card's memory in use (nvidia-smi), so the card must be otherwise idle; what it held before the
# run is recorded. The engine's environment switches in effect are recorded too. Exit 0 only if every length was
# scored and recorded.
set -uo pipefail
tag=${1:?usage: score.sh TAG OUT_NAME LENGTH...}; name=${2:?}; shift 2
[ $# -ge 1 ] || { echo "usage: score.sh TAG OUT_NAME LENGTH..."; exit 2; }
W=${WINDOWS_DIR:-/workspace/data/ppl-windows}
scorer=/workspace/build/$tag/lfm3-score; out=/workspace/build/$name; mkdir -p "$out"
for L in "$@"; do
  test ! -e "$out/nll-pascal-$L.f32" || { echo "exists: $out/nll-pascal-$L.f32"; exit 1; }
  windows_sha=$(sha256sum "$W/windows-$L.i32" | cut -c1-64); scorer_sha=$(sha256sum "$scorer" | cut -c1-64)
  [ ${#windows_sha} = 64 ] && [ ${#scorer_sha} = 64 ] || { echo "length $L: cannot hash the windows or the scorer"; exit 1; }
  baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  switches=$(env | grep -E '^(LFM_|LFM3_)[A-Z0-9_]*=' | sort | tr '\n' ' ')
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) > "$out/memory-$L.txt" 2>/dev/null &
  sampler=$!
  start=$(date +%s.%N)
  "$scorer" /workspace/models/LFM3-Omni-2B "$W/windows-$L.i32" "$L" "$out/nll-pascal-$L.f32" > "$out/score-$L.json" 2> "$out/score-$L.log"
  rc=$?; end=$(date +%s.%N); kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  [ $rc = 0 ] || { echo "length $L: scorer exit $rc: $(tail -c 400 "$out/score-$L.log")"; exit $rc; }
  python3 - "$out" "$L" "$scorer_sha" "$tag" "$windows_sha" "$start" "$end" "$baseline" "$switches" <<'PY' || { echo "length $L: the record could not be written"; exit 1; }
import hashlib, json, sys
out, length, scorer_sha, tag, windows_sha, start, end = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7])
baseline, switches = int(sys.argv[8]), sys.argv[9].split()
peak = max(int(x) for x in open(f"{out}/memory-{length}.txt").read().split())
score = json.loads(open(f"{out}/score-{length}.json").read().strip().splitlines()[-1])
nll_sha = hashlib.sha256(open(f"{out}/nll-pascal-{length}.f32", "rb").read()).hexdigest()
json.dump({"engine": "pascal", "length": length, "windows_file": f"windows-{length}.i32", "windows_sha256": windows_sha,
           "nll_sha256": nll_sha, "scorer": f"build/{tag}/lfm3-score", "scorer_sha256": scorer_sha,
           "hashes_taken": "inside the lfm3-pascal container, at the scorer's paths, just before it ran",
           "peak_device_mib": peak, "device_mib_before": baseline, "environment_switches": switches,
           "wall_seconds": round(end - start, 2), "scorer_output": score},
          open(f"{out}/nll-pascal-{length}.meta.json", "w"), indent=2)
print(f"length {length}: {json.dumps(score)}\n  peak device memory {peak} MiB, wall {end - start:.1f} s, nll sha256 {nll_sha[:16]}")
PY
done
