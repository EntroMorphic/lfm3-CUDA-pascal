#!/usr/bin/env bash
# Full perplexity evaluation, reference then native. Runs on the Thor host and drives both containers:
# the HF reference in lfm3-omni-dev, the native scorer in lfm3-profile-v1 (pinned runtime).
# Writes OUT.status.json after every stage (liveness for watchers), and for every native NLL file a
# nll-native-L.meta.json binding it to the windows it scored and the scorer binary that produced it.
# Usage (Thor host): run_ppl.sh OUT_NAME [SCORER_DIR_NAME]
set -uo pipefail
name=$1; scorer_dir=${2:-score-v1}
B=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA/build; out="$B/$name"; C="/workspace/liquid-CUDA/build/$name"
scorer="$B/$scorer_dir/lfm3-score"
status(){ printf '{"stage":"%s","rc":%s,"unix":%s}\n' "$1" "$2" "$(date +%s)" > "$out.status.json"; }
test ! -e "$out" || { echo "exists: $out"; exit 1; }
test -x "$scorer" || { echo "missing scorer: $scorer"; exit 1; }
status reference null
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev \
  python /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt --output "$C" \
  --lengths 256 257 4096 32768 > "$out.reference.log" 2>&1
rc=$?; [ $rc = 0 ] || { status reference $rc; exit $rc; }
for L in 256 257 4096 32768; do
  status native-$L null
  # Hash what the scorer will read, where it reads it (container paths), immediately before it runs.
  read_sha=$(docker exec lfm3-profile-v1 sha256sum "$C/windows-$L.i32" "/workspace/liquid-CUDA/build/$scorer_dir/lfm3-score") \
    || { status native-$L-hash 1; exit 1; }
  windows_read=$(echo "$read_sha" | sed -n 1p | cut -c1-64); scorer_sha=$(echo "$read_sha" | sed -n 2p | cut -c1-64)
  docker exec lfm3-profile-v1 "/workspace/liquid-CUDA/build/$scorer_dir/lfm3-score" /workspace/models/LFM3-Omni-2B \
    "$C/windows-$L.i32" $L "$C/nll-native-$L.f32" >> "$out.native.log" 2>&1
  rc=$?; [ $rc = 0 ] || { status native-$L $rc; exit $rc; }
  python3 - "$out" $L "$scorer_sha" "$scorer_dir" "$windows_read" <<'PY' || { status native-$L-meta 1; exit 1; }
import hashlib, json, sys
out, length, scorer_sha, scorer_dir, windows_read = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
summary = json.load(open(f"{out}/summary.json"))
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
windows_sha = summary["windows"][str(length)]["sha256"]
if windows_read != windows_sha:
    raise SystemExit("the scorer read windows that differ from the reference's")
json.dump({"engine": "native", "length": length, "windows_file": f"windows-{length}.i32", "windows_sha256": windows_sha,
           "nll_sha256": digest(f"{out}/nll-native-{length}.f32"), "scorer": f"build/{scorer_dir}/lfm3-score",
           "scorer_sha256": scorer_sha, "hashes_taken": "inside lfm3-profile-v1, at the scorer's paths, just before it ran"}, open(f"{out}/nll-native-{length}.meta.json", "w"), indent=2)
PY
done
status "done" 0
