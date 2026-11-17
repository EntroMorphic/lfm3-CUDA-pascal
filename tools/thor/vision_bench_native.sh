#!/usr/bin/env bash
# validation/images-v1 rule 3, native side: all 10,500 MMStar and POPE items (tools/lfm3_vision_bench_native.py) through
# lfm3-speech --concurrent 8, the list in slices of 300 (--concurrent prepares a slice's requests, pixel values
# included, before it starts), then collected into HF's row format and scored against HF's run
# (build/images-v1/hf-bench/run1.jsonl; tools/score_lfm3_vision_bench.py). Concurrent rows are bit-identical to lone
# runs (checked on a 40-item subset before this run: 40/40 equal generated ids).
# Usage (Thor host): vision_bench_native.sh BUILD_TAG OUT_TAG
set -uo pipefail
T=${1:?BUILD_TAG}; O=${2:?OUT_TAG}
B=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
P="docker exec lfm3-profile-v1 /opt/venv/bin/python"
$P /workspace/liquid-CUDA/tools/lfm3_vision_bench_native.py requests /workspace/data/vision-bench $C/$O || exit 1
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O
mkdir -p $B/$O/slices
split -l 300 -d -a 3 $B/$O/list.txt $B/$O/slices/s
for s in $B/$O/slices/s???; do
  n=$(basename $s)
  # resume: a slice whose every output exists is done
  if awk '{print $2".json"}' $s | sed "s|^$C|$B|" | xargs -r ls >/dev/null 2>&1; then continue; fi
  echo "$(date +%T) slice $n"
  $X $C/$T/lfm3-speech $M --concurrent 8 $C/$O/slices/$n > $B/$O/slices/$n.log 2>&1 || echo "SLICE $n FAILED: $(tail -1 $B/$O/slices/$n.log)"
done
sha=$(sha256sum $B/$T/lfm3-speech | cut -c1-64)
echo "{\"engine\": \"native\", \"build\": \"$T\", \"lfm3_speech_sha256\": \"$sha\", \"mode\": \"text\", \"concurrent_rows\": 8}" > $B/$O/env.json
$P /workspace/liquid-CUDA/tools/lfm3_vision_bench_native.py collect /workspace/data/vision-bench $C/$O $C/$O/native.jsonl $C/$O/env.json
$P /workspace/liquid-CUDA/tools/score_lfm3_vision_bench.py $C/images-v1/hf-bench/run1.jsonl $C/$O/native.jsonl > $B/$O/score.txt 2>&1
echo "VISION BENCH DONE"
