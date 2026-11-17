#!/usr/bin/env bash
# The text-wait soak again (rule 7: 2 h, 12 clients, text share 0.25) with host_memory_sampler.py beside it, to find
# which part of the host's memory in use grows. Usage: soak_memdiag.sh TAG OUTNAME   Writes build/textwait/OUTNAME/*.
set -uo pipefail
TAG=${1:?TAG}; NAME=${2:?OUTNAME}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
O=$B/textwait/$NAME; CO=$C/textwait/$NAME; mkdir -p $O && cd $O
if ps -eo args | grep -E "[c]ompute-sanitizer|[l]fm3-speech |[l]iquid-lfm3-server" | grep -v defunct; then
  echo "NOT QUIET $(date +%T)"; exit 1
fi
sha256sum $B/$TAG/liquid-lfm3-server > binaries.sha256
rm -f stop; python3 $W/liquid-CUDA/tools/thor/host_memory_sampler.py $O/host-memory.jsonl $O/stop 30 &
SP=$!
echo "== soak $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_soak.py $C/$TAG/liquid-lfm3-server $M $C $L $CO/soak.json 7200 12 0.25 > soak.log 2>&1
tail -2 soak.log
sleep 60; touch stop; wait $SP
echo "MEMDIAG DONE $(date +%T)"
