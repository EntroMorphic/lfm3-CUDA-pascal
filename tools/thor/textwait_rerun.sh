#!/usr/bin/env bash
# validation/text-wait-v1 on Thor, the reruns after a change to the server's HTTP layer only (the worker binary
# identical, so the engine check and the sanitizers, rule 8, carry over): rule 1 over HTTP (as is and with
# CUDA_LAUNCH_BLOCKING=1), the timing rules 2, 3, 4 and 6's first case, rule 5's speed pairs, the fuzz run, the soak
# (rule 7) and the text gate, one at a time on a quiet Thor.
# Usage: textwait_rerun.sh TAG (the server build, e.g. twa-2)   Writes build/textwait/TAG/*.
set -uo pipefail
TAG=${1:?TAG}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
O=$B/textwait/$TAG; CO=$C/textwait/$TAG; mkdir -p $O && cd $O
quiet() {  # nothing else on the GPU (timing rules)
  if ps -eo args | grep -E "[c]ompute-sanitizer|[l]fm3-speech |[l]iquid-lfm3-server" | grep -v defunct; then
    echo "NOT QUIET $(date +%T)"; exit 1
  fi
}
sha256sum $B/$TAG/liquid-lfm3-server $B/$TAG/liquid-lfm3-worker $B/$TAG/lfm3-text-slice-check \
  $B/assure-s4/liquid-lfm3-server > binaries.sha256
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
for mode in normal blocking; do
  E=""; [ $mode = blocking ] && E="-e CUDA_LAUNCH_BLOCKING=1"
  quiet; echo "== rule 1 over HTTP, $mode $(date +%T)"
  docker exec $E -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 \
    /workspace/liquid-CUDA/tools/check_lfm3_text_wait_exact.py $C/$TAG/liquid-lfm3-server $C/assure-s4/liquid-lfm3-server \
    $M $C $L $CO/rule1-$mode.json > rule1-$mode.log 2>&1
  grep -v "^  " rule1-$mode.log | tail -4
done
quiet; echo "== timing rules 2, 3, 4, deadline $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_text_wait.py $C/$TAG/liquid-lfm3-server $M $C $L $CO/text-wait.json > text-wait.log 2>&1
tail -1 text-wait.log
quiet; echo "== rule 5 speed pairs $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_text_wait_speed.py $C/$TAG/liquid-lfm3-server $C/assure-s4/liquid-lfm3-server $M $C $L $CO/speed.json 5 > speed.log 2>&1
tail -1 speed.log
quiet; echo "== fuzz $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_fuzz.py $C/$TAG/liquid-lfm3-server $M $C $L $CO/fuzz.json 3000 8 2026 > fuzz.log 2>&1
tail -2 fuzz.log
quiet; echo "== soak, text share 0.25 $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_soak.py $C/$TAG/liquid-lfm3-server $M $C $L $CO/soak.json 7200 12 0.25 > soak.log 2>&1
tail -2 soak.log
quiet; echo "== text gate $(date +%T)"
rm -rf $O/text-gate
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_native_server.py --checkpoint $M --server $C/$TAG/liquid-lfm3-server \
  --worker $C/$TAG/liquid-lfm3-worker --control-result /workspace/tmp/control-codecfree-remediated-001/output.json \
  --output $CO/text-gate > text-gate.log 2>&1
echo "text gate exit $?"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json > /dev/null
echo "TEXTWAIT RERUN DONE $(date +%T)"
