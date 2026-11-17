#!/usr/bin/env bash
# validation/text-wait-v1 on Thor, after rule 1 (exactness): the timing rules 2, 3, 4 and 6's first case, rule 5's
# speed pairs, the fuzz run and the soak (rule 7) on a quiet Thor, then the sanitizers (rule 8), which need no quiet.
# Usage: textwait_round.sh TAG (the server build, e.g. twa-1)   Writes build/textwait/TAG/*.
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
sha256sum $B/$TAG/liquid-lfm3-server $B/$TAG/lfm3-text-slice-check $B/assure-s4/liquid-lfm3-server > binaries.sha256
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
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
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json > /dev/null
echo "== sanitizers (rule 8) $(date +%T)"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
$X $C/$TAG/lfm3-text-slice-check $M $CO/san-plain.json --small > san-plain.log 2>&1  # the same inputs, no sanitizer
echo "plain: exit $?, $(grep -c '^PASS' san-plain.log) PASS, $(grep -c '^FAIL' san-plain.log) FAIL"
for t in memcheck synccheck initcheck racecheck; do
  extra=""; [ $t = memcheck ] && extra="--leak-check full"
  t0=$(date +%s)
  $X $SAN --tool $t $extra --error-exitcode 99 --print-limit 50 $C/$TAG/lfm3-text-slice-check $M $CO/san-$t.json --small > san-$t.log 2>&1
  s=$?
  echo "$t: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E "ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY" san-$t.log | tr '\n' ' ') $(grep -c '^PASS' san-$t.log) PASS, $(grep -c '^FAIL' san-$t.log) FAIL"
done
echo "TEXTWAIT ROUND DONE $(date +%T)"
