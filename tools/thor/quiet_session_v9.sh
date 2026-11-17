#!/usr/bin/env bash
# Quiet window 9 (2026-11-17), after window 8: validation/crowd-identity-v1's probe. The tcgen05 OFF binaries
# (tgemm-off1, the ones whose crowd had 4 differences in window 7), a 600 s soak (12 clients, text share 0.25) through
# the tcgen05 branch's check_lfm3_tcgen05_soak.py, but with main's check_lfm3_audio_soak.py (its crowd diagnostics),
# then CROWD_ROUNDS=20. Reading in the README, set before the run.
# Usage (Thor host): quiet_session_v9.sh   (results: build/quiet-w9-crowd/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
O=quiet-w9-crowd; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e CROWD_ROUNDS=20 -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
until grep -q "^QUIET-V8-FINISHED\|^NOT QUIET" $B/quiet-w8.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
rm -rf tools && cp -r $B/src-tcgen05/tools tools && cp $W/liquid-CUDA/tools/check_lfm3_audio_soak.py tools/
(cd $B && sha256sum tgemm-off1/liquid-lfm3-server tgemm-off1/liquid-lfm3-worker) > binaries.sha256
sha256sum tools/check_lfm3_audio_soak.py tools/check_lfm3_tcgen05_soak.py >> binaries.sha256
echo "== crowd probe $(date +%T)"
$X python3 $C/$O/tools/check_lfm3_tcgen05_soak.py $C/tgemm-off1/liquid-lfm3-server $M $C $L $C/vimg-e2e-a4 \
  $C/$O/soak.json 600 12 0.25 > soak.log 2>&1; echo "exit $?"
grep "^crowd" soak.log | awk '{print $2, $3}' | sort | uniq -c
grep "^crowd" soak.log | grep -v "56/56" | head -20 | cut -c1-300
[ -f soak.crowd-differ.jsonl ] && python3 -c "
import json
for l in open('soak.crowd-differ.jsonl'):
    d=json.loads(l); print(d['name'], d['streamed'], d['status'], d['tries'], json.dumps(d.get('fields', d.get('events')))[:200])" | head -40
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V9-FINISHED $(date +%T)"
