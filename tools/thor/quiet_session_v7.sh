#!/usr/bin/env bash
# Quiet window 7 (2026-11-17): tcgen05-gemm-v1 rule 4b under amendments 2 and 2a, the reverse-order pair (its
# agent's build/src-tcgen05/tools/thor/quiet_tcgen05_4b_pair2.sh, as delivered), after window 6 has finished.
# Usage (Thor host): quiet_session_v7.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build
mkdir -p $B/$O && cd $B/$O
until grep -q "^QUIET-V6-FINISHED\|^NOT QUIET\|^FAIL power" $B/quiet-w6.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]cgen05_host_racecheck|[r]acecheck-after-w5|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
s=$B/src-tcgen05/tools/thor/quiet_tcgen05_4b_pair2.sh
echo "== $(basename $s) $(date +%T)"
if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -12 $(basename $s .sh).log; else echo "MISSING $s"; fi
echo "QUIET-V7-FINISHED $(date +%T)"
