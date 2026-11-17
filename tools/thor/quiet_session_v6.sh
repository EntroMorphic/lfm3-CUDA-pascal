#!/usr/bin/env bash
# Quiet window 6 (2026-11-17): prefill-stall-v1's memlock hypothesis. Window 5: 26 + 30 stalls in the container, 0 on
# the host, and the container's memory cgroup never at its limit (that hypothesis falsified). The difference found
# after: RLIMIT_MEMLOCK is 16 GB on the host and 8 MiB (soft and hard) in lfm3-profile-v1, root included (no
# CAP_IPC_LOCK); same libcuda, pinned libraries, glibc 2.39 and binary.
# Pre-registered reading: memlock is the trigger if both host runs at 8 MiB (ulimit -l 8192, lowering needs no
# privilege) stall (>= 5 of 630 each, with millions of minor faults) and the host run at the default 16 GB has none;
# anything else, it is not, and the README says so. Order 8M, default, 8M. Usage (Thor host): quiet_session_v6.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build
mkdir -p $B/$O && cd $B/$O
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]cgen05_host_racecheck|[r]acecheck-after-w5|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
for run in host-8m-1 host-default host-8m-2; do
  echo "== stall probe $run $(date +%T)"
  if [ $run = host-default ]; then
    STALL_HOST=1 bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-$run > probe-$run.log 2>&1
  else
    ( ulimit -S -l 8192 && ulimit -H -l 8192 && STALL_HOST=1 bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-$run ) > probe-$run.log 2>&1
  fi
  cat $B/$O-$run/limits.txt; grep -E "requests" probe-$run.log; grep -c "^STALL" probe-$run.log; grep "^STALL" probe-$run.log | head -3 | cut -c1-250
done
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V6-FINISHED $(date +%T)"
