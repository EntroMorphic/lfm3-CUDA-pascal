#!/usr/bin/env bash
# Quiet window 5 (2026-11-17, after the overnight sanitizers): prefill-stall-v1's container hypothesis. quiet-w3's probe
# (engine inside lfm3-profile-v1) stalled 28 of 630; stall_catch (engine on the host) 0 of 1,260; the container's
# memory cgroup has a 96 GiB limit, hit 33,326 times (memory.events "max"), with ~43 GB of page cache charged to it.
# Pre-registered reading: the hypothesis holds if both container runs stall (>= 5 of 630 each) with cg_events_max rising
# inside the stall windows, and the host run has none; anything else, it does not, and the README says so.
# Order container, host, container (state drift). Then tcgen05-gemm-v1's rule 4b off/on reruns. Usage (Thor host): quiet_session_v5.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build
mkdir -p $B/$O && cd $B/$O
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[s]anitiz|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
for run in container-1 host container-2; do
  echo "== stall probe $run $(date +%T)"
  free -m > free-$run.txt; cat /proc/buddyinfo > buddyinfo-$run.txt
  if [ $run = host ]; then H=1; else H=0; fi
  STALL_HOST=$H bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-$run > probe-$run.log 2>&1
  grep -E "requests" probe-$run.log; grep -c "^STALL" probe-$run.log; grep "^STALL" probe-$run.log | head -4 | cut -c1-300
done
s=$B/src-tcgen05/tools/thor/quiet_tcgen05_4b_offon.sh  # tcgen05-gemm-v1 rule 4b, off then on (its agent's script)
echo "== $(basename $s) $(date +%T)"
if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -8 $(basename $s .sh).log; else echo "MISSING $s"; fi
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V5-FINISHED $(date +%T)"
