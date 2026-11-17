#!/usr/bin/env bash
# Quiet window 3 (2026-11-16): the day's remaining timing rules, unattended, once no engine, sanitizer, probe or HF
# process runs. Each engineer's script (build/src-*/tools/thor/quiet_*.sh) runs as delivered, in turn; mine:
#   image-queue-v1 rule 2 (the stall with part 2: cache off, then on), rule 3 (peak RSS, 16 x 4K at once);
#   image-cache-v1 rule 2 (three-turn conversations, the new server against the old);
#   prefill-stall-v1: the stall probe and the attach-on-stall catcher on a quiet machine.
# Speed rounds' tegrastats runs as root (EMC and GR3D). Usage (Thor host): quiet_session_v3.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
NEW=vimg-srv-c2; OLD=vimg-srv-a3
mkdir -p $B/$O && cd $B/$O
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_|[f]rontend-" | grep -v "quiet_" | head -3; }
for i in $(seq 1 480); do [ -z "$(busy)" ] && break; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done
[ -n "$(busy)" ] && { echo "NOT QUIET after 4 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
sudo -n /usr/bin/tegrastats --interval 500 > tegrastats.log 2>&1 & TG=$!
free -m > free-start.txt; cat /proc/buddyinfo > buddyinfo-start.txt
echo "== image-queue rule 2: stall, part 2 alone (cache off) $(date +%T)"
$X python3 $C/../tools/measure_lfm3_image_stall.py $C/$NEW/liquid-lfm3-server $M $C/vimg-e2e-a1/req/cherry.json $C/$O/stall-new-nocache.json 3 --image-cache-mb 0 > stall-new-nocache.log 2>&1; tail -1 stall-new-nocache.log
echo "== image-queue rule 2: stall, with the cache $(date +%T)"
$X python3 $C/../tools/measure_lfm3_image_stall.py $C/$NEW/liquid-lfm3-server $M $C/vimg-e2e-a1/req/cherry.json $C/$O/stall-new-cache.json 3 > stall-new-cache.log 2>&1; tail -1 stall-new-cache.log
echo "== image-cache rule 2: turns, new and old $(date +%T)"
for n in cherry two; do for S in $NEW $OLD; do
  $X python3 $C/../tools/measure_lfm3_image_turns.py $C/$S/liquid-lfm3-server $M $C/vimg-e2e-a1/req/$n.json $C/$O/turns-$n-$S.json 4 > turns-$n-$S.log 2>&1; echo "$n $S"; tail -4 turns-$n-$S.log
done; done
echo "== image-queue rule 3: RSS $(date +%T)"
$X python3 $C/../tools/measure_lfm3_image_rss.py $C/$NEW/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/rss-new.json > rss-new.log 2>&1; tail -1 rss-new.log
for s in $B/src-audio/tools/thor/quiet_audio_frontend.sh $B/src-frontend/tools/thor/quiet_frontend.sh $B/src-tcgen05/tools/thor/quiet_tcgen05_gemm.sh; do
  echo "== $(basename $s) $(date +%T)"
  if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -5 $(basename $s .sh).log; else echo "MISSING $s"; fi
done
echo "== stall probe (quiet) $(date +%T)"
free -m > free-probe.txt; cat /proc/buddyinfo > buddyinfo-probe.txt
bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-stall > stall-probe.log 2>&1; grep -E "requests|STALL" stall-probe.log | head -20
kill -TERM $TG 2>/dev/null
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
python3 $W/liquid-CUDA/tools/tegrastats_summary.py tegrastats.log > tegrastats-summary.json
echo "QUIET-V3-FINISHED $(date +%T)"
