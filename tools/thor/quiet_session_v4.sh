#!/usr/bin/env bash
# Quiet window 4 (2026-11-16): unattended once no engine, sanitizer, probe, build or HF process runs.
#   image-queue-v1 rule 2 with the cache (window 3's run crashed in the tool; fixed in f2106db);
#   images-frontend (GPU and PNG rounds): build/src-frontend/tools/thor/quiet_frontend_server.sh as delivered;
#   tcgen05-gemm-v1: build/src-tcgen05/tools/thor/quiet_tcgen05_soak.sh as delivered;
#   prefill-stall-v1: tools/stall_catch.py at full speed on the quiet machine (quiet-w3 reproduced 28 stalls in 630),
#   gdb attaching on the first thread over 1M minor faults/s after the load, up to 4 catches.
# Usage (Thor host): quiet_session_v4.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
NEW=vimg-srv-c2
mkdir -p $B/$O && cd $B/$O
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
for i in $(seq 1 480); do [ -z "$(busy)" ] && break; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done
[ -n "$(busy)" ] && { echo "NOT QUIET after 4 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
sudo -n /usr/bin/tegrastats --interval 500 > tegrastats.log 2>&1 & TG=$!
free -m > free-start.txt; cat /proc/buddyinfo > buddyinfo-start.txt
echo "== image-queue rule 2: stall, with the cache $(date +%T)"
$X python3 $C/../tools/measure_lfm3_image_stall.py $C/$NEW/liquid-lfm3-server $M $C/vimg-e2e-a1/req/cherry.json $C/$O/stall-new-cache.json 3 > stall-new-cache.log 2>&1; tail -1 stall-new-cache.log
for s in $B/src-frontend/tools/thor/quiet_frontend_server.sh $B/src-tcgen05/tools/thor/quiet_tcgen05_soak.sh; do
  echo "== $(basename $s) $(date +%T)"
  if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -8 $(basename $s .sh).log; else echo "MISSING $s"; fi
done
echo "== stall catcher (quiet) $(date +%T)"
free -m > free-catch.txt; cat /proc/buddyinfo > buddyinfo-catch.txt
mkdir -p catch && sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$B/$O/catch/out/\1#; s#$C/#$B/#g" $B/speech-normalize-v1/dev/tts/list.txt > catch/list.txt
mkdir -p catch/out && cp $B/stall-v4/libptracer_any.so catch/
LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia \
  python3 $W/liquid-CUDA/tools/stall_catch.py $B/vimg-a5/lfm3-speech $W/models/LFM3-Omni-2B $B/$O/catch/list.txt $B/$O/catch $B/$O/catch/libptracer_any.so 4 > catch.log 2>&1
tail -2 catch.log; grep CATCH catch/monitor.txt | head -8
kill -TERM $TG 2>/dev/null
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
python3 $W/liquid-CUDA/tools/tegrastats_summary.py tegrastats.log > tegrastats-summary.json
echo "QUIET-V4-FINISHED $(date +%T)"
