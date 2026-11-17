#!/usr/bin/env bash
# Quiet window 8 (2026-11-17), after window 7 and the tcgen05 host racecheck: prefill-stall-v1, the user variable and
# a catch inside the container. Window 6 falsified memlock; the host has 0 stalls of 3,780, the container 56 of 1,260.
# Pre-registered reading: (1) container as root, a control: >= 5 of 630 expected (if not, the trigger has gone and the
# README says so). (2) container as uid 1000, twice: both >= 5 stalls, the user is not the trigger, and (3) runs the
# catcher as uid 1000 in the container (host gdb attaches; same uid); both 0 while (1) stalls, running as root is
# implicated; anything else, not decisive. (4) Then audio-frontend-v1's remaining server rules
# (build/audio-src-b/tools/thor/quiet_audio_rest.sh, as delivered). Merged 2026-11-17 08:50 from two sessions'
# versions before any of its data existed.
# Usage (Thor host): quiet_session_v8.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
mkdir -p $B/$O && cd $B/$O
until grep -q "^QUIET-V7-FINISHED\|^NOT QUIET" $B/quiet-w7.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
for run in container-root container-u1000 container-u1000-2; do
  echo "== stall probe $run $(date +%T)"
  if [ $run = container-root ]; then U=; else U=1000; fi
  STALL_USER=$U bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-$run > probe-$run.log 2>&1
  cat $B/$O-$run/limits.txt; grep -E "requests" probe-$run.log; grep -c "^STALL" probe-$run.log; grep "^STALL" probe-$run.log | head -3 | cut -c1-250
done
n=$(grep -c "^STALL" probe-container-u1000.log); n2=$(grep -c "^STALL" probe-container-u1000-2.log)
if [ "$n" -ge 5 ] && [ "$n2" -ge 5 ]; then
  echo "== stall catcher in the container as uid 1000 $(date +%T)"
  mkdir -p catch/out && cp $B/stall-v4/libptracer_any.so catch/
  sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$C/$O/catch/out/\1#" $B/speech-normalize-v1/dev/tts/list.txt > catch/list.txt
  STALL_PREFIX="docker exec -u 1000 -e LD_PRELOAD=$C/$O/catch/libptracer_any.so -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1" \
    python3 $W/liquid-CUDA/tools/stall_catch.py $C/vimg-a5/lfm3-speech $M $C/$O/catch/list.txt $B/$O/catch $C/$O/catch/libptracer_any.so 4 > catch.log 2>&1
  tail -2 catch.log; grep CATCH catch/monitor.txt | head -8; ls catch/catch-*.txt 2>/dev/null
else
  echo "== catcher skipped: uid 1000 had $n and $n2 stalls"
fi
s=$B/audio-src-b/tools/thor/quiet_audio_rest.sh
echo "== $(basename $s) $(date +%T)"
if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -12 $(basename $s .sh).log; else echo "MISSING $s"; fi
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V8-FINISHED $(date +%T)"
