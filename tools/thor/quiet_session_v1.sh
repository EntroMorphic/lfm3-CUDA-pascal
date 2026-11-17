#!/usr/bin/env bash
# The quiet-machine measurements of 2026-11-16's rounds, in one session after images-speed-v1's benchmark (each
# method in its round's README, set before the run). Waits until no engine, sanitizer, probe or HF process runs, then:
#   timing (quiet, MAXN, the power state checked around it, tegrastats through it):
#     1. image-queue-v1 rule 0: a 4K image's stall of 7 streaming rows, on the server before the change (vimg-srv-a3)
#     2. image-cache-v1 rule 0: three-turn conversations re-sending cherry (4K) and two (a pair), vimg-srv-a3
#     3. audio-speed-v1 / audio-frontend-v1 rule 0 (the audio engineer's harness, build/src-audio)
#     4. the language model's prefill at 4K (nsys, host): lfm3-speech vimg-a5 on cherry, 4 new tokens
#   then exactness (order no longer matters):
#     5. images-speed-v1 rule 6, the server checks (vimg-a5, vimg-srv-a3)
#     6. image-queue-v1 rule 1: the stage-0 images through the new server (vimg-srv-q1), simultaneous, no 429
# Usage (Thor host): quiet_session_v1.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
M=/workspace/models/LFM3-Omni-2B; T=$W/liquid-CUDA/tools/thor
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[t]cgen05|[a]udio-encoder-bench|[f]rontend-|[v]envs/hf-|[n]sys" | grep -v quiet_session | head -3; }
for i in $(seq 1 240); do [ -z "$(busy)" ] && break; [ $i = 1 ] && echo "waiting for quiet: $(busy)"; sleep 30; done
[ -n "$(busy)" ] && { echo "NOT QUIET after 2 h: $(busy)"; exit 1; }
echo "== quiet $(date +%T)"
bash $T/power_state.sh power-start.json > /dev/null || { echo "FAIL power (not MAXN)"; exit 1; }
ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15 > ps-start.txt
tegrastats --interval 500 > tegrastats.log 2>&1 & TG=$!
echo "== 1 image stall (before the change) $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/measure_lfm3_image_stall.py $C/vimg-srv-a3/liquid-lfm3-server $M \
  $C/vimg-e2e-a1/req/cherry.json $C/$O/image-stall-a3.json 3 > image-stall-a3.log 2>&1; tail -1 image-stall-a3.log
echo "== 2 image turns $(date +%T)"
for n in cherry two; do
  $X python3 /workspace/liquid-CUDA/tools/measure_lfm3_image_turns.py $C/vimg-srv-a3/liquid-lfm3-server $M \
    $C/vimg-e2e-a1/req/$n.json $C/$O/turns-$n-a3.json 4 > turns-$n-a3.log 2>&1; tail -4 turns-$n-a3.log
done
echo "== 3 audio rule 0 $(date +%T)"
LFM3_WORKSPACE=$W bash $B/src-audio/tools/thor/audio_speed_rule0.sh audio-bench-a1 vimg-a5 $O-audio > audio-rule0.log 2>&1; tail -3 audio-rule0.log
echo "== 4 prefill nsys $(date +%T)"
python3 - $B/vimg-e2e-a1/req/cherry.json $B/$O/prefill-cherry.json $W <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r["max_new_tokens"] = 4; r.pop("record_image", None)
for m in r["messages"]:
    if isinstance(m["content"], list):
        for p in m["content"]:
            if p.get("type") == "image": p["image"] = p["image"].replace("/workspace", sys.argv[3])
json.dump(r, open(sys.argv[2], "w"))
PY
nsys profile -o $B/$O/prefill-cherry --force-overwrite true -t cuda --stats=false \
  env LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia \
  $B/vimg-a5/lfm3-speech $W/models/LFM3-Omni-2B $B/$O/prefill-cherry.json $B/$O/prefill-out > nsys.log 2>&1 || { tail -3 nsys.log; echo "nsys FAILED"; }
test -e prefill-cherry.nsys-rep && nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv -o prefill-cherry prefill-cherry.nsys-rep > /dev/null 2>&1
ls prefill-cherry_* 2>/dev/null | head
kill -TERM $TG 2>/dev/null; ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15 > ps-end.txt
bash $T/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID: power state changed"
python3 $W/liquid-CUDA/tools/tegrastats_summary.py tegrastats.log > tegrastats-summary.json || echo "tegrastats summary FAILED"
echo "== TIMING DONE $(date +%T)"
echo "== 5 images-speed-v1 rule 6 server $(date +%T)"
E2E_ROUND=vimg-e2e-a4 bash $T/vision_rule6_server.sh vimg-a5 vimg-srv-a3 $O-rule6 > rule6-server.log 2>&1; tail -20 rule6-server.log
echo "== 6 image-queue rule 1 $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_image_server.py $C/vimg-srv-q1/liquid-lfm3-server $M $C/vimg-e2e-a4 \
  $C/$O/image-queue-rule1.json > image-queue-rule1.log 2>&1; tail -2 image-queue-rule1.log
echo "QUIET SESSION DONE $(date +%T)"
