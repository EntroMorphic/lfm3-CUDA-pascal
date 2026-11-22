# red-team-2 verification, part 2 (tools/pascal/remote.sh detach rt2-v1b 'bash validation/red-team-2/job2.sh'): the
# two continuation steps of job.sh again with continuation_check.py as round 7 tightened it (R7-1: the old server must
# fail by the watchdog, worker_unavailable after at least 295 s; the new one first shows a continuation within the cap
# reusing the seed). Same builds (text-v37, text-v38); the rest of the verification is job.sh's.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-v1b; M=/workspace/models/LFM3-Omni-2B; PY=$B/venv-asr/bin/python
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
for b in text-v37 text-v38; do (cd $B/$b && sha256sum liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  $b/#"; done | tee $O/binaries.sha256
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
for v in old:v37 new:v38; do e=${v%%:*}; b=text-${v##*:}
  $PY tools/pascal/continuation_check.py $B/$b/liquid-lfm3-worker $B/$b/liquid-lfm3-server $M $O/continuation-$e $e > $O/continuation-$e.log 2>&1
  echo "continuation $e ($b) rc=$?"; grep -E "^continuation|^CONTINUATION" $O/continuation-$e.log; clean
done
date
