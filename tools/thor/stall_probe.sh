#!/usr/bin/env bash
# The intermittent 3-4 s prefill stall (found 2026-11-16 by image-cache-v1 rule 0 and a scan of earlier runs: 86
# requests in 12 runs, TTS prefills included): lfm3-speech BUILD on speech-normalize-v1's 630 dev TTS requests (38
# stalls in their recorded run), while a sampler records every 100 ms the kernel's reclaim, compaction and fault
# counters (/proc/vmstat) and the engine process's own faults (/proc/PID/stat). tools/stall_probe_analyze.py then lines
# each request's prefill (end: its output file's mtime, less its generate time) up against the counters.
# Since quiet-w4 it also samples the container's memory cgroup (memory.events "max": the times the 96 GiB limit was
# hit; memory.stat's in-cgroup reclaim), invisible in the global counters. STALL_HOST=1 runs the same engine on the
# host instead (outside the container's cgroup, as stall_catch did: 0 stalls in 1,260 requests).
# STALL_USER=UID runs the container engine as that user (docker exec -u) instead of root.
# Usage (Thor host): [STALL_HOST=1 | STALL_USER=UID] stall_probe.sh BUILD OUT_TAG
set -uo pipefail
T=${1:?BUILD}; O=${2:?OUT_TAG}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O/out && cd $B/$O
if [ "${STALL_HOST:-0}" = 1 ]; then
  sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$B/$O/out/\1#; s#$C/#$B/#g" $B/speech-normalize-v1/dev/tts/list.txt > list.txt
  RUN="env LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia $B/$T/lfm3-speech $W/models/LFM3-Omni-2B --batch $B/$O/list.txt"
else
  sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$C/$O/out/\1#" $B/speech-normalize-v1/dev/tts/list.txt > list.txt
  U=${STALL_USER:+-u $STALL_USER}
  RUN="docker exec $U -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $C/$T/lfm3-speech $M --batch $C/$O/list.txt"
fi
CG=$(find /sys/fs/cgroup/system.slice -maxdepth 1 -type d -name "docker-$(docker inspect lfm3-profile-v1 --format '{{.Id}}')*" | head -1)
echo "user ${STALL_USER:-root (container) or host}; memlock $(ulimit -l) KiB (soft) $(ulimit -H -l) (hard); STALL_HOST=${STALL_HOST:-0} STALL_USER=${STALL_USER:-root}" > limits.txt
echo "$CG" > cgroup.txt; cat $CG/memory.max $CG/memory.events >> cgroup.txt 2>/dev/null
head -1 list.txt
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag > thp.txt
free -m > free-start.txt
CG=$CG O=$O python3 - > sampler.jsonl <<'PY' &
import json, os, subprocess, time
CG = os.environ.get("CG", "")
CG_STAT = ("pgscan", "pgsteal", "file", "anon", "pgfault", "thp_fault_alloc", "workingset_refault_file")
KEYS = ("compact_stall", "compact_fail", "compact_success", "allocstall_normal", "allocstall_movable", "pgmajfault",
        "pgsteal_direct", "pgscan_direct", "pgsteal_kswapd", "thp_fault_alloc", "thp_fault_fallback", "pswpin", "pswpout")
pid = None
end = time.time() + 6 * 3600
while time.time() < end:
    if pid is None:
        r = subprocess.run(["pgrep", "-f", "^/[^ ]*/lfm3-speech .*/" + os.environ["O"] + "/list.txt"], capture_output=True, text=True)
        pid = int(r.stdout.split()[0]) if r.stdout.split() else None
    v = {}
    for line in open("/proc/vmstat"):
        k, n = line.split()
        if k in KEYS: v[k] = int(n)
    if CG:
        try:
            for line in open(CG + "/memory.events"):
                k, n = line.split()
                if k in ("max", "high"): v["cg_events_" + k] = int(n)
            for line in open(CG + "/memory.stat"):
                k, n = line.split()
                if k in CG_STAT: v["cg_" + k] = int(n)
        except OSError:
            pass
    p = None
    if pid:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            p = dict(minflt=int(f[7]), majflt=int(f[9]), utime=int(f[11]), stime=int(f[12]), rss_pages=int(f[21]))
        except OSError:
            if os.path.exists("done.flag"): break
            pid = None
    print(json.dumps(dict(t=time.time(), vm=v, proc=p)), flush=True)
    time.sleep(0.1)
PY
SP=$!
echo "== run $(date +%T)"
$RUN > run.log 2>&1; echo "exit $?"
touch done.flag; sleep 1; kill -TERM $SP 2>/dev/null
free -m > free-end.txt; cat $CG/memory.events >> cgroup.txt 2>/dev/null
python3 $W/liquid-CUDA/tools/stall_probe_analyze.py $B/$O | tee analysis.txt
