#!/usr/bin/env bash
# validation/images-v1, speed against HF (method in its README, set before the run): the 11 stage-0 conversations
# (build/vimg-e2e-a1/req), native lfm3-speech then HF BF16, each one untimed pass and 3 timed passes, on a quiet Thor in
# MAXN; the power state recorded at the start and checked at the end. Per request, the median of the 3 passes.
# Usage (Thor host): vision_speed_round.sh BUILD_TAG OUT_TAG
set -uo pipefail
T=${1:?BUILD_TAG}; O=${2:?OUT_TAG}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
M=/workspace/models/LFM3-Omni-2B; R=vimg-e2e-a1/req
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
H="docker exec -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tmp/cuda-13.0-ptxas/ptxas lfm3-profile-v1 /workspace/venvs/hf-5.20.2/bin/python"
mkdir -p $B/$O/native && cd $B/$O
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3" | head -3)
[ -n "$busy" ] && { echo "NOT QUIET: $busy"; exit 1; }
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power (not MAXN)"; exit 1; }
# what the power check cannot see (images-speed-v1 rule 2's v2 round): rail power, temperatures and per-core CPU load
# through the round (with root: the memory clock and GPU load too), and the busiest processes at the start and the end
ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15 > ps-start.txt
# as root when allowed (sudoers: tegrastats only, 2026-11-16): the memory clock (EMC) and GPU load appear only then
if sudo -n -l /usr/bin/tegrastats > /dev/null 2>&1; then TGCMD="sudo -n /usr/bin/tegrastats"; else TGCMD=tegrastats; fi
$TGCMD --interval 500 > tegrastats.log 2>&1 & TG=$!
for p in warm p0 p1 p2; do for r in $B/$R/*.json; do n=$(basename $r .json); echo "$C/$R/$n.json $C/$O/native/$p-$n"; done; done > native-list.txt
echo "== native $(date +%T)"
$X $C/$T/lfm3-speech $M --batch $C/$O/native-list.txt > native.log 2>&1 || { echo "native FAILED"; tail -3 native.log; }
echo "== HF $(date +%T)"
reqs=$(for r in $B/$R/*.json; do echo "$C/$R/$(basename $r)"; done)  # the container's paths, listed on the host
$H /workspace/liquid-CUDA/tools/time_lfm3_vision_hf.py $M $C/$O/hf.jsonl 3 $reqs > hf.log 2>&1 || { echo "HF FAILED"; tail -3 hf.log; }
kill -TERM $TG 2>/dev/null; ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15 > ps-end.txt
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID: power state changed"
python3 $W/liquid-CUDA/tools/tegrastats_summary.py tegrastats.log > tegrastats-summary.json || echo "tegrastats summary FAILED"
python3 - $B/$O <<'PY'
import json, statistics, sys
from pathlib import Path
o = Path(sys.argv[1])
hf = [json.loads(l) for l in open(o / "hf.jsonl")][1:]
names = sorted({r["name"] for r in hf})
med = lambda xs: statistics.median(xs)
rows = {}
for n in names:
    h = [r for r in hf if r["name"] == n]
    nat = [json.load(open(o / "native" / f"{p}-{n}.json")) for p in ("p0", "p1", "p2")]
    nfe, nenc, npre, ngen = (med([j[k] for j in nat]) for k in ("front_end_seconds", "encode_seconds", "prefill_seconds", "generate_seconds"))
    ntok = len(nat[0]["text_tokens"])
    rows[n] = dict(
        hf=dict(front_end=med([r["front_end_seconds"] for r in h]), encode=med([r["encode_seconds"] for r in h]),
                first_token=med([r["first_token_seconds"] for r in h]), decode=med([r["decode_seconds"] for r in h]),
                end_to_end=med([r["end_to_end_seconds"] for r in h]), new_tokens=h[0]["new_tokens"]),
        native=dict(front_end=nfe, encode=nenc, prefill=npre, decode=ngen, end_to_end=nfe + npre + ngen, new_tokens=ntok))
    for side in ("hf", "native"):
        d = rows[n][side]
        d["decode_ms_per_token"] = 1000 * d["decode"] / max(1, d["new_tokens"] - 1)
print(f"{'case':8s} {'HF e2e':>8s} {'nat e2e':>8s} {'x':>6s} | {'HF fe':>6s} {'nat fe':>6s} | {'HF enc':>6s} {'nat enc':>7s} | {'HF ttft':>7s} {'nat pre':>7s} | {'HF ms/tok':>9s} {'nat ms/tok':>10s}")
for n, r in rows.items():
    h, v = r["hf"], r["native"]
    print(f"{n:8s} {h['end_to_end']:8.3f} {v['end_to_end']:8.3f} {h['end_to_end']/v['end_to_end']:6.2f} | {h['front_end']:6.3f} {v['front_end']:6.3f} | "
          f"{h['encode']:6.3f} {v['encode']:7.3f} | {h['first_token']:7.3f} {v['prefill']:7.3f} | {h['decode_ms_per_token']:9.2f} {v['decode_ms_per_token']:10.2f}")
(o / "summary.json").write_text(json.dumps(rows, indent=1))
PY
echo "VISION SPEED DONE $(date +%T)"
