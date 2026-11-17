#!/usr/bin/env bash
# G4 tier 2b on Thor, run once (validation/g4-gates-v2/PLAN.md and its amendment): score the exact FP32 anchor
# (eager attention in blocks) over the §9 windows, then compare the native engine's NLL against it and SDPA's.
# A memory monitor sends SIGTERM (never SIGKILL: it leaked driver memory) if available memory falls below 30 GB.
# Usage: g4_tier2b.sh ANCHOR_DIR_NAME OUT_JSON
set -u
name=$1; report=$2
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
test ! -e $B/$name || { echo "exists $name"; exit 1; }
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
echo "== sources"; (cd $R && git rev-parse HEAD 2>/dev/null; sha256sum tools/ppl_reference.py tools/ppl_fidelity.py tools/ppl_compare.py tools/thor/g4_tier2b.sh) | sed 's/^/  /'
echo "== anchor $(date +%T), available $(avail) GB"
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \
  -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 /opt/venv/bin/python3 \
  /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt --output $C/$name \
  --configs eager_blocked --dtype float32 > $B/$name.log 2>&1 &
job=$!; min=$(avail); stopped=0
while kill -0 $job 2>/dev/null; do
  a=$(avail); [ $a -lt $min ] && min=$a
  if [ $a -lt 30 ] && [ $stopped = 0 ]; then docker exec lfm3-profile-v1 pkill -TERM -f ppl_reference.py; stopped=1; echo "SIGTERM at $a GB"; fi
  sleep 1
done
wait $job; rc=$?
echo "anchor rc=$rc, minimum available $min GB, now $(avail) GB"; grep '^{' $B/$name.log | cut -c1-150
[ $rc = 0 ] || exit 1
python3 -c "
import json, sys; a=json.load(open('$B/$name/summary.json')); b=json.load(open('$B/ppl-v5/summary.json'))
same=all(a['windows'][k]['sha256']==b['windows'][k]['sha256'] for k in b['windows']); p=a['precision']
print('windows identical to ppl-v5:', same, p)
sys.exit(0 if same and a['dtype']=='float32' and not p['matmul_allow_tf32'] and not p['cudnn_allow_tf32'] else 1)" || { echo "FAIL: anchor windows or precision"; exit 1; }
echo "== tier 2b $(date +%T)"
# The engine files: attempt 3's native NLL (bitwise identical to the v6 build's, round4/sha256.txt).
grep 'nll-native-gemm' $R/validation/g4-native-gates-v1/round4/sha256.txt > /tmp/g4-expected-nll.sha256
[ "$(wc -l < /tmp/g4-expected-nll.sha256)" = 4 ] || { echo "FAIL: expected four engine NLL hashes"; exit 1; }
python3 $R/tools/ppl_fidelity.py --anchor $B/$name --references $B/ppl-v5 --engine-dir $B/ppl-g4-v3 --engine native-gemm \
  --expect-sha256 /tmp/g4-expected-nll.sha256 --output $report
rc=$?; echo "tier 2b rc=$rc"; exit $rc
