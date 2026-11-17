#!/usr/bin/env bash
# Native long-context perplexity (lfm3-score, cuBLAS-free) on Thor (host), after a pg19_ppl_round.sh HF round.
# Usage: pg19_native_round.sh TAG SCORER_TAG HF_ROUND MODEL_DIR "LENGTHS"
#   e.g. pg19_native_round.sh pg19-native-v1 lc-score-v1 pg19-1p7b-v1 LFM3-1.7B "32768 65536 131072 262144"
# 1. Regression: LFM3-Omni-2B on the G4 WikiText windows (ppl-g4-v3) must reproduce the accepted native NLL bit for bit.
# 2. Each length scores the HF round's windows (onepass/ if present, else chunked/) into build/TAG/nll-native-gemm-L.f32
#    with a meta.json binding it to those windows. build/TAG/memory.log: host memory around each run.
set -uo pipefail
tag=$1; scorer=$2; hf=$3; model=$4; lengths=$5
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
fail(){ echo "FAIL: $1"; exit 1; }
mem(){ awk -v t="$1" '/MemTotal/{T=$2}/MemFree/{F=$2}/MemAvailable/{A=$2}/^Cached/{K=$2}/Buffers/{U=$2}
  END{printf "%s %s used %.1f GiB available %.1f GiB\n", strftime("%T"), t, (T-F-K-U)/1048576, A/1048576}' /proc/meminfo | tee -a $O/memory.log; }
echo "== sources"; (cd $W/liquid-CUDA && sha256sum tools/thor/pg19_native_round.sh) | sed 's/^/  /'
sha256sum $B/$scorer/lfm3-score | sed 's/^/  /'; sed 's/^/  /' $B/$scorer/sources.sha256
readelf -d $B/$scorer/lfm3-score | grep -q cublas && fail "scorer links cuBLAS"
echo "== regression (LFM3-Omni-2B, ppl-g4-v3 WikiText windows)"
for L in 4096 32768; do
  $EX $C/$scorer/lfm3-score /workspace/models/LFM3-Omni-2B $C/ppl-g4-v3/windows-$L.i32 $L $C/$tag/regression-$L.f32 \
    > $O/regression-$L.log 2>&1 || { tail -3 $O/regression-$L.log; fail "regression $L"; }
  cmp -s $O/regression-$L.f32 $B/ppl-g4-v3/nll-native-gemm-$L.f32 || fail "$L: NLL differs from the accepted native"
  echo "  $L: bitwise identical to ppl-g4-v3"
done
for L in $lengths; do
  d=$B/$hf/onepass; [ -e $d/windows-$L.i32 ] || d=$B/$hf/chunked; [ -e $d/windows-$L.i32 ] || fail "no windows for $L"
  mem "before $L"
  $EX $C/$scorer/lfm3-score /workspace/models/$model ${d/$B/$C}/windows-$L.i32 $L $C/$tag/nll-native-gemm-$L.f32 \
    > $O/score-$L.log 2>&1; rc=$?
  mem "after $L rc=$rc"; tail -1 $O/score-$L.log; [ $rc = 0 ] || { tail -3 $O/score-$L.log; fail "score $L"; }
  python3 -c "
import hashlib, json, sys
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump(dict(engine='native-gemm', length=$L, windows_file='$d/windows-$L.i32', windows_sha256=h('$d/windows-$L.i32'),
               nll_sha256=h('$O/nll-native-gemm-$L.f32'), scorer=h('$B/$scorer/lfm3-score'), model='$model'),
          open('$O/nll-native-gemm-$L.meta.json','w'), indent=1)"
done
echo "ALL DONE $(date +%T)"
