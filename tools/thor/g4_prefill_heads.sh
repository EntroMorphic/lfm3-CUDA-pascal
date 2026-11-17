#!/usr/bin/env bash
# G4 prefill heads-per-call measurement (journal/g4-prefill-memory_synth.md, step 1), Thor host.
# Builds the default (native) engine from the current tree, then in one session, alternating settings:
#  - prefill seconds for LFM3_PREFILL_HEADS = 16, 8, 4 (3 repetitions; 128 forced decode steps) at: 3,968 prompt tokens,
#    capacity 4,096 (one 3,968-row tile: the case the split's memory saving is for); 4,096 and 32,000 prompt tokens,
#    capacity 32,768 (128-row tiles; the requests of the committed speed evidence);
#  - bit-invariance: the exact FP32 logits of all 128 steps must hash the same for every setting.
# Usage: g4_prefill_heads.sh TAG
set -uo pipefail
tag=$1
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
O=$B/$tag; test ! -e $O && test ! -e $B/$tag-build || { echo "exists $tag"; exit 1; }; mkdir -p $O
echo "== sources"; (cd $R && sha256sum native/src/lfm3_main.cu native/include/lfm/gemm.cuh tools/thor/g4_prefill_heads.sh) | sed 's/^/  /'
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$tag-build; rc=\$?; chown -R 1000:1000 $C/$tag-build $C/$tag-build.cmake.log $C/$tag-build.build.log 2>/dev/null; exit \$rc" > $O/build.log 2>&1 || { echo "FAIL build"; exit 1; }
dyn=$(readelf -d $B/$tag-build/liquid-lfm3) || { echo "FAIL: readelf"; exit 1; }  # not `| grep -q` (SIGPIPE under pipefail)
[ "$(grep -c libcublas <<< "$dyn")" = 0 ] || { echo "FAIL: default build links cuBLAS"; exit 1; }
cp $B/g4-evidence-v1-decode-rate/req-4096.json $B/g4-evidence-v1-decode-rate/req-32000.json $O/
python3 - $O $B/ppl-v5/windows-32768.i32 <<'PY'
import json, sys
import numpy as np
ids = np.fromfile(sys.argv[2], np.int32).tolist()  # same tokens as the speed evidence
json.dump(dict(input_ids=ids[:3968], teacher_ids=ids[3968:4096], max_new_tokens=0, capacity=4096, warmup_runs=1),
          open(f"{sys.argv[1]}/req-3968.json", "w"))
for L in (3968, 4096, 32000):  # a traced copy for the bit check (logits only)
    r = json.load(open(f"{sys.argv[1]}/req-{L}.json")); r.pop("warmup_runs", None)
    json.dump(r, open(f"{sys.argv[1]}/trace-req-{L}.json", "w"))
PY
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia"
echo "== timing $(date +%T)"
for L in 3968 4096 32000; do for rep in 1 2 3; do for H in 16 8 4; do
  $EX -e LFM3_PREFILL_HEADS=$H lfm3-profile-v1 $C/$tag-build/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B \
    $C/$tag/req-$L.json $C/$tag/time-$H-$L-$rep.json > $O/time-$H-$L-$rep.log 2>&1 || { echo "FAIL time $H $L"; tail -2 $O/time-$H-$L-$rep.log; }
done; done; done
echo "== bit-invariance $(date +%T)"
for L in 3968 4096 32000; do for H in 16 8 4; do
  python3 - $O/trace-req-$L.json $C/$tag/trace-$H-$L > $O/trace-req-$H-$L.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r.update(trace_directory=sys.argv[2], trace_logits_only=True); print(json.dumps(r))
PY
  $EX -e LFM3_PREFILL_HEADS=$H lfm3-profile-v1 $C/$tag-build/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B \
    $C/$tag/trace-req-$H-$L.json $C/$tag/trace-$H-$L.json > $O/trace-$H-$L.log 2>&1 || { echo "FAIL trace $H $L"; tail -2 $O/trace-$H-$L.log; }
  echo "$L heads=$H: $(cd $O/trace-$H-$L && cat $(ls *.logits.f32 | sort -n) | sha256sum | cut -c1-16) over $(ls $O/trace-$H-$L/*.logits.f32 | wc -l) steps"
done; done
python3 - $O <<'PY'
import glob, json, statistics as st, sys
O = sys.argv[1]
print("prompt heads  prefill_s (median of 3)  decode_tok/s")
for L in (3968, 4096, 32000):
    for H in (16, 8, 4):
        r = [json.load(open(f)) for f in sorted(glob.glob(f"{O}/time-{H}-{L}-*.json"))]
        print(f"{L:6d} {H:5d}  {st.median(x['prefill_seconds'] for x in r):10.4f}  "
              f"{st.median(x['decode_forward_calls']/x['decode_seconds'] for x in r):14.1f}   ({len(r)} runs)")
PY
echo "ALL DONE $(date +%T)"
