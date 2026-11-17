#!/usr/bin/env bash
# G4 native-GEMM evidence that round 4's script did not produce cleanly, rerun from this committed script (Thor
# host): the speed comparison, http-batch with exact engine-buffer accounting, and the FP32-anchored spread with
# the current compare_trace_spread.py (which checks exact logit traces). Writes TAG-* under build/ and a log.
# Usage: g4_native_evidence.sh DEFAULT_BUILD NATIVE_BUILD SPREAD_ROOT TAG
#   e.g. g4_native_evidence.sh cmake-g4-default-v6 cmake-g4-native-v6 spread-v2 g4-evidence-v1
set -uo pipefail
default=$1; native=$2; spread=$3; tag=$4
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
for o in $tag-decode-rate $tag-http-batch; do test ! -e $B/$o || { echo "exists $o"; exit 1; }; done
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia"
CK=/workspace/models/LFM3-Omni-2B
step(){ echo "== $1 $(date +%T)"; }
# Speed depends on the power mode and clocks (validation/g4-prefill-budget-v1/): record them before and after.
power(){ echo "power $1: $(nvpmodel -q 2>/dev/null | head -1); $(for f in /sys/class/devfreq/gpu-*; do echo -n "$(basename $f) $(cat $f/governor) min=$(cat $f/min_freq) max=$(cat $f/max_freq); "; done)"; }
power before
step decode-rate
O=$B/$tag-decode-rate; mkdir -p $O
sha256sum $B/$default/liquid-lfm3 $B/$native/liquid-lfm3 > $O/binaries.sha256
# Prompts of L WikiText-2 tokens (window file of the 32K perplexity run), then 128 teacher-forced steps.
python3 - $B/ppl-v5/windows-32768.i32 $O <<'PY'
import json, sys
import numpy as np
ids = np.fromfile(sys.argv[1], np.int32).tolist()
for length in (86, 1024, 4096, 32000):
    capacity = length + 128 if length + 128 <= 4096 else 32768
    request = dict(input_ids=ids[:length], teacher_ids=ids[length:length + 128], max_new_tokens=0,
                   capacity=capacity, warmup_runs=1)
    if capacity > 4096:
        request["experimental_long_context"] = True
    json.dump(request, open(f"{sys.argv[2]}/req-{length}.json", "w"))
PY
for L in 86 1024 4096 32000; do for rep in 1 2 3; do for b in default:$default native:$native; do n=${b%%:*}; d=${b#*:}
  $EX lfm3-profile-v1 $C/$d/liquid-lfm3 --experimental $CK $C/$tag-decode-rate/req-$L.json $C/$tag-decode-rate/$n-$L-$rep.json \
    > $O/$n-$L-$rep.log 2>&1 || { echo "FAIL $n $L $rep"; tail -2 $O/$n-$L-$rep.log; }
done; done; done
python3 $R/validation/g4-native-gates-v1/decode-rate/summarize.py $O > $O/summary.json && echo "decode-rate summary written"
step http-batch-metrics
$EX -e LFM_ALLOCATION_METRICS=1 lfm3-profile-v1 /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/check_lfm3_http_batch.py \
  --checkpoint $CK --worker $C/$native/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$tag-http-batch \
  > $B/$tag-http-batch.log 2>&1
echo "rc=$?"
python3 - $B/$tag-http-batch/summary.json <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
e, p = j["empty_session_gpu_baseline"], j.get("post_close_gpu", {})
s = [x["gpu"] for x in j.get("samples", [])]
print(json.dumps(dict(status=j.get("status"), samples=len(s),
                      app_live_bytes=dict(empty=e.get("app_live_bytes"), post_close=p.get("app_live_bytes"),
                                          soak_distinct=sorted({x.get("app_live_bytes") for x in s})),
                      post_close_minus_empty_device_mib=(p["cuda_free_bytes"] - e["cuda_free_bytes"]) >> 20)))
PY
step spread-compare
python3 $R/tools/compare_trace_spread.py $B/$spread/fp32 $B/$tag-fp32-rank.json --cases baseline,arithmetic,window257,single \
  --candidate eager $B/$spread/eager/reference $B/$spread/eager --candidate sdpa $B/$spread/sdpa/reference $B/$spread/sdpa \
  --candidate native $B/$spread/eager/native $B/$spread/eager
power after
echo "ALL DONE $(date +%T)"
