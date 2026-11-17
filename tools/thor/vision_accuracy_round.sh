#!/usr/bin/env bash
# validation/images-speed-v1 rule 1 (and images-v1 rule 2's measure): lfm3-vision-check in a precision mode on all 31
# images (the 11 stage-0 cases and the 20 held-out MMStar images), each compared with HF's FP32 and BF16 dumps by
# tools/compare_lfm3_vision_stages.py; the summary checks rule 1's thresholds (error <= 0.15 x HF BF16's, minimum
# token cosine >= 0.9999) on every image.
# Usage (Thor host): vision_accuracy_round.sh BUILD_TAG OUT_TAG LINEAR ATTENTION   (modes: fp32, x1, x2)
set -uo pipefail
T=${1:?BUILD_TAG}; O=${2:?OUT_TAG}; LIN=${3:?LINEAR}; ATT=${4:?ATTENTION}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
P="docker exec lfm3-profile-v1 /opt/venv/bin/python"
mkdir -p $B/$O && cd $B/$O
cases=""
for d in $B/vision-ref/hf-5.20.2/float32/*/; do cases="$cases hf-5.20.2:$(basename $d)"; done
for d in $B/vision-ref/heldout/float32/*/; do cases="$cases heldout:$(basename $d)"; done
for c in $cases; do
  set_=${c%%:*}; n=${c#*:}; R=$C/vision-ref/$set_
  $X $C/$T/lfm3-vision-check /workspace/models/LFM3-Omni-2B $R/float32/$n $C/$O/$n --linear $LIN --attention $ATT > $n.log 2>&1 \
    || { echo "RUN FAILED $n"; tail -2 $n.log; continue; }
  $P /workspace/liquid-CUDA/tools/compare_lfm3_vision_stages.py $C/$O/$n $R/float32/$n $R/bfloat16/$n --json $C/$O/$n.compare.json > $n.compare.log 2>&1
done
python3 - $B/$O <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1]); rows = {}; ok = True
for f in sorted(o.glob("*.compare.json")):
    j = json.loads(f.read_text())["stages"]["image_features"]
    passed = j["ratio"] <= 0.15 and j["native_min_cos"] >= 0.9999
    ok &= passed
    rows[f.name.split(".")[0]] = dict(ratio=j["ratio"], native_min_cos=j["native_min_cos"], passed=passed)
print(f"{len(rows)} images; ratio {min(r['ratio'] for r in rows.values()):.3f}-{max(r['ratio'] for r in rows.values()):.3f}; "
      f"min cosine {min(r['native_min_cos'] for r in rows.values()):.7f}; failing: {[k for k, r in rows.items() if not r['passed']]}")
print("RULE 1", "PASS" if ok and len(rows) == 31 else "FAIL")
(o / "rule1.json").write_text(json.dumps(dict(passed=ok and len(rows) == 31, images=rows), indent=1))
PY
