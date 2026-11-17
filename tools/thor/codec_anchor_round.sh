#!/usr/bin/env bash
# #2 step 2 (A2): HF codec anchors (bf16 / eager / streamed / fp32 / fp64) from the reference's forced codes (Thor host).
# Usage: codec_anchor_round.sh TAG TRACE_TAG   (TRACE_TAG: a tts_trace_round.sh output under build/)
set -uo pipefail
tag=$1; trace=$2
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; O=$W/liquid-CUDA/build/$tag; C=/workspace/liquid-CUDA/build
test ! -e $O || { echo "exists $tag"; exit 1; }
EX="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
echo "== sources"; (cd $W/liquid-CUDA && sha256sum tools/lfm3_codec_anchor.py tools/thor/codec_anchor_round.sh) | sed 's/^/  /'
echo "power: $(nvpmodel -q 2>/dev/null | head -1)"
cases=(); for d in $W/liquid-CUDA/build/$trace/*/; do test -e $d/sequences.pt && cases+=("$C/$trace/$(basename $d)"); done
[ ${#cases[@]} -gt 0 ] || { echo "FAIL: no cases in $trace"; exit 1; }
echo "== anchor $(date +%T)"
$EX /workspace/liquid-CUDA/tools/lfm3_codec_anchor.py /workspace/models/LFM3-Omni-2B $C/$tag --dump tts-fox-female "${cases[@]}" \
  > $W/liquid-CUDA/build/$tag.log 2>&1; rc=$?
mkdir -p $O; mv $W/liquid-CUDA/build/$tag.log $O/anchor.log 2>/dev/null
cut -c1-400 $O/anchor.log | tail -12
echo "ALL DONE rc=$rc $(date +%T)"
