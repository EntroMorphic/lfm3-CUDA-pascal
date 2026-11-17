#!/usr/bin/env bash
# PG-19 long-context perplexity of an LFM3 checkpoint with HF (BF16 SDPA), on Thor (host).
# Usage: pg19_ppl_round.sh TAG MODEL_DIR MODEL_CLASS "ONE_PASS_LENGTHS" "CHUNKED_LENGTHS" CHUNK_ROWS
#   e.g. pg19_ppl_round.sh pg19-1p7b-v1 LFM3-1.7B causal "32768 65536 131072" "65536 131072 262144" 32768
# Writes build/TAG/{onepass,chunked}/ (ppl_reference.py outputs) and build/TAG/memory.log (host memory around each run).
set -uo pipefail
tag=$1; model=$2; cls=$3; onepass=$4; chunked=$5; chunk=$6
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
PY="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
mem(){ awk -v t="$1" '/MemTotal/{T=$2}/MemFree/{F=$2}/MemAvailable/{A=$2}/^Cached/{K=$2}/Buffers/{U=$2}
  END{printf "%s %s used %.1f GiB available %.1f GiB\n", strftime("%T"), t, (T-F-K-U)/1048576, A/1048576}' /proc/meminfo | tee -a $O/memory.log; }
echo "== sources"; (cd $W/liquid-CUDA && sha256sum tools/ppl_reference.py tools/thor/pg19_ppl_round.sh) | sed 's/^/  /'
grep EnableSystemMemoryPools /proc/driver/nvidia/params | sed 's/^/  /'
run(){ name=$1; shift; mem "before $name"
  $PY /workspace/liquid-CUDA/tools/ppl_reference.py --model /workspace/models/$model --model-class $cls \
    --corpus-dir /workspace/data/pg19-test --output $C/$tag/$name --configs sdpa --memory-fraction 0.85 "$@" 2>&1 \
    | grep -v -i "warn\|Loading weights" ; rc=${PIPESTATUS[0]}; mem "after $name rc=$rc"; }
run onepass --lengths $onepass
run chunked --lengths $chunked --chunk-rows $chunk
echo "ALL DONE $(date +%T)"
