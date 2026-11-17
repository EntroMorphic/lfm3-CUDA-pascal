#!/usr/bin/env bash
# Coherency check of the text engines on Thor (tools/lfm3_coherency.py): greedy chat generations through the native
# "messages" path (chat template, tokenizer, text out) at every prefill heads-per-call band, for each build named.
# Usage: g4_coherency.sh TAG NAME:BUILD [NAME:BUILD ...]   e.g. g4_coherency.sh g4-coh-v1 native:g4-rt3-default cublas:g4-final-v2-cublas
set -uo pipefail
tag=$1; shift
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="$EX /opt/venv/bin/python3 -I /workspace/liquid-CUDA/tools/lfm3_coherency.py"
CK=/workspace/models/LFM3-Omni-2B
echo "== sources"; (cd $R && sha256sum tools/lfm3_coherency.py tools/thor/g4_coherency.sh) | sed 's/^/  /'
for b in "$@"; do sha256sum $B/${b#*:}/liquid-lfm3 | sed "s/^/  ${b%%:*} /"; done
$PY cases $CK $C/ppl-v5/windows-32768.i32 $C/$tag || { echo "FAIL cases"; exit 1; }
echo "== generate $(date +%T)"
for req in $O/req-*.json; do name=$(basename $req .json); name=${name#req-}
  for b in "$@"; do n=${b%%:*}; d=${b#*:}
    $EX $C/$d/liquid-lfm3 --experimental $CK $C/$tag/req-$name.json $C/$tag/$n-$name.json > $O/$n-$name.log 2>&1 \
      || { echo "FAIL $n $name"; tail -2 $O/$n-$name.log; }
  done
done
echo "== report $(date +%T)"
names=(); for b in "$@"; do names+=("${b%%:*}"); done
$PY report $CK $C/$tag "${names[@]}" > $O/report.md && echo "report written: $O/report.md"
echo "ALL DONE $(date +%T)"
