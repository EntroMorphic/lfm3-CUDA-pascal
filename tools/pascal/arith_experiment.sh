#!/usr/bin/env bash
# Runs in the container: validation/prefill-arithmetic-v1. Scores windows of 65 tokens (positions 1-64) with the
# prefill GEMM's arithmetic varied: plain (nearest), the tensor-core model at every site or at one kind of site,
# and the plain kernel rounding every multiply-add toward zero or away from zero.
#   tools/pascal/arith_experiment.sh [VARIANT...]
set -uo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); B=/workspace/build; out=arith-v1
"$repo/tools/pascal/build_text.sh" ar-plain | grep -v links
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL "$repo/tools/pascal/build_text.sh" ar-exact | grep -v links
EXTRA_FLAGS=-DLFM_GEMM_PLAIN_ROUND=1 "$repo/tools/pascal/build_text.sh" ar-rz | grep -v links
EXTRA_FLAGS=-DLFM_GEMM_PLAIN_ROUND=2 "$repo/tools/pascal/build_text.sh" ar-away | grep -v links
run() {  # NAME BUILD [SITES]
  local name=$1 build=$2 sites=${3:-}
  [ -e "$B/$out/$name/nll-pascal-65.f32" ] && { echo "$name: exists"; return; }
  echo "== $name (build $build${sites:+, LFM_TCM_SITES=$sites})"
  if [ -n "$sites" ]; then export LFM_TCM_SITES=$sites; else unset LFM_TCM_SITES; fi
  WINDOWS_DIR=/workspace/data/ppl-windows-65 "$repo/tools/pascal/score.sh" "$build" "$out/$name" 65 || echo "$name FAILED"
  unset LFM_TCM_SITES
}
want=${*:-plain exact rz away head nohead mlp conv attnproj attn}
for v in $want; do
  case $v in
    plain) run plain ar-plain ;; exact) run exact ar-exact ;; rz) run rz ar-rz ;; away) run away ar-away ;;
    head) run head ar-exact 1 ;; nohead) run nohead ar-exact 62 ;; mlp) run mlp ar-exact 2 ;; conv) run conv ar-exact 4 ;;
    attnproj) run attnproj ar-exact 8 ;; attn) run attn ar-exact 16 ;;
    thor257) WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/thor257" 257 ;;
    thor256) WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/thor256" 256 ;;
    *) echo "unknown variant $v"; exit 2 ;;
  esac
done
