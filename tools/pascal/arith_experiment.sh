#!/usr/bin/env bash
# Runs in the container: validation/prefill-arithmetic-v1. Scores windows of 65 tokens (positions 1-64) with the
# prefill GEMM's arithmetic varied: plain (nearest), the tensor-core model at every site or at one kind of site,
# and the plain kernel rounding every multiply-add toward zero or away from zero.
#   tools/pascal/arith_experiment.sh [VARIANT...]
# A variant whose result already exists is skipped ("exists"), so the run can be resumed: results in
# /workspace/build/arith-v1 are from the tree that was there when each was scored (see their meta.json).
# Exit 0 only if every build and every requested variant succeeded.
set -uo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); B=/workspace/build; out=arith-v1; fail=0
unset LFM_TCM_ROWS LFM_TCM_SITES LFM_ONCE_ROWS LFM3_COMPACT_WINDOWS LFM3_PREFILL_HEADS EXTRA_FLAGS  # nothing inherited from the caller
build() { "$repo/tools/pascal/build_text.sh" "$1" | grep -v links || { echo "build $1 FAILED"; exit 1; }; }
build ar-plain
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL build ar-exact
EXTRA_FLAGS=-DLFM_GEMM_PLAIN_ROUND=1 build ar-rz
EXTRA_FLAGS=-DLFM_GEMM_PLAIN_ROUND=2 build ar-away
run() {  # NAME BUILD [SITES] [ROWS]
  local name=$1 build=$2 sites=${3:-} rows=${4:-}
  [ -e "$B/$out/$name/nll-pascal-65.f32" ] && { echo "$name: exists"; return; }
  echo "== $name (build $build${sites:+, LFM_TCM_SITES=$sites}${rows:+, LFM_TCM_ROWS=$rows})"
  ( [ -z "$rows" ] || export LFM_TCM_ROWS=$rows; [ -z "$sites" ] || export LFM_TCM_SITES=$sites
    WINDOWS_DIR=/workspace/data/ppl-windows-65 "$repo/tools/pascal/score.sh" "$build" "$out/$name" 65 ) || { echo "$name FAILED"; fail=1; }
}
want=${*:-plain exact rz away head nohead mlp conv attnproj attn}
for v in $want; do
  status=0
  case $v in
    plain) run plain ar-plain ;; exact) run exact ar-exact ;; rz) run rz ar-rz ;; away) run away ar-away ;;
    head) run head ar-exact 1 ;; nohead) run nohead ar-exact 62 ;; mlp) run mlp ar-exact 2 ;; conv) run conv ar-exact 4 ;;
    attnproj) run attnproj ar-exact 8 ;; attn) run attn ar-exact 16 ;;
    bosexact) run bosexact ar-exact "" 1 ;; bosplain) run bosplain ar-exact "" -1 ;;
    first8exact) run first8exact ar-exact "" 8 ;;
    bos257) LFM_TCM_ROWS=1 WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/bos257" 257 || status=1 ;;
    bos256) LFM_TCM_ROWS=1 WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/bos256" 256 || status=1 ;;
    thor257) WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/thor257" 257 || status=1 ;;
    thor256) WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-exact "$out/thor256" 256 || status=1 ;;
    # validation/round-once-v1: the [BOS] row summed exactly and rounded once, the plain kernel for every other row
    once65) EXTRA_FLAGS=-DLFM_GEMM_ROUND_ONCE build ar-once
            LFM_ONCE_ROWS=1 WINDOWS_DIR=/workspace/data/ppl-windows-65 "$repo/tools/pascal/score.sh" ar-once "$out/once65" 65 || status=1 ;;
    once256|once257) EXTRA_FLAGS=-DLFM_GEMM_ROUND_ONCE build ar-once
            LFM_ONCE_ROWS=1 WINDOWS_DIR=/workspace/data/ppl-windows "$repo/tools/pascal/score.sh" ar-once "$out/$v" "${v#once}" || status=1 ;;
    *) echo "unknown variant $v"; exit 2 ;;
  esac
  [ $status = 0 ] || { echo "$v FAILED"; fail=1; }
done
exit $fail
