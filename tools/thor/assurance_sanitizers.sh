#!/usr/bin/env bash
# audio-assurance-v1 rule 1: compute-sanitizer on lfm3-speech speed-a4, 4 rows staggered and one at a time.
# Usage: [T=TAG] assurance_sanitizers.sh [TOOLS...]   (default: memcheck initcheck synccheck racecheck; TAG speed-a4)
# Also (rule 11): TAG's fp32-linear-check under each tool. MODES (default "conc batch") picks the lfm3-speech runs; MODES=none
# runs only the fp32-linear-check part (to run the slow racecheck jobs side by side on idle cores; exactness does not
# depend on timing). FP32=0 skips the fp32-linear-check part. Memory: lfm3-profile-v1 is limited to 48 GiB; racecheck
# with 4 rows takes ~30 GB and one at a time ~20 GB, so the two racecheck runs cannot share the container.
set -uo pipefail
R=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA
T=${T:-speed-a4}; SUF=$([ $T = speed-a4 ] && echo "" || echo "-$T")
B=$R/build/assure/san$SUF; C=/workspace/liquid-CUDA/build/assure/san$SUF; E=/workspace/liquid-CUDA/build
BIN=$E/$T/lfm3-speech; M=/workspace/models/LFM3-Omni-2B
SAN=$E/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
TOOLS=${*:-memcheck initcheck synccheck racecheck}; MODES=${MODES:-conc batch}; [ "$MODES" = none ] && MODES=""
mkdir -p $B && cd $B
REQS="$E/speech-e2e-v1/requests/tts-fox-female-s1.req.json
$E/speech-e2e-v1/requests/tts-fox-male-s2.req.json
$E/speed-final2/conc/requests/asr-1089-134686-0000.req.json
$E/speed-final2/conc/requests/spoken-q0.req.json
$E/speech-e2e-v1/requests/il-reflective-s1.req.json"
list() { for r in $REQS; do n=$(basename $r .req.json); echo "$r $C/$1/$n"; done > $1.txt; rm -rf $1; mkdir -p $1; }
run() {  # NAME MODE(batch|conc) [TOOL]
  local name=$1 mode=$2 tool=${3:-}; list $name
  local args; [ $mode = batch ] && args="--batch $C/$name.txt" || args="--concurrent 4 $C/$name.txt 3"
  local pre=""; case $tool in
    memcheck) pre="$SAN --tool memcheck --leak-check full --error-exitcode 99 --print-limit 50";;
    "") pre="";;
    *) pre="$SAN --tool $tool --error-exitcode 99 --print-limit 50";;
  esac
  local t0=$(date +%s); $EXEC $pre $BIN $M $args > $name.log 2>&1; local s=$?
  echo "$name: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY' $name.log | tr '\n' ' ')"
}
sha256sum $R/build/$T/lfm3-speech $R/build/speed-a4/fp32-linear-check > binary.sha256
for m in batch conc; do [ -d plain-$m ] || run plain-$m $m; done
for t in $TOOLS; do for m in $MODES; do
  run $t-$m $m $t
  python3 $R/tools/compare_speech_outputs.py plain-$m $t-$m $t-$m-compare.json > /dev/null && echo "  outputs identical to plain-$m" || echo "  OUTPUTS DIFFER from plain-$m"
done; done
[ "${FP32:-1}" = 0 ] || for t in $TOOLS; do
  t0=$(date +%s); $EXEC $SAN --tool $t --error-exitcode 99 --print-limit 50 $E/speed-a4/fp32-linear-check > fp32-$t.log 2>&1; s=$?
  echo "fp32-linear-check $t: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' fp32-$t.log | tr '\n' ' ') $(grep -cE 'PASS' fp32-$t.log) PASS lines"
done
python3 $R/tools/compare_speech_outputs.py plain-batch plain-conc plain-conc-vs-batch.json > /dev/null && echo "plain conc == plain batch" || echo "PLAIN CONC DIFFERS FROM BATCH"
echo "SANITIZERS DONE $(date +%T)"
