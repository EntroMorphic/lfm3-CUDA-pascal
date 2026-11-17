#!/usr/bin/env bash
# G4 prefill heads-per-call budget rule: verification on Thor (journal/g4-prefill-memory_synth.md, steps 3-4).
# The rule only changes the batch count of bit-invariant GEMMs, so verification is: NLL bitwise identical to attempt 3
# at all four lengths, every gate against the UNCHANGED pins (TIERS=golden), then speed and exact memory.
# Usage: g4_prefill_budget_round.sh TAG CUBLAS_BUILD     (e.g. g4-budget-v1 g4-final-v2-cublas)
set -uo pipefail
tag=$1; cublas=$2
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
D=$tag-default; S=$tag-score; P=$tag-ppl
for o in $D $S $P $tag-golden-gates; do test ! -e $B/$o || { echo "exists $o"; exit 1; }; done
step(){ echo "== $1 $(date +%T)"; }; fail(){ echo "FAIL: $1"; exit 1; }
step sources; (cd $R && sha256sum native/CMakeLists.txt native/src/*.cu native/src/*.cuh native/include/lfm/gemm.cuh tools/*.py tools/thor/*.sh) | sed 's/^/  /'
step build
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$D; rc=\$?; chown -R 1000:1000 $C/$D $C/$D.cmake.log $C/$D.build.log 2>/dev/null; exit \$rc" || fail build
echo "warnings: $(grep -ci warning $B/$D.build.log)"
# Not `readelf | grep -q` (a SIGPIPE under pipefail can hide a match); a failed readelf is a failure, not a pass.
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server; do dyn=$(readelf -d $B/$D/$f) || fail "readelf $f"
  [ "$(grep -c libcublas <<< "$dyn")" = 0 ] || fail "$f links cuBLAS"; done
(cd $B/$D && sha256sum liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle lfm3-decode-batch-probe) > $B/$D.sha256
step nll-bit-identity; mkdir -p $B/$S $B/$P
docker run --rm --network none -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_scorer.sh $C/$S; rc=\$?; chown -R 1000:1000 $C/$S; exit \$rc" || fail scorer
cp -p $B/ppl-v5/windows-*.i32 $B/$P/
for L in 256 257 4096 32768; do
  docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $C/$S/lfm3-score /workspace/models/LFM3-Omni-2B \
    $C/$P/windows-$L.i32 $L $C/$P/nll-native-gemm-$L.f32 >> $B/$P.log 2>&1 || fail "score $L"
  cmp -s $B/$P/nll-native-gemm-$L.f32 $B/ppl-g4-v3/nll-native-gemm-$L.f32 || fail "$L: NLL differs from attempt 3"
  echo "$L: NLL bitwise identical to attempt 3"
done
step gates-golden
KEEP_GOING=1 TIERS=golden PIN_DIR=/workspace/liquid-CUDA/validation/g4-gates-v2/pins bash $R/tools/thor/run_native_gates.sh $D $tag-golden
rc=$?; echo "golden gates rc=$rc"; cat $B/$tag-golden-gates/failures 2>/dev/null; cat $B/$tag-golden-gates/status.json
[ $rc = 0 ] || fail "golden gates"
step speed-and-memory
bash $R/tools/thor/g4_native_evidence.sh $cublas $D spread-v2 $tag-evidence || fail evidence
echo "ALL DONE $(date +%T)"
