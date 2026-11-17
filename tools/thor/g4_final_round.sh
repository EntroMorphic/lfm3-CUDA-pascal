#!/usr/bin/env bash
# G4 final round on Thor (validation/g4-gates-v2/PLAN.md), run only after tier 2b passes:
#  1. build the default (now the native GEMM) and the cuBLAS A/B configuration; check identities and linkage;
#  2. tier 3 on the default scorer: perplexity NLL bitwise identical to attempt 3 at all four lengths;
#  3. every gate on the default build with TIERS=pin (tier 1 pins, tier 2a, exact memory), then again with
#     TIERS=golden (tier 1 reproduces bit for bit).
# Usage: g4_final_round.sh TAG     (writes build/TAG-* and validation/g4-gates-v2/pins/ in the mounted tree)
set -uo pipefail
tag=$1
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
D=$tag-default; A=$tag-cublas; S=$tag-score; P=$tag-ppl; PINS=$R/validation/g4-gates-v2/pins
for o in $D $A $S $P $tag-pin-gates $tag-golden-gates; do test ! -e $B/$o || { echo "exists $o"; exit 1; }; done
test ! -e $PINS || { echo "pins already exist: $PINS"; exit 1; }
step(){ echo "== $1 $(date +%T)"; }
fail(){ echo "FAIL: $1"; exit 1; }
step sources; (cd $R && git rev-parse HEAD 2>/dev/null; sha256sum native/CMakeLists.txt native/src/*.cu native/src/*.cuh native/include/lfm/gemm.cuh tools/*.py tools/thor/*.sh) | sed 's#^#  #'
build(){ docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$1 ${2:-}; rc=\$?; chown -R 1000:1000 $C/$1 $C/$1.cmake.log $C/$1.build.log 2>/dev/null; exit \$rc"; }
needed(){ readelf -d "$1" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | tr '\n' ' '; }
same(){ objcopy --strip-all "$1" /tmp/$tag-a; objcopy --strip-all "$2" /tmp/$tag-b; cmp -s /tmp/$tag-a /tmp/$tag-b && echo same || echo DIFF; }

step build-default; build $D || { echo FAIL; exit 1; }; echo "warnings: $(grep -ci warning $B/$D.build.log)"
step build-cublas; build $A -DLIQUID_LFM3_NATIVE_GEMM=OFF || { echo FAIL; exit 1; }; echo "warnings: $(grep -ci warning $B/$A.build.log)"
step identities
# The default build inherits v6's evidence only if it is v6; the cuBLAS option must still be the v3 engine.
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server lfm3-decode-batch-probe; do
  n=$(same $B/cmake-g4-native-v6/$f $B/$D/$f); c=$(same $B/cmake-server-v3/$f $B/$A/$f)
  echo "$f default vs native-v6: $n; cublas option vs cmake-server-v3: $c"
  [ "$n" = same ] && [ "$c" = same ] || fail "$f identity"
done
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server; do
  echo "$f default: $(stat -c %s $B/$D/$f) bytes; needed: $(needed $B/$D/$f)"
  needed $B/$D/$f | grep -q cublas && fail "$f default links cuBLAS"
done
(cd $B/$D && sha256sum liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle lfm3-decode-batch-probe) > $B/$D.sha256

step tier3-scorer; mkdir -p $B/$S $B/$P
docker run --rm --network none -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_scorer.sh $C/$S; rc=\$?; chown -R 1000:1000 $C/$S; exit \$rc" || { echo FAIL scorer; exit 1; }
echo "scorer mode: $(cat $B/$S/mode.txt); needed: $(needed $B/$S/lfm3-score)"
[ "$(cat $B/$S/mode.txt)" = native-gemm ] || fail "scorer is not native-gemm"
cp -p $B/ppl-v5/windows-*.i32 $B/$P/
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
for L in 256 257 4096 32768; do
  $EX $C/$S/lfm3-score /workspace/models/LFM3-Omni-2B $C/$P/windows-$L.i32 $L $C/$P/nll-native-gemm-$L.f32 >> $B/$P.log 2>&1 || { echo "FAIL score $L"; exit 1; }
  cmp -s $B/$P/nll-native-gemm-$L.f32 $B/ppl-g4-v3/nll-native-gemm-$L.f32 || fail "$L: NLL differs from attempt 3"
  echo "$L: NLL bitwise identical to attempt 3"
done

step gates-pin
# Strict: the first failing gate stops the run, and a pin set exists only if every gate passed.
mkdir -p $PINS
KEEP_GOING=0 TIERS=pin PIN_DIR=/workspace/liquid-CUDA/validation/g4-gates-v2/pins bash $R/tools/thor/run_native_gates.sh $D $tag-pin
rc=$?; echo "pin gates rc=$rc"; cat $B/$tag-pin-gates/status.json
if [ $rc != 0 ]; then rm -rf $PINS; echo "pins removed: the pin run did not pass"; exit 1; fi
step gates-golden
KEEP_GOING=1 TIERS=golden PIN_DIR=/workspace/liquid-CUDA/validation/g4-gates-v2/pins \
  bash $R/tools/thor/run_native_gates.sh $D $tag-golden; rc=$?; echo "golden gates rc=$rc"
cat $B/$tag-golden-gates/failures 2>/dev/null; cat $B/$tag-golden-gates/status.json
[ $rc = 0 ] || { rm -rf $PINS; fail "golden gates (pins removed: they did not reproduce)"; }
echo "ALL DONE $(date +%T)"
