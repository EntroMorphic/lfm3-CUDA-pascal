#!/usr/bin/env bash
# #2 step 3 (A1/B1): build liquid-depth and check it against the HF depth-decoder anchors (Thor host).
# Usage: depth_round.sh TAG TRACE_TAG ANCHOR_TAG
set -uo pipefail
tag=$1; trace=$2; anchor=$3
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
step(){ echo "== $1 $(date +%T)"; }; fail(){ echo "FAIL: $1"; exit 1; }
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
step sources; (cd $R && sha256sum native/include/lfm/fp32_ops.cuh native/include/lfm/depth.cuh native/src/depth_main.cu \
  native/CMakeLists.txt tools/lfm3_depth_check.py tools/thor/depth_round.sh) | sed 's/^/  /'
echo "power: $(nvpmodel -q 2>/dev/null | head -1)"
step build
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "cmake -S /src/native -B $C/$tag/cmake > $C/$tag/cmake.log 2>&1 && cmake --build $C/$tag/cmake --target liquid-depth -j4 > $C/$tag/build.log 2>&1; rc=\$?; chown -R 1000:1000 $C/$tag; exit \$rc" \
  || { grep -iE "error" $O/build.log | head -20; fail build; }
echo "warnings: $(grep -ci warning $O/build.log)"; sha256sum $O/cmake/liquid-depth | sed 's/^/  /'
dyn=$(readelf -d $O/cmake/liquid-depth) || fail readelf; [ "$(grep -c libcublas <<< "$dyn")" = 0 ] || fail "links cuBLAS"
cases=(ref-interleaved interleaved-study tts-fox-female tts-fox-male tts-museum-female tts-museum-male)
for c in "${cases[@]}"; do
  step "$c"
  $EX $C/$tag/cmake/liquid-depth /workspace/models/LFM3-Omni-2B $C/$anchor/$c.hidden.bf16 $C/$anchor/$c.codes.i32 \
    $C/$anchor/$c.masked.u8 $C/$tag/$c.native.f32 | tee -a $O/runs.jsonl || fail "depth $c"
done
step check
docker exec lfm3-omni-dev /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/lfm3_depth_check.py $C/$trace $C/$anchor $C/$tag \
  $C/$tag/a1b1.json "${cases[@]}"; rc=$?
echo "ALL DONE rc=$rc $(date +%T)"
