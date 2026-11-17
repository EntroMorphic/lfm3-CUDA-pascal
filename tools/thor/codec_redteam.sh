#!/usr/bin/env bash
# Red-team of #2 step 2 (validation/codec-v1): untested decoder paths, refused inputs, FP64 anchor widening, and
# where the per-call time of 1-frame streaming goes (nsys). Usage: codec_redteam.sh TAG CODEC_TAG TRACE_TAG ANCHOR_TAG
set -uo pipefail
tag=$1; prev=$2; trace=$3; anchor=$4
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
step(){ echo "== $1 $(date +%T)"; }; fail(){ echo "FAIL: $1"; exit 1; }
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
T=/workspace/models/LFM3-Omni-2B/audio_tokenizer; BIN=$C/$tag/cmake/liquid-codec
step sources; (cd $R && sha256sum native/include/lfm/codec.cuh native/src/codec_main.cu tools/lfm3_codec_widening_check.py \
  tools/thor/codec_redteam.sh) | sed 's/^/  /'
step build
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "cmake -S /src/native -B $C/$tag/cmake > $C/$tag/cmake.log 2>&1 && cmake --build $C/$tag/cmake --target liquid-codec -j4 > $C/$tag/build.log 2>&1; rc=\$?; chown -R 1000:1000 $C/$tag; exit \$rc" \
  || { grep -iE "error" $O/build.log | head -20; fail build; }
echo "warnings: $(grep -ci warning $O/build.log)"
step "same output as $prev, and selftest (reset; 7- and 5-frame chunks; empty finalize; whole again)"
for d in $B/$trace/*/; do c=$(basename $d); test -e $B/$prev/$c.i32 || continue
  out=$(docker exec -e LFM_CODEC_SELFTEST=1 -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 \
    $BIN $T $C/$prev/$c.i32 $C/$tag/$c.whole.f32) || fail "selftest $c"
  cmp -s $O/$c.whole.f32 $B/$prev/$c.whole.f32 && same=true || same=false
  echo "$c: equal_to_$prev=$same $(grep -o '"chunk7.*' <<< "$out")"
  [ $same = true ] && ! grep -q false <<< "$out" || fail "$c"
done
step refused-inputs
python3 - $O <<'PY'
import numpy as np, sys
o = sys.argv[1]
np.full((3, 12), 2016, np.int32).tofile(f"{o}/eos.i32")          # the EOS code is not decodable
np.full((3, 12), -1, np.int32).tofile(f"{o}/neg.i32")
np.zeros(13, np.int32).tofile(f"{o}/ragged.i32")                  # not whole frames
open(f"{o}/empty.i32", "wb").close()
PY
for f in eos neg ragged empty; do
  if $EX $BIN $T $C/$tag/$f.i32 $C/$tag/$f.out.f32 > $O/$f.log 2>&1; then fail "$f accepted"; fi
  echo "$f: refused: $(tail -1 $O/$f.log)"; done
for a in -1 x 3x; do if $EX $BIN $T $C/$prev/tts-fox-female.i32 $C/$tag/bad.f32 $a > /dev/null 2>&1; then fail "chunk '$a' accepted"; fi
  echo "chunk '$a': refused"; done
step fp64-widening
docker exec lfm3-omni-dev /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/lfm3_codec_widening_check.py \
  /workspace/models/LFM3-Omni-2B $C/$trace/tts-fox-female $C/$anchor $C/$prev/tts-fox-female.whole.f32 2>&1 | grep '^{' | tee $O/widening.json
step nsys-1-frame-chunks
docker exec lfm3-profile-v1 bash -c "command -v nsys" > /dev/null 2>&1 && P="$EX nsys" || P=""
if [ -z "$P" ]; then
  # nsys only on the host: run the binary there with the container's pinned CUDA libraries
  nsys profile -o $O/chunk1 --force-overwrite true -t cuda --stats=false env LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia \
    $O/cmake/liquid-codec $W/models/LFM3-Omni-2B/audio_tokenizer $B/$prev/tts-fox-female.i32 $O/chunk1.f32 1 > $O/nsys.log 2>&1 || { tail -5 $O/nsys.log; echo "nsys run failed"; }
  test -e $O/chunk1.nsys-rep && nsys stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $O/chunk1 $O/chunk1.nsys-rep > /dev/null 2>&1
  ls $O/chunk1_* 2>/dev/null
fi
echo "ALL DONE $(date +%T)"
