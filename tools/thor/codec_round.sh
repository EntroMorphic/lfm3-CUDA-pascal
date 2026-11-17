#!/usr/bin/env bash
# #2 step 2 (A2): build liquid-codec and check it against the HF codec anchors (Thor host).
# Decodes every traced case whole and streamed (12-frame and 1-frame chunks); dumps fox-female's intermediates.
# Usage: codec_round.sh TAG TRACE_TAG ANCHOR_TAG
set -uo pipefail
tag=$1; trace=$2; anchor=$3
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O/dump
step(){ echo "== $1 $(date +%T)"; }; fail(){ echo "FAIL: $1"; exit 1; }
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="docker exec lfm3-omni-dev /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/lfm3_codec_check.py"
step sources; (cd $R && sha256sum native/include/lfm/codec.cuh native/src/codec_main.cu native/CMakeLists.txt \
  tools/lfm3_codec_check.py tools/thor/codec_round.sh) | sed 's/^/  /'
echo "power: $(nvpmodel -q 2>/dev/null | head -1)"
step build
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "cmake -S /src/native -B $C/$tag/cmake > $C/$tag/cmake.log 2>&1 && cmake --build $C/$tag/cmake --target liquid-codec -j4 > $C/$tag/build.log 2>&1; rc=\$?; chown -R 1000:1000 $C/$tag; exit \$rc" \
  || { grep -iE "error" $O/build.log $O/cmake.log | head -20; fail build; }
echo "warnings: $(grep -ci warning $O/build.log)"; sha256sum $O/cmake/liquid-codec | sed 's/^/  /'
dyn=$(readelf -d $O/cmake/liquid-codec) || fail readelf; [ "$(grep -cE 'libcublas|libcufft' <<< "$dyn")" = 0 ] || fail "links cuBLAS/cuFFT"
cases=(); for d in $B/$trace/*/; do test -e $d/sequences.pt && cases+=("$(basename $d)"); done
for c in "${cases[@]}"; do
  step "$c"
  $PY codes $C/$trace/$c $C/$tag/$c.i32 > /dev/null || fail "codes $c"
  dump=(); [ $c = tts-fox-female ] && dump=(-e LFM_CODEC_DUMP=$C/$tag/dump)
  docker exec "${dump[@]}" -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 \
    $C/$tag/cmake/liquid-codec /workspace/models/LFM3-Omni-2B/audio_tokenizer $C/$tag/$c.i32 $C/$tag/$c.whole.f32 \
    | tee -a $O/runs.jsonl || fail "whole $c"
  for k in 12 1; do
    $EX $C/$tag/cmake/liquid-codec /workspace/models/LFM3-Omni-2B/audio_tokenizer $C/$tag/$c.i32 $C/$tag/$c.chunk$k.f32 $k \
      | tee -a $O/runs.jsonl || fail "chunk$k $c"
  done
done
step stages; $PY stages $C/$anchor/tts-fox-female.intermediates.pt $C/$tag/dump | tee $O/stages.txt
step report; $PY report $C/$anchor $C/$tag $C/$tag/a2.json "${cases[@]}"; rc=$?
step asr  # coherency: the native audio must still say the words (same Whisper and normalization as tts-trace-v1)
for c in "${cases[@]}"; do mkdir -p $O/asr/$c; cp $B/$trace/$c/summary.json $O/asr/$c/
  docker exec lfm3-omni-dev /opt/venv/bin/python3 -c "import numpy as np, soundfile as sf, sys; sf.write(sys.argv[2], np.fromfile(sys.argv[1], np.float32), 24000)" \
    $C/$tag/$c.whole.f32 $C/$tag/asr/$c/response.wav || fail "wav $c"; done
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/lfm3_asr.py \
  /workspace/models/whisper-large-v3 $C/$tag/asr.json $(for c in "${cases[@]}"; do echo $C/$tag/asr/$c; done) 2>&1 | grep WER || fail asr
echo "ALL DONE rc=$rc $(date +%T)"
