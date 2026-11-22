#!/usr/bin/env bash
# Runs in the container (tools/pascal/remote.sh run 'tools/pascal/build_speech.sh TAG'): the text CLI and scorer
# (tools/pascal/build_text.sh), then lfm3-speech (native/apps/lfm3_speech.cu: text and audio in, text and speech out),
# the server with speech and images (liquid-lfm3-worker, liquid-lfm3-server: -DLFM3_TEXT_ONLY=0 -DLFM3_VISION=1),
# lfm3-vision-check (the encoder's stages, images-v1 rule 2) and text-sampling-oracle for sm_61 into
# /workspace/build/TAG, with the hashes of their sources and binaries. The vision encoder computes in FP32 on the CUDA
# cores (lfm/vision_encoder.cuh: no tensor cores here). VISION=0 leaves vision out (lfm/vision_absent.hpp), as phase 4
# built. EXTRA_FLAGS as for build_text.sh (the parity configuration: -DLFM_GEMM_TENSOR_CORE_MODEL).
set -euo pipefail
tag=${1:?usage: build_speech.sh TAG}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N=$(cd "$here/../../native" && pwd)
"$here/build_text.sh" "$tag"
out=/workspace/build/$tag
L=/workspace/build/ljt-3.2.0
bash "$here/../build_libjpeg_turbo.sh" "$L" > /dev/null
cd "$out"
rm -f lfm3-speech liquid-lfm3-worker liquid-lfm3-server lfm3-vision-check text-sampling-oracle image_*.o \
  serve-binaries.sha256 serve-sources.sha256 serve-flags.txt speech-binaries.sha256 speech-sources.sha256 speech-flags.txt  # either script's records
(cd "$N" && sha256sum apps/lfm3_speech.cu src/lfm3_main.cu src/lfm3_batch.cuh src/lfm3_speech.cuh src/safetensors.cpp \
  src/lfm3_worker.cu src/lfm3_server.cu src/lfm3_decode_batch.cuh src/image_decode.cpp src/image_preprocess.cpp \
  tests/text_sampling_oracle.cpp tests/lfm3_vision_check.cu include/lfm/*.hpp include/lfm/*.cuh) > speech-sources.sha256
flags=(-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -DLFM3_TEXT_ONLY=0 -DLFM3_VISION=${VISION:-1}
       ${EXTRA_FLAGS:-} -I "$N/include" -I "$L/ljt/include")
printf "%s\n" "${flags[*]}" > speech-flags.txt
fail() { { grep -E "error" "$1.build.log" || tail -5 "$1.build.log"; } | sort | uniq -c | head -20; echo "build_speech.sh: $1 FAILED"; exit 1; }
for s in image_decode image_preprocess; do
  g++ -std=c++20 -O2 -I "$N/include" -I "$L/ljt/include" -c "$N/src/$s.cpp" -o "$s.o" > "$s.build.log" 2>&1 || fail "$s"
done
built() { echo "$1: warnings $(grep -ci warning "$1.build.log" || true), sha256 $(sha256sum "$1" | cut -d' ' -f1), $(stat -c %s "$1") bytes"; }
nvcc "${flags[@]}" -o lfm3-speech "$N/apps/lfm3_speech.cu" "$N/src/safetensors.cpp" image_decode.o image_preprocess.o \
  "$L/ljt/lib/libjpeg.a" -lpthread > lfm3-speech.build.log 2>&1 || fail lfm3-speech
built lfm3-speech
for b in worker server; do
  nvcc "${flags[@]}" -o "liquid-lfm3-$b" "$N/src/lfm3_$b.cu" "$N/src/safetensors.cpp" image_decode.o image_preprocess.o \
    "$L/ljt/lib/libjpeg.a" -lpthread > "liquid-lfm3-$b.build.log" 2>&1 || fail "liquid-lfm3-$b"
  built "liquid-lfm3-$b"
done
if [ "${VISION:-1}" = 1 ]; then
  nvcc "${flags[@]}" -o lfm3-vision-check "$N/tests/lfm3_vision_check.cu" "$N/src/safetensors.cpp" -lpthread \
    > lfm3-vision-check.build.log 2>&1 || fail lfm3-vision-check
  built lfm3-vision-check
fi
g++ -std=c++17 -O2 -I "$N/include" -o text-sampling-oracle "$N/tests/text_sampling_oracle.cpp" > text-sampling-oracle.build.log 2>&1 \
  || fail text-sampling-oracle
built text-sampling-oracle
sha256sum lfm3-speech liquid-lfm3-worker liquid-lfm3-server $( [ "${VISION:-1}" = 1 ] && echo lfm3-vision-check) text-sampling-oracle > speech-binaries.sha256
