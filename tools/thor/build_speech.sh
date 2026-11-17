#!/usr/bin/env bash
# Build lfm3-speech (#2 step 4) in the pinned compiler image, with the engine's flags (native GEMM, no cuBLAS).
# Image input (#4): the native front end (src/image_decode.cpp, src/image_preprocess.cpp) is compiled by g++ with the
# flags its validation used (tools/build_image_decode_check.sh: -std=c++20 -O2) and linked with the vendored
# libjpeg-turbo 3.2.0 built on Thor (build/images-v1/decode-aarch64/ljt; the image's g++ is the host's, 13.3.0).
# Usage (Thor host): build_speech.sh OUT_TAG   -> build/OUT_TAG/{lfm3-speech, sources.sha256, build-lfm3-speech.log}
set -euo pipefail
tag=$1; W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; C=/workspace/liquid-CUDA/build/$tag
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
test -f $W/liquid-CUDA/build/images-v1/decode-aarch64/ljt/lib/libjpeg.a || { echo "no libjpeg-turbo build (tools/build_image_decode_check.sh build/images-v1/decode-aarch64)"; exit 1; }
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=/workspace/liquid-CUDA/native
  L=/workspace/liquid-CUDA/build/images-v1/decode-aarch64/ljt
  F='-O3 --fmad=false -std=c++20 -arch=sm_110 -I '\$N/include
  (cd /workspace/liquid-CUDA && sha256sum native/src/lfm3_speech.cuh native/src/lfm3_batch.cuh native/src/lfm3_main.cu native/apps/lfm3_speech.cu native/include/lfm/audio_*.hpp native/include/lfm/audio_encoder.cuh native/include/lfm/wav.hpp native/include/lfm/base64.hpp native/include/lfm/fp32_ops.cuh native/include/lfm/safetensors_header.hpp native/include/lfm/speech_text.hpp native/include/lfm/speech_normalize.hpp native/include/lfm/speech_run.hpp native/include/lfm/lfm3_text.hpp native/include/lfm/codec.cuh native/include/lfm/depth.cuh native/include/lfm/vision_encoder.cuh native/include/lfm/image_*.hpp native/src/image_*.cpp) > sources.sha256
  g++ -std=c++20 -O2 -Wall -Wextra -I \$N/include -I \$L/include -c \$N/src/image_decode.cpp -o image_decode.o > build-image.log 2>&1
  g++ -std=c++20 -O2 -Wall -Wextra -I \$N/include -c \$N/src/image_preprocess.cpp -o image_preprocess.o >> build-image.log 2>&1
  nvcc \$F -DLFM3_NATIVE_GEMM -o lfm3-speech \$N/apps/lfm3_speech.cu \$N/src/safetensors.cpp image_decode.o image_preprocess.o \$L/lib/libjpeg.a > build-lfm3-speech.log 2>&1
  ! grep -iE ' error|warning' build-*.log; chown -R 1000:1000 ."
readelf -d $W/liquid-CUDA/build/$tag/lfm3-speech | grep -q cublas && { echo "lfm3-speech links cuBLAS"; exit 1; }
(cd $W/liquid-CUDA/build/$tag && sha256sum lfm3-speech)
