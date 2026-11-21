#!/usr/bin/env bash
# Runs in the container (tools/pascal/remote.sh run 'tools/pascal/build_serve.sh TAG'): builds the text CLI, the scorer
# (tools/pascal/build_text.sh) and the text server for sm_61 into /workspace/build/TAG: liquid-lfm3-worker,
# liquid-lfm3-server (text only, PLAN decision 8: -DLFM3_TEXT_ONLY=1) and text-sampling-oracle, with the hashes of
# their sources and binaries. The image front end (host code, as imported) is compiled at -O2 with the vendored
# libjpeg-turbo (tools/build_libjpeg_turbo.sh, built once into /workspace/build/ljt-3.2.0), as native/CMakeLists.txt
# builds it. EXTRA_FLAGS as for build_text.sh (e.g. the parity configuration).
set -euo pipefail
tag=${1:?usage: build_serve.sh TAG}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N=$(cd "$here/../../native" && pwd)
"$here/build_text.sh" "$tag"
out=/workspace/build/$tag
L=/workspace/build/ljt-3.2.0
bash "$here/../build_libjpeg_turbo.sh" "$L" > /dev/null
cd "$out"
rm -f lfm3-speech liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle image_*.o \
  serve-binaries.sha256 serve-sources.sha256 serve-flags.txt speech-binaries.sha256 speech-sources.sha256 speech-flags.txt  # either script's records
(cd "$N" && sha256sum src/lfm3_worker.cu src/lfm3_server.cu src/lfm3_decode_batch.cuh src/image_decode.cpp \
  src/image_preprocess.cpp tests/text_sampling_oracle.cpp) > serve-sources.sha256
flags=(-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -DLFM3_TEXT_ONLY=1
       ${EXTRA_FLAGS:-} -I "$N/include" -I "$L/ljt/include")
printf "%s\n" "${flags[*]}" > serve-flags.txt
fail() { { grep -E "error" "$1.build.log" || tail -5 "$1.build.log"; } | sort | uniq -c | head -20; echo "build_serve.sh: $1 FAILED"; exit 1; }
for s in image_decode image_preprocess; do
  g++ -std=c++20 -O2 -I "$N/include" -I "$L/ljt/include" -c "$N/src/$s.cpp" -o "$s.o" > "$s.build.log" 2>&1 || fail "$s"
done
built() { echo "$1: warnings $(grep -ci warning "$1.build.log" || true), sha256 $(sha256sum "$1" | cut -d' ' -f1), $(stat -c %s "$1") bytes"; }
for b in worker server; do
  nvcc "${flags[@]}" -o "liquid-lfm3-$b" "$N/src/lfm3_$b.cu" "$N/src/safetensors.cpp" image_decode.o image_preprocess.o \
    "$L/ljt/lib/libjpeg.a" -lpthread > "liquid-lfm3-$b.build.log" 2>&1 || fail "liquid-lfm3-$b"
  built "liquid-lfm3-$b"
done
g++ -std=c++17 -O2 -I "$N/include" -o text-sampling-oracle "$N/tests/text_sampling_oracle.cpp" > text-sampling-oracle.build.log 2>&1 \
  || fail text-sampling-oracle
built text-sampling-oracle
sha256sum liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle > serve-binaries.sha256
