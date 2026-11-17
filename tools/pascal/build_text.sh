#!/usr/bin/env bash
# Runs in the container (tools/pascal/remote.sh run 'tools/pascal/build_text.sh TAG'): builds the text CLI and the
# scorer for sm_61 into /workspace/build/TAG, with the hashes of the sources and binaries and the linked libraries.
set -euo pipefail
tag=${1:?usage: build_text.sh TAG}
N=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../native" && pwd)
out=/workspace/build/$tag
mkdir -p "$out" && cd "$out"
# EXTRA_FLAGS: e.g. -DLFM_GEMM_TENSOR_CORE_MODEL for the validation build with Blackwell's tensor-core arithmetic.
flags=(-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM ${EXTRA_FLAGS:-} -I "$N/include")
echo "flags: ${flags[*]}" > /dev/null
(cd "$N" && sha256sum src/lfm3_main.cu apps/lfm3_score.cu src/safetensors.cpp include/lfm/*.hpp include/lfm/*.cuh) > sources.sha256
printf "%s\n" "${flags[*]}" > flags.txt; nvcc --version | tail -2 > toolchain.txt; gcc --version | head -1 >> toolchain.txt
build() {  # NAME SOURCE
  if ! nvcc "${flags[@]}" -o "$1" "$N/$2" "$N/src/safetensors.cpp" > "$1.build.log" 2>&1; then
    grep -E "error" "$1.build.log" | sort | uniq -c | head -20; echo "build_text.sh: $1 FAILED"; exit 1
  fi
  echo "$1: warnings $(grep -ci warning "$1.build.log" || true), sha256 $(sha256sum "$1" | cut -d' ' -f1), $(stat -c %s "$1") bytes"
  ldd "$1" | awk '{print $1}' | sort > "$1.ldd.txt"; echo "  links: $(tr '\n' ' ' < "$1.ldd.txt")"
}
build liquid-lfm3 src/lfm3_main.cu
build lfm3-score apps/lfm3_score.cu
sha256sum liquid-lfm3 lfm3-score > binaries.sha256
