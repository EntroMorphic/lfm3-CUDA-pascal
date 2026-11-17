#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
output="${LIQUID_BUILD_DIR:-$root/build}"
mkdir -p "$output"
profile_flags=()
if [[ "${LIQUID_PROFILE_NVTX:-0}" == 1 ]]; then
  profile_flags+=(-DLFM_PROFILE_NVTX)
fi

nvcc \
  -O3 \
  -std=c++20 \
  --expt-relaxed-constexpr \
  --default-stream per-thread \
  -arch=sm_110 \
  "${profile_flags[@]}" \
  -I"$root/include" \
  "$root/src/main.cu" \
  "$root/src/safetensors.cpp" \
  "$root/src/tokenizer.cpp" \
  -lcublas \
  -lcublasLt \
  -lpcre2-8 \
  -o "$output/liquid-native"
