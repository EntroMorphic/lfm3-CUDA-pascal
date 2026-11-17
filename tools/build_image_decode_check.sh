#!/usr/bin/env bash
# Build the vendored libjpeg-turbo 3.2.0 (static, Release, its default SIMD where the platform has it: NEON on
# aarch64, NASM on x86 if installed) and lfm3-image-decode-check against it. Host only, no network.
# Usage: build_image_decode_check.sh OUT_DIR [extra compiler flags, e.g. -fsanitize=address,undefined]
set -euo pipefail
OUT=$(realpath -m "${1:?OUT_DIR}"); shift
N=$(cd "$(dirname "$0")/../native" && pwd)
mkdir -p "$OUT"
bash "$(dirname "${BASH_SOURCE[0]}")/build_libjpeg_turbo.sh" "$OUT"
g++ -std=c++20 -O2 -g -Wall -Wextra "$@" -I "$N/include" -I "$OUT/ljt/include" \
  "$N/tests/lfm3_image_decode_check.cpp" "$N/src/image_decode.cpp" "$N/src/image_preprocess.cpp" \
  "$OUT/ljt/lib/libjpeg.a" \
  -o "$OUT/lfm3-image-decode-check"
"$OUT/lfm3-image-decode-check" --selftest
