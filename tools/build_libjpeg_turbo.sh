#!/usr/bin/env bash
# Build the vendored libjpeg-turbo 3.2.0 (native/third_party/libjpeg-turbo; static, Release, its default SIMD where
# the platform has it: NEON on aarch64, NASM on x86 if installed) into OUT_DIR/ljt/{include,lib/libjpeg.a}, once:
# an existing OUT_DIR/ljt/lib/libjpeg.a is kept. Host only, no network. Used by tools/build_image_decode_check.sh,
# tools/thor/build_speech.sh's prerequisite, and the host tests that link the image front end.
# Usage: build_libjpeg_turbo.sh OUT_DIR
set -euo pipefail
OUT=$(realpath -m "${1:?OUT_DIR}")
N=$(cd "$(dirname "${BASH_SOURCE[0]}")/../native" && pwd)
[ -f "$OUT/ljt/lib/libjpeg.a" ] && exit 0
mkdir -p "$OUT"
rm -rf "$OUT/ljt-src" "$OUT/ljt-build"
mkdir -p "$OUT/ljt-src"
echo "6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e  $N/third_party/libjpeg-turbo/libjpeg-turbo-3.2.0.tar.gz" | sha256sum -c --quiet
tar xzf "$N/third_party/libjpeg-turbo/libjpeg-turbo-3.2.0.tar.gz" -C "$OUT/ljt-src" --strip-components=1
cmake -S "$OUT/ljt-src" -B "$OUT/ljt-build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OUT/ljt" \
  -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$OUT/ljt-cmake.log" 2>&1
cmake --build "$OUT/ljt-build" -j"$(nproc)" > "$OUT/ljt-build.log" 2>&1
cmake --install "$OUT/ljt-build" > /dev/null
grep -i "simd" "$OUT/ljt-cmake.log" | head -3 || true
