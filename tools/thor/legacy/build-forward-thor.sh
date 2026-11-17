#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root (these scripts lived there until the 2026-11 reorganization)
docker run --rm --runtime=nvidia --gpus all \
  --mount "type=bind,source=$PWD,target=/src" \
  entromorphic-dev-viz:r38.3.arm64-sbsa-cu130-24.04 \
  bash -lc 'cmake -S /src/native -B /src/build/cmake -DCMAKE_BUILD_TYPE=Release -DLIQUID_LFM3_OPS_TESTS=ON && cmake --build /src/build/cmake --target liquid-lfm3 lfm3-projection-probe -j6'
