#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root (these scripts lived there until the 2026-11 reorganization)
mkdir -p build
docker run --rm --runtime=nvidia --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD,target=/src" \
  entromorphic-dev-viz:r38.3.arm64-sbsa-cu130-24.04 \
  nvcc -std=c++20 -O2 --fmad=false -arch=sm_110 --shared --cudart=static \
  -Xcompiler=-fPIC -I/src/native/include /src/native/tests/lfm3_ops_bridge.cu \
  -o /src/build/liblfm3_ops.so
