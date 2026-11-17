#!/usr/bin/env bash
# Integrated native path with fail-closed library tracking controls.
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root (these scripts lived there until the 2026-11 reorganization)
PROJECT="$(dirname "$PWD")"
GATES="${GATES:?Set GATES to a completed correctness evidence directory}"
OUT="${OUT:?Set OUT to a fresh evidence directory relative to this repo}"
SANITIZER="${SANITIZER:-$PWD/build/sanitizer-13.3.27/cuda_sanitizer_api-linux-sbsa-13.3.27-archive/compute-sanitizer}"
test -f "$GATES/summary.json"
test -x "$SANITIZER/compute-sanitizer"
docker run --rm --runtime=nvidia --gpus all \
  --mount "type=bind,source=$PROJECT,target=/workspace" \
  --mount "type=bind,source=$SANITIZER,target=/opt/sanitizer,readonly" \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib \
  lfm3-omni-minimal:dev /opt/venv/bin/python \
  /workspace/liquid-CUDA/tools/check_lfm3_sanitizer.py \
  --sanitizer /opt/sanitizer/compute-sanitizer \
  --probe /workspace/liquid-CUDA/build/cmake/lfm3-projection-probe \
  --binary /workspace/liquid-CUDA/build/cmake/liquid-lfm3 \
  --checkpoint /workspace/models/LFM3-Omni-2B \
  --request "/workspace/liquid-CUDA/$GATES/window257/generation-request.json" \
  --output "/workspace/liquid-CUDA/$OUT" --exclude-nvjet-synccheck
