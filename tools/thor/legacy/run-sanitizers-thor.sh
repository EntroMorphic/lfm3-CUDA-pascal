#!/usr/bin/env bash
# Run from the Thor project copy. Does not alter the running sandbox/image.
set -euo pipefail
cd "$(dirname "$0")/../../.."  # the repository root (these scripts lived there until the 2026-11 reorganization)
ROOT="$(dirname "$PWD")"
SANITIZER="${SANITIZER:-$PWD/build/cuda-13.0.88-sanitizer}"
PREFIX="${PREFIX:-remediated-v2025_3_1-cpu-oracle}"
mkdir -p build
for tool in ${TOOLS:-memcheck racecheck synccheck initcheck}; do
  for suite in ${SUITES:-boundary convolution ops}; do
    extra=()
    if [[ "$suite" != boundary ]]; then
      extra+=(--checkpoint /workspace/models/LFM3-Omni-2B)
    fi
    if [[ "$suite" == ops ]]; then
      extra+=(--attention-reference cpu)
    fi
    docker run --rm --runtime=nvidia --gpus all \
      -e TRITON_PTXAS_PATH=/workspace/tools/ptxas \
      -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \
      --mount "type=bind,source=$ROOT,target=/workspace" \
      --mount "type=bind,source=$SANITIZER,target=/opt/sanitizer,readonly" \
      lfm3-omni-minimal:dev /opt/sanitizer/compute-sanitizer \
      --tool "$tool" --error-exitcode 86 \
      --log-file "/workspace/liquid-CUDA/build/$PREFIX-$tool-$suite.log" \
      /opt/venv/bin/python "/workspace/liquid-CUDA/tools/check_lfm3_$suite.py" \
      --library /workspace/liquid-CUDA/build/liblfm3_ops.so "${extra[@]}" \
      --output "/workspace/liquid-CUDA/build/$PREFIX-$tool-$suite.json" \
      > "build/$PREFIX-$tool-$suite-python.log" 2>&1
    echo "$tool/$suite: PASS"
    tail -n 2 "build/$PREFIX-$tool-$suite.log"
  done
done
