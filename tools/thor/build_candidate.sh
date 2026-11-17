#!/usr/bin/env bash
# Build a generated runtime candidate mirror inside the pinned compiler image (run as root there).
# Usage (in container): build_candidate.sh /workspace/liquid-CUDA/build/NAME
set -euo pipefail
cd "$1"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 || { tail -20 cmake.log; exit 1; }
cmake --build build --target liquid-lfm3 liquid-lfm3-worker lfm3-decode-batch-probe -j4 > build.log 2>&1 || { grep -iE " error|error:" build.log | head -10; exit 1; }
sha256sum build/liquid-lfm3 build/liquid-lfm3-worker build/lfm3-decode-batch-probe
