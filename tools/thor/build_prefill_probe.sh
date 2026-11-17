#!/usr/bin/env bash
# Build prefill probe variants inside the pinned compiler image. Usage (in container):
#   build_prefill_probe.sh OUTDIR S:NT [S:NT...]
set -euo pipefail
out=$1; shift
H=/workspace/liquid-CUDA/build/wmma-runtime-candidate-v4/native/tests/lfm3_wmma_runtime_generated.cuh
SRC=/workspace/liquid-CUDA/native/probes/lfm3_prefill_probe.cu
cd "$out"
sha256sum "$H" "$SRC" /workspace/liquid-CUDA/native/probes/lfm3_thin_v5_probe.cu > sources.sha256
for v in "$@"; do
  S=${v%%:*}; NT=${v#*:}; bin="prefill-v9-s${S}-nt${NT}"
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 "-DV4_HEADER=\"$H\"" -DPREFILL_S="$S" -DPREFILL_NT="$NT" \
       -Xptxas -v -o "$bin" "$SRC" -lcublasLt -lcublas > "$bin.build.log" 2>&1 || { echo "$bin BUILD FAILED"; grep -i error "$bin.build.log" | head -5; exit 1; }
  spill=$(grep -A2 "v920mma_prefill_partials" "$bin.build.log" | grep -oE "[0-9]+ bytes spill stores" | head -1 || true)
  echo "$bin built: ${spill:-spill ?}; sha256 $(sha256sum "$bin" | cut -c1-16)"
done
