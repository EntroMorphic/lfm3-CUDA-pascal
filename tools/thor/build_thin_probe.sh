#!/usr/bin/env bash
# Build thin-GEMM probe variants inside the pinned compiler image. Usage (in container):
#   build_thin_probe.sh OUTDIR CANDIDATE_NAMESPACE PREFIX S...
# Fails (nonzero) if any variant fails to compile; prints register/spill use per variant.
set -euo pipefail
out=$1; ns=$2; prefix=$3; shift 3
H=/workspace/liquid-CUDA/build/wmma-runtime-candidate-v4/native/tests/lfm3_wmma_runtime_generated.cuh
SRC=/workspace/liquid-CUDA/native/probes/lfm3_thin_v5_probe.cu
cd "$out"
sha256sum "$H" "$SRC" > sources.sha256
for S in "$@"; do
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 "-DV4_HEADER=\"$H\"" -DUNROLL="$S" "-DCANDIDATE=$ns::mma_thin_partials" \
       -Xptxas -v -o "$prefix-s$S" "$SRC" -lcublasLt -lcublas > "build-s$S.log" 2>&1 || { echo "S=$S BUILD FAILED"; grep -i error "build-s$S.log" | head -5; exit 1; }
  regs=$(grep -A1 "${ns}17mma_thin" "build-s$S.log" | grep -oE "Used [0-9]+ registers" | head -1 || true)
  spill=$(grep -A2 "${ns}17mma_thin" "build-s$S.log" | grep -oE "[0-9]+ bytes spill stores" | head -1 || true)
  echo "S=$S built: ${regs:-registers ?}; ${spill:-spill ?}; sha256 $(sha256sum "$prefix-s$S" | cut -c1-16)"
done
