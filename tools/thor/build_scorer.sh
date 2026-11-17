#!/usr/bin/env bash
# Build the native perplexity scorer in the pinned compiler image (same flags as liquid-lfm3).
# Usage (in container): build_scorer.sh OUTDIR [native-gemm|cublas]   (default native-gemm, like the engine; G4)
#   native-gemm: the engine without cuBLAS (LFM3_NATIVE_GEMM, lfm::gemm; G4)
set -euo pipefail
out=$1; mode=${2:-native-gemm}; N=/workspace/liquid-CUDA/native
case $mode in
  cublas) flags=(); libs=(-lcublas) ;;
  native-gemm) flags=(-DLFM3_NATIVE_GEMM); libs=() ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
cd "$out"
sha256sum $N/apps/lfm3_score.cu $N/src/lfm3_main.cu $N/src/safetensors.cpp $N/include/lfm/gemm.cuh > sources.sha256
echo "$mode" > mode.txt
nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 "${flags[@]}" -I $N/include -o lfm3-score $N/apps/lfm3_score.cu $N/src/safetensors.cpp "${libs[@]}" > build.log 2>&1 || { grep -iE "error" build.log | head -10; exit 1; }
echo "lfm3-score built sha256 $(sha256sum lfm3-score | cut -c1-16)"
