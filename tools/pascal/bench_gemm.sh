#!/usr/bin/env bash
# Runs in the container: build and run native/bench/pascal_gemm_bench.cu with the plain kernel and with the exact one.
#   tools/pascal/bench_gemm.sh [TAG=bench]
set -euo pipefail
tag=${1:-bench}; N=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../native" && pwd); out=/workspace/build/$tag; mkdir -p "$out"
flags=(-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -I "$N/include")
nvcc "${flags[@]}" -o "$out/gemm-plain" "$N/bench/pascal_gemm_bench.cu"
nvcc "${flags[@]}" -DLFM_GEMM_TENSOR_CORE_MODEL -o "$out/gemm-exact" "$N/bench/pascal_gemm_bench.cu"
nvcc "${flags[@]}" -DLFM_GEMM_TENSOR_CORE_MODEL=2 -o "$out/gemm-exact-ref" "$N/bench/pascal_gemm_bench.cu"
echo "tree $(sha256sum "$N/include/lfm/gemm.cuh" "$N/include/lfm/tensor_core_model.hpp" | sha256sum | cut -c1-16)"
for b in gemm-plain gemm-exact gemm-exact-ref; do "$out/$b" ${REPEATS:-5}; done
