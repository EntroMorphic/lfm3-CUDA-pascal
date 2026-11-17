#!/usr/bin/env bash
# Build the batched-decode tools (P0 speed) in the pinned compiler image, with the engine's flags (native GEMM, no cuBLAS).
# Usage (Thor host): build_batch_tools.sh OUT_TAG   -> build/OUT_TAG/{lfm3-batch,lfm3-batch-teacher,lfm3-gpu-select-check,
#                                                    decode-gemm-bench,tile-sweep} and sources.sha256
set -euo pipefail
tag=$1; W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; C=/workspace/liquid-CUDA/build/$tag
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=/workspace/liquid-CUDA/native
  F='-O3 --fmad=false -std=c++20 -arch=sm_110 -I '\$N/include
  (cd /workspace/liquid-CUDA && sha256sum native/src/lfm3_batch.cuh native/src/lfm3_main.cu native/include/lfm/gemm.cuh native/tests/lfm3_batch*.cu native/tests/lfm3_gpu_select_check.cu native/tests/lfm3_decode_*.cu) > sources.sha256
  for t in lfm3_batch:lfm3-batch lfm3_batch_teacher:lfm3-batch-teacher lfm3_gpu_select_check:lfm3-gpu-select-check; do
    nvcc \$F -DLFM3_NATIVE_GEMM -o \${t#*:} \$N/tests/\${t%%:*}.cu \$N/src/safetensors.cpp > build-\${t#*:}.log 2>&1; done
  nvcc \$F -o decode-gemm-bench \$N/bench/lfm3_decode_gemm_bench.cu > build-bench.log 2>&1
  nvcc \$F -o tile-sweep \$N/bench/lfm3_decode_tile_sweep.cu > build-sweep.log 2>&1
  ! grep -iE ' error|warning' build-*.log; chown -R 1000:1000 ."
for b in lfm3-batch lfm3-batch-teacher lfm3-gpu-select-check; do readelf -d $W/liquid-CUDA/build/$tag/$b | grep -q cublas && { echo "$b links cuBLAS"; exit 1; }; done
(cd $W/liquid-CUDA/build/$tag && sha256sum lfm3-batch lfm3-batch-teacher lfm3-gpu-select-check decode-gemm-bench tile-sweep)
