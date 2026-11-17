#!/usr/bin/env bash
# Build and run the GEMV measurement tools used by native/AUDIT.md.
#
#   gemv_bench       cold-cache bandwidth of cuBLAS, a streaming warp-per-row
#                    GEMV, and a pure streaming read, for every projection
#                    shape the decode path uses
#   gemv_order       which accumulation order reproduces cuBLAS bit-exactly on
#                    the model's own weights
#   gemv_concurrent  whether two independent projections overlap usefully
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

model="${MODEL:-models/LFM2.5-350M/model.safetensors}"
out="${GEMV_OUTPUT:-native/build}"
mkdir -p "$out"

common=(-O3 -std=c++20 --expt-relaxed-constexpr -arch=sm_110 -I native/include)

echo "== building =="
nvcc "${common[@]}" native/bench/gemv_bench.cu -lcublas -o "$out/gemv_bench"
nvcc "${common[@]}" native/probes/gemv_order.cu native/src/safetensors.cpp \
  -lcublas -o "$out/gemv_order"
nvcc "${common[@]}" native/bench/gemv_concurrent.cu native/src/safetensors.cpp \
  -lcublas -o "$out/gemv_concurrent"

echo
echo "== cold-cache projection bandwidth =="
"$out/gemv_bench"

echo
echo "== accumulation order vs cuBLAS on the real weights =="
"$out/gemv_order" "$model"

echo
echo "== two independent projections, sequential vs concurrent =="
"$out/gemv_concurrent" "$model"
