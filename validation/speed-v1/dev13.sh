# speed-v1 development: the value product's block shape and probability staging, timed alone, each twice.
set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev13; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
for v in "0 4" "0 8" "0 16" "1 4" "1 8" "1 16"; do set -- $v
  nvcc $flags -DLFM_GEMM_WIDE_SMEM=$1 -DLFM_GEMM_WIDE_SPLITS=$2 -o $out/b native/bench/decode_attention_bench.cu && echo "smem $1 splits/block $2" && $out/b | grep -E "(8000|31000) values" && $out/b | grep "31000 values"
done
