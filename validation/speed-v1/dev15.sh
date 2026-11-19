# speed-v1 development: the value product's columns per thread and keys ahead, timed alone (each twice); check each.
set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev15; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
for v in "8 4 4" "4 4 4" "4 8 4" "4 4 2" "4 8 2" "8 8 2"; do set -- $v
  d="-DLFM_GEMM_WIDE_COLS=$1 -DLFM_GEMM_WIDE_AHEAD=$2 -DLFM_GEMM_WIDE_SPLITS=$3"
  nvcc $flags $d -o $out/c native/tests/decode_attention_check.cu && echo "cols $1 ahead $2 splits/block $3: $($out/c | tail -1)"
  nvcc $flags $d -o $out/b native/bench/decode_attention_bench.cu && $out/b | grep -E "(8000|31000) values" && $out/b | grep "31000 values"
done
