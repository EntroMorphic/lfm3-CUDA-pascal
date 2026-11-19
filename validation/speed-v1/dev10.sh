# speed-v1 development (not a rule run): decode attention kernel variants, timed alone.
set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev10; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
for v in "1 4" "4 4" "8 4" "4 2" "4 8" "8 1"; do set -- $v
  nvcc $flags -DLFM_GEMM_WIDE_AHEAD=$1 -DLFM_GEMM_WIDE_SPLITS=$2 -o $out/b native/bench/decode_attention_bench.cu && echo "ahead $1 splits/block $2" && $out/b
done
