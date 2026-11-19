# speed-v1 development (not a rule run): the key product with read-ahead, at several grid caps; each variant twice.
set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev11; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/d native/tests/decode_attention_check.cu && $out/d | tail -1
for c in 1024 512 256 128 64; do
  nvcc $flags -DLFM_GEMM_K128_BLOCKS=$c -DLFM_GEMM_WIDE_SPLITS=2 -o $out/b native/bench/decode_attention_bench.cu && echo "cap $c" && $out/b | grep scores && $out/b | grep "31000 scores"
done
