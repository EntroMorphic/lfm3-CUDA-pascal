set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev18; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/c native/tests/decode_attention_check.cu && $out/c | tail -1
nvcc $flags -o $out/b native/bench/decode_attention_bench.cu && $out/b | grep values && $out/b | grep "31000 values"
