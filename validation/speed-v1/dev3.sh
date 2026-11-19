# speed-v1 development (not a rule run): paired heads only for long caches; the read ceiling in decode's shape.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev3; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/decode_attention_check native/tests/decode_attention_check.cu > $out/d.log 2>&1; echo "build rc=$?"; head -5 $out/d.log
$out/decode_attention_check | tail -30
nvcc -O3 -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -o $out/read_rate native/bench/read_rate.cu && $out/read_rate
tools/pascal/build_text.sh text-v14
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700" tools/pascal/speed.sh text-v14 speed-dev3-v14 1
date
