# speed-v1 development (not a rule run): four query heads per read of the keys and values; the red-team's tests.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev7; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
for t in decode_attention_check wide2_check; do
  nvcc $flags -o $out/$t native/tests/$t.cu > $out/$t.log 2>&1; echo "build $t rc=$? warnings=$(grep -c 'warning #' $out/$t.log)"; head -5 $out/$t.log
  $out/$t > $out/$t.txt 2>&1; echo "$t rc=$? $(tail -1 $out/$t.txt)"; grep FAIL $out/$t.txt | head -5
done
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 $out/wide2_check > $out/wide2_check.memcheck.txt 2>&1; echo "wide2_check under memcheck rc=$? $(grep 'ERROR SUMMARY' $out/wide2_check.memcheck.txt)"
nvcc $flags -DLFM_GEMM_FIRST_TX=0 -o $out/bad native/src/lfm3_main.cu native/src/safetensors.cpp > $out/bad.log 2>&1; echo "FIRST_TX=0 (must fail) rc=$?"
nvcc -O3 -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -o $out/read_rate native/bench/read_rate.cu && $out/read_rate > $out/read_rate.txt; echo "read_rate rc=$?"; grep "31.46" $out/read_rate.txt | head -3
tools/pascal/build_text.sh text-v20
CASES="decode-1k 1024 300 700
decode-8k 8192 7000 1000
decode-32k 32768 31000 700" tools/pascal/speed.sh text-v20 speed-dev7-v20 1
date
