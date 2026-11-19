# speed-baseline-v1, second job: the read ceiling with enough loads in flight; the 32,768-token prefill (the first job
# asked for a token beyond the capacity); decode-1k with older builds, since text-v10 measured 15.7 ms per step where
# prefill-speed-v1 recorded 14.7 for the same request. Then, not part of the baseline: wide2_check and the GEMM bench
# with the phase 2 kernel (checksums must equal the first job's).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-baseline-v1-card
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc -O3 -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -o $out/card_ceilings2 native/bench/card_ceilings.cu; echo "build ceilings rc=$?"
$out/card_ceilings2 3 | tee $out/ceilings-3.txt
CASES="prefill-32768 32768 32768 0" tools/pascal/speed.sh text-v10 speed-baseline-v1-p32768 3; echo "speed rc=$?"
for t in text-v1 text-v4 text-v6 text-v10; do CASES="decode-1k 1024 300 700" tools/pascal/speed.sh $t speed-baseline-v1-$t 2; echo "speed $t rc=$?"; done
nvcc $flags -o $out/wide2_check native/tests/wide2_check.cu > $out/wide2_check.log 2>&1; echo "build wide2_check rc=$? warnings=$(grep -c 'warning #' $out/wide2_check.log)"
$out/wide2_check > $out/wide2_check.txt 2>&1; echo "wide2_check rc=$? $(tail -1 $out/wide2_check.txt)"; grep FAIL $out/wide2_check.txt | head
nvcc $flags -o $out/gemm_bench2 native/bench/pascal_gemm_bench.cu; echo "build gemm bench rc=$?"
$out/gemm_bench2 5 | tee $out/gemm-bench-wide2.txt
date
