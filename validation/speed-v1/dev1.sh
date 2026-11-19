# speed-v1 development (not a rule run): wide2 at every row count, and the one-row position-0 launch.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev1; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/wide2_check native/tests/wide2_check.cu > $out/w.log 2>&1; echo "build wide2_check rc=$? warnings=$(grep -c 'warning #' $out/w.log)"; head -5 $out/w.log
$out/wide2_check > $out/wide2_check.txt 2>&1; echo "wide2_check rc=$? $(tail -1 $out/wide2_check.txt)"; grep FAIL $out/wide2_check.txt | head
nvcc $flags -o $out/first_row_check native/tests/first_row_check.cu native/tests/first_row_model.cpp > $out/f.log 2>&1; echo "build first_row_check rc=$? warnings=$(grep -c 'warning #' $out/f.log)"; head -5 $out/f.log
$out/first_row_check > $out/first_row_check.txt 2>&1; echo "first_row_check rc=$? $(tail -1 $out/first_row_check.txt)"; grep FAIL $out/first_row_check.txt | head
for tx in 8 32 64 128 256; do
  nvcc $flags -DLFM_GEMM_FIRST_TX=$tx -o $out/frb$tx native/bench/first_row_bench.cu > $out/frb$tx.log 2>&1 || { echo "build TX=$tx failed"; head -3 $out/frb$tx.log; continue; }
  echo "TX=$tx"; $out/frb$tx 20
done
nvcc $flags -o $out/gemm_bench native/bench/pascal_gemm_bench.cu && $out/gemm_bench 5
date
