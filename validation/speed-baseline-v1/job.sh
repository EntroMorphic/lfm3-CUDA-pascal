# speed-baseline-v1 (tools/pascal/remote.sh detach speed-baseline-v1 'bash validation/speed-baseline-v1/job.sh'):
# measurement only. The card's ceilings, then the shipping engine (text-v10, the red-team-1 build) on phase 2's cases.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-baseline-v1-card; mkdir -p $out
nvcc -O3 -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -o $out/card_ceilings native/bench/card_ceilings.cu; echo "build ceilings rc=$?"
$out/card_ceilings 3 | tee $out/ceilings-1.txt
tools/pascal/speed.sh text-v10 speed-baseline-v1 3; echo "speed rc=$?"
$out/card_ceilings 3 | tee $out/ceilings-2.txt
nvcc -O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include -o $out/gemm_bench native/bench/pascal_gemm_bench.cu; echo "build gemm bench rc=$?"
$out/gemm_bench 5 | tee $out/gemm-bench.txt
date
