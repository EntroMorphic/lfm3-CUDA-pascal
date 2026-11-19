set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev19; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/n native/tests/norm_warp_check.cu native/src/safetensors.cpp > $out/n.log 2>&1; echo "build rc=$? warnings=$(grep -c 'warning #' $out/n.log)"; grep -m5 -E "error|warning" $out/n.log
$out/n | tail -8
bash validation/speed-v1/dev16.sh ${1:-text-v27} 2>&1 | grep -v "^+" | grep -v links | grep -E "FIXT|ms per step|steady|norm|probabilities|split_kn_partial_wide|thin_k128_kernel  .*256x4"
