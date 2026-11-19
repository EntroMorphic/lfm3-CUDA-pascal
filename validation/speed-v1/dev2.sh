# speed-v1 development (not a rule run): paired query heads in decode attention; then a decode profile.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev2; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/decode_attention_check native/tests/decode_attention_check.cu > $out/d.log 2>&1; echo "build rc=$? warnings=$(grep -c 'warning #' $out/d.log)"; head -5 $out/d.log
$out/decode_attention_check | tail -30; echo "decode_attention_check rc=${PIPESTATUS[0]}"
bash validation/speed-v1/profile_decode.sh text-v13 2>&1 | grep -v "^+" | grep -E "steady state, steps|thin_k128|split_kn|lm_head|mlp w1" | head -30
date
