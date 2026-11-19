# speed-v1 development (not a rule run): eight value rows in flight in the decode P x V.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev8; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
nvcc $flags -o $out/decode_attention_check native/tests/decode_attention_check.cu > $out/d.log 2>&1; echo "build rc=$?"; head -3 $out/d.log
$out/decode_attention_check > $out/d.txt; echo "decode_attention_check rc=$? $(tail -1 $out/d.txt)"
tools/pascal/build_text.sh ${1:-text-v21}
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700" tools/pascal/speed.sh ${1:-text-v21} speed-dev8-${1:-text-v21} 1
bash validation/speed-v1/profile_decode.sh ${1:-text-v21} > /dev/null 2>&1
python3 tools/pascal/decode_atoms.py $B/profile-decode-${1:-text-v21}/d32k.sqlite 100 | grep -E "steady state, steps|split_kn_partial|thin_k128" | head -6 | cut -c1-150
date
