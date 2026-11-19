# speed-v1 development (not a rule run): the greedy choice in two stages, its test; staged probabilities; engine timing.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev12; mkdir -p $out
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
for t in greedy_pick_check decode_attention_check; do
  nvcc $flags -o $out/$t native/tests/$t.cu > $out/$t.log 2>&1; echo "build $t rc=$? warnings=$(grep -c 'warning #' $out/$t.log)"; head -3 $out/$t.log
  $out/$t > $out/$t.txt 2>&1; echo "$t rc=$? $(tail -1 $out/$t.txt)"; grep FAIL $out/$t.txt | head -3
done
nvcc $flags -o $out/b native/bench/decode_attention_bench.cu && $out/b
tools/pascal/build_text.sh ${1:-text-v24}
tools/pascal/run_text_fixtures.sh ${1:-text-v24} 1; echo "fixtures rc=$?"
CASES="decode-1k 1024 300 700
decode-8k 8192 7000 1000
decode-32k 32768 31000 700" tools/pascal/speed.sh ${1:-text-v24} speed-dev12-v23 1
bash validation/speed-v1/profile_decode.sh ${1:-text-v24} > /dev/null 2>&1
python3 tools/pascal/decode_atoms.py $B/profile-decode-${1:-text-v24}/d32k.sqlite 100 | sed -n '/steady state, steps/,/2048-column/p' | head -22 | cut -c1-150
date
