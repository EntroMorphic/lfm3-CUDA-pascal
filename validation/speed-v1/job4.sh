# speed-v1, after the final review (tools/pascal/remote.sh detach speed-v1d 'bash validation/speed-v1/job4.sh'): the
# greedy header included in both builds (the cuBLAS build had stopped compiling) and decode_attention_check closer to
# the engine (its grid cap, a cleared workspace, lfm::gemm::run). The native engine must be the one job3.sh verified.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/sv4-matrix; mkdir -p $M/strip
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -I $N/include"
tools/pascal/build_text.sh text-v31; echo "build rc=$?"
for f in liquid-lfm3 lfm3-score; do strip -o $M/strip/a $B/text-v30/$f; strip -o $M/strip/b $B/text-v31/$f
  cmp -s $M/strip/a $M/strip/b && echo "$f: code IDENTICAL to text-v30 (symbols stripped)" || echo "$f: DIFFERS from text-v30"; done
nvcc $flags -o $M/cublas native/src/lfm3_main.cu native/src/safetensors.cpp -lcublas > $M/cublas.log 2>&1; echo "cuBLAS build rc=$? errors=$(grep -c ' error' $M/cublas.log) warnings=$(grep -c 'warning #' $M/cublas.log)"
nvcc $flags -DLFM3_NATIVE_GEMM -o $M/decode_attention_check native/tests/decode_attention_check.cu > $M/d.log 2>&1; echo "build rc=$? warnings=$(grep -c 'warning #' $M/d.log)"
$M/decode_attention_check > $M/decode_attention_check.txt 2>&1; echo "decode_attention_check rc=$? $(tail -1 $M/decode_attention_check.txt)"; grep FAIL $M/decode_attention_check.txt | head
date
# Then (run after the above): the device code function by function, the fixtures, and every speed case's ids against text-v30.
for f in liquid-lfm3 lfm3-score; do for v in 30 31; do cuobjdump -sass $B/text-v$v/$f > $M/$f.$v.sass; done; echo "$f device code: $(python3 validation/speed-v1/sass_compare.py $M/$f.30.sass $M/$f.31.sass)"; done
tools/pascal/run_text_fixtures.sh text-v31 2; echo "fixtures rc=$?"
tools/pascal/speed.sh text-v31 sv4-speed 1; echo "speed rc=$?"
python3 - <<'PY'
import json, os
a, b = "/workspace/build/sv3-speed", "/workspace/build/sv4-speed"
ok = True
for f in sorted(os.listdir(b)):
    if not f.endswith(".1.json"): continue
    x, y = json.load(open(f"{a}/{f}")), json.load(open(f"{b}/{f}"))
    same = x["teacher_argmax"] == y["teacher_argmax"] and x["generated_ids"] == y["generated_ids"]
    ok &= same; print(f"{f}: ids {'same as text-v30' if same else 'DIFFER from text-v30'}")
print("TEXT-V31 IDS", "PASS" if ok else "FAIL")
PY
