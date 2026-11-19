# speed-v1, after the second red-team review (tools/pascal/remote.sh detach speed-v1f 'bash validation/speed-v1/job5.sh'):
# the dead split_kn_partial<1, 4> launch removed, two tuning switches checked, decode_attention_check with 3 to 8 rows,
# sass_compare refusing empty input. The engine must be job3.sh's (text-v30) but for the removed function.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/sv5-matrix
[ -e $M ] && { echo "$M exists"; exit 2; }; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -I $N/include"
tools/pascal/build_text.sh text-v32; echo "build rc=$?"
for f in liquid-lfm3 lfm3-score; do for v in 30 32; do cuobjdump -sass $B/text-v$v/$f > $M/$f.$v.sass; done
  python3 validation/speed-v1/sass_compare.py $M/$f.30.sass $M/$f.32.sass --only-in-a 'split_kn_partialILi1ELi4EE' > $M/$f.sass-compare.txt
  echo "$f device code rc=$? $(head -1 $M/$f.sass-compare.txt)"; grep removed: $M/$f.sass-compare.txt; done
sha256sum $M/*.sass > $M/sass.sha256; gzip -9 $M/*.sass
tools/pascal/gate_pins.sh parity-v32; echo "gate rc=$?"
nvcc $flags -o $M/cublas native/src/lfm3_main.cu native/src/safetensors.cpp -lcublas > $M/cublas.log 2>&1; echo "cuBLAS build rc=$? errors=$(grep -c ' error' $M/cublas.log) warnings=$(grep -c 'warning #' $M/cublas.log)"
# Switches: the defaults and in-range values build; 0 must stop the build.
for d in "" "-DLFM_GEMM_WIDE_SPLITS=8" "-DLFM_GEMM_K128_BLOCKS=1" "-DLFM_GEMM_WIDE_SPLITS=0" "-DLFM_GEMM_K128_BLOCKS=0" "-DLFM_GEMM_WIDE_SPLITS=65"; do
  nvcc $flags -DLFM3_NATIVE_GEMM $d -o $M/switch native/tests/decode_attention_check.cu > $M/switch.log 2>&1; echo "switch '$d' build rc=$? $(grep -o 'static assertion failed[^\"]*\"[^\"]*' $M/switch.log | head -1)"; done
rm -f $M/switch
for t in first_row_check wide2_check decode_attention_check greedy_pick_check; do
  extra=; [ $t = first_row_check ] && extra=native/tests/first_row_model.cpp
  nvcc $flags -DLFM3_NATIVE_GEMM -o $M/$t native/tests/$t.cu $extra > $M/$t.log 2>&1; echo "build $t rc=$? warnings=$(grep -c 'warning #' $M/$t.log)"
  $M/$t > $M/$t.txt 2>&1; echo "$t rc=$? $(tail -1 $M/$t.txt)"
done
grep -c '^ok   rows [3-8]' $M/decode_attention_check.txt | sed 's/^/decode_attention_check, 3 to 8 rows, lines ok: /'
for t in wide2_check decode_attention_check; do
  /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 $M/$t > $M/$t.memcheck.txt 2>&1; echo "$t under memcheck rc=$? $(grep 'ERROR SUMMARY' $M/$t.memcheck.txt)"; done
tools/pascal/run_text_fixtures.sh text-v32 2; echo "fixtures rc=$?"
tools/pascal/speed.sh text-v32 sv5-speed 1; echo "speed rc=$?"
python3 - <<'PY'
import json, os
a, b = "/workspace/build/sv3-speed", "/workspace/build/sv5-speed"
cases = ["prefill-32", "prefill-257", "prefill-1024", "prefill-4096", "prefill-32768", "decode-1k", "decode-8k", "decode-32k"]
ok = True
for c in cases:
    f = f"{c}.1.json"
    if not os.path.exists(f"{b}/{f}"): ok = False; print(f"{c}: MISSING"); continue
    x, y = json.load(open(f"{a}/{f}")), json.load(open(f"{b}/{f}"))
    same = x["teacher_argmax"] == y["teacher_argmax"] and x["generated_ids"] == y["generated_ids"]
    ok &= same; print(f"{c}: ids {'same as text-v30' if same else 'DIFFER from text-v30'} ({len(y['teacher_argmax'])} argmaxes)")
print("TEXT-V32 IDS (all 8 cases)", "PASS" if ok else "FAIL")
PY
# The read rate again, with the card's clocks and state sampled throughout.
nvcc -O3 -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -o $M/read_rate native/bench/read_rate.cu; echo "build read_rate rc=$?"
( while :; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu --format=csv,noheader; sleep 0.2; done ) > $M/read_rate.clocks.txt 2>/dev/null &
s=$!; $M/read_rate > $M/read_rate.txt; echo "read_rate rc=$?"; kill $s; wait $s 2>/dev/null
grep 31.46 $M/read_rate.txt | awk '{print $NF" "$(NF-1)}' | sort -n | sed -n '1p;$p' | tr '\n' ' ' | sed 's/^/read_rate 31.46 MB, lowest and highest: /'; echo
awk -F', ' '{print $3" "$4}' $M/read_rate.clocks.txt | sort | uniq -c | sed 's/^/clocks during read_rate: /'
date
