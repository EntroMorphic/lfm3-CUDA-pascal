# The job as run in the container for red-team-1 (tools/pascal/remote.sh detach red-team-1 'bash validation/red-team-1/job.sh').
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I $N/include"
# A. Before the fixes (binary text-v6, phase 1's): the two findings reproduced.
XPROMPTS="9 129 136 137 257 300 513 1025 2049 3329" PROMPTS="" tools/pascal/boundary_sweep.sh text-v6 rt1-before-xcap; echo "before xcap rc=$?"
tools/pascal/tail_tile_check.sh text-v6 rt1-before-tail; echo "before tail rc=$?"
# B. Builds: shipping, the parity gate's, and every switch (compile only); a bad switch value must not compile.
tools/pascal/build_text.sh text-v7; echo "build rc=$?"
mkdir -p $B/rt1-matrix
for d in LFM_GEMM_FIRST_ROW=0 LFM_GEMM_TENSOR_CORE_MODEL=2 LFM_GEMM_ROUND_ONCE LFM_GEMM_PLAIN_ROUND=1 LFM_GEMM_PLAIN_ROUND=2; do
  nvcc $flags -D$d -o $B/rt1-matrix/ok native/src/lfm3_main.cu native/src/safetensors.cpp > $B/rt1-matrix/$d.log 2>&1; echo "compile -D$d rc=$?"
done
for d in LFM_GEMM_TENSOR_CORE_MODEL=3 LFM_GEMM_TENSOR_CORE_MODEL=0 LFM_GEMM_FIRST_ROW=2 LFM_GEMM_ROUND_ONCE=0 LFM_GEMM_PLAIN_ROUND=3; do
  nvcc $flags -D$d -o $B/rt1-matrix/bad native/src/lfm3_main.cu native/src/safetensors.cpp > $B/rt1-matrix/$d.log 2>&1; echo "compile -D$d (must fail) rc=$? $(grep -c '#error' $B/rt1-matrix/$d.log) #error line(s)"
done
nvcc $flags -o $B/rt1-matrix/first_row_check native/tests/first_row_check.cu > $B/rt1-matrix/first_row_check.log 2>&1; echo "build first_row_check rc=$?"
# C. The rules.
$B/rt1-matrix/first_row_check; echo "first_row_check rc=$?"
tools/pascal/gate_pins.sh parity-v7; echo "gate rc=$?"
tools/pascal/run_text_fixtures.sh text-v7 2; echo "fixtures rc=$?"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v7 2; echo "fixtures compact rc=$?"
tools/pascal/run_text_fixtures.sh text-v7 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck fixtures rc=$?"
STEPS=24 PROMPTS="257 513" XPROMPTS="" tools/pascal/boundary_sweep.sh text-v7 rt1-memcheck-sweep /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck sweep rc=$?"
python3 tools/pascal/round_once_check.py $B/rt1-round-once > $B/rt1-matrix/round_once_check.txt 2>&1; echo "round_once_check rc=$? $(tail -1 $B/rt1-matrix/round_once_check.txt)"
tools/pascal/boundary_sweep.sh text-v7 rt1-sweep; echo "sweep rc=$?"
tools/pascal/tail_tile_check.sh text-v7 rt1-tail; echo "tail rc=$?"
tools/pascal/long_decode.sh text-v7 rt1-long-decode; echo "long decode rc=$?"
python3 - <<'PY'
import json
ok = True
for case in ("control", "short", "middle", "end", "free"):
    new = json.load(open(f"/workspace/build/rt1-long-decode/{case}.default.json")); old = json.load(open(f"/workspace/build/long-decode-v1/{case}.default.json"))
    same = new["logits_sha256"] == old["logits_sha256"] and new["teacher_argmax"] == old["teacher_argmax"] and new["generated_ids"] == old["generated_ids"]
    ok &= same; print(f"long decode {case}: {'IDENTICAL to long-decode-v1' if same else 'DIFFERS from long-decode-v1'}")
print("LONG DECODE UNCHANGED", "PASS" if ok else "FAIL")
PY
tools/pascal/score.sh text-v7 ppl-v7 256 257 4096 32768; echo "score rc=$?"
for L in 256 257 4096 32768; do cmp $B/ppl-v6/nll-pascal-$L.f32 $B/ppl-v7/nll-pascal-$L.f32 && echo "nll $L: IDENTICAL to text-correct-v2" || echo "nll $L: DIFFERS"; done
date
