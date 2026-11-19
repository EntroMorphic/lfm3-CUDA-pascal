# The second job, on the final tree (tools/pascal/remote.sh detach red-team-1b 'bash validation/red-team-1/job2.sh'):
# every rule again with builds text-v8 and parity-v8. job.sh ran first, on the tree of commit 862daec.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/rt2-matrix; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I $N/include"
tools/pascal/build_text.sh text-v8; echo "build rc=$?"
for d in LFM_GEMM_FIRST_ROW=0 LFM_GEMM_TENSOR_CORE_MODEL=2 LFM_GEMM_ROUND_ONCE LFM_GEMM_PLAIN_ROUND=1 LFM_GEMM_PLAIN_ROUND=2; do
  nvcc $flags -D$d -o $M/ok native/src/lfm3_main.cu native/src/safetensors.cpp > $M/$d.log 2>&1; echo "compile -D$d rc=$? warnings=$(grep -c 'warning #' $M/$d.log)"
done
for d in LFM_GEMM_TENSOR_CORE_MODEL=3 LFM_GEMM_TENSOR_CORE_MODEL=0 LFM_GEMM_FIRST_ROW=2 LFM_GEMM_ROUND_ONCE=0 LFM_GEMM_PLAIN_ROUND=3; do
  nvcc $flags -D$d -o $M/bad native/src/lfm3_main.cu native/src/safetensors.cpp > $M/$d.log 2>&1; echo "compile -D$d (must fail) rc=$? $(grep -c '#error' $M/$d.log) #error line(s)"
done
nvcc $flags -o $M/probe native/probes/lfm3_decode_batch_probe.cu native/src/safetensors.cpp > $M/probe.log 2>&1; echo "compile decode-batch probe (the guard's caller) rc=$? warnings=$(grep -c 'warning #' $M/probe.log)"
# The batch, tree and worker programs need Thor-only kernels (lfm/gemm_lossless.cuh) and do not build for sm_61:
# what is checked is that their errors are all in that file, none in the files the guards were added to.
for t in "batch native/apps/lfm3_batch.cu" "tree native/apps/lfm3_tree.cu" "worker native/src/lfm3_worker.cu"; do
  set -- $t; nvcc $flags -o $M/$1 $2 native/src/safetensors.cpp > $M/$1.log 2>&1
  echo "compile $1 rc=$? errors: $(grep -c ': error' $M/$1.log), outside gemm_lossless.cuh: $(grep ': error' $M/$1.log | grep -vc gemm_lossless.cuh)"
done
nvcc $flags -o $M/first_row_check native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_check.log 2>&1; echo "build first_row_check rc=$?"
$M/first_row_check; echo "first_row_check rc=$?"
tools/pascal/gate_pins.sh parity-v8; echo "gate rc=$?"
tools/pascal/run_text_fixtures.sh text-v8 2; echo "fixtures rc=$?"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v8 2; echo "fixtures compact rc=$?"
tools/pascal/run_text_fixtures.sh text-v8 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck fixtures rc=$?"
STEPS=24 PROMPTS="257 513" XPROMPTS="" tools/pascal/boundary_sweep.sh text-v8 rt2-memcheck-sweep /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck sweep rc=$?"
python3 tools/pascal/round_once_check.py $B/rt2-round-once > $M/round_once_check.txt 2>&1; echo "round_once_check rc=$? $(tail -1 $M/round_once_check.txt)"
tools/pascal/boundary_sweep.sh text-v8 rt2-sweep; echo "sweep rc=$?"
tools/pascal/tail_tile_check.sh text-v8 rt2-tail; echo "tail rc=$?"
tools/pascal/long_decode.sh text-v8 rt2-long-decode; echo "long decode rc=$?"
python3 - <<'PY'
import json
ok = True
for case in ("control", "short", "middle", "end", "free"):
    new = json.load(open(f"/workspace/build/rt2-long-decode/{case}.default.json")); old = json.load(open(f"/workspace/build/long-decode-v1/{case}.default.json"))
    same = new["logits_sha256"] == old["logits_sha256"] and new["teacher_argmax"] == old["teacher_argmax"] and new["generated_ids"] == old["generated_ids"]
    ok &= same; print(f"long decode {case}: {'IDENTICAL to long-decode-v1' if same else 'DIFFERS from long-decode-v1'}")
print("LONG DECODE UNCHANGED", "PASS" if ok else "FAIL")
PY
tools/pascal/score.sh text-v8 ppl-v8 256 257 4096 32768; echo "score rc=$?"
for L in 256 257 4096 32768; do cmp $B/ppl-v6/nll-pascal-$L.f32 $B/ppl-v8/nll-pascal-$L.f32 && echo "nll $L: IDENTICAL to text-correct-v2" || echo "nll $L: DIFFERS"; done
date
