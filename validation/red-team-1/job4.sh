# The fourth job (tools/pascal/remote.sh detach red-team-1d 'bash validation/red-team-1/job4.sh'), after the second
# review: non-numeric build switches, a sweep that ran nothing, the generation case of the sweep, and attention's
# several-heads products in first_row_check. The engine's source changed only in preprocessor checks.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/rt4-matrix; mkdir -p $M/strip
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I $N/include"
tools/pascal/build_text.sh text-v10; echo "build rc=$?"
tools/pascal/gate_pins.sh parity-v10; echo "gate rc=$?"
for t in "text-v8 text-v10 liquid-lfm3" "text-v8 text-v10 lfm3-score" "parity-v8 parity-v10 liquid-lfm3" "parity-v8 parity-v10 lfm3-score"; do
  set -- $t; strip -o $M/strip/a $B/$1/$3; strip -o $M/strip/b $B/$2/$3
  cmp $M/strip/a $M/strip/b && echo "$2/$3: code IDENTICAL to $1 (symbols stripped)" || echo "$2/$3: DIFFERS from $1"
done
for d in LFM_GEMM_FIRST_ROW=0 LFM_GEMM_FIRST_ROW=1 LFM_GEMM_FIRST_ROW LFM_GEMM_TENSOR_CORE_MODEL LFM_GEMM_TENSOR_CORE_MODEL=2 LFM_GEMM_ROUND_ONCE LFM_GEMM_PLAIN_ROUND=0 LFM_GEMM_PLAIN_ROUND=1 LFM_GEMM_PLAIN_ROUND=2; do
  nvcc $flags -D$d -o $M/ok native/src/lfm3_main.cu native/src/safetensors.cpp > $M/$d.log 2>&1; echo "compile -D$d rc=$? warnings=$(grep -c 'warning #' $M/$d.log)"
done
for d in LFM_GEMM_FIRST_ROW=on LFM_GEMM_FIRST_ROW=yes LFM_GEMM_FIRST_ROW=2 LFM_GEMM_FIRST_ROW= LFM_GEMM_PLAIN_ROUND=away LFM_GEMM_PLAIN_ROUND=3 LFM_GEMM_PLAIN_ROUND= LFM_GEMM_TENSOR_CORE_MODEL=on LFM_GEMM_ROUND_ONCE=on; do
  nvcc $flags -D$d -o $M/bad native/src/lfm3_main.cu native/src/safetensors.cpp > $M/bad.log 2>&1; echo "compile -D$d (must fail) rc=$? $(grep -c '#error' $M/bad.log) #error line(s)"
done
nvcc $flags -o $M/first_row_check native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_check.log 2>&1; echo "build first_row_check rc=$? warnings=$(grep -c 'warning #' $M/first_row_check.log)"
$M/first_row_check > $M/first_row_check.txt 2>&1; echo "first_row_check rc=$? $(tail -1 $M/first_row_check.txt)"
# The check must catch the defect it was extended for: position 0's launch reaching only the first head.
mkdir -p $M/mutant && cp -r $N/include $M/mutant/ && sed -i 's/^    first\.keep_nonfinite = true;$/    first.keep_nonfinite = true; first.batch = first.group;/' $M/mutant/include/lfm/gemm.cuh
cmp -s $N/include/lfm/gemm.cuh $M/mutant/include/lfm/gemm.cuh && echo "mutant: NOT APPLIED"
nvcc ${flags% -I*} -I $M/mutant/include -o $M/first_row_mutant native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_mutant.log 2>&1; echo "build mutant rc=$?"
$M/first_row_mutant > $M/first_row_mutant.txt 2>&1; echo "mutant first_row_check rc=$? (must be 1) $(tail -1 $M/first_row_mutant.txt)"
PROMPTS="" XPROMPTS="" tools/pascal/boundary_sweep.sh text-v10 rt4-sweep-empty; echo "empty sweep rc=$? (must be 2)"
tools/pascal/boundary_sweep.sh text-v10 rt4-sweep; echo "sweep rc=$?"
date
