# The third job (tools/pascal/remote.sh detach red-team-1c 'bash validation/red-team-1/job3.sh'): job2.sh's tree had
# first_row_check's old shapes, so the check aborted on the GEMM's alignment rule before testing anything. Only that
# test changed afterwards. This builds the final tree again, shows the engine is the one job2.sh verified, and runs the check.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/rt3-matrix; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I $N/include"
tools/pascal/build_text.sh text-v9; echo "build rc=$?"
for f in liquid-lfm3 lfm3-score; do
  cmp $B/text-v8/$f $B/text-v9/$f && echo "$f: IDENTICAL to text-v8" || echo "$f: DIFFERS from text-v8"
done
nvcc $flags -o $M/first_row_check native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_check.log 2>&1; echo "build first_row_check rc=$? warnings=$(grep -c 'warning #' $M/first_row_check.log)"
$M/first_row_check > $M/first_row_check.txt 2>&1; echo "first_row_check rc=$? $(tail -1 $M/first_row_check.txt)"
date
