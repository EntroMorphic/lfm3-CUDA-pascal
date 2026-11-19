# The fifth job (tools/pascal/remote.sh detach red-team-1e 'bash validation/red-team-1/job5.sh'): in job4.sh the deliberately
# broken kernel failed the scores of heads 2 and 3 but not P x V, whose BF16 outputs hid the difference. first_row_check now
# also reads P x V in FP32. Run on the final tree, and on the same broken kernel, which must now fail both products.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/rt5-matrix; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM"
nvcc $flags -I $N/include -o $M/first_row_check native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_check.log 2>&1; echo "build first_row_check rc=$? warnings=$(grep -c 'warning #' $M/first_row_check.log)"
$M/first_row_check > $M/first_row_check.txt 2>&1; echo "first_row_check rc=$? $(tail -1 $M/first_row_check.txt)"
mkdir -p $M/mutant && cp -r $N/include $M/mutant/ && sed -i 's/^    first\.keep_nonfinite = true;$/    first.keep_nonfinite = true; first.batch = first.group;/' $M/mutant/include/lfm/gemm.cuh
cmp -s $N/include/lfm/gemm.cuh $M/mutant/include/lfm/gemm.cuh && echo "mutant: NOT APPLIED"
nvcc $flags -I $M/mutant/include -o $M/first_row_mutant native/tests/first_row_check.cu native/tests/first_row_model.cpp > $M/first_row_mutant.log 2>&1; echo "build mutant rc=$?"
$M/first_row_mutant > $M/first_row_mutant.txt 2>&1; echo "mutant first_row_check rc=$? (must be 1) $(tail -1 $M/first_row_mutant.txt)"
grep -c "FAIL: attention scores" $M/first_row_mutant.txt; grep -c "FAIL: attention P x V f32" $M/first_row_mutant.txt
grep attention $M/first_row_check.txt
date
