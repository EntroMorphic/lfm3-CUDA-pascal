# speed-v1's rule run (tools/pascal/remote.sh detach speed-v1 'bash validation/speed-v1/job.sh'): builds text-v17 and
# parity-v17 from this tree; rules 1 to 6 (outputs unchanged), then rules 6 to 14 (speed) against text-v10 measured in
# the same job.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; N=native; M=$B/sv1-matrix; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I $N/include"
tools/pascal/build_text.sh text-v17; echo "build rc=$?"
tools/pascal/gate_pins.sh parity-v17; echo "rule 1 gate rc=$?"
tools/pascal/run_text_fixtures.sh text-v17 2; echo "rule 2 fixtures rc=$?"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v17 2; echo "rule 2 fixtures compact rc=$?"
tools/pascal/run_text_fixtures.sh text-v17 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "rule 2 memcheck fixtures rc=$?"
STEPS=24 PROMPTS="257 513" XPROMPTS="" tools/pascal/boundary_sweep.sh text-v17 sv1-memcheck-sweep /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "rule 2 memcheck sweep rc=$?"
for t in first_row_check wide2_check decode_attention_check; do
  extra=; [ $t = first_row_check ] && extra=native/tests/first_row_model.cpp
  nvcc $flags -o $M/$t native/tests/$t.cu $extra > $M/$t.log 2>&1; echo "build $t rc=$? warnings=$(grep -c 'warning #' $M/$t.log)"
  $M/$t > $M/$t.txt 2>&1; echo "rule 5 $t rc=$? $(tail -1 $M/$t.txt)"
done
python3 tools/pascal/round_once_check.py $B/sv1-round-once > $M/round_once_check.txt 2>&1; echo "rule 5 round_once_check rc=$? $(tail -1 $M/round_once_check.txt)"
tools/pascal/boundary_sweep.sh text-v17 sv1-sweep; echo "rule 4 sweep rc=$?"
tools/pascal/long_decode.sh text-v17 sv1-long-decode; echo "long decode rc=$?"
python3 - <<'PY'
import json
ok = True
for case in ("control", "short", "middle", "end", "free"):
    new = json.load(open(f"/workspace/build/sv1-long-decode/{case}.default.json")); old = json.load(open(f"/workspace/build/long-decode-v1/{case}.default.json"))
    same = new["logits_sha256"] == old["logits_sha256"] and new["teacher_argmax"] == old["teacher_argmax"] and new["generated_ids"] == old["generated_ids"]
    ok &= same; print(f"long decode {case}: {'IDENTICAL to long-decode-v1' if same else 'DIFFERS from long-decode-v1'}")
print("rule 4 LONG DECODE UNCHANGED", "PASS" if ok else "FAIL")
PY
tools/pascal/score.sh text-v17 ppl-v17 256 257 4096 32768; echo "score rc=$?"
for L in 256 257 4096 32768; do cmp $B/ppl-v6/nll-pascal-$L.f32 $B/ppl-v17/nll-pascal-$L.f32 && echo "rule 3 nll $L: IDENTICAL to text-correct-v2" || echo "rule 3 nll $L: DIFFERS"; done
tools/pascal/speed.sh text-v10 sv1-speed-baseline 3; echo "speed baseline rc=$?"
tools/pascal/speed.sh text-v17 sv1-speed 3; echo "speed rc=$?"
python3 tools/pascal/analysis/speed_report.py $B/sv1-speed-baseline $B/sv1-speed --output $B/sv1-speed/report.txt; echo "speed rules rc=$?"
date
