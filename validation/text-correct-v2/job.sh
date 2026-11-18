# The job as run in the container for text-correct-v2 (tools/pascal/remote.sh detach text-v6 'bash validation/text-correct-v2/job.sh').
set -x; date; cd /workspace/lfm3-CUDA-pascal; sha256sum /workspace/build/text-v6/lfm3-score /workspace/build/text-v6/liquid-lfm3
tools/pascal/gate_pins.sh parity-v6; echo "gate rc=$?"
EXTRA_FLAGS=-DLFM_GEMM_FIRST_ROW=0 tools/pascal/build_text.sh text-v6-plain; tools/pascal/run_text_fixtures.sh text-v6-plain 2; echo "plain fixtures rc=$?"
tools/pascal/run_text_fixtures.sh text-v6 2; echo "fixtures rc=$?"
tools/pascal/run_text_fixtures.sh text-v6 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck rc=$?"; grep -h "ERROR SUMMARY" /workspace/build/text-v6/fixtures-wrapped/*.log
tools/pascal/score.sh text-v6 ppl-v6 256 257 4096 32768; echo "score rc=$?"
date
