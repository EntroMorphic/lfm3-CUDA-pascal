set -x; date -u; cd /workspace/lfm3-CUDA-pascal; sha256sum /workspace/build/text-v1/lfm3-score /workspace/build/text-v1/liquid-lfm3; rm -rf /workspace/build/probe-score
tools/pascal/score.sh text-v1 ppl-v1 256 257 4096; echo "score rc=$?"
tools/pascal/score.sh text-v1 ppl-v1-repeat 256; echo "repeat rc=$?"; sha256sum /workspace/build/ppl-v1/nll-pascal-256.f32 /workspace/build/ppl-v1-repeat/nll-pascal-256.f32
tools/pascal/run_text_fixtures.sh text-v1 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "memcheck rc=$?"; grep -h "ERROR SUMMARY" /workspace/build/text-v1/fixtures-wrapped/*.log
tools/pascal/score.sh text-v1 ppl-v1 32768; echo "score32k rc=$?"
date -u
