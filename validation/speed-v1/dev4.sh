# speed-v1 development (not a rule run): fused norms and rotations; quick output check and decode speed.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build
tools/pascal/build_text.sh text-v15
tools/pascal/run_text_fixtures.sh text-v15 1; echo "fixtures rc=$?"
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700
prefill-32 4096 32 0" tools/pascal/speed.sh text-v15 speed-dev4-v15 1
date
