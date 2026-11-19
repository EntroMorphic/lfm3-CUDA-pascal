# speed-v1 development (not a rule run): the logits copied straight into the result; decode speed and the gap.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build
tools/pascal/build_text.sh text-v16
tools/pascal/run_text_fixtures.sh text-v16 1; echo "fixtures rc=$?"
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700" tools/pascal/speed.sh text-v16 speed-dev5-v16 1
bash validation/speed-v1/profile_decode.sh text-v16 > /dev/null 2>&1
python3 validation/speed-v1/gaps.py $B/profile-decode-text-v16/d1k.sqlite
date
