# speed-v1 development (not a rule run): build TAG; fixtures; decode timings (2 runs); decode-32k profile.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; tag=${1:?}
tools/pascal/build_text.sh $tag
tools/pascal/run_text_fixtures.sh $tag 1 | tail -1
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700" tools/pascal/speed.sh $tag speed-$tag 2 | grep "ms per step"
bash validation/speed-v1/profile_decode.sh $tag > /dev/null 2>&1
python3 tools/pascal/decode_atoms.py $B/profile-decode-$tag/d32k.sqlite 100 | sed -n '/steady state, steps/,/2048-column/p' | head -26 | cut -c1-150
date
