# speed-v1 development (not a rule run): decode steps of build TAG under Nsight Systems, read by decode_atoms.py.
#   bash validation/speed-v1/profile_decode.sh TAG
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; tag=${1:?}; out=$B/profile-decode-$tag; mkdir -p $out
test -x $B/$tag/liquid-lfm3 || tools/pascal/build_text.sh $tag
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda*/bin/nsys 2>/dev/null | head -1)
python3 - $out <<'PY'
import json, sys
from array import array
out = sys.argv[1]
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read())
json.dump({"input_ids": list(a[:300]), "teacher_ids": list(a[300:600]), "capacity": 1024}, open(f"{out}/d1k.json", "w"))
json.dump({"input_ids": list(a[:31000]), "teacher_ids": list(a[31000:31300]), "capacity": 32768, "experimental_long_context": True}, open(f"{out}/d32k.json", "w"))
PY
for c in d1k d32k; do
  $NSYS profile -t cuda -o $out/$c -f true $B/$tag/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B $out/$c.json $out/$c.out > $out/$c.nsys.log 2>&1
  $NSYS export -t sqlite -o $out/$c.sqlite -f true $out/$c.nsys-rep > /dev/null 2>&1
  python3 tools/pascal/decode_atoms.py $out/$c.sqlite 100 | sed -n '/steady state/,$p'
done
date
