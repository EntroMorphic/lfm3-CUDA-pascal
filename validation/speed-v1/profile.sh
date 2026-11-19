# speed-v1 development (not a rule run): build TAG from this tree and profile the prefill cases with Nsight Systems,
# kernels summed by name.  bash validation/speed-v1/profile.sh TAG
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; tag=${1:?}; out=$B/profile-$tag; mkdir -p $out
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda*/bin/nsys 2>/dev/null | head -1)
test -x $B/$tag/liquid-lfm3 || tools/pascal/build_text.sh $tag
python3 - $out <<'PY'
import json, sys
from array import array
out = sys.argv[1]
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read())
for n in (32, 257, 4096):
    json.dump({"input_ids": list(a[:n]), "capacity": 4096}, open(f"{out}/p{n}.json", "w"))
PY
for n in 32 257 4096; do
  $B/$tag/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B $out/p$n.json $out/p$n.warm.out > /dev/null 2>&1
  $NSYS profile -t cuda -o $out/p$n -f true $B/$tag/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B $out/p$n.json $out/p$n.out > $out/p$n.nsys.log 2>&1
  python3 -c "import json;print('p$n prefill', json.load(open('$out/p$n.out'))['prefill_seconds'])"
  $NSYS stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $out/p$n $out/p$n.nsys-rep > /dev/null 2>&1
  python3 - $out/p${n}_cuda_gpu_kern_sum.csv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
total = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"  GPU kernel time {total/1e6:.1f} ms in {sum(int(r['Instances']) for r in rows)} launches")
for r in rows[:12]:
    print(f"  {float(r['Total Time (ns)'])/1e6:9.2f} ms {int(r['Instances']):6d}x  {r['Name'][:110]}")
PY
done
date
