# speed-v1 development (not a rule run): the greedy choice on the card. Ids against records, decode speed, the gap.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; out=$B/speed-dev6; mkdir -p $out
tools/pascal/build_text.sh text-v18
tools/pascal/run_text_fixtures.sh text-v18 1; echo "fixtures rc=$?"
# long-decode-v1's free case (600 generated tokens from a 6,000-token prompt at capacity 32,768) without tracing
python3 - $out <<'PY'
import json, sys
from array import array
out = sys.argv[1]
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read())
w = a[32768:65536]
json.dump({"input_ids": list(w[:6000]), "capacity": 32768, "experimental_long_context": True, "max_new_tokens": 600}, open(out + "/free.json", "w"))
PY
$B/text-v18/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B $out/free.json $out/free.out > $out/free.log 2>&1; echo "free rc=$?"
python3 -c "
import json
new=json.load(open('$out/free.out')); old=json.load(open('$B/long-decode-v1/free.default.json'))
print('free generation:', 'IDENTICAL ids to long-decode-v1' if new['generated_ids']==old['generated_ids'] else 'DIFFERENT ids', len(new['generated_ids']), 'tokens')"
CASES="decode-1k 1024 300 700
decode-32k 32768 31000 700
prefill-32 4096 32 0" tools/pascal/speed.sh text-v18 speed-dev6-v18 1
python3 tools/pascal/analysis/speed_report.py $B/sv1-speed-baseline $B/speed-dev6-v18 2>&1 | grep -E "rule 6|rule 1[12]|rule 10"
bash validation/speed-v1/profile_decode.sh text-v18 > /dev/null 2>&1
python3 validation/speed-v1/gaps.py $B/profile-decode-text-v18/d1k.sqlite
date
