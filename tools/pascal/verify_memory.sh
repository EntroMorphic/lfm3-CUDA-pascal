#!/usr/bin/env bash
# Runs in the container: memory-v1's on-card checks of build NEW against build OLD (rules 3, 4, 5, 6, 7).
#   tools/pascal/verify_memory.sh OLD_TAG NEW_TAG OLD_PPL_NAME [fast|slow|all]
# fast: rules 3, 4 and the subsets of rule 5 (minutes). slow: every window at 256, 257 and 4,096, then 32,768 (hours).
set -uo pipefail
old=${1:?}; new=${2:?}; oldppl=${3:?}; part=${4:-all}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); B=/workspace/build; M=/workspace/models/LFM3-Omni-2B
fail=0; say(){ echo "== $*"; }
say "binaries"; sha256sum $B/$old/liquid-lfm3 $B/$old/lfm3-score $B/$new/liquid-lfm3 $B/$new/lfm3-score

if [ "$part" != slow ]; then
say "rule 3: fixtures, LFM3_COMPACT_WINDOWS unset"
"$repo/tools/pascal/run_text_fixtures.sh" "$new" 2 || fail=1
say "rule 3: fixtures, LFM3_COMPACT_WINDOWS=1"
LFM3_COMPACT_WINDOWS=1 "$repo/tools/pascal/run_text_fixtures.sh" "$new" 2 || fail=1

say "rule 4: long decode"
L=$B/$new/long-decode; rm -rf "$L"; mkdir -p "$L/old" "$L/new"
for side in old new; do
  tag=$old; [ $side = new ] && tag=$new
  python3 - "$repo/validation/memory-v1/long-decode.request.json" "$L/$side" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); r["trace_directory"] = sys.argv[2] + "/trace"
json.dump(r, open(sys.argv[2] + "/request.json", "w"))
PY
  if [ $side = new ]; then export LFM3_COMPACT_WINDOWS=1; else unset LFM3_COMPACT_WINDOWS; fi
  "$B/$tag/liquid-lfm3" --experimental $M "$L/$side/request.json" "$L/$side/native.json" > "$L/$side/log" 2>&1 || { echo "long decode $side FAILED: $(tail -c 300 "$L/$side/log")"; fail=1; }
  (cd "$L/$side/trace" && ls | wc -l && sha256sum $(ls *.logits.f32 | sort -n) | sha256sum) > "$L/$side/trace.sha256" 2>/dev/null
  rm -rf "$L/$side/trace"
done
unset LFM3_COMPACT_WINDOWS
python3 - "$L" <<'PY' || fail=1
import json, sys
L = sys.argv[1]; a, b = (json.load(open(f"{L}/{s}/native.json")) for s in ("old", "new"))
ta, tb = (open(f"{L}/{s}/trace.sha256").read().split() for s in ("old", "new"))
ok = a["teacher_argmax"] == b["teacher_argmax"] and len(a["teacher_argmax"]) == 701 and ta == tb and int(ta[0]) >= 701
print(f"long decode: {len(a['teacher_argmax'])} argmax ids {'equal' if a['teacher_argmax'] == b['teacher_argmax'] else 'DIFFER'}; "
      f"{ta[0]} trace files, logits digest old {ta[1][:16]} new {tb[1][:16]}: {'IDENTICAL' if ta == tb else 'DIFFER'}; "
      f"decode {a['decode_seconds']:.1f} s old, {b['decode_seconds']:.1f} s new")
sys.exit(0 if ok else 1)
PY

say "rule 5 (unset): subsets"
P=/workspace/data/ppl-subset; mkdir -p $P
head -c $((64*256*4)) /workspace/data/ppl-windows/windows-256.i32 > $P/windows-256.i32
head -c $((64*257*4)) /workspace/data/ppl-windows/windows-257.i32 > $P/windows-257.i32
head -c $((2*4096*4)) /workspace/data/ppl-windows/windows-4096.i32 > $P/windows-4096.i32
rm -rf $B/$new-subset-old $B/$new-subset-new
WINDOWS_DIR=$P "$repo/tools/pascal/score.sh" "$old" "$new-subset-old" 256 257 4096 || fail=1
WINDOWS_DIR=$P "$repo/tools/pascal/score.sh" "$new" "$new-subset-new" 256 257 4096 || fail=1
for L in 256 257 4096; do cmp "$B/$new-subset-old/nll-pascal-$L.f32" "$B/$new-subset-new/nll-pascal-$L.f32" && echo "subset $L: IDENTICAL" || { echo "subset $L: DIFFER"; fail=1; }; done

fi
if [ "$part" != fast ]; then
say "rules 5, 6 (LFM3_COMPACT_WINDOWS=1): all windows"
rm -rf $B/$new-ppl
LFM3_COMPACT_WINDOWS=1 "$repo/tools/pascal/score.sh" "$new" "$new-ppl" 256 257 4096 || fail=1
for L in 256 257 4096; do cmp "$B/$oldppl/nll-pascal-$L.f32" "$B/$new-ppl/nll-pascal-$L.f32" && echo "full $L: IDENTICAL to $oldppl" || { echo "full $L: DIFFER"; fail=1; }; done

say "rule 7: 32,768"
"$repo/tools/pascal/score.sh" "$new" "$new-ppl" 32768; echo "32768 rc=$?"
fi
say "VERIFY $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
