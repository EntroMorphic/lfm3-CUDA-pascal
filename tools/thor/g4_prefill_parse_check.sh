#!/usr/bin/env bash
# Verification of the red-team remediation of the prefill budget rule (strict LFM3_PREFILL_HEADS parsing; comments).
# The default path must be unchanged: NLL bitwise identical to attempt 3 at all four lengths, and the logits of a
# traced request identical for the default and every valid override. Malformed overrides must be refused.
# Usage: g4_prefill_parse_check.sh TAG
set -uo pipefail
tag=$1
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
D=$tag-default; S=$tag-score; P=$tag-ppl; O=$B/$tag
for o in $D $S $P $tag; do test ! -e $B/$o || { echo "exists $o"; exit 1; }; done
step(){ echo "== $1 $(date +%T)"; }; fail(){ echo "FAIL: $1"; exit 1; }
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia"
step sources; (cd $R && sha256sum native/src/lfm3_main.cu tools/thor/g4_prefill_parse_check.sh) | sed 's/^/  /'
step build
docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$D; rc=\$?; chown -R 1000:1000 $C/$D $C/$D.cmake.log $C/$D.build.log 2>/dev/null; exit \$rc" || fail build
echo "warnings: $(grep -ci warning $B/$D.build.log)"
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server; do dyn=$(readelf -d $B/$D/$f) || fail "readelf $f"
  [ "$(grep -c libcublas <<< "$dyn")" = 0 ] || fail "$f links cuBLAS"; done
step nll-bit-identity; mkdir -p $B/$S $B/$P $O
docker run --rm --network none -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_scorer.sh $C/$S; rc=\$?; chown -R 1000:1000 $C/$S; exit \$rc" || fail scorer
cp -p $B/ppl-v5/windows-*.i32 $B/$P/
for L in 256 257 4096 32768; do
  $EX lfm3-profile-v1 $C/$S/lfm3-score /workspace/models/LFM3-Omni-2B $C/$P/windows-$L.i32 $L $C/$P/nll-native-gemm-$L.f32 >> $B/$P.log 2>&1 || fail "score $L"
  cmp -s $B/$P/nll-native-gemm-$L.f32 $B/ppl-g4-v3/nll-native-gemm-$L.f32 || fail "$L: NLL differs from attempt 3"
  echo "$L: NLL bitwise identical to attempt 3"
done
step override-parsing
# Capacity 2,560 (the 8-head band, never run before) and 32,768; 128 teacher-forced steps, logits traced.
python3 - $B/ppl-v5/windows-32768.i32 $O <<'PY'
import json, sys
import numpy as np
ids = np.fromfile(sys.argv[1], np.int32).tolist()
for length, capacity in ((2432, 2560), (4096, 32768)):
    r = dict(input_ids=ids[:length], teacher_ids=ids[length:length + 128], max_new_tokens=0, capacity=capacity)
    if capacity > 4096: r["experimental_long_context"] = True
    json.dump(r, open(f"{sys.argv[2]}/req-{capacity}.json", "w"))
PY
run(){ # NAME ENV CAPACITY
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r.update(trace_directory=sys.argv[2], trace_logits_only=True); print(json.dumps(r))" \
    $O/req-$3.json $C/$tag/trace-$1-$3 > $O/req-$1-$3.json
  $EX $2 lfm3-profile-v1 $C/$D/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B $C/$tag/req-$1-$3.json $C/$tag/out-$1-$3.json > $O/$1-$3.log 2>&1; }
for cap in 2560 32768; do
  for h in default 4 8 16; do e=""; [ $h = default ] || e="-e LFM3_PREFILL_HEADS=$h"; run $h "$e" $cap || fail "heads $h at $cap"
    echo "$cap heads=$h: $(cd $O/trace-$h-$cap && cat $(ls *.logits.f32 | sort -n) | sha256sum | cut -c1-16) over $(ls $O/trace-$h-$cap/*.logits.f32 | wc -l) steps"
  done
done
for bad in abc 16x "" 0 -4 3 12 99999999999; do
  if run bad "-e LFM3_PREFILL_HEADS=$bad" 2560; then fail "LFM3_PREFILL_HEADS='$bad' was accepted"; fi
  [ "$(grep -c 'LFM3_PREFILL_HEADS must' $O/bad-2560.log)" != 0 ] || { tail -3 $O/bad-2560.log; fail "'$bad' failed for another reason"; }
  echo "refused '$bad': $(grep -o 'LFM3_PREFILL_HEADS must[^"]*' $O/bad-2560.log | head -1)"
  rm -rf $O/trace-bad-2560
done
echo "ALL DONE $(date +%T)"
