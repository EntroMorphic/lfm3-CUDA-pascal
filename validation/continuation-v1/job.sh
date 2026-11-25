# continuation-v1 (tools/pascal/remote.sh detach cont-v1 'bash validation/continuation-v1/job.sh'): rules 1-6
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/cont-v1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; V=validation/speech-v1/inputs/reference; T=/workspace/data/ppl-windows/windows-32768.i32
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
cp validation/speech-v1/records/sp-v1/same.py $O/same.py

# rule 1
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v42 > $O/build-parity-v42.log 2>&1; echo "build parity-v42 rc=$?"
tools/pascal/build_speech.sh text-v42 > $O/build-text-v42.log 2>&1; echo "build text-v42 rc=$?"
tools/pascal/build_serve.sh text-v42s > $O/build-text-v42s.log 2>&1; echo "build text-v42s rc=$?"
grep -h "warnings" $O/build-*.log
for b in text-v42 parity-v42; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server lfm3-vision-check) | sed "s#  #  $b/#"; done > $O/binaries.sha256
(cd $B/text-v42 && sha256sum lfm3-continuation-check) | sed "s#  #  text-v42/#" >> $O/binaries.sha256
(cd $B/text-v42s && sha256sum liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  text-v42s/#" >> $O/binaries.sha256
grep -c "177-D" $B/text-v42s/liquid-lfm3-worker.build.log $B/text-v39s/liquid-lfm3-worker.build.log
diff <(grep -o 'warning #[0-9]*-D: [^"]*"[^"]*"' $B/text-v42s/liquid-lfm3-worker.build.log | sort) <(grep -o 'warning #[0-9]*-D: [^"]*"[^"]*"' $B/text-v39s/liquid-lfm3-worker.build.log | sort) && echo "text-only worker warnings: the same list as text-v39s"

# rule 2: the exact identity, three layouts, and memcheck
C=$B/text-v42/lfm3-continuation-check
$C $M $T exact > $O/exact.jsonl 2> $O/exact.err; echo "rule 2 exact rc=$? $(tail -1 $O/exact.jsonl)"
LFM3_COMPACT_WINDOWS=1 $C $M $T exact > $O/exact-compact.jsonl 2> $O/exact-compact.err; echo "rule 2 exact compact rc=$? $(tail -1 $O/exact-compact.jsonl)"
LFM3_COMPACT_WINDOWS=0 $C $M $T exact > $O/exact-full.jsonl 2> $O/exact-full.err; echo "rule 2 exact full-length rc=$? $(tail -1 $O/exact-full.jsonl)"
for f in exact exact-compact exact-full; do echo "rule 2 $f: $(grep -c '"ok":true' $O/$f.jsonl) of $(grep -c '"ends"' $O/$f.jsonl) cases ok"; done
SAN="/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9"
$SAN $C $M $T exact-small > $O/memcheck.log 2>&1; echo "rule 2 memcheck rc=$? $(grep 'ERROR SUMMARY' $O/memcheck.log) $(grep -o '{"exact":"[A-Z]*"}' $O/memcheck.log)"
LFM3_COMPACT_WINDOWS=1 $SAN $C $M $T exact-small > $O/memcheck-compact.log 2>&1; echo "rule 2 memcheck compact rc=$? $(grep 'ERROR SUMMARY' $O/memcheck-compact.log) $(grep -o '{"exact":"[A-Z]*"}' $O/memcheck-compact.log)"

# rule 3: speed
$C $M $T speed > $O/speed.jsonl 2> $O/speed.err; echo "rule 3 runs rc=$?"
python3 - $O/speed.jsonl > $O/speed.txt <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
ok = len(rows) == 14
for r in rows:
    e, w, t, n = r["extend_s"], r["whole_prefill_s"], r["token_at_a_time_s"], r["added"]
    good = e <= 1.03 * w and (n < 8 or t < 0 or e < t)
    if (r["capacity"], r["prefix"], n) == (32768, 31744, 1024): good &= e <= 2.0
    if (r["capacity"], r["prefix"], n) == (4096, 3072, 1024): good &= e <= 1.0
    ok &= good
    print(f"rule 3 capacity {r['capacity']}, {r['prefix']} cached + {n}: extend {e:.3f} s, whole prefill {w:.3f} s, "
          f"token at a time {'-' if t < 0 else f'{t:.3f} s'}: {'ok' if good else 'FAIL'}")
print("RULE 3", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
echo "rule 3 rc=$?"; cat $O/speed.txt

# rule 4: the served paths
python3 tools/pascal/continuation_serve_check.py $B/text-v42/liquid-lfm3-worker $B/text-v42/liquid-lfm3-server $M $O/serve-shipping shipping > $O/serve-shipping.log 2>&1; echo "rule 4 shipping rc=$? $(tail -1 $O/serve-shipping.log)"; clean
python3 tools/pascal/continuation_serve_check.py $B/text-v42s/liquid-lfm3-worker $B/text-v42s/liquid-lfm3-server $M $O/serve-text-only shipping > $O/serve-text-only.log 2>&1; echo "rule 4 text-only rc=$? $(tail -1 $O/serve-text-only.log)"; clean
python3 tools/pascal/continuation_serve_check.py $B/parity-v42/liquid-lfm3-worker - $M $O/serve-parity parity > $O/serve-parity.log 2>&1; echo "rule 4 parity rc=$? $(tail -1 $O/serve-parity.log)"
grep -hE '^(worker|server):' $O/serve-*.log

# rule 5, device code: A = v42, B = the baseline; only the continuation's two kernels may be in v42 alone
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server lfm3-vision-check; do for k in text parity; do
  [ $k = text ] && base=text-v41 || base=parity-v40
  cuobjdump -sass $B/$k-v42/$f > $O/e1/a.sass; cuobjdump -sass $B/$base/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass --only-in-a convolution7_continue > $O/e1/$f.$k.txt; echo "rule 5 $f $k-v42 against $base rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done
for f in liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server; do
  cuobjdump -sass $B/text-v42s/$f > $O/e1/a.sass; cuobjdump -sass $B/text-v39s/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass --only-in-a convolution7_continue > $O/e1/$f.text-only.txt; echo "rule 5 $f text-v42s against text-v39s rc=$? $(head -1 $O/e1/$f.text-only.txt)"
done; rm -f $O/e1/*.sass

# rule 5, outputs
python3 tools/pascal/compare_pins.py parity-v42 > $O/gate.log 2>&1; echo "gate rc=$? $(tail -1 $O/gate.log)"
tools/pascal/run_text_fixtures.sh text-v42 2 > $O/fixtures.log 2>&1; echo "fixtures rc=$? $(tail -1 $O/fixtures.log)"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v42 2 > $O/fixtures-compact.log 2>&1; echo "fixtures compact rc=$? $(tail -1 $O/fixtures-compact.log)"
tools/pascal/run_text_fixtures.sh text-v42 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 > $O/fixtures-memcheck.log 2>&1; echo "fixtures memcheck rc=$? $(tail -1 $O/fixtures-memcheck.log)"

P=$B/rt2-ppl-subset
for k in text parity; do WINDOWS_DIR=$P tools/pascal/score.sh $k-v42 cont1-score-$k-v42 256 257 4096 > $O/score-$k-v42.log 2>&1 || echo "score $k-v42 FAILED"
  for L in 256 257 4096; do cmp $B/img1-score-$k-v40/nll-pascal-$L.f32 $B/cont1-score-$k-v42/nll-pascal-$L.f32 && echo "scorer $k $L: v42 IDENTICAL to v40" || echo "scorer $k $L: DIFFER"; done; done

python3 - $O/a1 <<'PY'
import json, sys
from array import array
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir()
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read()); w = a[:32768]
for L in (129, 136, 137, 300):
    (out / f"p{L}.request.json").write_text(json.dumps(dict(input_ids=list(w[:L]), teacher_ids=list(w[L:L + 16]),
                                                            capacity=4097, experimental_long_context=True, trace_logits_only=True)))
PY
for k in text parity; do [ $k = text ] && base=v41 || base=v40; for v in $base v42; do for L in 129 136 137 300; do
  d=$O/a1/$k-$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/$k-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 $k-$v p$L run FAILED"
  (cd $d/trace && shopt -s nullglob && f=(*.logits.f32) && echo "${#f[@]}" && { [ ${#f[@]} -gt 0 ] && sha256sum $(printf '%s\n' "${f[@]}" | sort -n) | sha256sum | cut -c1-16 || echo none; }) > $d/logits.sha; rm -rf $d/trace
done; done
for L in 129 136 137 300; do a=$(tr '\n' ' ' < $O/a1/$k-$base-p$L/logits.sha); b=$(tr '\n' ' ' < $O/a1/$k-v42-p$L/logits.sha)
  echo "above 4,096 $k p$L: $base [$a] v42 [$b] $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFER)$([[ $a == "17 "* && $b == "17 "* ]] || echo ' INCOMPLETE')"; done; done

# rule 5, speech
R3=$O/rule3; mkdir -p $R3/out; RP=$B/sp-refs-parity-s1
python3 - $RP validation/speech-v1/inputs/thor-clips.json $R3 <<'PY'
import json, sys
from pathlib import Path
refs, clips, o = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
lines = []
for run in sorted({c["run"] for c in clips}):
    split = run.startswith(("tts-long-male", "probe-other-long")); d = refs / ("speech-chunk-native-v1" if split else "speech-e2e-v1")
    lines.append(f"{d}/requests/{run}.req.json {o}/out/{run}")
(o / "list").write_text("\n".join(lines) + "\n")
PY
$B/parity-v42/lfm3-speech $M --batch $R3/list > $R3/log 2> $R3/err; echo "speech-v1 rule 3 runs rc=$? $(tail -1 $R3/err)"
python3 - $R3/out validation/speech-v1/inputs/thor-clips.json > $O/rule3.txt <<'PY'
import hashlib, json, sys
from pathlib import Path
out, clips = Path(sys.argv[1]), json.load(open(sys.argv[2]))
same = 0
for c in clips:
    wav = out / f"{c['run']}.wav"
    ok = wav.exists() and hashlib.sha256(wav.read_bytes()).hexdigest() == c["sha256"]; same += ok
    print(f"rule 3 {c['run']} ({c['set']}): {'equal' if ok else 'DIFFERS'}")
runs = {c["run"] for c in clips}
print(f"rule 3: {same} of {len(clips)} clips ({len(runs)} runs) equal to the Thor engine's"); sys.exit(0 if same == len(clips) == 16 and len(runs) == 15 else 1)
PY
echo "speech-v1 rule 3 rc=$? $(tail -1 $O/rule3.txt)"
N3=$O/rule3-native; mkdir -p $N3/requests $N3/out
python3 - $RP validation/speech-v1/inputs/thor-clips.json $N3 <<'PY2'
import ast, json, sys
from pathlib import Path
refs, clips, n3 = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
lines = []
for run in sorted({c["run"] for c in clips}):
    split = run.startswith(("tts-long-male", "probe-other-long")); d = refs / ("speech-chunk-native-v1" if split else "speech-e2e-v1")
    r = json.loads((d / "requests" / f"{run}.req.json").read_text())
    conv = r["conversation"]; conv = ast.literal_eval(conv) if isinstance(conv, str) else conv
    body = dict(messages=conv, mode=r["mode"], max_new_tokens=r["max_new_tokens"], seed=r["seed"], split=split)
    (n3 / "requests" / f"{run}.req.json").write_text(json.dumps(body)); lines.append(f"{n3}/requests/{run}.req.json {n3}/out/{run}")
(n3 / "list").write_text("\n".join(lines) + "\n")
PY2
$B/parity-v42/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "speech-v1 rule 3 rendering runs rc=$? $(tail -1 $N3/err)"
python3 - $R3/out $N3 $RP > $O/rule3-native.txt <<'PY2'
import json, sys
from pathlib import Path
ids_run, n3, refs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); same = 0
runs = sorted(p.name[:-9] for p in (n3 / "requests").glob("*.req.json"))
for run in runs:
    split = run.startswith(("tts-long-male", "probe-other-long"))
    r = json.loads((refs / ("speech-chunk-native-v1" if split else "speech-e2e-v1") / "requests" / f"{run}.req.json").read_text())
    L = lambda v: json.loads(v) if isinstance(v, str) else v
    got, want = (json.loads((d / f"{run}.json").read_text()) for d in (n3 / "out", ids_run))
    ok = got.get("prompt") == want.get("prompt") and len(got.get("chunks", [])) == len(want.get("chunks", [])) and \
        (not split or len(got["chunks"]) == len(L(r["chunks"]))) and \
        (n3 / "out" / f"{run}.wav").read_bytes() == (ids_run / f"{run}.wav").read_bytes()
    same += ok; print(f"rule 3 rendering {run}: {'same ids and WAV' if ok else 'DIFFERS'}")
print(f"rule 3 rendering: {same} of {len(runs)} equal"); sys.exit(0 if same == len(runs) == 15 else 1)
PY2
echo "speech-v1 rule 3 rendering rc=$? $(tail -1 $O/rule3-native.txt)"
E=$O/e2; mkdir -p $E/text $E/text-split; RT=$B/sp-refs-text-s1; S1=$B/sp-v1/e2
{ for f in $(ls $RT/speech-e2e-v1/requests | grep -- '-s1\.req\.json$'); do echo "$RT/speech-e2e-v1/requests/$f $E/text/${f%.req.json}"; done
  echo "$RT/speech-chunk-native-v1/requests/tts-long-male-s1.req.json $E/text-split/tts-long-male-s1"; } > $E/text.list
$B/text-v42/lfm3-speech $M --batch $E/text.list > $E/text.log 2> $E/text.err; echo "E2 shipping speech runs rc=$? $(tail -1 $E/text.err)"
python3 $O/same.py $E/text $S1/text $(grep -v text-split $E/text.list | awk '{print $2}' | xargs -n1 basename) | tail -1; echo "E2 shipping speech rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-split $S1/text-split tts-long-male-s1 | tail -1; echo "E2 split rc=${PIPESTATUS[0]}"
python3 -c "import json,sys; print(' '.join(sorted(c['case'] for c in json.load(open(sys.argv[1]))['cases'])))" $V/audio-in-v1/round2-full-check.json > $E/libri-stems.txt
for b in text parity; do mkdir -p $E/libri-$b
  for s in $(cat $E/libri-stems.txt); do echo "$B/sp-dev3/requests/$s.req.json $E/libri-$b/$s"; done > $E/libri-$b.list
  $B/$b-v42/lfm3-speech $M --batch $E/libri-$b.list > $E/libri-$b.log 2> $E/libri-$b.err; echo "E2 LibriSpeech $b runs rc=$? $(tail -1 $E/libri-$b.err)"
  python3 $O/same.py $E/libri-$b $S1/libri-$b $(cat $E/libri-stems.txt) | tail -1; echo "E2 LibriSpeech $b rc=${PIPESTATUS[0]}"
done

# rule 5, paired decode
python3 tools/pascal/paired_decode_check.py $B/text-v42/liquid-lfm3-worker $M $T $O/paired > $O/paired.log 2>&1; echo "paired decode rc=$?"; grep -E 'PAIRED' $O/paired.log

# rule 6: continuations after decoded positions
$C $M $T decoded > $O/decoded.jsonl 2> $O/decoded.err; echo "rule 6 rc=$? $(grep -c '"mode":"decoded"' $O/decoded.jsonl) of 8 cases"; cat $O/decoded.jsonl $O/decoded.err
LFM3_COMPACT_WINDOWS=1 $C $M $T decoded > $O/decoded-compact.jsonl 2> $O/decoded-compact.err; echo "rule 6 compact rc=$? $(grep -c '"mode":"decoded"' $O/decoded-compact.jsonl) of 8 cases"; cat $O/decoded-compact.jsonl $O/decoded-compact.err
date
