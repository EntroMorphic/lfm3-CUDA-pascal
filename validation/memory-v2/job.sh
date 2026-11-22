# memory-v2 (tools/pascal/remote.sh detach mem-v2 'bash validation/memory-v2/job.sh'): Runner::operand, rules 1-4
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/mem-v2; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; V=validation/speech-v1/inputs/reference
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
cp validation/speech-v1/records/sp-v1/same.py $O/same.py

# rule 1
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v39 > $O/build-parity-v39.log 2>&1; echo "build parity-v39 rc=$?"
tools/pascal/build_speech.sh text-v39 > $O/build-text-v39.log 2>&1; echo "build text-v39 rc=$?"
tools/pascal/build_serve.sh text-v39s > $O/build-text-v39s.log 2>&1; echo "build text-v39s rc=$?"
grep -h "warnings" $O/build-*.log
for b in text-v39 parity-v39; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  $b/#"; done > $O/binaries.sha256
(cd $B/text-v39s && sha256sum liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  text-v39s/#" >> $O/binaries.sha256

# rule 2
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server; do for k in text parity; do
  cuobjdump -sass $B/$k-v39/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-v38/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.$k.txt; echo "rule 2 $f $k-v39 against $k-v38 rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done
for f in liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server; do
  cuobjdump -sass $B/text-v39s/$f > $O/e1/a.sass; cuobjdump -sass $B/text-v38s/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.text-only.txt; echo "rule 2 $f text-v39s against text-v38s rc=$? $(head -1 $O/e1/$f.text-only.txt)"
done; rm -f $O/e1/*.sass

# rule 3
python3 tools/pascal/compare_pins.py parity-v39 > $O/gate.log 2>&1; echo "gate rc=$? $(tail -1 $O/gate.log)"
tools/pascal/run_text_fixtures.sh text-v39 2 > $O/fixtures.log 2>&1; echo "fixtures rc=$? $(tail -1 $O/fixtures.log)"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v39 2 > $O/fixtures-compact.log 2>&1; echo "fixtures compact rc=$? $(tail -1 $O/fixtures-compact.log)"
tools/pascal/run_text_fixtures.sh text-v39 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 > $O/fixtures-memcheck.log 2>&1; echo "fixtures memcheck rc=$? $(tail -1 $O/fixtures-memcheck.log)"

P=$B/rt2-ppl-subset
for k in text parity; do WINDOWS_DIR=$P tools/pascal/score.sh $k-v39 mem2-score-$k-v39 256 257 4096 > $O/score-$k-v39.log 2>&1 || echo "score $k-v39 FAILED"
  for L in 256 257 4096; do cmp $B/rt2-score-$k-v38/nll-pascal-$L.f32 $B/mem2-score-$k-v39/nll-pascal-$L.f32 && echo "scorer $k $L: v39 IDENTICAL to v38" || echo "scorer $k $L: DIFFER"; done; done

# rule 3, above 4,096
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
for k in text parity; do for v in v38 v39; do for L in 129 136 137 300; do
  d=$O/a1/$k-$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/$k-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 $k-$v p$L run FAILED"
  (cd $d/trace && shopt -s nullglob && f=(*.logits.f32) && echo "${#f[@]}" && { [ ${#f[@]} -gt 0 ] && sha256sum $(printf '%s\n' "${f[@]}" | sort -n) | sha256sum | cut -c1-16 || echo none; }) > $d/logits.sha; rm -rf $d/trace
done; done
for L in 129 136 137 300; do a=$(tr '\n' ' ' < $O/a1/$k-v38-p$L/logits.sha); b=$(tr '\n' ' ' < $O/a1/$k-v39-p$L/logits.sha)
  echo "above 4,096 $k p$L: v38 [$a] v39 [$b] $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFER)$([[ $a == "17 "* && $b == "17 "* ]] || echo ' INCOMPLETE')"; done; done


# rule 3, speech
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
$B/parity-v39/lfm3-speech $M --batch $R3/list > $R3/log 2> $R3/err; echo "rule 3 runs rc=$? $(tail -1 $R3/err)"
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
$B/parity-v39/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "rule 3 native rendering runs rc=$? $(tail -1 $N3/err)"
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
$B/text-v39/lfm3-speech $M --batch $E/text.list > $E/text.log 2> $E/text.err; echo "E2 shipping speech runs rc=$? $(tail -1 $E/text.err)"
python3 $O/same.py $E/text $S1/text $(grep -v text-split $E/text.list | awk '{print $2}' | xargs -n1 basename) | tail -1; echo "E2 shipping speech rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-split $S1/text-split tts-long-male-s1 | tail -1; echo "E2 split rc=${PIPESTATUS[0]}"
python3 -c "import json,sys; print(' '.join(sorted(c['case'] for c in json.load(open(sys.argv[1]))['cases'])))" $V/audio-in-v1/round2-full-check.json > $E/libri-stems.txt
for b in text parity; do mkdir -p $E/libri-$b
  for s in $(cat $E/libri-stems.txt); do echo "$B/sp-dev3/requests/$s.req.json $E/libri-$b/$s"; done > $E/libri-$b.list
  $B/$b-v39/lfm3-speech $M --batch $E/libri-$b.list > $E/libri-$b.log 2> $E/libri-$b.err; echo "E2 LibriSpeech $b runs rc=$? $(tail -1 $E/libri-$b.err)"
  python3 $O/same.py $E/libri-$b $S1/libri-$b $(cat $E/libri-stems.txt) | tail -1; echo "E2 LibriSpeech $b rc=${PIPESTATUS[0]}"
done

# rule 3, paired decode
python3 tools/pascal/paired_decode_check.py $B/text-v39/liquid-lfm3-worker $M /workspace/data/ppl-windows/windows-32768.i32 $O/paired > $O/paired.log 2>&1; echo "paired decode rc=$?"; grep -E 'PAIRED' $O/paired.log

# rule 4
$PY tools/pascal/speech_memory.py $B/text-v39/liquid-lfm3-server $M 1 $LIBRI $O/memory-1 7740 29696 > $O/memory-1.log 2>&1
echo "rule 12 rows 1 total 29696 rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-1.log; clean
date
