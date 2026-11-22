# images-v1 (tools/pascal/remote.sh detach img-v1 'bash validation/images-v1/job.sh'): rules 1-8
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/img-v1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; V=validation/speech-v1/inputs/reference
REF=/workspace/data/vision-ref/hf-5.20.2; BENCH=/workspace/data/vision-bench; I=validation/images-v1/inputs/reference
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
cp validation/speech-v1/records/sp-v1/same.py $O/same.py
echo "data: manifest $(sha256sum < $BENCH/manifest.jsonl | cut -c1-8) ($(wc -l < $BENCH/manifest.jsonl) rows), $(ls $BENCH/images | wc -l) images, $(ls $REF/float32 | wc -l) FP32 and $(ls $REF/bfloat16 | wc -l) BF16 cases"
CASES="cherry china flower gray mirror noise rgba small tall two wide"

# rule 1
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v40 > $O/build-parity-v40.log 2>&1; echo "build parity-v40 rc=$?"
tools/pascal/build_speech.sh text-v40 > $O/build-text-v40.log 2>&1; echo "build text-v40 rc=$?"
grep -h "warnings" $O/build-*.log
for b in text-v40 parity-v40; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server lfm3-vision-check) | sed "s#  #  $b/#"; done > $O/binaries.sha256

# rule 2, device code: A = v40, B = v39; only the vision encoder's kernels may be in v40 alone
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server; do for k in text parity; do
  cuobjdump -sass $B/$k-v40/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-v39/$f > $O/e1/b.sass
  case $f in liquid-lfm3|lfm3-score) only=();; *) only=(--only-in-a lfm3_vision);; esac
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass "${only[@]}" > $O/e1/$f.$k.txt; echo "rule 2 $f $k-v40 against $k-v39 rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done; rm -f $O/e1/*.sass

# rule 2, outputs
python3 tools/pascal/compare_pins.py parity-v40 > $O/gate.log 2>&1; echo "gate rc=$? $(tail -1 $O/gate.log)"
tools/pascal/run_text_fixtures.sh text-v40 2 > $O/fixtures.log 2>&1; echo "fixtures rc=$? $(tail -1 $O/fixtures.log)"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v40 2 > $O/fixtures-compact.log 2>&1; echo "fixtures compact rc=$? $(tail -1 $O/fixtures-compact.log)"
tools/pascal/run_text_fixtures.sh text-v40 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 > $O/fixtures-memcheck.log 2>&1; echo "fixtures memcheck rc=$? $(tail -1 $O/fixtures-memcheck.log)"

P=$B/rt2-ppl-subset
for k in text parity; do WINDOWS_DIR=$P tools/pascal/score.sh $k-v40 img1-score-$k-v40 256 257 4096 > $O/score-$k-v40.log 2>&1 || echo "score $k-v40 FAILED"
  for L in 256 257 4096; do cmp $B/mem2-score-$k-v39/nll-pascal-$L.f32 $B/img1-score-$k-v40/nll-pascal-$L.f32 && echo "scorer $k $L: v40 IDENTICAL to v39" || echo "scorer $k $L: DIFFER"; done; done

# rule 2, above 4,096
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
for k in text parity; do for v in v39 v40; do for L in 129 136 137 300; do
  d=$O/a1/$k-$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/$k-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 $k-$v p$L run FAILED"
  (cd $d/trace && shopt -s nullglob && f=(*.logits.f32) && echo "${#f[@]}" && { [ ${#f[@]} -gt 0 ] && sha256sum $(printf '%s\n' "${f[@]}" | sort -n) | sha256sum | cut -c1-16 || echo none; }) > $d/logits.sha; rm -rf $d/trace
done; done
for L in 129 136 137 300; do a=$(tr '\n' ' ' < $O/a1/$k-v39-p$L/logits.sha); b=$(tr '\n' ' ' < $O/a1/$k-v40-p$L/logits.sha)
  echo "above 4,096 $k p$L: v39 [$a] v40 [$b] $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFER)$([[ $a == "17 "* && $b == "17 "* ]] || echo ' INCOMPLETE')"; done; done


# rule 2, speech
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
$B/parity-v40/lfm3-speech $M --batch $R3/list > $R3/log 2> $R3/err; echo "rule 3 runs rc=$? $(tail -1 $R3/err)"
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
$B/parity-v40/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "rule 3 native rendering runs rc=$? $(tail -1 $N3/err)"
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
$B/text-v40/lfm3-speech $M --batch $E/text.list > $E/text.log 2> $E/text.err; echo "E2 shipping speech runs rc=$? $(tail -1 $E/text.err)"
python3 $O/same.py $E/text $S1/text $(grep -v text-split $E/text.list | awk '{print $2}' | xargs -n1 basename) | tail -1; echo "E2 shipping speech rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-split $S1/text-split tts-long-male-s1 | tail -1; echo "E2 split rc=${PIPESTATUS[0]}"
python3 -c "import json,sys; print(' '.join(sorted(c['case'] for c in json.load(open(sys.argv[1]))['cases'])))" $V/audio-in-v1/round2-full-check.json > $E/libri-stems.txt
for b in text parity; do mkdir -p $E/libri-$b
  for s in $(cat $E/libri-stems.txt); do echo "$B/sp-dev3/requests/$s.req.json $E/libri-$b/$s"; done > $E/libri-$b.list
  $B/$b-v40/lfm3-speech $M --batch $E/libri-$b.list > $E/libri-$b.log 2> $E/libri-$b.err; echo "E2 LibriSpeech $b runs rc=$? $(tail -1 $E/libri-$b.err)"
  python3 $O/same.py $E/libri-$b $S1/libri-$b $(cat $E/libri-stems.txt) | tail -1; echo "E2 LibriSpeech $b rc=${PIPESTATUS[0]}"
done

# rule 2, paired decode
python3 tools/pascal/paired_decode_check.py $B/text-v40/liquid-lfm3-worker $M /workspace/data/ppl-windows/windows-32768.i32 $O/paired > $O/paired.log 2>&1; echo "paired decode rc=$?"; grep -E 'PAIRED' $O/paired.log

# rule 3: the encoder's stages on HF's pixel values, against HF FP32 and BF16
mkdir -p $O/vision
for c in $CASES; do
  /usr/bin/time -f "%e s" $B/text-v40/lfm3-vision-check $M $REF/float32/$c $O/vision/$c > $O/vision/$c.log 2>&1 || echo "vision-check $c FAILED"
  python3 tools/compare_lfm3_vision_stages.py $O/vision/$c $REF/float32/$c $REF/bfloat16/$c --json $O/vision/$c.compare.json > $O/vision/$c.compare.log 2>&1
done
python3 - $O/vision $I/thor-fp32-compare $CASES <<'PY2'
import gzip, json, sys
from pathlib import Path
o, thor, cases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]; ok = True
for c in cases:
    try: j = json.loads((o / f"{c}.compare.json").read_text())["stages"]["image_features"]
    except Exception as e: print(f"rule 3 {c}: NO RESULT ({e!r})"); ok = False; continue
    t = json.loads(gzip.open(thor / f"{c}.compare.json.gz", "rt").read())["stages"]["image_features"]
    good = j["ratio"] <= 1.0 and j["native_min_cos"] >= j["bf16_min_cos"]; ok &= good
    print(f"rule 3 {c}: error {j['native_rel']:.3e} (HF BF16 {j['bf16_rel']:.3e}, ratio {j['ratio']:.4f}; Thor kFp32 {t['ratio']:.4f}), "
          f"min cosine {j['native_min_cos']:.7f} (HF BF16 {j['bf16_min_cos']:.7f}; Thor {t['native_min_cos']:.7f}): {'ok' if good else 'FAIL'}")
print("RULE 3", "PASS" if ok and len(cases) == 11 else "FAIL")
PY2

# rule 5: the front end on the benchmark images, against HF's record
bash tools/build_image_decode_check.sh $B/img1-front > $O/front-build.log 2>&1; echo "front build rc=$? $(tail -1 $O/front-build.log)"
$B/img1-front/lfm3-image-decode-check --preprocess bench=$BENCH/images > $O/front-native.jsonl 2> $O/front-native.err; echo "front run rc=$? rows $(wc -l < $O/front-native.jsonl)"
zcat $I/front-aarch64.jsonl.gz > $O/front-hf.jsonl
python3 tools/compare_lfm3_image_front.py $O/front-native.jsonl $O/front-hf.jsonl > $O/front-compare.log 2>&1; echo "rule 5 rc=$? $(tail -2 $O/front-compare.log | tr '\n' ' ')"

# rule 6: the 11 conversations through lfm3-speech, then the same through the server
tools/pascal/vision_e2e_round.sh text-v40 $O/e2e > $O/e2e.log 2>&1; echo "e2e round rc=$?"; grep -E '^== ' $O/e2e.log
$PY tools/check_lfm3_image_server.py $B/text-v40/liquid-lfm3-server $M $O/e2e $O/image-server.json > $O/image-server.log 2>&1; echo "rule 6 server rc=$? $(tail -1 $O/image-server.log)"; clean

# rule 4: MMStar and POPE, all 10,500 items
date
$PY tools/lfm3_vision_bench_native.py requests $BENCH $O/bench; echo "bench requests rc=$? $(wc -l < $O/bench/list.txt)"
/usr/bin/time -f "%e s" $B/text-v40/lfm3-speech $M --batch $O/bench/list.txt > $O/bench/run.log 2>&1; echo "bench run rc=$? $(tail -2 $O/bench/run.log | tr '\n' ' ')"
echo "{\"engine\": \"native\", \"build\": \"text-v40\", \"lfm3_speech_sha256\": \"$(sha256sum < $B/text-v40/lfm3-speech | cut -c1-64)\", \"mode\": \"text\", \"device\": \"GTX 1080\", \"batch\": true}" > $O/bench/env.json
$PY tools/lfm3_vision_bench_native.py collect $BENCH $O/bench $O/native.jsonl $O/bench/env.json; echo "collect rc=$?"
zcat $I/hf-bench-run1.jsonl.gz > $O/hf-bf16.jsonl; zcat $I/hf-fp32-mmstar.jsonl.gz > $O/hf-fp32-mmstar.jsonl
python3 tools/score_lfm3_vision_bench.py $O/hf-bf16.jsonl $O/native.jsonl > $O/score.txt 2>&1; echo "score rc=$?"; cat $O/score.txt | grep -v '^env differs'
python3 tools/score_lfm3_vision_fp32_agreement.py $O/hf-fp32-mmstar.jsonl $O/hf-bf16.jsonl $O/native.jsonl --json $O/rule4b.json > $O/rule4b.log 2>&1; echo "fp32 agreement rc=$?"; tail -12 $O/rule4b.log
python3 - $O/score.txt $O/rule4b.json <<'PY2'
import json, re, sys
s = open(sys.argv[1]).read(); b = json.load(open(sys.argv[2]))
m = re.search(r"pope: n=(\d+).*?95% CI \[([-+0-9.]+), ([-+0-9.]+)\]", s, re.S)
pope = m and int(m.group(1)) == 9000 and float(m.group(2)) >= -0.5
mm = b.get("items") == 1500 and b.get("passed") is True
print(f"rule 4 POPE: {m.group(1) if m else '?'} items, lower bound {m.group(2) if m else '?'}: {'ok' if pope else 'FAIL'}; "
      f"MMStar FP32 agreement native {b.get('agreement')} over {b.get('items')} items: {'ok' if mm else 'FAIL'}")
print("RULE 4", "PASS" if pope and mm else "FAIL")
PY2
date

# rule 7: memory at the defaults, with the image configuration
IMAGES=1 $PY tools/pascal/speech_memory.py $B/text-v40/liquid-lfm3-server $M 1 $LIBRI $O/memory-1 7740 29696 > $O/memory-1.log 2>&1
echo "rule 7 rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-1.log; clean

# rule 8: memcheck on the smallest, two-image and largest cases, outputs against the plain run
for c in small two cherry; do
  /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 $B/text-v40/lfm3-vision-check $M $REF/float32/$c $O/memcheck-$c > $O/memcheck-$c.log 2>&1
  rc=$?; same=$(cd $O/vision/$c && for f in *.f32; do cmp -s $f $O/memcheck-$c/$f || echo DIFFER; done | sort -u)
  echo "rule 8 $c rc=$rc $(grep -E 'ERROR SUMMARY' $O/memcheck-$c.log) outputs ${same:-identical}"
done
date
