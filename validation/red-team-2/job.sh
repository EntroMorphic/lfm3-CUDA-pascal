# red-team-2 verification (tools/pascal/remote.sh detach rt2-v1 'bash validation/red-team-2/job.sh'): the final tree,
# builds text-v38 and parity-v38 (tools/pascal/build_speech.sh) and text-v38s (build_serve.sh, the text-only server).
# Rules 2 to 4 of the README: what the fixes reach runs again; what they cannot reach is carried on identical device
# code and byte-identical reruns (decision 11's E1 and E2). Each result line is printed with its step's exit status,
# and a step whose comparison depends on an earlier step says so (red-team-2 C7, D10). The host suite runs on the
# development machine.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-v1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; V=validation/speech-v1/inputs/reference
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; local rc=$?; echo "$n rc=$rc $(grep -E 'PASS|FAIL|INCONCLUSIVE' $O/$n.log | tr '\n' ' ' | cut -c1-1500) $(tail -1 $O/$n.log)"; clean; return $rc; }
cp validation/speech-v1/records/sp-v1/same.py $O/same.py  # the speech-v1 rule run's comparison (outputs without timings)

# Builds
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v38 > $O/build-parity-v38.log 2>&1; echo "build parity-v38 rc=$?"
tools/pascal/build_speech.sh text-v38 > $O/build-text-v38.log 2>&1; echo "build text-v38 rc=$?"
tools/pascal/build_serve.sh text-v38s > $O/build-text-v38s.log 2>&1; echo "build text-v38s rc=$?"
grep -h "warnings" $O/build-*.log
for b in text-v38 parity-v38 text-v35 parity-v35; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  $b/#"; done > $O/binaries.sha256
(cd $B/text-v38s && sha256sum liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server) | sed "s#  #  text-v38s/#" >> $O/binaries.sha256

# Rule 3, E1: every binary's device code against v35's (speech-v1's rule run; v35's CLI and scorer equal text-v33's)
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server; do for k in text parity; do
  cuobjdump -sass $B/$k-v38/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-v35/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.$k.txt; echo "E1 $f $k-v38 against $k-v35 rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done
# the text-only build (build_serve.sh) against serve-v1's final one, text-v34
for f in liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server; do
  cuobjdump -sass $B/text-v38s/$f > $O/e1/a.sass; cuobjdump -sass $B/text-v34/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.text-only.txt; echo "E1 $f text-v38s against text-v34 rc=$? $(head -1 $O/e1/$f.text-only.txt)"
done; rm -f $O/e1/*.sass

# Rule 4: the parity gate; fixtures (compact windows unset and 1; under memcheck)
python3 tools/pascal/compare_pins.py parity-v38 > $O/gate.log 2>&1; echo "gate rc=$? $(tail -1 $O/gate.log)"
tools/pascal/run_text_fixtures.sh text-v38 2 > $O/fixtures.log 2>&1; echo "fixtures rc=$? $(tail -1 $O/fixtures.log)"; cp -r $B/text-v38/fixtures $O/fixtures-unset
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v38 2 > $O/fixtures-compact.log 2>&1; echo "fixtures compact rc=$? $(tail -1 $O/fixtures-compact.log)"; cp -r $B/text-v38/fixtures $O/fixtures-compact
tools/pascal/run_text_fixtures.sh text-v38 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9 > $O/fixtures-memcheck.log 2>&1; echo "fixtures memcheck rc=$? $(tail -1 $O/fixtures-memcheck.log)"

# C1: text-correct-v2 rule 2's compact-windows leg, on text-v6's own binary (7464f580...), in a directory of its own
mkdir -p $B/text-v6-c1 && cp $B/text-v6/liquid-lfm3 $B/text-v6-c1/
[ "$(sha256sum < $B/text-v6-c1/liquid-lfm3 | cut -c1-64)" = 7464f580b1f5ec7eb610e3167abc26b687a490a27d120e25fd230744e1eb45a0 ] && echo "C1 binary is text-v6's" || echo "C1 binary NOT text-v6's"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v6-c1 2 > $O/c1-text-v6-compact.log 2>&1; echo "C1 text-v6 fixtures compact rc=$? $(tail -1 $O/c1-text-v6-compact.log)"

# A1 (information, not a rule): the CLI at capacity 4,097, teacher-forced 16 steps, against v35. Expected: the
# shipping build identical at every prompt (unchanged); the parity build identical at 137 and 300 (no short last tile),
# and at 129 and 136 whatever its v35 gave (its last tile of 1 or 8 rows now takes the decode kernels, the Thor
# engine's; a 1-row tile can round to the same values). Each side must have written 17 logits files.
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
for k in text parity; do for v in v35 v38; do for L in 129 136 137 300; do
  d=$O/a1/$k-$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/$k-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 $k-$v p$L run FAILED"
  (cd $d/trace && shopt -s nullglob && f=(*.logits.f32) && echo "${#f[@]}" && { [ ${#f[@]} -gt 0 ] && sha256sum $(printf '%s\n' "${f[@]}" | sort -n) | sha256sum | cut -c1-16 || echo none; }) > $d/logits.sha; rm -rf $d/trace
done; done
for L in 129 136 137 300; do a=$(tr '\n' ' ' < $O/a1/$k-v35-p$L/logits.sha); b=$(tr '\n' ' ' < $O/a1/$k-v38-p$L/logits.sha)
  echo "A1 (information) $k p$L: v35 [$a] v38 [$b] $([ "$a" = "$b" ] && echo same || echo differ)$([[ $a == "17 "* && $b == "17 "* ]] || echo ' INCOMPLETE')"; done; done

# Scorer (E2): 64 windows at 256 and 257 and 2 at 4,096, v38 against v35, each build (the bar's lengths leave no tile
# of 1 to 8 rows, so A1's scorer change cannot reach them)
P=$B/rt2-ppl-subset; mkdir -p $P
head -c $((64*256*4)) /workspace/data/ppl-windows/windows-256.i32 > $P/windows-256.i32
head -c $((64*257*4)) /workspace/data/ppl-windows/windows-257.i32 > $P/windows-257.i32
head -c $((2*4096*4)) /workspace/data/ppl-windows/windows-4096.i32 > $P/windows-4096.i32
for k in text parity; do for v in v35 v38; do WINDOWS_DIR=$P tools/pascal/score.sh $k-$v rt2-score-$k-$v 256 257 4096 > $O/score-$k-$v.log 2>&1 || echo "score $k-$v FAILED"; done
  for L in 256 257 4096; do cmp $B/rt2-score-$k-v35/nll-pascal-$L.f32 $B/rt2-score-$k-v38/nll-pascal-$L.f32 && echo "scorer $k $L: v38 IDENTICAL to v35" || echo "scorer $k $L: DIFFER"; done; done

# speech-v1 rule 3 on parity-v38: the Thor engine's 16 clips, and the native rendering clause (as speech-v1's job.sh)
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
$B/parity-v38/lfm3-speech $M --batch $R3/list > $R3/log 2> $R3/err; echo "rule 3 runs rc=$? $(tail -1 $R3/err)"
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
$B/parity-v38/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "rule 3 native rendering runs rc=$? $(tail -1 $N3/err)"
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
# E2, speech: the shipping build's ten seed-1 cases and the split lighthouse (sp-v1's E2 runs), and the 19 committed
# LibriSpeech utterances on both builds (sp-v1's), byte for byte without timings
E=$O/e2; mkdir -p $E/text $E/text-split; RT=$B/sp-refs-text-s1; S1=$B/sp-v1/e2
{ for f in $(ls $RT/speech-e2e-v1/requests | grep -- '-s1\.req\.json$'); do echo "$RT/speech-e2e-v1/requests/$f $E/text/${f%.req.json}"; done
  echo "$RT/speech-chunk-native-v1/requests/tts-long-male-s1.req.json $E/text-split/tts-long-male-s1"; } > $E/text.list
$B/text-v38/lfm3-speech $M --batch $E/text.list > $E/text.log 2> $E/text.err; echo "E2 shipping speech runs rc=$? $(tail -1 $E/text.err)"
python3 $O/same.py $E/text $S1/text $(grep -v text-split $E/text.list | awk '{print $2}' | xargs -n1 basename) | tail -1; echo "E2 shipping speech rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-split $S1/text-split tts-long-male-s1 | tail -1; echo "E2 split rc=${PIPESTATUS[0]}"
python3 -c "import json,sys; print(' '.join(sorted(c['case'] for c in json.load(open(sys.argv[1]))['cases'])))" $V/audio-in-v1/round2-full-check.json > $E/libri-stems.txt
for b in text parity; do mkdir -p $E/libri-$b
  for s in $(cat $E/libri-stems.txt); do echo "$B/sp-dev3/requests/$s.req.json $E/libri-$b/$s"; done > $E/libri-$b.list
  $B/$b-v38/lfm3-speech $M --batch $E/libri-$b.list > $E/libri-$b.log 2> $E/libri-$b.err; echo "E2 LibriSpeech $b runs rc=$? $(tail -1 $E/libri-$b.err)"
  python3 $O/same.py $E/libri-$b $S1/libri-$b $(cat $E/libri-stems.txt) | tail -1; echo "E2 LibriSpeech $b rc=${PIPESTATUS[0]}"
done

# speech-v1 rule 12 at the defaults (ready lines at 1, 8 and 16 rows; memory with the full clip)
for r in 1 8 16; do $PY - $B/text-v38/liquid-lfm3-server $M $O $r <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out, rows = sys.argv[1:5]; tok = os.path.join(out, f"ready{rows}.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0", "--speech-rows", rows],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
want = {"1": 29696, "8": 15360, "16": 1024}[rows]
r = json.loads(line); print(f"rule 12 ready line at {rows} rows:", line.strip(), "exit", rc,
      "ok" if (r.get("total_capacity"), r.get("speech_rows"), rc) == (want, int(rows), 0) else "FAIL")
PY
clean; done
for rt in 1:29696 8:15360 16:1024; do r=${rt%%:*}; t=${rt##*:}
  $PY tools/pascal/speech_memory.py $B/text-v38/liquid-lfm3-server $M $r $LIBRI $O/memory-$r 7740 $t > $O/memory-$r.log 2>&1
  echo "rule 12 rows $r total $t rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-$r.log; clean
done

# B1: the batch bound, measured on both shipping servers (the speech server at 29,696; the text-only one at 32,768)
run batch-speech-server $PY tools/pascal/batch_budget.py $B/text-v38/liquid-lfm3-server $M $O/batch-speech 29696
run batch-text-server $PY tools/pascal/batch_budget.py $B/text-v38s/liquid-lfm3-server $M $O/batch-text 32768

# speech-v1 rule 13, both builds (with the 64-clip control)
for b in text parity; do $PY tools/pascal/speech_limits.py $B/$b-v38/liquid-lfm3-server $M $O/limits-$b > $O/limits-$b.log 2>&1
  echo "speech-v1 rule 13 $b rc=$?"; grep -E '^rule 13|SPEECH LIMITS' $O/limits-$b.log; clean; done

# speech-v1 rule 2 / serve-v1 rules 3 to 9: every stage, both builds; serve-v1 rule 10 (paired decode)
tools/pascal/serve_gates.sh text-v38 rt2-serve-shipping own > $O/serve-shipping.log 2>&1; echo "serve gates shipping rc=$?"; cat $B/rt2-serve-shipping/status.txt; clean
tools/pascal/serve_gates.sh parity-v38 rt2-serve-parity pinned > $O/serve-parity.log 2>&1; echo "serve gates parity rc=$?"; cat $B/rt2-serve-parity/status.txt; clean
tools/pascal/serve_gates.sh text-v38s rt2-serve-text-only own > $O/serve-text-only.log 2>&1; echo "serve gates text-only rc=$?"; cat $B/rt2-serve-text-only/status.txt; clean
python3 tools/pascal/paired_decode_check.py $B/text-v38/liquid-lfm3-worker $M /workspace/data/ppl-windows/windows-32768.i32 $O/paired > $O/paired.log 2>&1; echo "paired decode rc=$?"; grep -E 'PAIRED' $O/paired.log

# speech-v1 rule 11, all five checks, both builds, on these servers (D2), judged by decision 10
for b in text parity; do S=$B/$b-v38; R=$B/sp-refs-$b-s1; date
  run $b-speech-server $PY tools/pascal/check_lfm3_speech_server.py $S/liquid-lfm3-server $M $R $O/$b-speech-server.json
  run $b-speech-stream $PY tools/pascal/check_lfm3_speech_stream.py $S/liquid-lfm3-server $M $R $LIBRI $O/$b-speech-stream.json
  run $b-audio-in-server $PY tools/pascal/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/$b-audio-in-work $O/$b-audio-in-server.json
  run $b-speech-concurrent $PY tools/check_lfm3_speech_concurrent.py $S/lfm3-speech $M $R $LIBRI $O/$b-concurrent-work $O/$b-speech-concurrent.json
  run $b-batch-server $PY tools/pascal/check_lfm3_speech_batch_server.py $S/liquid-lfm3-server $S/liquid-lfm3-server $M $R $LIBRI $O/$b-batch-server.json
  # the references were made by the development build s1's lfm3-speech, whose outputs v38's equal (E2 above)
  python3 tools/pascal/judge_speech_server.py $O $b $B/$b-v38 $R $B/$b-s1; echo "speech-v1 rule 11 $b rc=$?"
done

# C10: prefill-arithmetic-v1's position-1 tables from the variants' NLL files on the card
$PY tools/pascal/analysis/arith_report.py $B/arith-v1 > $O/arith-position1.txt 2>&1; echo "arith_report rc=$?"
cmp $O/arith-position1.txt validation/prefill-arithmetic-v1/arith-position1.txt && echo "arith-position1.txt: equal to the committed one" || echo "arith-position1.txt: DIFFERS"
date
