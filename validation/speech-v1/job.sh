# speech-v1, the rule run (tools/pascal/remote.sh detach speech-v1 'bash validation/speech-v1/job.sh'): phase 4's
# final tree; builds text-v35 (shipping) and parity-v35 (parity configuration), each with the CLI, the scorer,
# lfm3-speech and the server with speech (tools/pascal/build_speech.sh). Decision 11 (README): a rule whose inputs no
# change since a development run could reach carries that run's result, on two pieces of evidence made here: (E1) the
# final binaries' device code equals the development builds' function for function, and the host sources that differ
# are listed; (E2) the final binaries, run again on a sample of those results' inputs, give byte-identical outputs.
# Everything a change could reach runs again. Rule 9 is the owner's (here only the final shipping clips' hashes; the
# key is checked on the development machine, as is the host suite).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-v1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; W=/workspace/models/whisper-large-v3; V=validation/speech-v1/inputs/reference
LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; echo "$n rc=$? $(grep -E 'PASS|FAIL' $O/$n.log | tr '\n' ' ' | cut -c1-1500) $(tail -1 $O/$n.log)"; clean; }
# same.py DIR_A DIR_B [STEM...]: every output file of each stem byte-identical (the .json without its timings)
cat > $O/same.py <<'PY'
import json, sys
from pathlib import Path
T = {"encode_seconds", "prefill_seconds", "generate_seconds", "lm_step_seconds", "text_pick_seconds", "depth_seconds",
     "sample_host_seconds", "codec_seconds", "real_time_factor", "real_time_factor_end_to_end"}
a, b, stems = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
same = 0
for s in stems:
    fa, fb = sorted(p.name for p in a.glob(s + ".*")), sorted(p.name for p in b.glob(s + ".*"))
    ok = bool(fa) and fa == fb
    for f in fa if ok else []:
        if f.endswith(".json"):
            ja, jb = [{k: v for k, v in json.loads((d / f).read_text()).items() if k not in T} for d in (a, b)]; ok &= ja == jb
        else: ok &= (a / f).read_bytes() == (b / f).read_bytes()
    same += ok
    if not ok: print(f"DIFFERS {s} ({len(fa)} and {len(fb)} files)")
print(f"{same} of {len(stems)} identical ({a} against {b})"); sys.exit(0 if same == len(stems) and stems else 1)
PY

# Builds
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v35 > $O/build-parity-v35.log 2>&1; echo "build parity-v35 rc=$?"
tools/pascal/build_speech.sh text-v35 > $O/build-text-v35.log 2>&1; echo "build text-v35 rc=$?"
for b in text-v35 parity-v35 text-s4 parity-s4 text-s1 parity-s1; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score lfm3-speech $(ls liquid-lfm3-worker liquid-lfm3-server 2>/dev/null)) | sed "s#  #  $b/#"; done > $O/binaries.sha256

# E1: device code of every binary against the development builds whose results are carried (s1: rules 4, 7, 8, 10 and
# the references; s4: rules 2, 11, 14), function for function; the host sources that differ, by the builds' hashes
mkdir -p $O/e1
for f in lfm3-speech liquid-lfm3-server liquid-lfm3-worker; do for k in text parity; do for d in s1 s4; do
  [ -e $B/$k-$d/$f ] || continue
  cuobjdump -sass $B/$k-v35/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-$d/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.$k-v35.$k-$d.txt; echo "E1 $f $k-v35 against $k-$d rc=$? $(head -1 $O/e1/$f.$k-v35.$k-$d.txt)"
done; done; done; rm -f $O/e1/*.sass
for d in s1 s4; do diff <(cut -d' ' -f1,3 $B/text-$d/speech-sources.sha256 | sort -k2) <(cut -d' ' -f1,3 $B/text-v35/speech-sources.sha256 | sort -k2) | grep '^>' | sed "s#^> [0-9a-f]* #E1 host source differing from $d: #"; done | tee $O/e1/sources.txt

# Rule 1: the parity gate; the text CLI's and scorer's device code against text-v33; fixtures
python3 tools/pascal/compare_pins.py parity-v35 > $O/gate.log 2>&1; echo "rule 1 gate rc=$?"; tail -1 $O/gate.log
for f in liquid-lfm3 lfm3-score; do cuobjdump -sass $B/text-v33/$f > $O/$f.v33.sass; cuobjdump -sass $B/text-v35/$f > $O/$f.v35.sass
  python3 validation/speed-v1/sass_compare.py $O/$f.v33.sass $O/$f.v35.sass > $O/$f.sass-compare.txt; echo "rule 1 $f device code against text-v33 rc=$? $(head -1 $O/$f.sass-compare.txt)"; done
tools/pascal/run_text_fixtures.sh text-v35 2 > $O/fixtures.log 2>&1; echo "rule 1 fixtures rc=$?"; tail -1 $O/fixtures.log
# Rule 5: lfm3-speech's device code, the two builds, outside lfm::gemm
for b in text parity; do cuobjdump -sass $B/$b-v35/lfm3-speech > $O/speech.$b.sass; done
python3 tools/pascal/sass_speech_only.py $O/speech.text.sass $O/speech.parity.sass > $O/rule5.txt; echo "rule 5 rc=$?"; head -1 $O/rule5.txt
sha256sum $O/*.sass > $O/sass.sha256; gzip -9 $O/*.sass

# Rule 3: the parity build on the Thor engine's 15 clips (the reference's rendered ids, by SHA-256), and the rendering
# clause (the same requests from their messages, rendered natively: the same prompt ids and WAVs)
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
$B/parity-v35/lfm3-speech $M --batch $R3/list > $R3/log 2> $R3/err; echo "rule 3 runs rc=$? $(tail -1 $R3/err)"
python3 - $R3/out validation/speech-v1/inputs/thor-clips.json > $O/rule3.txt <<'PY'
import hashlib, json, sys
from pathlib import Path
out, clips = Path(sys.argv[1]), json.load(open(sys.argv[2]))
same = 0
for c in clips:
    wav = out / f"{c['run']}.wav"
    ok = wav.exists() and hashlib.sha256(wav.read_bytes()).hexdigest() == c["sha256"]; same += ok
    print(f"rule 3 {c['run']} ({c['set']}): {'equal' if ok else 'DIFFERS'}")
runs = {c["run"] for c in clips}  # 16 entries, 15 runs: tts-long-male-s1 is in both listening sets
print(f"rule 3: {same} of {len(clips)} clips ({len(runs)} runs) equal to the Thor engine's"); sys.exit(0 if same == len(clips) == 16 and len(runs) == 15 else 1)
PY
echo "rule 3 rc=$?"; tail -1 $O/rule3.txt
N3=$O/rule3-native; mkdir -p $N3/requests $N3/out
python3 - $RP validation/speech-v1/inputs/thor-clips.json $N3 <<'PY'
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
PY
$B/parity-v35/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "rule 3 native rendering runs rc=$? $(tail -1 $N3/err)"
python3 - $R3/out $N3 $RP > $O/rule3-native.txt <<'PY'
import json, sys
from pathlib import Path
ids_run, n3, refs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); same = 0
runs = sorted(p.name[:-9] for p in (n3 / "requests").glob("*.req.json"))
for run in runs:
    split = run.startswith(("tts-long-male", "probe-other-long"))
    r = json.loads((refs / ("speech-chunk-native-v1" if split else "speech-e2e-v1") / "requests" / f"{run}.req.json").read_text())
    L = lambda v: json.loads(v) if isinstance(v, str) else v
    got, want = (json.loads((d / f"{run}.json").read_text()) for d in (n3 / "out", ids_run))
    # the json's prompt is what ran (the first chunk's, with the mode's appended token): equal to the run from the
    # reference's ids, with the same number of chunks; then the WAV
    ok = got.get("prompt") == want.get("prompt") and len(got.get("chunks", [])) == len(want.get("chunks", [])) and \
        (not split or len(got["chunks"]) == len(L(r["chunks"]))) and \
        (n3 / "out" / f"{run}.wav").read_bytes() == (ids_run / f"{run}.wav").read_bytes()
    same += ok; print(f"rule 3 rendering {run}: {'same ids and WAV' if ok else 'DIFFERS'}")
print(f"rule 3 rendering: {same} of {len(runs)} equal"); sys.exit(0 if same == len(runs) == 15 else 1)
PY
echo "rule 3 rendering rc=$?"; tail -1 $O/rule3-native.txt

# E2: the final lfm3-speech on the inputs of the carried results, byte for byte
#  shipping: the 30 runs and the 3 split lighthouse runs of rules 7 and 8 (sp-refs-text-s1, development 4), and the 24
#  escalation runs (development 6); both builds: the 19 committed LibriSpeech utterances and every 32nd other (rules 4
#  and 10, development 3); the parity build's 15 clips are rule 3's above (equal to s1's, which equal the Thor engine's)
E=$O/e2; mkdir -p $E; RT=$B/sp-refs-text-s1; X6=$B/sp-dev6/escalate; D3=$B/sp-dev3
{ for f in $(ls $RT/speech-e2e-v1/requests); do echo "$RT/speech-e2e-v1/requests/$f $E/text/${f%.req.json}"; done
  for s in 1 2 3; do echo "$RT/speech-chunk-native-v1/requests/tts-long-male-s$s.req.json $E/text-split/tts-long-male-s$s"; done
  for s in $(seq 4 15); do echo "$X6/long/requests/tts-numbers-female-s$s.req.json $E/text-escalate-numbers/tts-numbers-female-s$s"
    echo "$RT/speech-chunk-native-v1/requests/tts-long-male-s$s.req.json $E/text-escalate-split/tts-long-male-s$s"; done; } > $E/text.list
mkdir -p $E/text $E/text-split $E/text-escalate-numbers $E/text-escalate-split
$B/text-v35/lfm3-speech $M --batch $E/text.list > $E/text.log 2> $E/text.err; echo "E2 shipping runs rc=$? $(tail -1 $E/text.err)"
python3 $O/same.py $E/text $RT/speech-e2e-v1/native $(ls $RT/speech-e2e-v1/requests | sed 's/\.req\.json$//') | tail -2; echo "E2 rules 7-8 runs rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-split $RT/speech-chunk-native-v1/native tts-long-male-s1 tts-long-male-s2 tts-long-male-s3 | tail -1; echo "E2 split runs rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-escalate-numbers $X6/numbers $(seq -f tts-numbers-female-s%g 4 15) | tail -1; echo "E2 escalation numbers rc=${PIPESTATUS[0]}"
python3 $O/same.py $E/text-escalate-split $X6/split $(seq -f tts-long-male-s%g 4 15) | tail -1; echo "E2 escalation split rc=${PIPESTATUS[0]}"
python3 - $V/audio-in-v1/round2-full-check.json $D3/requests > $E/libri-stems.txt <<'PY'
import json, sys
from pathlib import Path
committed = sorted(c["case"] for c in json.load(open(sys.argv[1]))["cases"])
others = [s for s in sorted(p.name[:-9] for p in Path(sys.argv[2]).glob("*.req.json")) if s not in committed]
print(" ".join(committed + others[::32]))
PY
for b in text parity; do mkdir -p $E/libri-$b
  for s in $(cat $E/libri-stems.txt); do echo "$D3/requests/$s.req.json $E/libri-$b/$s"; done > $E/libri-$b.list
  $B/$b-v35/lfm3-speech $M --batch $E/libri-$b.list > $E/libri-$b.log 2> $E/libri-$b.err; echo "E2 LibriSpeech $b runs rc=$? $(tail -1 $E/libri-$b.err)"
  python3 $O/same.py $E/libri-$b $D3/$b-s1 $(cat $E/libri-stems.txt) | tail -2; echo "E2 LibriSpeech $b rc=${PIPESTATUS[0]}"
done

# Rule 6: the scorer reproduces the committed transcripts (the reference's HF and Thor seed-1 clips)
C=/workspace/data/speech-calibration
for k in hf engine; do $PY tools/lfm3_asr.py $W $O/calibration-asr-$k.json $C/$k/* > $O/calibration-asr-$k.log 2>&1; echo "rule 6 asr $k rc=$?"; done
python3 tools/pascal/asr_calibration.py compare $C $O/calibration-asr-hf.json $O/calibration-asr-engine.json $V $O/calibration.json > $O/rule6.log 2>&1; echo "rule 6 rc=$?"; tail -1 $O/rule6.log

# Rule 9: the final shipping build's seed-1 clips, for the key's hashes (checked on the development machine)
(cd $E && sha256sum text/*-s1.wav text-split/tts-long-male-s1.wav) > $O/rule9-shipping-clips.sha256

# Rule 11: the two checks whose card copies changed (audio-in: each case's mode; batch-server: every server's exit
# code), each build, against its development references (sp-refs-*-s1; E2); the other three carried (decision 11)
for b in text parity; do S=$B/$b-v35; R=$B/sp-refs-$b-s1
  run $b-audio-in-server $PY tools/pascal/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/$b-audio-in-work $O/$b-audio-in-server.json
  run $b-batch-server $PY tools/pascal/check_lfm3_speech_batch_server.py $S/liquid-lfm3-server $S/liquid-lfm3-server $M $R $LIBRI $O/$b-batch-server.json
done
cp $B/sp-dev12/text-speech-server.json $B/sp-dev11/text-speech-stream.json $B/sp-dev11/text-speech-concurrent.json \
   $B/sp-dev12/parity-speech-server.json $B/sp-dev12/parity-speech-stream.json $B/sp-dev12/parity-speech-concurrent.json $O/
for b in text parity; do python3 tools/pascal/judge_speech_server.py $O $b $B/$b-v35,$B/$b-s4,$B/$b-s1 $B/sp-refs-$b-s1; echo "rule 11 $b rc=$?"; done

# Rule 12: the server's ready line with its defaults, and memory at the limits (one speech row, total 30,720)
$PY - $B/text-v35/liquid-lfm3-server $M $O <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out = sys.argv[1:4]; tok = os.path.join(out, "ready.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
r = json.loads(line); print("rule 12 ready line (defaults):", line.strip(), "exit", rc,
      "ok" if (r.get("total_capacity"), r.get("speech_rows"), rc) == (30720, 1, 0) else "FAIL")
PY
$PY tools/pascal/speech_memory.py $B/text-v35/liquid-lfm3-server $M 1 $LIBRI $O/memory 7740 30720 > $O/memory.log 2>&1; echo "rule 12 rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory.log; clean
# Rule 13: past the limits, both servers
for b in text parity; do $PY tools/pascal/speech_limits.py $B/$b-v35/liquid-lfm3-server $M $O/limits-$b > $O/limits-$b.log 2>&1
  echo "rule 13 $b rc=$?"; grep -E '^rule 13|SPEECH LIMITS' $O/limits-$b.log; clean; done
# Rule 2: serve-v1's native-server stage (the only one on liquid-lfm3-server, whose host code changed) on both builds;
# the other stages and serve-v1's rule 10 run the worker and the CLI, unchanged since development 13 (E1), and are carried
STAGES=native-server tools/pascal/serve_gates.sh text-v35 sp-v1-serve-shipping own > $O/serve-shipping.log 2>&1; echo "rule 2 shipping native-server rc=$?"; cat $B/sp-v1-serve-shipping/status.txt; clean
STAGES=native-server tools/pascal/serve_gates.sh parity-v35 sp-v1-serve-parity pinned > $O/serve-parity.log 2>&1; echo "rule 2 parity native-server rc=$?"; cat $B/sp-v1-serve-parity/status.txt; clean

# Carried (decision 11), their records copied here: rules 4 and 10 (development 3), 7 and 8 (developments 5 and 6),
# 14 (development 13), rule 2's other stages and serve-v1's rule 10 (development 13), 12 at 8 and 16 rows (development 15)
K=$O/carried; mkdir -p $K
cp $D3/rule4.json $D3/rule10.json $D3/rule4.log $D3/rule10.log $K/
cp $B/sp-dev5/c1c2.json $B/sp-dev5/asr-e2e.json $B/sp-dev5/asr-split.json $X6/numbers.json $X6/long.json $X6/asr-numbers.json $X6/asr-split.json $K/
mkdir -p $K/rule14 && cp -r $B/sp-dev13/m-text $B/sp-dev13/m-parity $K/rule14/ && find $K/rule14 \( -name "*.wav" -o -name "*.f32" -o -name "*.i32" \) -delete
cp -r $B/sp-dev13-serve-shipping $K/serve-shipping-dev13; cp -r $B/sp-dev13-serve-parity $K/serve-parity-dev13; cp $B/sp-dev13/paired.log $K/
cp -r $B/sp-dev15/memory-rows8-16384 $B/sp-dev15/memory-rows16-2048 $B/sp-dev15/memory-rows8-16384.log $B/sp-dev15/memory-rows16-2048.log $K/
grep -h "^SPEECH MEMORY\|PAIRED\|ERROR SUMMARY" $K/*.log $K/rule14/*/memcheck.* 2>/dev/null | sort | uniq -c

# Recorded, not judged: the real-time factor of each case through the CLI (the shipping build's E2 runs; the parity
# build's development 4 runs); the server's in rule 11's reports
for d in $E/text $B/sp-refs-parity-s1/speech-e2e-v1/native; do python3 - $d <<'PY'
import json, statistics, sys
from pathlib import Path
per = {}
for j in sorted(Path(sys.argv[1]).glob("*-s[0-9]*.json")):
    case = j.stem.rsplit("-s", 1)[0]; per.setdefault(case, []).append(json.loads(j.read_text())["real_time_factor"])
for case, v in per.items(): print(f"speed {sys.argv[1]} CLI {case}: real-time factor {statistics.mean(v):.2f} (seeds {len(v)})")
PY
done
date
