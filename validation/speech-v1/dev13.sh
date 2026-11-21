# speech-v1 development 13 (tools/pascal/remote.sh detach speech-dev13 'bash validation/speech-v1/dev13.sh'; it waits
# for development 12 to finish). The rules not yet run, on s4 (the tree of developments 11 and 12): rule 1 (the parity
# gate, the text CLI's and scorer's device code against text-v33, fixtures), rule 2 (serve-v1 rules 3 to 9 on both
# builds' servers with speech, and rule 10), rule 5 (tools/pascal/sass_speech_only.py), rule 13
# (tools/pascal/speech_limits.py on both servers) and rule 14 (memcheck on lfm3-speech: one TTS, one interleaved, one
# audio in, both builds, against plain runs of the same requests).
until [ -e /workspace/build/jobs/speech-dev12.rc ]; do sleep 60; done
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev13; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; SAN="/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9"
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
# Rule 1
python3 tools/pascal/compare_pins.py parity-s4 > $O/gate.log 2>&1; echo "rule 1 gate rc=$?"; tail -3 $O/gate.log
for f in liquid-lfm3 lfm3-score; do cuobjdump -sass $B/text-v33/$f > $O/$f.v33.sass; cuobjdump -sass $B/text-s4/$f > $O/$f.s4.sass
  python3 validation/speed-v1/sass_compare.py $O/$f.v33.sass $O/$f.s4.sass > $O/$f.sass-compare.txt; echo "rule 1 $f device code against text-v33 rc=$? $(head -1 $O/$f.sass-compare.txt)"; done
tools/pascal/run_text_fixtures.sh text-s4 2 > $O/fixtures.log 2>&1; echo "rule 1 fixtures rc=$?"; tail -2 $O/fixtures.log
# Rule 5
for b in text parity; do cuobjdump -sass $B/$b-s4/lfm3-speech > $O/speech.$b.sass; done
python3 tools/pascal/sass_speech_only.py $O/speech.text.sass $O/speech.parity.sass > $O/rule5.txt; echo "rule 5 rc=$?"; head -3 $O/rule5.txt
sha256sum $O/*.sass > $O/sass.sha256; gzip -9 $O/*.sass
# Rule 13
for b in text parity; do $PY tools/pascal/speech_limits.py $B/$b-s4/liquid-lfm3-server $M $O/limits-$b > $O/limits-$b.log 2>&1
  echo "rule 13 $b rc=$?"; grep -E '^rule 13|SPEECH LIMITS' $O/limits-$b.log; clean; done
# Rule 14: memcheck. The requests: TTS (fox, female, seed 1), interleaved (the fifth spoken question: audio in,
# interleaved out), audio in (ASR of a LibriSpeech utterance, text out).
W=$B/sp-dev11/text-concurrent-work/requests
for b in text parity; do
  for k in plain memcheck; do mkdir -p $O/m-$b/$k
    printf '%s %s\n' $B/sp-refs-$b-s1/speech-e2e-v1/requests/tts-fox-female-s1.req.json $O/m-$b/$k/tts-fox-female-s1 \
      $W/spoken-q4.req.json $O/m-$b/$k/spoken-q4 $W/asr-121-123859-0002.req.json $O/m-$b/$k/asr-121-123859-0002 > $O/m-$b/$k.list
    w=""; [ $k = memcheck ] && w=$SAN
    /usr/bin/time -f "%e s" $w $B/$b-s4/lfm3-speech $M --batch $O/m-$b/$k.list > $O/m-$b/$k.log 2> $O/m-$b/$k.err; echo "rule 14 $b $k rc=$? $(tail -1 $O/m-$b/$k.err)"
  done
  grep -E "ERROR SUMMARY|errors" $O/m-$b/memcheck.log $O/m-$b/memcheck.err | tail -2
  python3 - $O/m-$b <<'PY'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); T = {"encode_seconds", "prefill_seconds", "generate_seconds", "lm_step_seconds", "text_pick_seconds",
     "depth_seconds", "sample_host_seconds", "codec_seconds", "real_time_factor", "real_time_factor_end_to_end"}
same = 0
for n in ("tts-fox-female-s1", "spoken-q4", "asr-121-123859-0002"):
    ok = True
    for ext in (".wav", ".f32", ".frames.i32", ".raw.i32"):
        a, b = d / "plain" / (n + ext), d / "memcheck" / (n + ext)
        ok &= a.exists() == b.exists() and (not a.exists() or a.read_bytes() == b.read_bytes())
    ja, jb = [{k: v for k, v in json.loads((d / s / (n + ".json")).read_text()).items() if k not in T} for s in ("plain", "memcheck")]
    ok &= ja == jb; same += ok; print(f"rule 14 {d.name} {n}: outputs {'identical' if ok else 'DIFFER'}")
print(f"rule 14 {d.name}: {same} of 3 identical")
PY
done
# Rule 2: serve-v1 rules 3 to 9 on the servers with speech, then rule 10
tools/pascal/serve_gates.sh text-s4 sp-dev13-serve-shipping own > $O/serve-shipping.log 2>&1; echo "rule 2 shipping rc=$?"; cat $B/sp-dev13-serve-shipping/status.txt; clean
tools/pascal/serve_gates.sh parity-s4 sp-dev13-serve-parity pinned > $O/serve-parity.log 2>&1; echo "rule 2 parity rc=$?"; cat $B/sp-dev13-serve-parity/status.txt; clean
python3 tools/pascal/paired_decode_check.py $B/text-s4/liquid-lfm3-worker $M /workspace/data/ppl-windows/windows-32768.i32 $O/paired > $O/paired.log 2>&1; echo "rule 2 (serve-v1 rule 10) rc=$?"; grep -E 'lane|PAIRED' $O/paired.log
date
