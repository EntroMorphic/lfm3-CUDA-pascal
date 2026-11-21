# speech-v1 development 11 (tools/pascal/remote.sh detach speech-dev11 'bash validation/speech-v1/dev11.sh'). Development
# 10: one speech row passes rule 12 at a total capacity of 31,744 (every configuration; the owner kept rule 12 as
# written). The server now defaults to one speech row and, when it serves speech, to a total capacity of 31,744, and its
# ready line reports both (lfm3_server.cu; host code only). Here: both builds again (s4); lfm3-speech against s3's,
# byte for byte; the ready line with the defaults; then rule 11's five server checks on the shipping build, then on the
# parity build (tools/pascal/ adaptations for the audio-in and batch-server checks), against the CLI runs of
# development 4 (sp-refs-*-s1: lfm3-speech outputs identical to s3's, development 9).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev11; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s4 > $O/build-parity-s4.log 2>&1; echo "build parity-s4 rc=$?"
tools/pascal/build_speech.sh text-s4 > $O/build-text-s4.log 2>&1; echo "build text-s4 rc=$?"
for b in text parity; do cmp $B/$b-s3/lfm3-speech $B/$b-s4/lfm3-speech && echo "$b-s4 lfm3-speech identical to $b-s3's"; done
$PY - $B/text-s4/liquid-lfm3-server $M $O <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out = sys.argv[1:4]; tok = os.path.join(out, "ready.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
r = json.loads(line); print("ready line (defaults):", line.strip(), "exit", rc,
      "ok" if (r.get("total_capacity"), r.get("speech_rows"), rc) == (31744, 1, 0) else "FAIL")
PY
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; echo "$n rc=$? $(grep -E 'PASS|FAIL' $O/$n.log | tr '\n' ' ' | cut -c1-1500) $(tail -1 $O/$n.log)"; }
for b in text parity; do
  S=$B/$b-s4; R=$B/sp-refs-$b-s1; date
  run $b-speech-server $PY tools/check_lfm3_speech_server.py $S/liquid-lfm3-server $M $R $O/$b-speech-server.json
  run $b-speech-stream $PY tools/check_lfm3_speech_stream.py $S/liquid-lfm3-server $M $R $LIBRI $O/$b-speech-stream.json
  run $b-audio-in-server $PY tools/pascal/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/$b-audio-in-work $O/$b-audio-in-server.json
  run $b-speech-concurrent $PY tools/check_lfm3_speech_concurrent.py $S/lfm3-speech $M $R $LIBRI $O/$b-concurrent-work $O/$b-speech-concurrent.json
  run $b-batch-server $PY tools/pascal/check_lfm3_speech_batch_server.py $S/liquid-lfm3-server $S/liquid-lfm3-server $M $R $LIBRI $O/$b-batch-server.json
done
date
