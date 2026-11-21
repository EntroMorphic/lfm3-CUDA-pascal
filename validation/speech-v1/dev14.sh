# speech-v1 development 14 (tools/pascal/remote.sh detach speech-dev14 'bash validation/speech-v1/dev14.sh'): after the
# red-team review. The server (s5) scales its default total capacity down by 2,048 tokens for each speech row past the
# first; speech_memory.py now sends audio in at its limit while the text session at its limit is open and filled;
# speech_limits.py's text case is within every other limit; job.sh's rule 3 rendering clause. Here: build text-s5 and
# parity-s5; the ready line at 1, 8 and 16 rows; rule 12 at one row (31,744), 8 rows (17,408) and 16 rows (1,024);
# rule 13 on text-s5; the rendering clause on parity-s5 (against development 4's parity references, whose lfm3-speech
# outputs equal s5's: device code unchanged since s1).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev14; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s5 > $O/build-parity-s5.log 2>&1; echo "build parity-s5 rc=$?"
tools/pascal/build_speech.sh text-s5 > $O/build-text-s5.log 2>&1; echo "build text-s5 rc=$?"
for b in text parity; do echo "$b lfm3-speech s4/s5 bytes differing: $(cmp -l $B/$b-s4/lfm3-speech $B/$b-s5/lfm3-speech | wc -l)"; done
for rows in 1 8 16; do $PY - $B/text-s5/liquid-lfm3-server $M $O $rows <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out, rows = sys.argv[1:5]; tok = os.path.join(out, "ready.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0", "--speech-rows", rows], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
print(f"ready line at {rows} rows:", line.strip(), "exit", rc)
PY
done
for rt in "1 31744" "8 17408" "16 1024"; do set -- $rt
  $PY tools/pascal/speech_memory.py $B/text-s5/liquid-lfm3-server $M $1 $LIBRI $O/memory-rows$1 7740 $2 > $O/memory-rows$1.log 2>&1
  echo "rule 12 rows $1 total $2 rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-rows$1.log | cut -c1-400; clean
done
$PY tools/pascal/speech_limits.py $B/text-s5/liquid-lfm3-server $M $O/limits-text > $O/limits-text.log 2>&1; echo "rule 13 text rc=$?"; grep -E '^rule 13|SPEECH LIMITS' $O/limits-text.log; clean
# the rendering clause, as job.sh runs it (its two Python steps), on parity-s5 against sp-refs-parity-s1
N3=$O/rule3-native; mkdir -p $N3/requests $N3/out; R=$B/sp-refs-parity-s1
sed -n '/^python3 - \$B\/sp-v1-refs-parity validation\/speech-v1\/inputs\/thor-clips.json \$N3 <<.PY.$/,/^PY$/p' validation/speech-v1/job.sh | sed '1d;$d' > $O/render.py
sed -n '/^python3 - \$B\/sp-v1-refs-parity \$N3 > \$O\/rule3-native.txt <<.PY.$/,/^PY$/p' validation/speech-v1/job.sh | sed '1d;$d' > $O/compare.py
python3 $O/render.py $R validation/speech-v1/inputs/thor-clips.json $N3; echo "render rc=$?"
$B/parity-s5/lfm3-speech $M --batch $N3/list > $N3/log 2> $N3/err; echo "rendering runs rc=$? $(tail -1 $N3/err)"
python3 $O/compare.py $R $N3; echo "rule 3 rendering rc=$?"
date
