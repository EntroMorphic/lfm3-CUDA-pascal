# speech-v1 development 15 (tools/pascal/remote.sh detach speech-dev15 'bash validation/speech-v1/dev15.sh'). Development
# 14: with audio in at its limit beside the text session at its limit (the red-team review's configuration), one speech
# row peaks at 7,781 MiB at a total capacity of 31,744 (limit 7,740); 8 rows at 17,408 pass; the 16-row run stopped in
# speech_memory.py's own calibration (a 4,096-token session over a total of 1,024; fixed). Rule 12: the next multiple of
# 1,024 down. The server (s6) defaults to 30,720 at one row, 2,048 less for each further row, at least 2,048. Here:
# text-s6 and parity-s6; the ready line at 1, 8 and 16 rows; rule 12 at one row at 30,720 (and 29,696, should it fail),
# 8 rows at 16,384 and 16 rows at 2,048.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev15; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s6 > $O/build-parity-s6.log 2>&1; echo "build parity-s6 rc=$?"
tools/pascal/build_speech.sh text-s6 > $O/build-text-s6.log 2>&1; echo "build text-s6 rc=$?"
for rows in 1 8 16; do $PY - $B/text-s6/liquid-lfm3-server $M $O $rows <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out, rows = sys.argv[1:5]; tok = os.path.join(out, "ready.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0", "--speech-rows", rows], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
print(f"ready line at {rows} rows:", line.strip(), "exit", rc)
PY
done
m() { $PY tools/pascal/speech_memory.py $B/text-s6/liquid-lfm3-server $M $1 $LIBRI $O/memory-rows$1-$2 7740 $2 > $O/memory-rows$1-$2.log 2>&1; local rc=$?
  echo "rule 12 rows $1 total $2 rc=$rc"; grep -E '^rule 12|^SPEECH' $O/memory-rows$1-$2.log | cut -c1-400; clean; return $rc; }
m 1 30720 || m 1 29696
m 8 16384; m 16 2048
date
