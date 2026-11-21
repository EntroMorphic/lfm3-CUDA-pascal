# speech-v1 development 4 (tools/pascal/remote.sh detach speech-dev4 'bash validation/speech-v1/dev4.sh'): each build's
# validated runs (tools/pascal/speech_refs.sh: the server checks' BUILD_DIR, and rule 7's seeds 1-3), then the
# reference's five speech server checks on the shipping build (text-s1), each against its own CLI. The batch check's
# OLD_SERVER (on Thor an earlier, single-request binary) is the same server serving one speech request at a time
# (--speech-rows 1, a wrapper script). Memory is sampled throughout (nvidia-smi, every second).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev4; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 1000 > $O/memory.csv & SAMPLER=$!
for b in text-s1 parity-s1; do /usr/bin/time -f "%e s" tools/pascal/speech_refs.sh $b sp-refs-$b > $O/refs-$b.log 2>&1; echo "refs $b rc=$? $(tail -2 $O/refs-$b.log | tr '\n' ' ')"; done
S=$B/text-s1; R=$B/sp-refs-text-s1
printf '#!/bin/sh\nexec %s "$@" --speech-rows 1\n' $S/liquid-lfm3-server > $O/server-rows1; chmod +x $O/server-rows1
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; echo "$n rc=$? $(tail -1 $O/$n.log)"; }
run speech-server $PY tools/check_lfm3_speech_server.py $S/liquid-lfm3-server $M $R $O/speech-server.json
run audio-in-server $PY tools/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/audio-in-work $O/audio-in-server.json
run speech-stream $PY tools/check_lfm3_speech_stream.py $S/liquid-lfm3-server $M $R $LIBRI $O/speech-stream.json
run speech-concurrent $PY tools/check_lfm3_speech_concurrent.py $S/lfm3-speech $M $R $LIBRI $O/concurrent-work $O/speech-concurrent.json
run speech-batch-server $PY tools/check_lfm3_speech_batch_server.py $S/liquid-lfm3-server $O/server-rows1 $M $R $LIBRI $O/speech-batch-server.json
kill $SAMPLER; sort -t, -k2 -n $O/memory.csv | tail -1
date
