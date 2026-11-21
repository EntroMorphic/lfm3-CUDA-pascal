# speech-v1 development 12 (tools/pascal/remote.sh detach speech-dev12 'bash validation/speech-v1/dev12.sh'). Development
# 11: on the shipping build every output of rule 11's five checks was exact, three failing only by time; decision 10
# (timing recorded, not judged). Its parity half failed at once: check_lfm3_speech_server.py's 900 s client deadline
# ran out on the worst case, the check died with its server still running, and the four checks after it found the
# card full. Here, with the tools/pascal/ copies (longer client deadlines; the speech-server check's longer session
# idle timeout; the batch crowd's longer queue deadline): the speech-server copy on the shipping build, then all five
# on the parity build; any server a check leaves behind is stopped before the next. Then judge_speech_server.py.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev12; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; echo "$n rc=$? $(grep -E 'PASS|FAIL' $O/$n.log | tr '\n' ' ' | cut -c1-1500) $(tail -1 $O/$n.log)"; clean; }
run text-speech-server $PY tools/pascal/check_lfm3_speech_server.py $B/text-s4/liquid-lfm3-server $M $B/sp-refs-text-s1 $O/text-speech-server.json
S=$B/parity-s4; R=$B/sp-refs-parity-s1
run parity-speech-server $PY tools/pascal/check_lfm3_speech_server.py $S/liquid-lfm3-server $M $R $O/parity-speech-server.json
run parity-speech-stream $PY tools/pascal/check_lfm3_speech_stream.py $S/liquid-lfm3-server $M $R $LIBRI $O/parity-speech-stream.json
run parity-audio-in-server $PY tools/pascal/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/parity-audio-in-work $O/parity-audio-in-server.json
run parity-speech-concurrent $PY tools/check_lfm3_speech_concurrent.py $S/lfm3-speech $M $R $LIBRI $O/parity-concurrent-work $O/parity-speech-concurrent.json
run parity-batch-server $PY tools/pascal/check_lfm3_speech_batch_server.py $S/liquid-lfm3-server $S/liquid-lfm3-server $M $R $LIBRI $O/parity-batch-server.json
for n in speech-stream audio-in-server speech-concurrent batch-server; do cp $B/sp-dev11/text-$n.json $O/text-$n.json; done
$PY tools/pascal/judge_speech_server.py $O text; $PY tools/pascal/judge_speech_server.py $O parity
date
