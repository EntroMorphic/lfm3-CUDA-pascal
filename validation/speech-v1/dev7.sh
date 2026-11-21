# speech-v1 development 7 (tools/pascal/remote.sh detach speech-dev7 'bash validation/speech-v1/dev7.sh'): rule 12's
# memory through the shipping server (text-s2) at the default 8 speech rows (tools/pascal/speech_memory.py: serve-v1's
# five text configurations with every speech row generating, and audio in at the body limit); then the three server
# checks that run at the default rows (speech-server, audio-in-server, speech-stream) against the CLI's runs of
# development 4 (sp-refs-text-s1: text-s1's lfm3-speech, device code identical to text-s2's, development 6).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev7; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; S=$B/text-s2; R=$B/sp-refs-text-s1
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
run() { local n=$1; shift; /usr/bin/time -f "%e s" "$@" > $O/$n.log 2>&1; echo "$n rc=$? $(grep -E '^rule 12|PASS|FAIL' $O/$n.log | tr '\n' ' ' | cut -c1-1500) $(tail -1 $O/$n.log)"; }
run memory-rows8 $PY tools/pascal/speech_memory.py $S/liquid-lfm3-server $M 8 $LIBRI $O/memory-rows8
run speech-server $PY tools/check_lfm3_speech_server.py $S/liquid-lfm3-server $M $R $O/speech-server.json
run audio-in-server $PY tools/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/audio-in-work $O/audio-in-server.json
run speech-stream $PY tools/check_lfm3_speech_stream.py $S/liquid-lfm3-server $M $R $LIBRI $O/speech-stream.json
date
