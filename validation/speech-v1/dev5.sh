# speech-v1 development 5 (tools/pascal/remote.sh detach speech-dev5 'bash validation/speech-v1/dev5.sh'): rules 7 and
# 8 on the shipping build's runs from development 4 (speech_refs.sh text-s1): tools/lfm3_asr.py over the ten cases at
# seeds 1-3 and the split lighthouse at seeds 1-3, then tools/pascal/score_speech.py c1c2 against HF's committed records.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev5; PY=$B/venv-asr/bin/python
R=$B/sp-refs-text-s1; W=/workspace/models/whisper-large-v3
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
nvidia-smi --query-gpu=memory.used --format=csv,noheader
/usr/bin/time -f "%e s" $PY tools/lfm3_asr.py $W $O/asr-e2e.json $R/speech-e2e-v1/native-cases/* > $O/asr-e2e.log 2>&1; echo "asr e2e rc=$? $(tail -1 $O/asr-e2e.log)"
/usr/bin/time -f "%e s" $PY tools/lfm3_asr.py $W $O/asr-split.json $R/speech-chunk-native-v1/native-cases/tts-long-male-s* > $O/asr-split.log 2>&1; echo "asr split rc=$? $(tail -1 $O/asr-split.log)"
python3 tools/pascal/score_speech.py c1c2 validation/speech-e2e-v1/prompts.json $O/asr-e2e.json $O/asr-split.json \
  $R/speech-e2e-v1/native-cases $R/speech-chunk-native-v1/native validation/speech-v1/inputs/reference $O/c1c2.json
echo "rules 7 and 8 rc=$?"
date
