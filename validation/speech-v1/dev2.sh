# speech-v1 development 2 (tools/pascal/remote.sh detach speech-dev2 'bash validation/speech-v1/dev2.sh'): rule 6's
# check, the scorer on the card. tools/lfm3_asr.py as imported (Whisper large-v3, FP16, greedy) over the 25 clips of the
# reference's listening sets whose transcripts are committed (tools/pascal/asr_calibration.py prepare, run on the
# development machine and copied to /workspace/data/speech-calibration), then each transcript against the committed one.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev2; PY=$B/venv-asr/bin/python
W=/workspace/data/speech-calibration; R=validation/speech-v1/inputs/reference
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
nvidia-smi --query-gpu=memory.used --format=csv,noheader
(cd $W && sha256sum */*/response.wav */*/summary.json) > $O/inputs.sha256
for k in hf engine; do
  /usr/bin/time -f "%e s" $PY tools/lfm3_asr.py /workspace/models/whisper-large-v3 $O/asr-$k.json $W/$k/* > $O/asr-$k.log 2>&1; echo "asr $k rc=$? $(tail -1 $O/asr-$k.log)"
done
python3 tools/pascal/asr_calibration.py compare $W $O/asr-hf.json $O/asr-engine.json $R $O/calibration.json; echo "rule 6 rc=$?"
date
