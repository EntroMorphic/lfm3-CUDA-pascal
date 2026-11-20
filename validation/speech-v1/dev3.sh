# speech-v1 development 3 (tools/pascal/remote.sh detach speech-dev3 'bash validation/speech-v1/dev3.sh'): audio in over
# LibriSpeech test-clean, as audio-in-v1 ran it. The corpus (the development machine's copy, its SHA-256 checked) is
# extracted; tools/librispeech_cases.py writes the 16-bit WAVs and the cases (DUMP 20, OFFSETS 0,65, as audio-in-v1);
# tools/lfm3_audio_in_hf.py writes the requests (no model loaded); each build's lfm3-speech runs them all (--batch: the
# model loaded once); tools/pascal/score_audio_in.py scores the parity build against the Thor engine (rule 4) and the
# shipping build against HF (rule 10).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev3; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; D=/workspace/data/librispeech; R=validation/speech-v1/inputs/reference
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
(cd $D && sha256sum -c test-clean.tar.gz.sha256); echo "corpus sha256 rc=$?"
[ -e $D/extracted/LibriSpeech/test-clean ] || { mkdir -p $D/extracted && tar xzf $D/test-clean.tar.gz -C $D/extracted; }
$PY tools/librispeech_cases.py $D/extracted/LibriSpeech/test-clean $D/wav $O/cases.json 20 0,65 > $O/cases.log 2>&1; echo "cases rc=$? $(tail -1 $O/cases.log)"
$PY tools/lfm3_audio_in_hf.py $M $O/cases.json $O requests > $O/requests.log 2>&1; echo "requests rc=$? $(ls $O/requests | wc -l)"
for build in parity-s1 text-s1; do
  mkdir -p $O/$build; for f in $(ls $O/requests); do echo "$O/requests/$f $O/$build/${f%.req.json}"; done > $O/$build.list
  /usr/bin/time -f "%e s %M KiB" $B/$build/lfm3-speech $M --batch $O/$build.list > $O/$build.log 2> $O/$build.err; echo "$build rc=$? $(tail -2 $O/$build.err | tr '\n' ' ')"
done
$PY tools/pascal/score_audio_in.py parity $M /workspace/models/whisper-large-v3 $O/cases.json $O/parity-s1 $R/audio-in-v1/round2-full-check.json $O/rule4.json > $O/rule4.log 2>&1; echo "rule 4 rc=$?"; tail -14 $O/rule4.log
$PY tools/pascal/score_audio_in.py shipping $M /workspace/models/whisper-large-v3 $O/cases.json $O/text-s1 $O/rule4.json $R/audio-in-v1/round2-full-check.json $O/rule10.json > $O/rule10.log 2>&1; echo "rule 10 rc=$?"; tail -14 $O/rule10.log
date
