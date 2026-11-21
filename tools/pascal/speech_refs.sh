#!/usr/bin/env bash
# Runs in the container: tools/pascal/speech_refs.sh BUILD_TAG OUT_TAG
# The validated runs the reference's speech server checks read (their BUILD_DIR), made by BUILD_TAG's own lfm3-speech
# into /workspace/build/OUT_TAG in the reference's layout (speech-v1 rule 11: each check against the same build's CLI):
#   speech-e2e-v1/{requests,native,native-cases}   prompts.json, the ten cases at seeds 1-3 (tools/lfm3_speech_hf.py
#                                                  requests; tools/lfm3_speech_cases.py packages them for the scorer)
#   speech-chunk-native-v1/{requests,native}       prompts-chunked.json split, seeds 1-3 (lighthouse and bread)
#   speech-split-b-full-49/{requests,native}       prompts-split-phase-b.json split at the 49-word cap, seed 1
#   audio-in-listen/                               the five spoken questions: the build's own TTS of them, resampled to
#                                                  16 kHz (tools/lfm3_audio_in_listening.py tts, then prepare)
# Requests are rendered as the reference renders them (no model loaded). The e2e runs are also rule 7's seeds 1-3.
set -euo pipefail
build=${1:?usage: speech_refs.sh BUILD_TAG OUT_TAG}; out=/workspace/build/${2:?usage: speech_refs.sh BUILD_TAG OUT_TAG}
B=/workspace/build; M=/workspace/models/LFM3-Omni-2B; PY=$B/venv-asr/bin/python; P=validation/speech-e2e-v1
S=$B/$build/lfm3-speech; LIBRI=/workspace/data/librispeech/wav
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -e "$out" ] && { echo "speech_refs.sh: $out exists"; exit 2; }
mkdir -p "$out"; sha256sum "$S" > "$out/lfm3-speech.sha256"
run_all() {  # REQUEST_DIR NATIVE_DIR [PATTERN]: every request through one lfm3-speech --batch (the model loaded once)
  mkdir -p "$2"
  for f in $(ls "$1" | grep -E -- "${3:-.}"); do echo "$1/$f $2/${f%.req.json}"; done > "$2.list"
  "$S" "$M" --batch "$2.list" > "$2.log" 2> "$2.err" || { tail -3 "$2.err"; echo "speech_refs.sh: $2 FAILED"; exit 1; }
  echo "$2: $(grep -c . "$2.list") runs"
}
E=$out/speech-e2e-v1
$PY tools/lfm3_speech_hf.py $M $P/prompts.json $E requests > $E.requests.log 2>&1
run_all $E/requests $E/native
$PY tools/lfm3_speech_cases.py $M $E/requests $E/native $E/native-cases $(ls $E/requests | sed 's/\.req\.json$//') > $E.cases.log 2>&1
C=$out/speech-chunk-native-v1
$PY tools/lfm3_speech_chunked_hf.py $M $P/prompts-chunked.json $C requests > $C.requests.log 2>&1
run_all $C/requests $C/native '-s[123]\.req\.json$'
$PY tools/lfm3_speech_cases.py $M $C/requests $C/native $C/native-cases $(ls $C/requests | grep -E -- '-s[123]\.req\.json$' | sed 's/\.req\.json$//') > $C.cases.log 2>&1
A=$out/speech-split-b-full-49
$PY tools/lfm3_speech_chunked_hf.py $M $P/prompts-split-phase-b.json $A requests > $A.requests.log 2>&1
run_all $A/requests $A/native '^abbrev-s1\.req\.json$'
L=$out/audio-in-listen
$PY tools/lfm3_audio_in_listening.py tts $L > $L.tts.log 2>&1
for i in 0 1 2 3 4; do "$S" $M $L/tts-q$i.req.json $L/tts-q$i > /dev/null 2>> $L.tts.log; done
$PY tools/lfm3_audio_in_listening.py prepare $L $LIBRI > $L.prepare.log 2>&1
ls $L/q?-16k.wav | wc -l
(cd "$out" && find . -name "*.wav" | sort | xargs sha256sum) > "$out/wav.sha256"
echo "speech_refs.sh: $out done"
