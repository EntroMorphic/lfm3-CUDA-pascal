#!/usr/bin/env bash
# #2 step 4: a native-only round over a prompt file (no HF generation): requests, lfm3-speech, the case layout, ASR.
# Usage: [CHUNKED=1] speech_native_round.sh TAG TOOLS_TAG PROMPTS(file in validation/speech-e2e-v1)
#   CHUNKED=1: long-form TTS by sentences (tools/lfm3_speech_chunked_hf.py requests; LFM3_SPLIT_RULE/CAP pass through)
set -uo pipefail
tag=$1; tools=$2; prompts=$3
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O/native
EXN="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="docker exec -e LFM3_SPLIT_RULE -e LFM3_SPLIT_CAP lfm3-omni-dev /opt/venv/bin/python3"  # split rule (chunked tool)
T=/workspace/liquid-CUDA/tools; M=/workspace/models/LFM3-Omni-2B; P=/workspace/liquid-CUDA/validation/speech-e2e-v1/$prompts
fail(){ echo "FAIL: $1"; exit 1; }
bash $R/tools/thor/power_state.sh $O/power-start.json || fail "power state"
if [ "${CHUNKED:-0}" = 1 ]; then  # the requests are split by the chunked tool: record its rule and source
  { echo "rule=${LFM3_SPLIT_RULE:-sentences (default)} cap=${LFM3_SPLIT_CAP:-none}"; sha256sum $R/tools/lfm3_speech_chunked_hf.py; } | tee $O/split-rule.txt
fi
if [ "${CHUNKED:-0}" = 1 ]; then REQ="$T/lfm3_speech_chunked_hf.py $M $P $C/$tag requests"; else REQ="$T/lfm3_speech_hf.py $M $P $C/$tag requests"; fi
$PY $REQ > $O/requests.log 2>&1 || fail requests
names=$(ls $O/requests | sed 's/\.req\.json$//')
echo "== native $(date +%T)"
for n in $names; do $EXN $C/$tools/lfm3-speech $M $C/$tag/requests/$n.req.json $C/$tag/native/$n > /dev/null || fail "native $n"; done
$PY $T/lfm3_speech_cases.py $M $C/$tag/requests $C/$tag/native $C/$tag/native-cases $names > $O/native.log 2>&1 || fail cases
[ "$(ls $O/native-cases | wc -l)" = "$(echo $names | wc -w)" ] || fail "native-cases incomplete"
echo "== ASR $(date +%T)"
docker exec lfm3-omni-dev bash -c "/opt/venv/bin/python3 $T/lfm3_asr.py /workspace/models/whisper-large-v3 $C/$tag/asr.json $C/$tag/native-cases/*" > $O/asr-full.log 2>&1 || fail asr
grep WER $O/asr-full.log > $O/asr.log
bash $R/tools/thor/power_state.sh $O/power-end.json $O/power-start.json || fail "power state changed"
echo "ALL DONE $(date +%T)"
