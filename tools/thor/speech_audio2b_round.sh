#!/usr/bin/env bash
# LFM3-Audio-2B TTS round (validation/speech-audio2b-v1): tools/lfm3_audio2b_tts_hf.py on a prompt file, then ASR.
# Usage (Thor host): speech_audio2b_round.sh TAG PROMPTS_JSON ARM   (PROMPTS_JSON: a path under the repo; ARM: eval|card)
set -uo pipefail
tag=$1; prompts=$2; arm=$3
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; mkdir -p $O
PY="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
fail(){ echo "FAIL: $1"; exit 1; }
bash $R/tools/thor/power_state.sh $O/power-$arm.json > /dev/null || fail "power state"
echo "== TTS $arm $(date +%T)"
$PY /workspace/liquid-CUDA/tools/lfm3_audio2b_tts_hf.py /workspace/models/LFM3-Audio-2B /workspace/liquid-CUDA/$prompts $C/$tag $arm \
  > $O/tts-$arm.log 2>&1 || fail "tts $arm"
grep -c "frames" $O/tts-$arm.log
echo "== ASR $(date +%T)"
docker exec lfm3-omni-dev bash -c "/opt/venv/bin/python3 /workspace/liquid-CUDA/tools/lfm3_asr.py /workspace/models/whisper-large-v3 $C/$tag/asr-$arm.json $C/$tag/audio2b-$arm/*" \
  > $O/asr-$arm-full.log 2>&1 || fail "asr $arm"
grep WER $O/asr-$arm-full.log > $O/asr-$arm.log
echo "ALL DONE $arm $(date +%T)"
