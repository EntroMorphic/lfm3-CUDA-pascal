#!/usr/bin/env bash
# audio-assurance-v1: the final builds (lfm3-speech A, server S) through every exactness and server check, then the
# hostile-input run (rule 5), then the soak (rule 4) last, on a quiet machine.
# Usage: assurance_server.sh A S [PHASES]   PHASES: any of "engine server fuzz soak" (default all)
set -uo pipefail
A=${1:?A}; S=${2:?S}; PHASES=${3:-engine server fuzz soak}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
L=/workspace/data/librispeech/wav
O=$B/assure/final; CO=$C/assure/final; mkdir -p $O && cd $O
sha256sum $B/$A/lfm3-speech $B/$S/liquid-lfm3-server $B/$S/liquid-lfm3-worker > binaries-$(date +%H%M%S).sha256
P=$(date +%H%M%S); bash $W/liquid-CUDA/tools/thor/power_state.sh power-start-$P.json > /dev/null || { echo "FAIL power"; exit 1; }
if [[ " $PHASES " == *" engine "* ]]; then
  echo "== speech regression $(date +%T)"
  bash $W/liquid-CUDA/tools/thor/speech_regression_check.sh $A > speech-regression.log 2>&1
  echo "same $(grep -c '^SAME' speech-regression.log)"; grep -E "DIFFER|RUN FAILED" speech-regression.log
  echo "== engine exactness $(date +%T)"
  rm -rf $O/conc
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_concurrent.py $C/$A/lfm3-speech $M $C $L $CO/conc $CO/engine-exact.json $C/speed-final2/conc/alone > engine-exact.log 2>&1; grep -E "reference|CONCURRENT" engine-exact.log
fi
if [[ " $PHASES " == *" server "* ]]; then
  echo "== batch server $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_batch_server.py $C/$S/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L $CO/batch-server.json > batch-server.log 2>&1; tail -1 batch-server.log
  echo "== stream check $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_stream.py $C/$S/liquid-lfm3-server $M $C $L $CO/stream-check.json > stream-check.log 2>&1; tail -1 stream-check.log
  echo "== audio-in server $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_in_server.py $C/$S/liquid-lfm3-server $C/$A/lfm3-speech $M $L $CO/work $CO/audio-in-server.json > audio-in-server.log 2>&1; tail -1 audio-in-server.log
  echo "== TTS server $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_server.py $C/$S/liquid-lfm3-server $M $C $CO/speech-check.json > speech-check.log 2>&1; tail -1 speech-check.log
  echo "== text gate $(date +%T)"
  rm -rf $O/text-gate
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_native_server.py --checkpoint $M --server $C/$S/liquid-lfm3-server --worker $C/$S/liquid-lfm3-worker --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $CO/text-gate > text-gate.log 2>&1; echo "text gate exit $?"
fi
if [[ " $PHASES " == *" fuzz "* ]]; then
  echo "== fuzz (rule 5) $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_fuzz.py $C/$S/liquid-lfm3-server $M $C $L $CO/fuzz.json 3000 8 2026 > fuzz.log 2>&1; tail -2 fuzz.log
fi
if [[ " $PHASES " == *" soak "* ]]; then
  echo "== soak (rule 4) $(date +%T)"
  $X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_soak.py $C/$S/liquid-lfm3-server $M $C $L $CO/soak.json 7200 12 > soak.log 2>&1; tail -2 soak.log
fi
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end-$P.json power-start-$P.json > /dev/null || echo "WARN power changed"
echo "ASSURANCE SERVER DONE $(date +%T)"
