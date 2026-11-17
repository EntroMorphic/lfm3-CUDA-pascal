#!/usr/bin/env bash
# audio-assurance-v1 rules 2 and 3: all of LibriSpeech test-clean through lfm3-speech speed-a4, alone then 8 rows staggered.
set -uo pipefail
R=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA
B=$R/build/assure; C=/workspace/liquid-CUDA/build/assure; A2=/workspace/liquid-CUDA/build/audioin-v2
BIN=/workspace/liquid-CUDA/build/speed-a4/lfm3-speech
mkdir -p $B && cd $B
sha256sum $R/build/speed-a4/lfm3-speech > corpus-binary.sha256
bash $R/tools/thor/power_state.sh power-corpus-start.json > /dev/null || { echo "FAIL power"; exit 1; }
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
rm -rf alone rows8 && mkdir -p alone rows8
sed "s#$A2/native-full/#$C/alone/#" $R/build/audioin-v2/list-full.txt > list-alone.txt
sed "s#$A2/native-full/#$C/rows8/#" $R/build/audioin-v2/list-full.txt > list-rows8.txt
echo "== alone $(date +%T)"
$EXEC $BIN /workspace/models/LFM3-Omni-2B --batch $C/list-alone.txt > alone.log 2>&1; echo "exit $?"; tail -1 alone.log
echo "== rule 2 $(date +%T)"
python3 $R/tools/compare_speech_outputs.py $R/build/audioin-v2/native-full alone rule2-slots.json --slots-only
echo "== rows 8 $(date +%T)"
$EXEC $BIN /workspace/models/LFM3-Omni-2B --concurrent 8 $C/list-rows8.txt 7 > rows8.log 2>&1; echo "exit $?"; tail -2 rows8.log
echo "== rule 3 $(date +%T)"
python3 $R/tools/compare_speech_outputs.py alone rows8 rule3-files.json
bash $R/tools/thor/power_state.sh power-corpus-end.json power-corpus-start.json > /dev/null || echo "WARN power changed"
echo "CORPUS DONE $(date +%T)"
