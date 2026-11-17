#!/usr/bin/env bash
# audio-assurance-v1 rules 7-9: the engine in the tree against the builds fidelity was measured on.
set -uo pipefail
R=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA; B=$R/build; E=/workspace/liquid-CUDA/build
O=$B/assure/rerun; C=$E/assure/rerun; M=/workspace/models/LFM3-Omni-2B
BIN=$E/speed-a4/lfm3-speech
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
rm -rf $O && mkdir -p $O && cd $O
same3() {  # REF_DIR NEW_DIR LABEL: raw.i32 frames.i32 f32 of every stem in REF_DIR
  local n=0 d=0
  for f in $1/*.raw.i32; do s=$(basename $f .raw.i32)
    for x in raw.i32 frames.i32 f32; do n=$((n+1)); cmp -s $1/$s.$x $2/$s.$x || { d=$((d+1)); echo "  DIFFER $3 $s.$x"; }; done
  done; echo "$3: $((n-d)) of $n identical"; }
echo "== rule 7: depth $(date +%T)"
rm -rf $B/assure-depth; bash $R/tools/thor/depth_round.sh assure-depth tts-trace-v2 depth-anchor-v1 > depth.log 2>&1; echo "depth exit $?"
grep -iE "tripwire|PASS|FAIL|median|ALL DONE" depth.log | tail -12
echo "== rule 8: 40 dumped clips $(date +%T)"
mkdir -p dump edge
sed "s#$E/audioin-v2/native-dump/#$C/dump/#" $B/audioin-v2/list-dump.txt > list-dump.txt
$EXEC $BIN $M --batch $C/list-dump.txt > dump.log 2>&1; echo "exit $?"
python3 $R/tools/compare_speech_outputs.py $B/audioin-v2/native-dump dump rule8-dump.json
echo "== rule 8: edge cases $(date +%T)"
sed "s#$E/server-stream-v1-check/edge/native/#$C/edge/#" $B/server-stream-v1-check/edge/list.txt > list-edge.txt
$EXEC $BIN $M --batch $C/list-edge.txt > edge.log 2>&1; echo "exit $?"
python3 $R/tools/compare_speech_outputs.py $B/server-stream-v1-check/edge/native edge rule8-edge.json
echo "== rule 9: e2e and split $(date +%T)"
mkdir -p e2e split
for r in $B/speech-e2e-v1/requests/*.req.json; do n=$(basename $r .req.json); echo "$E/speech-e2e-v1/requests/$n.req.json $C/e2e/$n"; done > list-e2e.txt
for r in $B/speech-chunk-native-v1/requests/*.req.json; do n=$(basename $r .req.json); echo "$E/speech-chunk-native-v1/requests/$n.req.json $C/split/$n"; done > list-split.txt
$EXEC $BIN $M --batch $C/list-e2e.txt > e2e.log 2>&1; echo "exit $?"
$EXEC $BIN $M --batch $C/list-split.txt > split.log 2>&1; echo "exit $?"
same3 $B/speech-e2e-v1/native e2e e2e | tee rule9.txt
same3 $B/speech-chunk-native-v1/native split split | tee -a rule9.txt
echo "RERUN DONE $(date +%T)"
