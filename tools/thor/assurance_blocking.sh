#!/usr/bin/env bash
# audio-assurance-v1 rule 12 (and the fixed build against speed-a4): the engine check's 41 requests, 8 rows staggered,
# normal and with CUDA_LAUNCH_BLOCKING=1, on lfm3-speech TAG.
set -uo pipefail
T=${1:?TAG}
R=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA; B=$R/build; E=/workspace/liquid-CUDA/build
O=$B/assure/blocking-$T; C=$E/assure/blocking-$T; M=/workspace/models/LFM3-Omni-2B
rm -rf $O && mkdir -p $O/normal $O/blocking && cd $O
sha256sum $B/$T/lfm3-speech > binary.sha256
for d in normal blocking; do sed "s#$E/speed-final2/conc/conc4/#$C/$d/#" $B/speed-final2/conc/conc4/list.txt > list-$d.txt; done
echo "== normal $(date +%T)"
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $E/$T/lfm3-speech $M --concurrent 8 $C/list-normal.txt 11 > normal.log 2>&1; echo "exit $?"
echo "== blocking $(date +%T)"
docker exec -e CUDA_LAUNCH_BLOCKING=1 -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $E/$T/lfm3-speech $M --concurrent 8 $C/list-blocking.txt 11 > blocking.log 2>&1; echo "exit $?"
python3 $R/tools/compare_speech_outputs.py normal blocking rule12.json
python3 $R/tools/compare_speech_outputs.py $B/speed-final2/conc/alone normal vs-speed-a4-alone.json
echo "BLOCKING DONE $(date +%T)"
