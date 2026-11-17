#!/usr/bin/env bash
# #2 step 4, C1-C4 round on Thor (host): HF and native over validation/speech-e2e-v1/prompts.json, every seed.
# Usage: [PROMPTS=prompts-long.json] speech_e2e_round.sh TAG TOOLS_TAG   (TOOLS_TAG: the build dir holding lfm3-speech)
set -uo pipefail
tag=$1; tools=$2
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O/native
EXN="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
T=/workspace/liquid-CUDA/tools; M=/workspace/models/LFM3-Omni-2B; P=/workspace/liquid-CUDA/validation/speech-e2e-v1/${PROMPTS:-prompts.json}
mem(){ awk -v t="$1" '/MemTotal/{T=$2}/MemFree/{F=$2}/MemAvailable/{A=$2}/^Cached/{K=$2}/Buffers/{U=$2}END{printf "%s %s used %.1f GiB available %.1f GiB\n", strftime("%T"), t, (T-F-K-U)/1048576, A/1048576}' /proc/meminfo | tee -a $O/memory.log; }
fail(){ echo "FAIL: $1"; exit 1; }
bash $R/tools/thor/power_state.sh $O/power-start.json || fail "power state"
echo "== sources"; (cd $R && sha256sum tools/lfm3_speech_hf.py tools/lfm3_speech_cases.py tools/lfm3_speech_report.py tools/lfm3_asr.py tools/thor/speech_e2e_round.sh validation/speech-e2e-v1/prompts.json) | sed 's/^/  /'
sha256sum $B/$tools/lfm3-speech | sed 's/^/  /'; grep EnableSystemMemoryPools /proc/driver/nvidia/params | sed 's/^/  /'; mem start
echo "== HF $(date +%T)"; $PY $T/lfm3_speech_hf.py $M $P $C/$tag > $O/hf-full.log 2>&1 || fail hf
grep -v -iE "warn|Loading weights" $O/hf-full.log > $O/hf.log
mem "after HF"
echo "== native $(date +%T)"
names=$(ls $O/requests | sed 's/\.req\.json$//')
for n in $names; do $EXN $C/$tools/lfm3-speech $M $C/$tag/requests/$n.req.json $C/$tag/native/$n > /dev/null || fail "native $n"; done
mem "after native"
$PY $T/lfm3_speech_cases.py $M $C/$tag/requests $C/$tag/native $C/$tag/native-cases $names > $O/native.log 2>&1 || fail cases
[ "$(ls $O/native-cases | wc -l)" = "$(echo $names | wc -w)" ] && [ "$(ls $O/hf | wc -l)" = "$(echo $names | wc -w)" ] || fail "cases incomplete"
echo "== ASR $(date +%T)"
for e in hf native-cases; do docker exec lfm3-omni-dev bash -c "/opt/venv/bin/python3 $T/lfm3_asr.py /workspace/models/whisper-large-v3 $C/$tag/asr-$e.json $C/$tag/$e/*" > $O/asr-$e-full.log 2>&1 || fail "asr $e"; grep WER $O/asr-$e-full.log > $O/asr-$e.log; done
echo "== report"
$PY $T/lfm3_speech_report.py $P $C/$tag/asr-native-cases.json $C/$tag/asr-hf.json $C/$tag/native-cases $C/$tag/hf $C/$tag/report.json > $O/report.md 2>&1 || fail report
cat $O/report.md
mem end; bash $R/tools/thor/power_state.sh $O/power-end.json $O/power-start.json || fail "power state changed"
echo "ALL DONE $(date +%T)"
