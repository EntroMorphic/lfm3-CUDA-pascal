#!/usr/bin/env bash
# #2 step 1: measured trace of the HF text -> text + audio path, then the ASR round-trip of each case (Thor host).
# Runs tools/lfm3_audio_trace.py in the reference environment (lfm3-omni-dev, as test_interleaved.py) and
# tools/lfm3_asr.py (Whisper large-v3). The first case is test_interleaved.py's, compared with its committed output.
# Usage: tts_trace_round.sh TAG
set -uo pipefail
tag=$1
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; O=$W/liquid-CUDA/build/$tag; C=/workspace/liquid-CUDA/build/$tag
test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
# The model's causal conv is a Triton kernel; Triton's bundled ptxas does not know sm_110a, so point it at Thor's
# (copied to tools/ptxas by run-minimal-thor.sh, as resume-tests-thor.sh does).
EX="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
T=/workspace/liquid-CUDA/tools; M=/workspace/models/LFM3-Omni-2B
echo "== sources"; (cd $W/liquid-CUDA && sha256sum tools/lfm3_audio_trace.py tools/lfm3_asr.py tools/thor/tts_trace_round.sh) | sed 's/^/  /'
echo "power: $(nvpmodel -q 2>/dev/null | head -1)"
IL="Respond with interleaved text and audio."
TF="Perform TTS in American English using a female voice."; TM="Perform TTS in American English using a male voice."
LONG="On March 3rd, 2024, the museum reopened after a two-year renovation; visitors could finally see the restored murals, which date back nearly four hundred years."
run(){ echo "== $1 $(date +%T)"; $EX $T/lfm3_audio_trace.py $M $C "$@" > $O/$1.log 2>&1 && tail -1 $O/$1.log || { echo "FAIL $1"; tail -5 $O/$1.log; }; }
run ref-interleaved interleaved 42 2048 "$IL" "Hypothetically, might reflective recursion be a function of cognition?" /workspace/outputs/reflective-recursion
run interleaved-study interleaved 42 2048 "$IL" "What are three tips for staying focused while studying?"
run tts-fox-female audio 42 256 "$TF" "The quick brown fox jumps over the lazy dog"
run tts-fox-male audio 42 256 "$TM" "The quick brown fox jumps over the lazy dog"
run tts-museum-female audio 42 512 "$TF" "$LONG"
run tts-museum-male audio 42 512 "$TM" "$LONG"
echo "== asr $(date +%T)"
cases=(); for d in $O/*/; do test -e $d/summary.json && cases+=("$C/$(basename $d)"); done
[ ${#cases[@]} -gt 0 ] || { echo "FAIL: no case produced output"; exit 1; }
$EX $T/lfm3_asr.py /workspace/models/whisper-large-v3 $C/asr.json "${cases[@]}" > $O/asr.log 2>&1; grep WER $O/asr.log || tail -5 $O/asr.log
echo "ALL DONE $(date +%T)"
