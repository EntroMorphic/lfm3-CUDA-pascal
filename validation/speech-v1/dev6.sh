# speech-v1 development 6 (tools/pascal/remote.sh detach speech-dev6 'bash validation/speech-v1/dev6.sh'). Development 4
# found the s1 builds without the server (built before build_speech.sh built it) and lfm3-speech --concurrent 16 out of
# memory ("cudaMalloc model payload": lfm3_speech::Batch's rows allocated, then the speech weights do not fit). Here:
# both builds again with the server (text-s2, parity-s2), lfm3-speech compared with s1's; then the concurrent check's
# own 41-request list (sp-dev4, staggered by seed 12) through the shipping build's lfm3-speech --concurrent at 15 down
# to 9 rows, each run's exit, error and peak device memory (nvidia-smi every 200 ms).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev6; M=/workspace/models/LFM3-Omni-2B
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
# Rule 7's escalation (development 5 flagged tts-numbers-female and the split tts-long-male): seeds 4-15 by text-s1's
# lfm3-speech, the binary of seeds 1-3 (sp-refs-text-s1), scored against HF's committed 15 seeds.
PY=$B/venv-asr/bin/python; W=/workspace/models/whisper-large-v3; R=$B/sp-refs-text-s1; X=$O/escalate
$PY tools/lfm3_speech_hf.py $M validation/speech-e2e-v1/prompts-long.json $X/long requests > $X.requests.log 2>&1; echo "requests long rc=$?"
for s in 4 5 6 7 8 9 10 11 12 13 14 15; do
  echo "$X/long/requests/tts-numbers-female-s$s.req.json $X/numbers/tts-numbers-female-s$s"
  echo "$R/speech-chunk-native-v1/requests/tts-long-male-s$s.req.json $X/split/tts-long-male-s$s"
done > $X.list; mkdir -p $X/numbers $X/split
$B/text-s1/lfm3-speech $M --batch $X.list > $X.log 2> $X.err; echo "escalation runs rc=$? $(tail -1 $X.err)"
n=$(seq -f tts-numbers-female-s%g 4 15); l=$(seq -f tts-long-male-s%g 4 15)
$PY tools/lfm3_speech_cases.py $M $X/long/requests $X/numbers $X/numbers-cases $n > $X.cases.log 2>&1; echo "cases numbers rc=$?"
$PY tools/lfm3_speech_cases.py $M $R/speech-chunk-native-v1/requests $X/split $X/split-cases $l >> $X.cases.log 2>&1; echo "cases split rc=$?"
$PY tools/lfm3_asr.py $W $X/asr-numbers.json $X/numbers-cases/* > $X.asr.log 2>&1; echo "asr numbers rc=$?"
$PY tools/lfm3_asr.py $W $X/asr-split.json $X/split-cases/* >> $X.asr.log 2>&1; echo "asr split rc=$?"
V=validation/speech-v1/inputs/reference; D5=$B/sp-dev5
$PY tools/pascal/score_speech.py escalate tts-numbers-female $D5/asr-e2e.json,$X/asr-numbers.json $V $X/numbers.json; echo "escalate numbers rc=$?"
$PY tools/pascal/score_speech.py escalate tts-long-male $D5/asr-split.json,$X/asr-split.json $V $X/long.json; echo "escalate long rc=$?"
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s2 > $O/build-parity-s2.log 2>&1; echo "build parity-s2 rc=$?"
tools/pascal/build_speech.sh text-s2 > $O/build-text-s2.log 2>&1; echo "build text-s2 rc=$?"
for b in text parity; do for f in $B/$b-s2/speech-binaries.sha256; do cat $f; done; echo "$b lfm3-speech s1: $(sha256sum $B/$b-s1/lfm3-speech | cut -c1-64)"; done
L=$B/sp-dev4/concurrent-work/conc16-staggered/list.txt
for rows in 15 14 13 12 11 10 9; do
  mkdir -p $O/rows$rows; sed "s#$B/sp-dev4/concurrent-work/conc16-staggered/#$O/rows$rows/#" $L > $O/rows$rows/list.txt
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $O/rows$rows/memory.txt & S=$!
  $B/text-s2/lfm3-speech $M --concurrent $rows $O/rows$rows/list.txt 12 > $O/rows$rows/out.log 2> $O/rows$rows/err.log; rc=$?
  kill $S; same=0; for w in $B/sp-dev4/concurrent-work/alone/*.wav; do cmp -s $w $O/rows$rows/$(basename $w) && same=$((same+1)); done
  echo "rows $rows rc=$rc peak $(sort -n $O/rows$rows/memory.txt | tail -1) MiB, WAVs equal to alone: $same $(tail -1 $O/rows$rows/err.log) $(grep '^{"rows"' $O/rows$rows/out.log)"
done
date
