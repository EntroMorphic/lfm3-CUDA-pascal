# speech-v1 development 9 (tools/pascal/remote.sh detach speech-dev9 'bash validation/speech-v1/dev9.sh'). Development 8:
# with text at its limits, no number of speech rows passes rule 12 (one row: 7,745 MiB on /generate_batch, limit
# 7,740; each row about 245 MiB more). Each row's codec decoder held its own copy of the codec checkpoint (147 MB);
# the s3 builds share one (lfm/codec.cuh, lfm3_speech.cuh; host code only). Here: both builds again (s3); their
# outputs against s1's, byte for byte (rule 3's 15 clips through --batch; s1 parity equals the Thor engine on them);
# the concurrent check's 41 requests at 8, 12 and 16 rows against development 4's alone runs, with peak memory; then
# tools/pascal/speech_memory.py (now with a longer speech load, staggered sends) from 8 rows down to the first pass.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev9; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; D1=$B/sp-dev1
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s3 > $O/build-parity-s3.log 2>&1; echo "build parity-s3 rc=$?"
tools/pascal/build_speech.sh text-s3 > $O/build-text-s3.log 2>&1; echo "build text-s3 rc=$?"
for b in text parity; do
  mkdir -p $O/$b
  for f in $(ls $D1/single/requests | grep -- '-s1\.req\.json$' | grep -v tts-long-male); do echo "$D1/single/requests/$f $O/$b/${f%.req.json} $D1/$b-s1/${f%.req.json}.wav"; done > $O/$b.map
  for f in $(ls $D1/split/requests | grep -E -- '-s[123]\.req\.json$'); do n=${f%.req.json}; w=$D1/$b-s1/$n.wav; [ ${n#tts-long-male} != $n ] && w=$D1/$b-s1-split/$n.wav; echo "$D1/split/requests/$f $O/$b/$n $w"; done >> $O/$b.map
  cut -d' ' -f1,2 $O/$b.map > $O/$b.list
  $B/$b-s3/lfm3-speech $M --batch $O/$b.list > $O/$b.log 2> $O/$b.err; echo "$b-s3 batch rc=$? $(tail -1 $O/$b.err)"
  same=0; n=0; while read r s w; do n=$((n+1)); cmp -s $s.wav $w && same=$((same+1)) || echo "DIFFERS $s"; done < $O/$b.map; echo "$b-s3 against $b-s1: $same of $n WAVs equal"
done
L=$B/sp-dev4/concurrent-work/conc16-staggered/list.txt
for rows in 8 12 16; do
  mkdir -p $O/rows$rows; sed "s#$B/sp-dev4/concurrent-work/conc16-staggered/#$O/rows$rows/#" $L > $O/rows$rows/list.txt
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $O/rows$rows/memory.txt & S=$!
  $B/text-s3/lfm3-speech $M --concurrent $rows $O/rows$rows/list.txt 12 > $O/rows$rows/out.log 2> $O/rows$rows/err.log; rc=$?
  kill $S; same=0; for w in $B/sp-dev4/concurrent-work/alone/*.wav; do cmp -s $w $O/rows$rows/$(basename $w) && same=$((same+1)); done
  echo "rows $rows rc=$rc peak $(sort -n $O/rows$rows/memory.txt | tail -1) MiB, WAVs equal to alone: $same $(tail -1 $O/rows$rows/err.log)"
done
for rows in 8 7 6 5 4 3 2 1; do
  $PY tools/pascal/speech_memory.py $B/text-s3/liquid-lfm3-server $M $rows $LIBRI $O/memory-rows$rows > $O/memory-rows$rows.log 2>&1; rc=$?
  echo "memory rows $rows rc=$rc"; grep -E '^rule 12|^SPEECH' $O/memory-rows$rows.log | cut -c1-400
  [ $rc = 0 ] && break
done
date
