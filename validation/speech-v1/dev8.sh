# speech-v1 development 8 (tools/pascal/remote.sh detach speech-dev8 'bash validation/speech-v1/dev8.sh'): rule 12's
# fallback. Development 7 found 8 speech rows over the card (the worker gone at the first configuration, one-32768,
# while the eight rows generated: 503 worker_unavailable). tools/pascal/speech_memory.py at 7 rows and down, stopping at
# the first number at which every configuration passes.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev8; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav; S=$B/text-s2
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
for rows in 7 6 5 4 3 2 1; do
  $PY tools/pascal/speech_memory.py $S/liquid-lfm3-server $M $rows $LIBRI $O/memory-rows$rows > $O/memory-rows$rows.log 2>&1; rc=$?
  echo "rows $rows rc=$rc"; grep -E '^rule 12|^SPEECH' $O/memory-rows$rows.log | cut -c1-400
  [ $rc = 0 ] && break
done
date
