# speech-v1 development 10 (tools/pascal/remote.sh detach speech-dev10 'bash validation/speech-v1/dev10.sh'). Development
# 9: with the codec checkpoint shared, one speech row still exceeds rule 12 at a total capacity of 32,768
# (/generate_batch on 4,096 + 28,672: 7,745 MiB, limit 7,740); each row about 102 MiB more. Rule 12: the total capacity
# is lowered as serve-v1 rule 11 says (multiples of 1,024, the configurations scaled). speech_memory.py now takes the
# total and judges /generate_batch by the drain (the server runs it only once the speech rows are empty). Here: one row
# at 31,744, 30,720 and 28,672 (the slope of the peaks with capacity), on text-s3's server. Stopped by hand once
# 31,744 had passed (every configuration; the batch at 7,693 MiB), during 30,720: the owner kept rule 12 as written
# (one row, the largest passing capacity), so the slope was no longer needed.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev10; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
for total in 31744 30720 28672; do
  $PY tools/pascal/speech_memory.py $B/text-s3/liquid-lfm3-server $M 1 $LIBRI $O/memory-rows1-$total 7740 $total > $O/memory-rows1-$total.log 2>&1
  echo "memory rows 1 total $total rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-rows1-$total.log | cut -c1-400
done
date
