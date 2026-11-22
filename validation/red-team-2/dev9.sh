# red-team-2 development 9 (tools/pascal/remote.sh detach rt2-dev9 'bash validation/red-team-2/dev9.sh'): the batch's
# worst case at the lowered bound, greedy, sampled, and sampled after a continuation of a cached prompt (R5-1), 2,048 new tokens a lane (development 4: sampled at 4,096 ran past
# the engine timeout and the server failed closed), on text-v37's server. Its timing only: v37 still allows 4,096, so
# the refusal of 2,049 is not expected here (the verification checks it on v38).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-dev9; M=/workspace/models/LFM3-Omni-2B
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
$B/venv-asr/bin/python tools/pascal/batch_budget.py $B/text-v37/liquid-lfm3-server $M $O/batch 29696 > $O/batch.log 2>&1; echo "B1 rc=$?"; grep -E "batch budget|BATCH|Error" $O/batch.log
pkill -TERM -f liquid-lfm3-server; date
