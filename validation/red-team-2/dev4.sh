# red-team-2 development 4 (tools/pascal/remote.sh detach rt2-dev4 'bash validation/red-team-2/dev4.sh'): the batch's
# worst case greedy and sampled by the host sampler's slowest path (R4-1), on text-v37's server.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-dev4; M=/workspace/models/LFM3-Omni-2B
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
$B/venv-asr/bin/python tools/pascal/batch_budget.py $B/text-v37/liquid-lfm3-server $M $O/batch 29696 > $O/batch.log 2>&1; echo "B1 rc=$?"; grep -E "batch budget|BATCH|Error" $O/batch.log
pkill -TERM -f liquid-lfm3-server; date
