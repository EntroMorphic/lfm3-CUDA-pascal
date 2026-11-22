# red-team-2 development 3 (tools/pascal/remote.sh detach rt2-dev3 'bash validation/red-team-2/dev3.sh'): batch_budget.py
# with number-list prompts (development 2's prose prompt stopped one lane after 200 tokens), on text-v37's server.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-dev3; M=/workspace/models/LFM3-Omni-2B
mkdir -p $O; [ -e $O/batch2 ] && { echo "$O/batch2 exists"; exit 2; }
$B/venv-asr/bin/python tools/pascal/batch_budget.py $B/text-v37/liquid-lfm3-server $M $O/batch2 29696 > $O/batch2.log 2>&1; echo "B1 rc=$?"; tail -3 $O/batch2.log
pkill -TERM -f liquid-lfm3-server; date
