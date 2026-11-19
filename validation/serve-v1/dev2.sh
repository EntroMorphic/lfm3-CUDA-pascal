# serve-v1 development (tools/pascal/remote.sh detach serve-dev2 'bash validation/serve-v1/dev2.sh'): rule 10's driver
# in traced continuations of 32 tokens, at full length, on dev1's shipping build.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; M=$B/serve-dev2; [ -e $M ] && exit 2; mkdir -p $M
python3 tools/pascal/paired_decode_check.py $B/serve-d1/liquid-lfm3-worker /workspace/models/LFM3-Omni-2B /workspace/data/ppl-windows/windows-32768.i32 $M/paired > $M/paired.log 2>&1; echo "paired rc=$?"; grep -E 'lane|FAIL|PASS|Error' $M/paired.log
date
