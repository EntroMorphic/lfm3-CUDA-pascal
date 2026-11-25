# continuation-v1 development 2 (tools/pascal/remote.sh detach cont-dev2 'bash validation/continuation-v1/dev/dev2.sh'):
# after the review's changes to the check (NaN in the shared pair before every run; 400-step decoded cases) and the
# two warnings' fix: the shipping build's warnings, exact-small (unset, =1) and decoded (unset, =1).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/cont-dev2; M=/workspace/models/LFM3-Omni-2B
T=/workspace/data/ppl-windows/windows-32768.i32
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
tools/pascal/build_speech.sh cont-dev2-text > $O/build-text.log 2>&1; echo "build text rc=$?"; grep -h warnings $O/build-text.log
C=$B/cont-dev2-text/lfm3-continuation-check
$C $M $T exact-small > $O/exact-small.jsonl 2> $O/exact-small.err; echo "exact-small rc=$? $(tail -1 $O/exact-small.jsonl)"; cat $O/exact-small.err
LFM3_COMPACT_WINDOWS=1 $C $M $T exact-small > $O/exact-small-compact.jsonl 2> $O/exact-small-compact.err; echo "exact-small compact rc=$? $(tail -1 $O/exact-small-compact.jsonl)"; cat $O/exact-small-compact.err
$C $M $T decoded > $O/decoded.jsonl 2> $O/decoded.err; echo "decoded rc=$? $(grep -c decoded $O/decoded.jsonl)"; cat $O/decoded.jsonl $O/decoded.err
LFM3_COMPACT_WINDOWS=1 $C $M $T decoded > $O/decoded-compact.jsonl 2> $O/decoded-compact.err; echo "decoded compact rc=$? $(grep -c decoded $O/decoded-compact.jsonl)"; cat $O/decoded-compact.jsonl $O/decoded-compact.err
date
