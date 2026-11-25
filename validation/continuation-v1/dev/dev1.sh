# continuation-v1 development 1 (tools/pascal/remote.sh detach cont-dev1 'bash validation/continuation-v1/dev/dev1.sh'):
# builds the shipping text CLI with lfm3-continuation-check, and the speech and parity servers (compile check), then
# runs the check's modes once.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/cont-dev1; M=/workspace/models/LFM3-Omni-2B
T=/workspace/data/ppl-windows/windows-32768.i32
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O; ls -l $T
tools/pascal/build_speech.sh cont-dev1-text > $O/build-text.log 2>&1; echo "build text rc=$?"; grep -h warnings $O/build-text.log
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh cont-dev1-parity > $O/build-parity.log 2>&1; echo "build parity rc=$?"; grep -h warnings $O/build-parity.log
C=$B/cont-dev1-text/lfm3-continuation-check
$C $M $T exact-small > $O/exact-small.jsonl 2> $O/exact-small.err; echo "exact-small rc=$?"; cat $O/exact-small.jsonl $O/exact-small.err
$C $M $T exact > $O/exact.jsonl 2> $O/exact.err; echo "exact rc=$?"; cat $O/exact.jsonl $O/exact.err
LFM3_COMPACT_WINDOWS=1 $C $M $T exact-small > $O/exact-small-compact.jsonl 2> $O/exact-small-compact.err; echo "exact-small compact rc=$?"; cat $O/exact-small-compact.jsonl $O/exact-small-compact.err
$C $M $T speed > $O/speed.jsonl 2> $O/speed.err; echo "speed rc=$?"; cat $O/speed.jsonl $O/speed.err
$C $M $T decoded > $O/decoded.jsonl 2> $O/decoded.err; echo "decoded rc=$?"; cat $O/decoded.jsonl $O/decoded.err
date
