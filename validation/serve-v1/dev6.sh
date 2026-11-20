# serve-v1 development (tools/pascal/remote.sh detach serve-dev6 'bash validation/serve-v1/dev6.sh'): the parity
# build's native-server check with the finer cancellation delays; rules 11 and 12 with the driver's full-queue fix.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; PY=/workspace/build/venv-ref/bin/python; mkdir -p $B/serve-dev6
LFM_ALLOCATION_METRICS=1 $PY tools/pascal/check_lfm3_native_server_parity.py --checkpoint /workspace/models/LFM3-Omni-2B --server $B/serve-d1p/liquid-lfm3-server --worker $B/serve-d1p/liquid-lfm3-worker --control-result validation/serve-v1/inputs/control.output.json --output $B/serve-dev6/native-server > $B/serve-dev6/native-server.log 2>&1; echo "parity native-server rc=$?"
$PY tools/pascal/serve_memory.py $B/serve-d1/liquid-lfm3-server /workspace/models/LFM3-Omni-2B $B/serve-dev4-shipping/control.output.json $B/serve-dev6/memory > $B/serve-dev6/memory.log 2>&1; echo "memory rc=$?"; grep -E 'rule|SERVE|rror' $B/serve-dev6/memory.log | tail -12
date
