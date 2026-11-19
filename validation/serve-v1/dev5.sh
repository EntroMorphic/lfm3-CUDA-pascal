# serve-v1 development (tools/pascal/remote.sh detach serve-dev5 'bash validation/serve-v1/dev5.sh'): the parity build's
# gates with its timing scaled (the amendment); rules 11 and 12 through the shipping build's server.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build
tools/pascal/serve_gates.sh serve-d1p serve-dev5-parity pinned; echo "parity rc=$?"
/workspace/build/venv-ref/bin/python tools/pascal/serve_memory.py $B/serve-d1/liquid-lfm3-server /workspace/models/LFM3-Omni-2B $B/serve-dev4-shipping/control.output.json $B/serve-dev5-memory > $B/serve-dev5-memory.log 2>&1; echo "memory rc=$?"; grep -E 'rule|SERVE|Error|error' $B/serve-dev5-memory.log | tail -20
date
