# serve-v1 development (tools/pascal/remote.sh detach serve-dev3 'bash validation/serve-v1/dev3.sh'): the reference's
# serving gates on dev1's builds: the shipping build against its own control, the parity build against the pinned one.
set -x; date; cd /workspace/lfm3-CUDA-pascal
tools/pascal/serve_gates.sh serve-d1 serve-dev3-shipping own; echo "shipping rc=$?"
tools/pascal/serve_gates.sh serve-d1p serve-dev3-parity pinned; echo "parity rc=$?"
date
