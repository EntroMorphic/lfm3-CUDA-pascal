# serve-v1 development (tools/pascal/remote.sh detach serve-dev4 'bash validation/serve-v1/dev4.sh'): dev3 again with the
# pinned front-end stack (venv-ref) and the 8 GB worker check.
set -x; date; cd /workspace/lfm3-CUDA-pascal
tools/pascal/serve_gates.sh serve-d1 serve-dev4-shipping own; echo "shipping rc=$?"
tools/pascal/serve_gates.sh serve-d1p serve-dev4-parity pinned; echo "parity rc=$?"
date
