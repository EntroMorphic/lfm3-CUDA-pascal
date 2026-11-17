#!/usr/bin/env bash
# The parity gate (PLAN.md decision 4). Runs in the container: builds the validation engine (the prefill GEMM with
# Blackwell's tensor-core arithmetic) from this tree and requires every trace file of the four forward fixtures to
# equal the Thor engine's pins, 7,826 of 7,826. Exit 0 only then.
#   tools/pascal/remote.sh run 'tools/pascal/gate_pins.sh [TAG=parity]'
set -euo pipefail
tag=${1:-parity}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL "$here/build_text.sh" "$tag"
python3 "$here/compare_pins.py" "$tag"
