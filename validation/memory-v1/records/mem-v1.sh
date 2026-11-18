set -x; date -u; cd /workspace/lfm3-CUDA-pascal
tools/pascal/build_text.sh text-v2 || exit 1
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_text.sh text-tc || exit 1
tools/pascal/verify_memory.sh text-v1 text-v2 ppl-v1 fast; echo "verify fast rc=$?"
python3 tools/pascal/compare_pins.py text-tc; echo "pins tc rc=$?"
python3 tools/pascal/compare_pins.py text-v2 arithmetic single; echo "pins plain rc=$?"
date -u
tools/pascal/verify_memory.sh text-v1 text-v2 ppl-v1 slow; echo "verify slow rc=$?"
date -u
