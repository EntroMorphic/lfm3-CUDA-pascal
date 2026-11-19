# serve-v1 development (tools/pascal/remote.sh detach serve-dev1 'bash validation/serve-v1/dev1.sh'): the text-only
# server build in both configurations; paired decode on the compact caches (40 steps); the reference's grouped-worker check.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; M=$B/serve-dev1; [ -e $M ] && exit 2; mkdir -p $M
tools/pascal/build_serve.sh serve-d1; echo "build rc=$?"
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_serve.sh serve-d1p; echo "build parity rc=$?"
python3 tools/pascal/paired_decode_check.py $B/serve-d1/liquid-lfm3-worker /workspace/models/LFM3-Omni-2B /workspace/data/ppl-windows/windows-32768.i32 $M/paired 40; echo "paired rc=$?"
R=validation/serve-v1/inputs; python3 tools/check_lfm3_batch_worker.py --checkpoint /workspace/models/LFM3-Omni-2B --worker $B/serve-d1/liquid-lfm3-worker --request $R/generation-request.json --oracle $B/serve-d1/text-sampling-oracle --long-request $R/decode_into32768.request.json --output $M/grouped-worker > $M/grouped-worker.log 2>&1; echo "grouped-worker rc=$? $(tail -3 $M/grouped-worker.log | tr '\n' ' ')"
date
