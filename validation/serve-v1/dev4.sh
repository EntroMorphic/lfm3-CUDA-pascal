# serve-v1 development (tools/pascal/remote.sh detach serve-dev4 'bash validation/serve-v1/dev4.sh'): dev3 again with the
# pinned front-end stack (venv-ref) and the 8 GB worker check.
set -x; date; cd /workspace/lfm3-CUDA-pascal
tools/pascal/serve_gates.sh serve-d1 serve-dev4-shipping own; echo "shipping rc=$?"
tools/pascal/serve_gates.sh serve-d1p serve-dev4-parity pinned; echo "parity rc=$?"

# How long the parity build's prefill of the reference's 32,767-token request takes (the gates' timeouts are Thor's).
python3 -c "import json; r=json.load(open('validation/serve-v1/inputs/decode_into32768.request.json')); json.dump(dict(input_ids=r['input_ids'][:32763], capacity=32768, experimental_long_context=True, max_new_tokens=1), open('/workspace/build/serve-dev4-parity/p32k.json','w'))"
for b in serve-d1 serve-d1p; do s=$(date +%s.%N); /workspace/build/$b/liquid-lfm3 --experimental /workspace/models/LFM3-Omni-2B /workspace/build/serve-dev4-parity/p32k.json /workspace/build/serve-dev4-parity/p32k.$b.json > /dev/null 2>&1; echo "$b 32,763-token prefill rc=$? wall $(python3 -c "import time;print(round(time.time()-$s,1))") s, prefill_seconds $(python3 -c "import json;print(json.load(open('/workspace/build/serve-dev4-parity/p32k.$b.json'))['prefill_seconds'])")"; done
date
