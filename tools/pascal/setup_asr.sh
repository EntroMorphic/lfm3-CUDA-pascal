#!/usr/bin/env bash
# The speech scorer on the card (decision 9): Whisper large-v3 at the reference's pinned revision
# (download_asr_model.py in the reference's parent repository), run by tools/lfm3_asr.py as imported, in its own
# environment /workspace/build/venv-asr (venv-ref, the serving checks' pinned stack, stays as it is). torch 2.7.1 for
# CUDA 12.6 is a build with sm_61 kernels; the reference ran torch 2.10 on Thor. Transformers is the reference's 5.22.1.
# Run in the container: bash tools/pascal/setup_asr.sh   (network; no GPU work but the final check)
set -euo pipefail
V=/workspace/build/venv-asr; W=/workspace/models/whisper-large-v3
[ -e $V ] || python3 -m venv $V
$V/bin/pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1"
$V/bin/pip install -q "transformers==5.22.1" "huggingface_hub" "librosa" "soundfile" "numpy"
$V/bin/python - "$W" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
MODEL_ID = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
         "normalizer.json", "model.safetensors", "README.md"]
destination = Path(sys.argv[1])
snapshot_download(MODEL_ID, revision=REVISION, local_dir=destination, allow_patterns=FILES)
(destination / "model-source.json").write_text(
    json.dumps({"repo_id": MODEL_ID, "revision": REVISION, "files": FILES}, indent=2) + "\n")
PY
(cd $W && sha256sum $(ls | grep -v '^\.' | sort)) > $W.sha256
$V/bin/pip freeze > $V.freeze.txt
$V/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0)); x = torch.ones(64, 64, device='cuda', dtype=torch.float16); print('fp16 matmul', float((x @ x)[0, 0]))"
