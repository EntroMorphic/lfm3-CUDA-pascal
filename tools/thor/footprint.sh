#!/usr/bin/env bash
# Report the on-disk footprint of the inference stacks (run inside the runtime container).
set -uo pipefail
B=/workspace/liquid-CUDA/build
mb(){ awk -v b="$(stat -Lc %s "$1")" 'BEGIN{printf "%9.1f MB", b/1e6}'; }
echo "== native binaries"
for f in $B/cmake-bounded-prefill-v1/liquid-lfm3 $B/thin-hybrid-runtime-candidate-v4/build/liquid-lfm3; do echo "$(mb $f)  $f"; done
echo "== shared libraries the production binary loads (pinned env)"
ldd $B/cmake-bounded-prefill-v1/liquid-lfm3 | awk '$3 ~ /^\// {print $1, $3}' | while read -r n p; do echo "$(mb "$p")  $n -> $p"; done
echo "== candidate (no-cuBLAS-linear) binary still links"
ldd $B/thin-hybrid-runtime-candidate-v4/build/liquid-lfm3 | awk '$3 ~ /^\// {print $1}' | tr '\n' ' '; echo
echo "== Python/Transformers stack"
SP=$(ls -d /opt/venv/lib/python3*/site-packages)
du -sh "$SP" "$SP"/torch "$SP"/nvidia "$SP"/transformers "$SP"/triton 2>/dev/null
echo "== model files"
du -sh /workspace/models/LFM3-Omni-2B/model.safetensors /workspace/models/LFM3-Omni-2B/audio_tokenizer /workspace/models/LFM3-Omni-2B 2>/dev/null
