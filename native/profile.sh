#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

model="${MODEL:-models/LFM2.5-350M/model.safetensors}"
tokenizer="${TOKENIZER:-models/LFM2.5-350M/tokenizer.json}"
output="${PROFILE_OUTPUT:-artifacts/native-profile}"
prompt='Hypothetically, might reflective recursion be a function of cognition?'

mkdir -p "$output"
native/build.sh
nvcc -O3 -std=c++20 --default-stream per-thread -arch=sm_110 \
  native/bench/roofline.cu -lcublas -o "$output/roofline"

for run in 1 2 3; do
  "$output/roofline" | tee "$output/roofline-$run.txt"
done

for run in 1 2 3; do
  native/build/liquid-native \
    --model "$model" \
    --tokenizer "$tokenizer" \
    --prompt "$prompt" \
    > "$output/control-graph-$run.txt"
done

nsys profile \
  --force-overwrite=true \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --output="$output/control-nodes" \
  native/build/liquid-native \
    --model "$model" \
    --tokenizer "$tokenizer" \
    --prompt "$prompt" \
    > "$output/control-nodes-run.txt"

nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_api_sum \
  "$output/control-nodes.nsys-rep" \
  > "$output/control-nodes-stats.txt"

for result in "$output"/control-graph-*.txt "$output/control-nodes-run.txt"; do
  grep -qx 'generated_tokens: 788' "$result"
  grep -qx 'generated_fnv1a: 15144752490453972299' "$result"
done
grep -hE '^(generated_tokens|generated_fnv1a|generated_tokens_per_second):' \
  "$output"/control-graph-*.txt "$output/control-nodes-run.txt"
