#!/usr/bin/env bash
# Compare the generated-token hash of two native binaries across a sweep of
# seed and sampling configurations.
#
# The batched decode work changed the runtime. The single-sequence path is
# supposed to be untouched, and the only way to know that is to compare the
# generated tokens of the old binary and the new one, configuration by
# configuration.
#
# usage: compare_hashes.sh BEFORE AFTER MODEL TOKENIZER [PROMPT]
set -euo pipefail

before=${1:?before binary required}
after=${2:?after binary required}
model=${3:?model required}
tokenizer=${4:?tokenizer required}
prompt=${5:-Hypothetically, might reflective recursion be a function of cognition?}

configs=(
  "--seed 1"
  "--seed 7 --temperature 0.528 --top-k 0"
  "--seed 42 --temperature 0.528 --top-k 0"
  "--seed 42 --temperature 0.528 --top-k 50"
  "--seed 42 --temperature 0.1 --top-k 50 --repetition-penalty 1.05"
  "--seed 42 --temperature 0.0 --top-k 65"
  "--seed 42 --top-p 0.9"
  "--seed 42 --min-p 0.05 --temperature 0.8"
  "--seed 123 --temperature 0.3 --top-k 20 --top-p 0.95"
  "--seed 999 --temperature 0.528 --top-k 0 --presence-penalty 0.2"
)

mismatches=0
checked=0
for config in "${configs[@]}"; do
  # shellcheck disable=SC2086
  old=$("$before" --model "$model" --tokenizer "$tokenizer" --prompt "$prompt" \
        $config 2>/dev/null | sed -n 's/^generated_fnv1a: //p')
  # shellcheck disable=SC2086
  new=$("$after" --model "$model" --tokenizer "$tokenizer" --prompt "$prompt" \
        $config 2>/dev/null | sed -n 's/^generated_fnv1a: //p')
  checked=$((checked + 1))
  if [[ "$old" == "$new" && -n "$old" ]]; then
    printf '  [same] %-70s %s\n' "$config" "$old"
  else
    printf '  [DIFF] %-70s before=%s after=%s\n' "$config" "$old" "$new"
    mismatches=$((mismatches + 1))
  fi
done
printf 'compared %d configurations, %d mismatches\n' "$checked" "$mismatches"
[[ "$mismatches" -eq 0 ]]
