#!/usr/bin/env bash
# Compare two native binaries on the attention paths the runtime supports.
#
# The decode attention kernel has three paths: the full prefix, the windowed
# path, and the sparse path. A change to the kernel has to leave all of them
# byte-identical, so this compares the generated-token hash of two binaries on
# each.
#
# usage: compare_attention_paths.sh BEFORE AFTER MODEL TOKENIZER
set -euo pipefail

before=${1:?before binary required}
after=${2:?after binary required}
model=${3:?model required}
tokenizer=${4:?tokenizer required}
prompt="Hypothetically, might reflective recursion be a function of cognition?"

paths=(
  ""
  "--attention-window 64"
  "--attention-window 64 --attention-sinks 4 --attention-sparse-after 8"
  "--attention-window 128 --attention-sinks 8 --attention-sparse-after 16"
)

mismatches=0
for path in "${paths[@]}"; do
  # shellcheck disable=SC2086
  old=$("$before" --model "$model" --tokenizer "$tokenizer" --prompt "$prompt" $path 2>/dev/null |
        sed -n 's/^generated_fnv1a: //p')
  # shellcheck disable=SC2086
  new=$("$after" --model "$model" --tokenizer "$tokenizer" --prompt "$prompt" $path 2>/dev/null |
        sed -n 's/^generated_fnv1a: //p')
  if [[ "$old" == "$new" && -n "$old" ]]; then
    printf '  [same] %-58s %s\n' "${path:-full prefix}" "$old"
  else
    printf '  [DIFF] %-58s before=%s after=%s\n' "${path:-full prefix}" "$old" "$new"
    mismatches=$((mismatches + 1))
  fi
done
printf 'compared %d attention paths, %d mismatches\n' "${#paths[@]}" "$mismatches"
[[ "$mismatches" -eq 0 ]]
