#!/usr/bin/env bash
# G4 tier 2b diagnostics on Thor, after the gate (validation/g4-gates-v2/tier2b/README.md): the same comparison
#  - recomputed for the gate's own inputs, to add Student-t intervals (tier2b-recomputed.json; the verdict stays tier2b.json's);
#  - with attempt 2 (BF16 logits, bit for bit attempt 3 upstream), the like-for-like engine (diagnostic-attempt2-bf16-logits.json);
#  - against the anchor as declared (FP32 SDPA, three lengths), for both engines (robust-ppl-g4-v{3,2}.json).
# CPU only (reads NLL files). Usage: g4_tier2b_diagnostics.sh OUTDIR
set -uo pipefail
out=$1; R=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA; B=$R/build
mkdir -p "$out"
echo "== sources"; (cd $R && git rev-parse HEAD 2>/dev/null; sha256sum tools/ppl_fidelity.py tools/ppl_compare.py tools/thor/g4_tier2b_diagnostics.sh) | sed 's/^/  /'
F="python3 $R/tools/ppl_fidelity.py --references $B/ppl-v5 --engine native-gemm"
grep 'nll-native-gemm' $R/validation/g4-native-gates-v1/round4/sha256.txt > "$out/expected-attempt3.sha256"
grep 'nll-native-gemm' $R/validation/g4-native-gemm-v2/sha256.txt > "$out/expected-attempt2.sha256"
run(){ name=$1; shift; echo "== $name: $*"; $F "$@" --output "$out/$name.json"; echo "rc=$?"; }
run tier2b-recomputed --anchor $B/ppl-fp32-v2 --engine-dir $B/ppl-g4-v3 --expect-sha256 "$out/expected-attempt3.sha256"
run diagnostic-attempt2-bf16-logits --anchor $B/ppl-fp32-v2 --engine-dir $B/ppl-g4-v2 --expect-sha256 "$out/expected-attempt2.sha256"
# The declared anchor has no summary.json (it stopped at 32,768): windows from ppl-v5, lengths 256 257 4096; its
# precision is recorded per file (dtype) and in the session (TF32 disabled by ppl_reference --dtype float32).
for e in v3:attempt3 v2:attempt2; do
  run robust-ppl-g4-${e%%:*} --anchor $B/ppl-fp32-v1 --anchor-engine hf-sdpa-fp32 --windows-dir $B/ppl-v5 --lengths 256 257 4096 \
    --engine-dir $B/ppl-g4-${e%%:*} --expect-sha256 "$out/expected-${e#*:}.sha256"
done
