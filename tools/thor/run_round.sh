#!/usr/bin/env bash
# One verification round on Thor (host), sequential so jobs never share the GPU; stops at first failure.
# The cuBLAS engine's regression round (G6): it builds -DLIQUID_LFM3_NATIVE_GEMM=OFF and the cuBLAS scorer explicitly,
# because the default engine is now lfm::gemm (G4) and its NLL baseline is the cuBLAS one (ppl-v1).
#   1 build native targets (cmake-g6-vR) and the scorer (score-vR) from the current source tree
#   2 perplexity: run_ppl.sh ppl-vR with score-vR, then ppl_compare (must PASS, inputs verified)
#   3 native NLL must be byte-identical to the committed baseline (validation-only scorer changes)
#   4 runtime red-team (fail-closed tokenizer, high-temperature sampling, scorer determinism and misuse)
#   5 every native gate (run_native_gates.sh)
# Usage: run_round.sh R > build/round-driver-vR.log 2>&1     Status: build/round-vR.status.json
# Refuses to start if any of this round's own outputs already exist (listed explicitly, so a driver
# log under another name cannot trip the check and nothing the round writes is missed).
set -uo pipefail
R=${1:-}; [[ "$R" =~ ^[0-9]+$ ]] || { echo "usage: run_round.sh ROUND_NUMBER"; exit 2; }
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; T=$W/liquid-CUDA/tools
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
st="$B/round-v$R.status.json"
status(){ printf '{"round":%s,"stage":"%s","rc":%s,"unix":%s}\n' "$R" "$1" "$2" "$(date +%s)" > "$st"; }
step(){ name=$1; shift; status "$name" null; "$@" > "$B/round-v$R.$name.log" 2>&1; rc=$?; [ $rc = 0 ] || { status "$name" $rc; exit $rc; }; }
outputs=("cmake-g6-v$R" "cmake-g6-v$R.cmake.log" "cmake-g6-v$R.build.log" "score-v$R" "ppl-v$R" "ppl-v$R.status.json"
         "ppl-v$R.reference.log" "ppl-v$R.native.log" "g6-redteam-v$R" "g6r$R-gates" "round-v$R.status.json")
for s in build-native build-scorer ppl compare nll-identity redteam gates; do outputs+=("round-v$R.$s.log"); done
for g in forward context sampling worker grouped-worker server cancellation http-cancellation 32k http-batch; do outputs+=("g6r$R-$g"); done
for o in "${outputs[@]}"; do test ! -e "$B/$o" || { echo "round $R output already exists: $B/$o"; exit 1; }; done
step build-native docker run --rm --network none -v $W/liquid-CUDA/native:/src/native:ro -v $W:/workspace $IMG \
  bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh /workspace/liquid-CUDA/build/cmake-g6-v$R -DLIQUID_LFM3_NATIVE_GEMM=OFF; rc=\$?; chown -R 1000:1000 /workspace/liquid-CUDA/build/cmake-g6-v$R /workspace/liquid-CUDA/build/cmake-g6-v$R.cmake.log /workspace/liquid-CUDA/build/cmake-g6-v$R.build.log 2>/dev/null; exit \$rc"
mkdir "$B/score-v$R"
step build-scorer docker run --rm --network none -v $W:/workspace $IMG \
  bash -c "bash /workspace/liquid-CUDA/tools/thor/build_scorer.sh /workspace/liquid-CUDA/build/score-v$R cublas; rc=\$?; chown -R 1000:1000 /workspace/liquid-CUDA/build/score-v$R; exit \$rc"
step ppl bash $T/thor/run_ppl.sh ppl-v$R score-v$R
step compare python3 $T/ppl_compare.py "$B/ppl-v$R"
status nll-identity null
for L in 256 257 4096 32768; do cmp -s "$B/ppl-v$R/nll-native-$L.f32" "$B/ppl-v1/nll-native-$L.f32" || { echo "native NLL differs at $L" > "$B/round-v$R.nll-identity.log"; status nll-identity 1; exit 1; }; done
echo "native NLL byte-identical to ppl-v1 at 256 257 4096 32768" > "$B/round-v$R.nll-identity.log"
step redteam bash $T/thor/redteam_g6_runtime.sh cmake-g6-v$R score-v$R g6-redteam-v$R ppl-v$R
step gates bash $T/thor/run_native_gates.sh cmake-g6-v$R g6r$R
status "done" 0
