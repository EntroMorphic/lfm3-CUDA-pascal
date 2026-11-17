#!/usr/bin/env bash
# Runtime red-team for G6 and the perplexity scorer on Thor (host). Writes OUT/report.txt; exits
# nonzero if any expectation fails. Checks:
#   1 liquid-lfm3 fails closed on corrupted tokenizers (an id gap; an id beyond the output layer)
#   2 high-temperature, unrestricted sampling never emits an id the tokenizer does not define
#   3 the scorer is deterministic (byte-identical rescoring of the 256-token windows)
#   4 the scorer refuses misuse: the 4096-token windows file scored as 32768-token windows
#   5 top_k between the token count and the output rows is accepted (clamped); above the rows, rejected
# Usage: redteam_g6_runtime.sh BUILD_DIR_NAME SCORER_DIR_NAME OUT_NAME PPL_DIR_NAME
set -uo pipefail
build=$1; scorer_dir=$2; name=$3; ppl=$4
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B="$W/liquid-CUDA/build"; C=/workspace/liquid-CUDA/build
out="$B/$name"; test ! -e "$out" || { echo "exists: $out"; exit 1; }; mkdir "$out"
EX=(docker exec lfm3-profile-v1); BIN="$C/$build/liquid-lfm3"; SCORER="$C/$scorer_dir/lfm3-score"; M="$W/models/LFM3-Omni-2B"
fail=0
say() { echo "$*" | tee -a "$out/report.txt"; }

# 1. Fail closed on a corrupted tokenizer (checkpoint copies: symlinked weights/config, edited tokenizer.json).
for kind in gap outofrange; do
  d="$out/ckpt-$kind"; mkdir "$d"
  for f in config.json model.safetensors; do ln -s "../../../../models/LFM3-Omni-2B/$f" "$d/$f"; done
  python3 - "$M/tokenizer.json" "$d/tokenizer.json" "$kind" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
if sys.argv[3] == "gap":
    k = next(k for k, v in t["model"]["vocab"].items() if v == 1000)
    del t["model"]["vocab"][k]                     # hole at id 1000
else:
    t["added_tokens"][-1]["id"] = 128000           # beyond the output layer
json.dump(t, open(sys.argv[2], "w"))
PY
  printf '{"input_ids":[124894,1000],"max_new_tokens":2}' > "$out/req-$kind.json"
  "${EX[@]}" "$BIN" --experimental "$C/$name/ckpt-$kind" "$C/$name/req-$kind.json" "$C/$name/res-$kind.json" > "$out/run-$kind.log" 2>&1
  rc=$?
  last=$(tail -1 "$out/run-$kind.log")
  if [ "$rc" -ne 0 ] && grep -qE "not contiguous|outside the model vocabulary" "$out/run-$kind.log"; then
    say "PASS fail-closed[$kind]: rc=$rc $last"
  else
    say "FAIL fail-closed[$kind]: rc=$rc $last"; fail=1
  fi
done

# 2. High-temperature, unrestricted sampling stays inside the tokenizer's ids.
IDS=$(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['input_ids']))" "$B/forward-bounded-prefill-v1/baseline/generation-request.json")
for seed in 1 2 3; do
  printf '{"input_ids":%s,"max_new_tokens":128,"do_sample":true,"temperature":3.0,"top_k":0,"top_p":1.0,"seed":%s}' "$IDS" "$seed" > "$out/req-sample-$seed.json"
  "${EX[@]}" "$BIN" --experimental /workspace/models/LFM3-Omni-2B "$C/$name/req-sample-$seed.json" "$C/$name/res-sample-$seed.json" > "$out/run-sample-$seed.log" 2>&1
  rc=$?
  r=$(python3 -c "import json,sys;o=json.load(open(sys.argv[1]));g=o['generated_ids'];print(len(g),max(g),sum(t>=o['live_token_count'] for t in g),o['live_token_count'])" "$out/res-sample-$seed.json" 2>&1)
  read -r n mx dead live <<< "$r"
  if [ "$rc" -eq 0 ] && [ "${dead:-x}" = 0 ] && [ "${live:-x}" = 125033 ]; then
    say "PASS sample[seed=$seed]: tokens=$n max_id=$mx dead=0 live=$live"
  else
    say "FAIL sample[seed=$seed]: rc=$rc $r"; fail=1
  fi
done

# 3. Scorer determinism: rescore L=256 and compare bytes with the round's own perplexity run.
"${EX[@]}" "$SCORER" /workspace/models/LFM3-Omni-2B "$C/$ppl/windows-256.i32" 256 "$C/$name/nll-native-256.rerun.f32" > "$out/rescore.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && cmp -s "$out/nll-native-256.rerun.f32" "$B/$ppl/nll-native-256.f32"; then
  say "PASS scorer deterministic: L=256 rerun byte-identical"
else
  say "FAIL scorer determinism: rc=$rc $(cmp "$out/nll-native-256.rerun.f32" "$B/$ppl/nll-native-256.f32" 2>&1 | head -1)"; fail=1
fi

# 4. Scorer misuse: 72 x 4096 windows = 9 x 32768 ids; scoring them as 32768-token windows must fail loudly.
"${EX[@]}" "$SCORER" /workspace/models/LFM3-Omni-2B "$C/$ppl/windows-4096.i32" 32768 "$C/$name/misuse.f32" > "$out/misuse.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && grep -q "not \[delimiter\] + content" "$out/misuse.log" && [ ! -e "$out/misuse.f32" ]; then
  say "PASS scorer refuses 4096 windows as 32768: $(tail -1 "$out/misuse.log")"
else
  say "FAIL scorer misuse accepted: rc=$rc $(tail -1 "$out/misuse.log")"; fail=1
fi
# 5. top_k contract: validated against the 128000 output rows, then clamped to the 125033 tokens.
for case in "127000:0" "128001:1"; do
  k=${case%%:*}; want=${case#*:}
  printf '{"input_ids":%s,"max_new_tokens":4,"do_sample":true,"temperature":1.0,"top_k":%s,"seed":5}' "$IDS" "$k" > "$out/req-topk-$k.json"
  "${EX[@]}" "$BIN" --experimental /workspace/models/LFM3-Omni-2B "$C/$name/req-topk-$k.json" "$C/$name/res-topk-$k.json" > "$out/run-topk-$k.log" 2>&1
  rc=$?
  bounded=$(python3 -c "import json,sys;o=json.load(open(sys.argv[1]));print(int(o['live_token_count']==125033 and all(t<125033 for t in o['generated_ids'])))" "$out/res-topk-$k.json" 2>/dev/null)
  if { [ "$want" = 0 ] && [ "$rc" -eq 0 ] && [ "$bounded" = 1 ]; } || \
     { [ "$want" = 1 ] && [ "$rc" -ne 0 ] && grep -q "top_k exceeds vocabulary size" "$out/run-topk-$k.log" && [ ! -e "$out/res-topk-$k.json" ]; }; then
    say "PASS top_k=$k: rc=$rc (expected $([ "$want" = 0 ] && echo accepted || echo rejected))"
  else
    say "FAIL top_k=$k: rc=$rc $(tail -1 "$out/run-topk-$k.log")"; fail=1
  fi
done
exit "$fail"
