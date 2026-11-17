#!/usr/bin/env bash
# lfm3-speech regression and refusal check (step 4 red-team): every regression must be bit-identical to the validated
# runs (split and unsplit generations; forced replays' hidden states and embeddings), every refusal must fail cleanly
# (nonzero status, a message, no outputs left). Usage (Thor host): speech_regression_check.sh BUILD_TAG
# Needs the validated runs in build/: speech-chunk-native-v1, speech-split-native-v3/requests, speech-e2e-v1, speech-v1.
set -uo pipefail
T=${1:?BUILD_TAG}
B=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $C/$T/lfm3-speech $M"
O=$B/$T-check; rm -rf $O; mkdir -p $O/req
same() { for x in $3; do cmp -s $B/$1.$x $O/$2.$x && echo "SAME   $2.$x" || echo "DIFFER $2.$x"; done; }
# regressions
for n in tts-long-male-s1 probe-other-long-s1; do $X $C/speech-split-native-v3/requests/$n.req.json $C/$T-check/split-$n >/dev/null || echo "RUN FAILED $n"; same speech-chunk-native-v1/native/$n split-$n "raw.i32 frames.i32 f32"; done
for n in tts-fox-female-s1 il-study-s1; do $X $C/speech-e2e-v1/requests/$n.req.json $C/$T-check/unsplit-$n >/dev/null || echo "RUN FAILED $n"; same speech-e2e-v1/native/$n unsplit-$n "raw.i32 frames.i32 f32"; done
for n in tts-fox-female tts-fox-male tts-museum-female tts-museum-male interleaved-study ref-interleaved; do $X $C/speech-v1/$n.forced.json $C/$T-check/forced-$n >/dev/null || echo "RUN FAILED forced $n"; same speech-v1/$n forced-$n "hidden.bf16 embeds.bf16 raw.i32 f32"; done
python3 - $O <<'PY'
import json, sys
o = sys.argv[1]
for n in ("tts-long-male-s1", "probe-other-long-s1"):
    j = json.load(open(f"{o}/split-{n}.json"))
    print(n, "chunks ended_by_eos (strict):", [c["ended_by_eos"] for c in j["chunks"]], "rtf", round(j["real_time_factor"], 2), "e2e", round(j["real_time_factor_end_to_end"], 2))
PY
# refusals
r=$(python3 -c "import json;print(json.dumps(json.load(open('$B/speech-e2e-v1/requests/tts-fox-female-s1.req.json'))['prompt_ids']))")
mk() { echo "$2" > $O/req/$1.json; out=$($X $C/$T-check/req/$1.json $C/$T-check/ref-$1 2>&1 >/dev/null); s=$?; left=$(ls $O/ref-$1.* 2>/dev/null | wc -l); echo "REFUSE $1: status $s, outputs left $left: $out"; }
mk empty-prompt '{"prompt_ids": [], "max_new_tokens": 64}'
mk empty-chunk "{\"chunks\": [$r, []], \"max_new_tokens\": 64}"
mk bad-mode "{\"prompt_ids\": $r, \"mode\": \"Audio\", \"max_new_tokens\": 64}"
mk placeholder "{\"prompt_ids\": [1, 125031, $(echo $r | tr -d '[]')], \"max_new_tokens\": 64}"
mk capacity "{\"prompt_ids\": $r, \"max_new_tokens\": 4090}"
mk split-no-messages "{\"prompt_ids\": $r, \"split\": true, \"max_new_tokens\": 64}"
# forced replays with an unused token, or an unused frame
python3 - $B $O <<'PY'
import json, sys
b, o = sys.argv[1:]
for name, key in (("forced-leftover-token", "forced_tokens"), ("forced-leftover-frame", "forced_frames")):
    r = json.load(open(f"{b}/speech-v1/tts-fox-female.forced.json"))
    r[key].append(r[key][-1])
    json.dump(r, open(f"{o}/req/{name}.json", "w"))
PY
for t in forced-leftover-token forced-leftover-frame; do out=$($X $C/$T-check/req/$t.json $C/$T-check/ref-$t 2>&1 >/dev/null); s=$?; echo "REFUSE $t: status $s, outputs left $(ls $O/ref-$t.* 2>/dev/null | wc -l): $out"; done
# a request file that is one of the stem's outputs is refused (and survives)
cp $B/speech-e2e-v1/requests/tts-fox-female-s1.req.json $O/self.json
out=$($X $C/$T-check/self.json $C/$T-check/self 2>&1 >/dev/null); st=$?; echo "REFUSE request-is-output: status $st, request kept $(test -s $O/self.json && echo yes || echo no): $out"
# stale optional outputs are removed: an earlier run left hidden/embeds under this stem
cp $O/forced-tts-fox-female.hidden.bf16 $O/unsplit-tts-fox-female-s1.hidden.bf16
$X $C/speech-e2e-v1/requests/tts-fox-female-s1.req.json $C/$T-check/unsplit-tts-fox-female-s1 >/dev/null
test -e $O/unsplit-tts-fox-female-s1.hidden.bf16 && echo "STALE hidden left" || echo "stale hidden removed"
