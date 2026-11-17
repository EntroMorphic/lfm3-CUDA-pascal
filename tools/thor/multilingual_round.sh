#!/usr/bin/env bash
# validation/multilingual-speech-v1 on Thor (host). Phases, in order (each may be run alone, later ones need prep):
#   prep      the fixed cases from FLEURS (tools/fleurs_cases.py) and every request (HF's processor, no model loaded)
#   exact     rules 1 and 8, exactness only (may share Thor): native rendering and splitting against HF's ids, and the
#             8-row batch against alone
#   tts long interleaved   rules 2, 3, 5: HF and native, Whisper scoring, reports
#   asr       rule 4: FLEURS recognition, HF and native
#   escalate  rule 2's escalation: seeds 4-15 for the cases tts's report flagged
#   server    rules 6 and 7 (exact; may share Thor)
#   listen    rule 9: Tripp's blind sets in outputs/multilingual/
# Usage: multilingual_round.sh TAG SPEECH_TAG SERVER_TAG PHASE [PHASE ...]
#   TAG: build/ml-TAG holds everything; SPEECH_TAG/SERVER_TAG: build dirs of lfm3-speech and liquid-lfm3-server.
# FLEURS must already be at data/fleurs/CONFIG/{test.tsv,audio/test/*.wav} (rule 0; downloaded only with Tripp's OK).
set -uo pipefail
tag=$1; st=$2; sv=$3; shift 3; phases="$*"
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; E=/workspace/liquid-CUDA/build
O=$B/ml-$tag; C=$E/ml-$tag; M=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools
F=/workspace/data/fleurs; WH=/workspace/models/whisper-large-v3
EXN="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="docker exec -e PYTHONUTF8=1 -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev /opt/venv/bin/python3"
BIN=$E/$st/lfm3-speech
fail(){ echo "FAIL: $1"; exit 1; }
mkdir -p $O && cd $O
has(){ case " $phases " in *" $1 "*) return 0;; esac; return 1; }
names(){ ls $O/$1/requests | sed 's/\.req\.json$//'; }
native_batch(){  # SET: lfm3-speech over SET/requests into SET/native, the model loaded once
  mkdir -p $O/$1/native; for n in $(names $1); do echo "$C/$1/requests/$n.req.json $C/$1/native/$n"; done > $O/$1/list.txt
  $EXN $BIN $M --batch $C/$1/list.txt > $O/$1/native.log 2>&1 || fail "native $1"
}
score(){  # SET ENGINE_DIR(relative to SET) PROMPTS
  docker exec -e PYTHONUTF8=1 lfm3-omni-dev bash -c "/opt/venv/bin/python3 $T/lfm3_asr_multilingual.py $3 $WH $C/$1/asr-$(basename $2).json $C/$1/$2/*" \
    > $O/$1/asr-$(basename $2).log 2>&1 || fail "asr $1 $2"
}
echo "== $tag: $phases $(date +%T)"; sha256sum $B/$st/lfm3-speech $B/$sv/liquid-lfm3-server > $O/binaries-$(date +%H%M%S).sha256
(cd $R && sha256sum tools/fleurs_cases.py tools/lfm3_asr_multilingual.py tools/lfm3_asr.py tools/check_lfm3_multilingual_asr.py \
  tools/check_lfm3_multilingual_server.py tools/lfm3_multilingual_report.py tools/lfm3_speech_report.py tools/lfm3_speech_hf.py \
  tools/lfm3_speech_chunked_hf.py tools/lfm3_audio_in_hf.py tools/thor/multilingual_round.sh) > $O/sources-$(date +%H%M%S).sha256
bash $R/tools/thor/power_state.sh $O/power-start-$(date +%H%M%S).json || fail "power state"

if has prep; then
  echo "== prep $(date +%T)"
  $PY $T/fleurs_cases.py $F $C/cases > prep.log 2>&1 || fail "cases"; cat prep.log
  for s in tts interleaved; do $PY $T/lfm3_speech_hf.py $M $C/cases/$s.json $C/$s requests >> prep.log 2>&1 || fail "requests $s"; done
  $PY $T/lfm3_speech_chunked_hf.py $M $C/cases/long.json $C/long requests >> prep.log 2>&1 || fail "requests long"
  $PY $T/lfm3_audio_in_hf.py $M $C/cases/asr-cases.json $C/asr requests >> prep.log 2>&1 || fail "requests asr"
  for s in tts long interleaved asr; do echo "  $s: $(ls $O/$s/requests | wc -l) requests"; done
fi

if has exact; then  # rules 1 and 8: exactness only; safe to run beside other GPU work (timings are not used)
  echo "== exact $(date +%T)"; X=$O/exact; mkdir -p $X/ids $X/messages $X/alone $X/conc
  # seed 1 of every TTS (sentence k0) and long case, every interleaved k0, the first 2 ASR clips per language
  pick=$( (ls $O/tts/requests | grep -E -- '-k0-s1\.req\.json$' | sed 's#^#tts/#'; ls $O/long/requests | grep -- '-s1\.req\.json$' | sed 's#^#long/#'
           ls $O/interleaved/requests | grep -E -- '-k0-s1\.req\.json$' | sed 's#^#interleaved/#'
           ls $O/asr/requests | awk -F- '{n[$2]++} n[$2]<=2' | sed 's#^#asr/#') | sed 's/\.req\.json$//')
  : > $X/list-ids.txt; : > $X/list-messages.txt
  for p in $pick; do n=$(basename $p)
    echo "$C/$p.req.json $C/exact/ids/$n" >> $X/list-ids.txt
    python3 - "$O/$p.req.json" "$X/messages/$n.req.json" "$O/cases/asr-cases.json" "$n" <<'EOF' || fail "messages request"
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
out = dict(mode=r["mode"], max_new_tokens=r["max_new_tokens"], seed=r["seed"])
if "audio" in r:  # audio-in requests carry no conversation: the case's system text and its clips as audio parts
    spec = json.load(open(sys.argv[3], encoding="utf-8"))
    c = next(c for c in spec["cases"] if c["case"] == sys.argv[4])
    out["messages"] = [{"role": "system", "content": spec["systems"][c["system"]]},
                       {"role": "user", "content": [{"type": "audio", "audio": a} for a in r["audio"]]}]
else:
    out["messages"] = r["conversation"]
    if "chunks" in r:
        out["split"] = True  # the native splitter against HF's split_text (the chunks' prompts)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
EOF
    echo "$C/exact/messages/$n.req.json $C/exact/messages/$n" >> $X/list-messages.txt
  done
  $EXN $BIN $M --batch $C/exact/list-ids.txt > $X/ids.log 2>&1 || fail "ids run"
  $EXN $BIN $M --batch $C/exact/list-messages.txt > $X/messages.log 2>&1 || fail "messages run"
  python3 $R/tools/compare_speech_outputs.py $X/ids $X/messages $X/rule1.json > /dev/null && echo "rule 1: messages == ids, PASS" || echo "rule 1: FAIL"
  # rule 8: the TTS/interleaved subset (one capacity) in an 8-row batch, staggered, against alone
  grep -E "/(tts|interleaved)/" $X/list-ids.txt | sed "s#$C/exact/ids/#$C/exact/alone/#" > $X/list-alone.txt
  sed "s#$C/exact/alone/#$C/exact/conc/#" $X/list-alone.txt > $X/list-conc.txt
  $EXN $BIN $M --batch $C/exact/list-alone.txt > $X/alone.log 2>&1 || fail "alone run"
  $EXN $BIN $M --concurrent 8 $C/exact/list-conc.txt 7 > $X/conc.log 2>&1 || fail "concurrent run"
  python3 $R/tools/compare_speech_outputs.py $X/alone $X/conc $X/rule8.json > /dev/null && echo "rule 8: batch == alone, PASS" || echo "rule 8: FAIL"
fi

for s in tts long interleaved; do has $s || continue
  echo "== $s $(date +%T)"
  if [ $s = long ]; then
    $PY $T/lfm3_speech_chunked_hf.py $M $C/cases/long.json $C/long fresh > $O/long/hf.log 2>&1 || fail "hf long"; hf=hf-fresh
  else
    $PY $T/lfm3_speech_hf.py $M $C/cases/$s.json $C/$s > $O/$s/hf.log 2>&1 || fail "hf $s"; hf=hf
  fi
  native_batch $s
  $PY $T/lfm3_speech_cases.py $M $C/$s/requests $C/$s/native $C/$s/native-cases $(names $s) > $O/$s/cases.log 2>&1 || fail "cases $s"
  score $s native-cases $C/cases/$s.json; score $s $hf $C/cases/$s.json
  $PY $T/lfm3_speech_report.py $C/cases/$s.json $C/$s/asr-native-cases.json $C/$s/asr-$hf.json $C/$s/native-cases $C/$s/$hf \
    $C/$s/report.json > $O/$s/report.md 2>&1 || fail "report $s"
  $PY $T/lfm3_multilingual_report.py $C/cases/$s.json $C/$s/report.json $C/$s/asr-native-cases.json $C/$s/asr-$hf.json \
    $C/$s/locales.json | tee $O/$s/locales.md
  if [ $s = long ]; then  # rule 3: chunks per paragraph and how each chunk ended, both engines
    python3 - $O/long <<'EOF' | tee $O/long/chunks.md
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {}
for f in sorted((d / "native").glob("*.json")):
    out.setdefault(f.stem, {})["native"] = [c["ended_by_eos"] for c in json.loads(f.read_text()).get("chunks", [])]
for f in sorted((d / "hf-fresh").glob("*/summary.json")):
    out.setdefault(f.parent.name, {})["hf"] = [c["ended_by_eos"] for c in json.loads(f.read_text())["chunks"]]
print("| case | chunks (native, HF) | ended by EOS (native, HF) |\n|---|---|---|")
for k, v in out.items():
    n, h = v.get("native", []), v.get("hf", [])
    print(f"| {k} | {len(n)}, {len(h)} | {sum(n)}/{len(n)}, {sum(h)}/{len(h)} |")
json.dump(out, open(d / "chunks.json", "w"), indent=1)
EOF
  fi
done

if has asr; then
  echo "== asr $(date +%T)"
  $PY $T/lfm3_audio_in_hf.py $M $C/cases/asr-cases.json $C/asr > $O/asr/hf.log 2>&1 || fail "hf asr"
  native_batch asr
  $PY $T/check_lfm3_multilingual_asr.py $M $WH $C/cases/asr-cases.json $C/asr $C/asr/native $C/asr/check.json \
    > $O/asr/check.log 2>&1; echo "asr check exit $?"; tail -3 $O/asr/check.log
fi

if has escalate; then  # rule 2: seeds 4-15 for every flagged TTS case, both engines
  echo "== escalate $(date +%T)"; mkdir -p $O/escalate
  python3 - $O/tts/report.json $O/cases/tts-escalate.json $O/escalate/prompts.json <<'EOF' || fail "escalation prompts"
import json, sys
flagged = [k for k, v in json.load(open(sys.argv[1]))["cases"].items() if v["c1_flag"]]
spec = json.load(open(sys.argv[2], encoding="utf-8"))
spec["cases"] = [c for c in spec["cases"] if c["case"] in flagged]
json.dump(spec, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(flagged)} flagged: {flagged}")
EOF
  $PY $T/lfm3_speech_hf.py $M $C/escalate/prompts.json $C/escalate > $O/escalate/hf.log 2>&1 || fail "hf escalate"
  native_batch escalate
  $PY $T/lfm3_speech_cases.py $M $C/escalate/requests $C/escalate/native $C/escalate/native-cases $(names escalate) > $O/escalate/cases.log 2>&1 || fail "cases escalate"
  score escalate native-cases $C/escalate/prompts.json; score escalate hf $C/escalate/prompts.json
  # seeds 1-15: this round's 4-15 with tts's 1-3 (lfm3_speech_report reads one directory per engine)
  for e in native-cases hf; do for d in $O/tts/$e/*-s[123]; do n=$(basename $d); case=${n%-s*}
    grep -q "\"$case\"" $O/escalate/prompts.json && cp -r $d $O/escalate/$e/; done; done
  python3 - $O/escalate/prompts.json $O/tts <<'EOF' || fail "seed merge"
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8")); p["seeds"] = list(range(1, 16))
json.dump(p, open(sys.argv[1].replace("prompts.json", "prompts-1-15.json"), "w", encoding="utf-8"), ensure_ascii=False)
for e in ("native-cases", "hf"):  # one ASR json per engine over seeds 1-15
    a = json.load(open(f"{sys.argv[2]}/asr-{e}.json", encoding="utf-8"))
    b_path = sys.argv[1].replace("prompts.json", f"asr-{e}.json")
    b = json.load(open(b_path, encoding="utf-8"))
    cases = {c["case"] for c in p["cases"]}
    b["results"] += [r for r in a["results"] if r["case"].rsplit("-s", 1)[0] in cases]
    json.dump(b, open(b_path.replace(".json", "-1-15.json"), "w", encoding="utf-8"), ensure_ascii=False)
EOF
  $PY $T/lfm3_speech_report.py $C/escalate/prompts-1-15.json $C/escalate/asr-native-cases-1-15.json $C/escalate/asr-hf-1-15.json \
    $C/escalate/native-cases $C/escalate/hf $C/escalate/report.json > $O/escalate/report.md 2>&1 || fail "report escalate"
  cat $O/escalate/report.md
fi

if has server; then
  echo "== server $(date +%T)"; mkdir -p $O/server
  docker exec -e PYTHONUTF8=1 -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 \
    $T/check_lfm3_multilingual_server.py $E/$sv/liquid-lfm3-server $BIN $M $C/cases $C/server/work $C/server/check.json \
    > $O/server/check.log 2>&1; echo "server check exit $?"; tail -1 $O/server/check.log
fi

if has listen; then  # rule 9: seed 1, sentence k0 of every locale x voice, and every long-form paragraph
  echo "== listen $(date +%T)"; L=$W/outputs/multilingual; mkdir -p $L
  python3 - $O/cases/tts.json $O/listen-tts.json <<'EOF' || fail "listening prompts"
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")); s["cases"] = [c for c in s["cases"] if c["case"].endswith("-k0")]
json.dump(s, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
EOF
  python3 $R/tools/lfm3_listening_set.py $O/listen-tts.json $O/tts/native-cases $O/tts/hf $L/tts $O/listen-tts.key.json
  python3 $R/tools/lfm3_listening_set.py $O/cases/long.json $O/long/native-cases $O/long/hf-fresh $L/long $O/listen-long.key.json
fi
bash $R/tools/thor/power_state.sh $O/power-end-$(date +%H%M%S).json || echo "power state changed (recorded)"
echo "MULTILINGUAL DONE $(date +%T)"
