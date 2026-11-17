#!/usr/bin/env bash
# Batched-decode evidence round (P0 speed) on Thor (host): every gate and measurement on one build of the tools.
# Usage: batch_round.sh TAG TOOLS_TAG   (TOOLS_TAG from build_batch_tools.sh)
# 1 row independence (6 chat prompts, alone vs together, greedy and sampled); 2 throughput (batch 1..64, warm-up, two
# runs each; sampled at 32); 3 teacher-forced decode fidelity vs FP32/SDPA (32 x 1,024 positions); 4 device token
# selection vs the host sampler; 5 GEMM bench and tile sweep; 6 coherency texts (decoded with the HF tokenizer).
set -uo pipefail
tag=$1; tools=$2
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=$B/$tag; test ! -e $O || { echo "exists $tag"; exit 1; }; mkdir -p $O
T=$C/$tools; M=/workspace/models/LFM3-Omni-2B
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
PY="docker exec -i lfm3-omni-dev /opt/venv/bin/python3"
fail(){ echo "FAIL: $1"; exit 1; }
mem(){ awk -v t="$1" '/MemTotal/{T=$2}/MemFree/{F=$2}/MemAvailable/{A=$2}/^Cached/{K=$2}/Buffers/{U=$2}
  END{printf "%s %s used %.1f GiB available %.1f GiB\n", strftime("%T"), t, (T-F-K-U)/1048576, A/1048576}' /proc/meminfo | tee -a $O/memory.log; }
echo "== sources"; (cd $R && sha256sum tools/thor/batch_round.sh tools/batch_decode_fidelity.py tools/check_gpu_select.py) | sed 's/^/  /'
sed 's/^/  /' $B/$tools/sources.sha256; (cd $B/$tools && sha256sum lfm3-batch lfm3-batch-teacher lfm3-gpu-select-check) | sed 's/^/  /'
grep EnableSystemMemoryPools /proc/driver/nvidia/params | sed 's/^/  /'; mem start
python3 - "$B" "$O" <<'P' || exit 1
import json, sys
B, O = sys.argv[1:]
names = ("sky", "code", "arith", "poem", "memory")
prompts = [json.load(open(f"{B}/g4-coh-v2/native-{n}-512.json"))["input_ids"] for n in names]
prompts.append(json.load(open(f"{B}/cli-demo/out2.json"))["input_ids"])
json.dump(dict(names=list(names) + ["reflective"], prompts=prompts), open(f"{O}/prompts.json", "w"))
for mode, t in (("greedy", 0), ("sampled", 0.528)):
    for serial in (False, True):
        json.dump(dict(capacity=4096, prompts=prompts, batch=6, max_new_tokens=512, temperature=t, seed=42, serial=serial),
                  open(f"{O}/rows-{mode}-{'serial' if serial else 'batch'}.json", "w"))
p = prompts[5]
for n in (1, 8, 16, 32, 64):
    json.dump(dict(capacity=4096, prompts=[p], batch=n, max_new_tokens=256, temperature=0, stop=[], warmup=True), open(f"{O}/rate-{n}.json", "w"))
json.dump(dict(capacity=4096, prompts=[p], batch=32, max_new_tokens=256, temperature=0.528, seed=42, stop=[], warmup=True), open(f"{O}/rate-32-sampled.json", "w"))
P
echo "== 1 row independence"
for f in rows-greedy-serial rows-greedy-batch rows-sampled-serial rows-sampled-batch; do
  $EX $T/lfm3-batch $M ${O/$B/$C}/$f.json ${O/$B/$C}/$f.out.json > /dev/null || fail $f; done
python3 - "$O" <<'P' || exit 1
import json, sys
O = sys.argv[1]
ok = True
for m in ("greedy", "sampled"):
    a, b = (json.load(open(f"{O}/rows-{m}-{k}.out.json"))["rows"] for k in ("serial", "batch"))
    same = [x["generated_ids"] == y["generated_ids"] for x, y in zip(a, b)]
    ok &= all(same)
    print(f"  {m}: {sum(same)}/{len(same)} rows identical alone and in the batch; tokens {[len(x['generated_ids']) for x in a]}")
sys.exit(0 if ok else 1)
P
echo "== 2 throughput"; mem "before rates"
for n in 1 8 16 32 64; do for i in 1 2; do printf "  batch $n run $i: "; $EX $T/lfm3-batch $M ${O/$B/$C}/rate-$n.json ${O/$B/$C}/rate-$n-$i.out.json || fail "rate $n"; done; done
for i in 1 2; do printf "  batch 32 sampled run $i: "; $EX $T/lfm3-batch $M ${O/$B/$C}/rate-32-sampled.json ${O/$B/$C}/rate-32-sampled-$i.out.json || fail sampled; done
python3 - "$O" <<'P' || exit 1
import json, sys
O = sys.argv[1]
ref = json.load(open(f"{O}/rate-1-1.out.json"))["rows"][0]["generated_ids"]
ok = all(r["generated_ids"] == ref for n in (1, 8, 16, 32, 64) for i in (1, 2) for r in json.load(open(f"{O}/rate-{n}-{i}.out.json"))["rows"])
print("  every row of every batch size equals batch 1:", ok); sys.exit(0 if ok else 1)
P
mem "after rates"
echo "== 3 teacher-forced decode fidelity"
$EX $T/lfm3-batch-teacher $M $C/ppl-v5/windows-4096.i32 4096 3072 32 ${O/$B/$C}/teacher || fail teacher
SHA=$(python3 -c "import json;print(json.load(open('$B/ppl-v5/summary.json'))['windows']['4096']['sha256'])")
$PY $C/../tools/batch_decode_fidelity.py ${O/$B/$C}/teacher 4096 3072 $C/ppl-fp32-v2/nll-hf-eager_blocked-fp32-4096.f32 \
  $C/ppl-v5/nll-hf-sdpa-4096.f32 $C/ppl-v5/nll-hf-eager-4096.f32 $C/ppl-v5/windows-4096.i32 $SHA ${O/$B/$C}/fidelity.json > $O/fidelity.log || fail fidelity
grep -E '"verdict"|"pass"' $O/fidelity.log | sed 's/^/  /'
echo "== 4 device token selection"; mkdir -p $O/select
$EX $T/lfm3-gpu-select-check ${O/$B/$C}/select > $O/select.log || fail select; cat $O/select.log | sed 's/^/  /'
$PY /workspace/liquid-CUDA/tools/check_gpu_select.py ${O/$B/$C}/select 4 | tee $O/select-chi2.log | sed 's/^/  /'; grep -q PASS $O/select-chi2.log || fail chi2
echo "== 5 GEMM bench, tile sweep"
$EX $T/decode-gemm-bench > $O/gemm-bench.log || fail bench; $EX $T/tile-sweep > $O/tile-sweep.log || fail sweep
tail -1 $O/gemm-bench.log | sed 's/^/  /'; tail -1 $O/tile-sweep.log | sed 's/^/  /'
echo "== 6 coherency texts"
$PY - "${O/$B/$C}" > $O/coherency.md <<'P'
import json, sys
from transformers import AutoTokenizer
O = sys.argv[1]
t = AutoTokenizer.from_pretrained("/workspace/models/LFM3-Omni-2B", trust_remote_code=True)
names = json.load(open(f"{O}/prompts.json"))["names"]
for m in ("greedy", "sampled"):
    for n, r in zip(names, json.load(open(f"{O}/rows-{m}-batch.out.json"))["rows"]):
        print(f"## {m} / {n} ({len(r['generated_ids'])} tokens, {r['end']})\n\n{t.decode(r['generated_ids'], skip_special_tokens=True)}\n")
P
wc -l $O/coherency.md | sed 's/^/  /'; mem end
echo "ALL DONE $(date +%T)"
