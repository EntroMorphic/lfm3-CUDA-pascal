#!/usr/bin/env bash
# Runs in the container: every fixture of validation/text-correct-v1/fixtures RUNS times with build TAG's liquid-lfm3;
# compares the ids with expected.json and between runs. Exit 0 only if everything agrees.
#   tools/pascal/run_text_fixtures.sh TAG [RUNS=2] [WRAPPER...]     (WRAPPER e.g. compute-sanitizer --tool memcheck)
set -uo pipefail
tag=${1:?usage: run_text_fixtures.sh TAG [RUNS] [WRAPPER...]}; runs=${2:-2}; shift; shift || true
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fx=$repo/validation/text-correct-v1/fixtures
bin=/workspace/build/$tag/liquid-lfm3
out=/workspace/build/$tag/fixtures${1:+-wrapped}; rm -rf "$out"; mkdir -p "$out"
echo "binary $(sha256sum "$bin" | cut -d' ' -f1)  tree $(cd "$repo" && sha256sum native/include/lfm/gemm.cuh | cut -c1-16) (gemm.cuh)"
for request in "$fx"/*.request.json; do
  name=$(basename "$request" .request.json)
  for run in $(seq 1 "$runs"); do
    "$@" "$bin" --experimental /workspace/models/LFM3-Omni-2B "$request" "$out/$name.$run.json" > "$out/$name.$run.log" 2>&1
    echo $? > "$out/$name.$run.rc"
  done
done
python3 - "$fx/expected.json" "$out" "$runs" <<'PY'
import json, sys, pathlib
expected = json.load(open(sys.argv[1]))["generated_ids"]; out = pathlib.Path(sys.argv[2]); runs = int(sys.argv[3]); ok = True
for name, want in expected.items():
    got = []
    for run in range(1, runs + 1):
        rc = int((out / f"{name}.{run}.rc").read_text())
        if rc != 0:
            print(f"{name} run {run}: exit {rc}: {(out / f'{name}.{run}.log').read_text()[-400:].strip()}"); ok = False; got.append(None); continue
        got.append(json.load(open(out / f"{name}.{run}.json")))
    if None in got: continue
    ids = [g["generated_ids"] for g in got]
    same = all(i == ids[0] for i in ids); match = ids[0] == want; ok &= same and match and all(g["reset_exact_logits_ok"] for g in got)
    print(f"{name}: ids {'MATCH' if match else 'DIFFER'} ({len(want)}), runs {'identical' if same else 'DIFFER'}, "
          f"reset {'ok' if all(g['reset_exact_logits_ok'] for g in got) else 'FAIL'}, input {len(got[0]['input_ids'])} tokens, "
          f"prefill {got[0]['prefill_seconds']:.3f} s, decode {got[0]['decode_seconds']:.3f} s / {got[0]['decode_forward_calls']} calls")
    if not match: print("  expected", want, "\n  got     ", ids[0])
    if "text" in got[0]: print("  text:", json.dumps(got[0]["text"]))
print("FIXTURES", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
