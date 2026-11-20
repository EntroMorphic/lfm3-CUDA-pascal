# serve-v1, the rule run (tools/pascal/remote.sh detach serve-v1 'bash validation/serve-v1/job.sh'): phase 3's final
# tree; builds text-v33 (shipping) and parity-v33 (parity configuration), each with the text server. Rules 1 to 12 as
# README.md states them (the host suite and the host-side byte check run on the development machine).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; M=$B/sv1-serve; PY=$B/venv-ref/bin/python
[ -e $M ] && { echo "$M exists"; exit 2; }; mkdir -p $M
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include"
# Builds (build_serve.sh builds the CLI and scorer too); rule 1's parity gate compares the pins with parity-v33's CLI
# (tools/pascal/gate_pins.sh is these two steps).
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_serve.sh parity-v33; echo "build parity-v33 rc=$?"
python3 tools/pascal/compare_pins.py parity-v33; echo "rule 1 gate rc=$?"
tools/pascal/build_serve.sh text-v33; echo "build text-v33 rc=$?"
# Rule 1: the serial path unchanged. Device code of the CLI and the scorer against text-v32, function by function
# (in place of the NLL and long-decode runs, as the rule allows if it is identical); fixtures; sanitizer; sweep.
for f in liquid-lfm3 lfm3-score; do for v in 32 33; do cuobjdump -sass $B/text-v$v/$f > $M/$f.$v.sass; done
  python3 validation/speed-v1/sass_compare.py $M/$f.32.sass $M/$f.33.sass > $M/$f.sass-compare.txt; echo "rule 1 $f device code against text-v32 rc=$? $(head -1 $M/$f.sass-compare.txt)"; done
sha256sum $M/*.sass > $M/sass.sha256; gzip -9 $M/*.sass
tools/pascal/run_text_fixtures.sh text-v33 2; echo "rule 1 fixtures rc=$?"
LFM3_COMPACT_WINDOWS=1 tools/pascal/run_text_fixtures.sh text-v33 2; echo "rule 1 fixtures compact rc=$?"
tools/pascal/run_text_fixtures.sh text-v33 1 /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "rule 1 memcheck fixtures rc=$?"
STEPS=24 PROMPTS="257 513" XPROMPTS="" tools/pascal/boundary_sweep.sh text-v33 sv1-serve-memcheck-sweep /usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 9; echo "rule 1 memcheck sweep rc=$?"
tools/pascal/boundary_sweep.sh text-v33 sv1-serve-sweep; echo "rule 1 sweep rc=$?"
# Rule 2: the text-only switch's values (1 builds; anything else but 0 stops the build; 0 is the Thor build, which needs
# kernels sm_61 does not have).
for v in 1 2 on 01 "" ; do
  nvcc $flags -DLFM3_TEXT_ONLY=$v -I $B/ljt-3.2.0/ljt/include -c -o $M/switch.o native/src/lfm3_worker.cu > $M/switch.log 2>&1; echo "rule 2 LFM3_TEXT_ONLY='$v' build rc=$? $(grep -o 'LFM3_TEXT_ONLY must be 0 or 1' $M/switch.log | head -1)"; done
rm -f $M/switch.o
# Rules 3 to 9: the reference's serving checks on both builds.
tools/pascal/serve_gates.sh text-v33 sv1-serve-shipping own; echo "rules 3-9 shipping rc=$?"
tools/pascal/serve_gates.sh parity-v33 sv1-serve-parity pinned; echo "rules 3-9 parity rc=$?"
# Rule 10: paired decode on the compact caches.
python3 tools/pascal/paired_decode_check.py $B/text-v33/liquid-lfm3-worker /workspace/models/LFM3-Omni-2B /workspace/data/ppl-windows/windows-32768.i32 $M/paired > $M/paired.log 2>&1; echo "rule 10 rc=$?"; grep -E 'lane|PAIRED' $M/paired.log
# Rules 11 and 12: memory at the limits and refusals past them, through the shipping server.
$PY tools/pascal/serve_memory.py $B/text-v33/liquid-lfm3-server /workspace/models/LFM3-Omni-2B $B/sv1-serve-shipping/control.output.json $M/memory > $M/memory.log 2>&1; echo "rules 11-12 rc=$?"; grep -E 'rule|SERVE' $M/memory.log
# Recorded, not judged: the control (128 tokens at most) through the server against the CLI, three runs each.
$PY - $B/text-v33 $M <<'PY'
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "tools")
from check_lfm3_native_server import NativeServer, Client
build, out = Path(sys.argv[1]), Path(sys.argv[2]); ck = "/workspace/models/LFM3-Omni-2B"
messages = [dict(role="system", content="Respond with text."), dict(role="user", content="Hypothetically, might reflective recursion be a function of cognition?")]
cli = []
for i in range(3):
    t = time.monotonic(); subprocess.run([str(build / "liquid-lfm3"), "--experimental", ck, "validation/serve-v1/inputs/control.request.json", str(out / "speed-cli.json")], check=True, capture_output=True)
    r = json.load(open(out / "speed-cli.json")); cli.append(dict(wall=time.monotonic() - t, prefill=r["prefill_seconds"], decode=r["decode_seconds"], tokens=len(r["generated_ids"])))
server = NativeServer(build / "liquid-lfm3-server", ck, out, "speed", None); client = Client(server.port, server.token, [])
sid = client.call("POST", "/sessions", dict(capacity=4096))["session_id"]; served = []
for i in range(4):
    client.call("POST", f"/sessions/{sid}/reset", {}); t = time.monotonic()
    r = client.call("POST", "/generate", dict(session_id=sid, messages=messages, max_new_tokens=128)); served.append(dict(wall=time.monotonic() - t, tokens=len(r["generated_ids"])))
server.stop()
json.dump(dict(cli=cli, server=served[1:], server_first=served[0]), open(out / "speed.json", "w"), indent=1)
for x in cli: print(f"speed CLI: {x['tokens']} tokens, prefill {x['prefill']:.3f} s, decode {1e3 * x['decode'] / max(1, x['tokens'] - 1):.2f} ms per token, process wall {x['wall']:.2f} s")
for x in served[1:]: print(f"speed server (warm): {x['tokens']} tokens, request wall {x['wall']:.3f} s")
PY
echo "speed record rc=$?"
date
