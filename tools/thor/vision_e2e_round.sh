#!/usr/bin/env bash
# Image input (#4), first end-to-end look: the 11 stage-0 conversations (tools/image_prompt_cases.py) through
# lfm3-speech, rendered natively from messages, text mode, greedy, 128 new tokens (as the HF dump's reply), with
# record_image. Per case: the native prompt (mode marker removed) against HF's input_ids, the native pixel_values
# against HF's (bytes), and the native reply next to HF FP32's (build/vision-ref/hf-5.20.2/report.json).
# Usage (Thor host): vision_e2e_round.sh BUILD_TAG OUT_TAG
set -euo pipefail
T=${1:?BUILD_TAG}; O=${2:?OUT_TAG}
B=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
D="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
P="docker exec -i lfm3-profile-v1 /opt/venv/bin/python"
mkdir -p $B/$O/req $B/$O/out
$P - $C/vision-ref/hf-5.20.2 $C/$O <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/liquid-CUDA/tools")
import image_prompt_cases as ipc
ref, out = Path(sys.argv[1]), Path(sys.argv[2])
img = lambda n: str(ref / "images" / f"{n}.png")
cases = {"china": ipc.conversation(ipc.DESCRIBE, [img("china")]),
         "flower": ipc.conversation(ipc.GROUNDED, [img("flower")], "What kind of flower is this, and what color is it?"),
         "two": ipc.conversation(ipc.GROUNDED, [img("china"), img("flower")], "What is different between these two images?")}
for n in ("cherry", "mirror", "small", "wide", "tall", "noise", "gray", "rgba"):
    cases[n] = ipc.conversation(ipc.DESCRIBE, [img(n)])
with open(out / "list.txt", "w") as lst:
    for n, conv in cases.items():
        (out / "req" / f"{n}.json").write_text(json.dumps(dict(messages=conv, mode="text", max_new_tokens=128,
                                                                record_image=True)))
        lst.write(f"{out}/req/{n}.json {out}/out/{n}\n")
PY
[ -n "${COMPARE_ONLY:-}" ] || $D $C/$T/lfm3-speech $M --batch $C/$O/list.txt > $B/$O/run.log 2>&1 || { echo "RUN FAILED"; tail -5 $B/$O/run.log; }
$P - $C/vision-ref/hf-5.20.2 $C/$O <<'PY'
import json, sys
from pathlib import Path
import numpy as np
ref, out = Path(sys.argv[1]), Path(sys.argv[2])
hf = json.load(open(ref / "report.json"))["runs"]
hf = {r["dtype"]: r["cases"] for r in hf}["torch.float32"]
rows = {}
for req in sorted((out / "req").glob("*.json")):
    n = req.stem
    j = json.load(open(out / "out" / f"{n}.json"))
    want = np.load(ref / "float32" / n / "input_ids.npy")[0].tolist()
    prompt_ok = j["prompt"][:len(want)] == want and len(j["prompt"]) <= len(want) + 1
    pv = np.load(ref / "float32" / n / "pixel_values.npy")
    k, nat = 0, []
    while (out / "out" / f"{n}.image{k}.pixel_values.f32").exists():
        nat.append(np.fromfile(out / "out" / f"{n}.image{k}.pixel_values.f32", dtype=np.float32)); k += 1
    nat = np.concatenate(nat) if nat else np.zeros(0, np.float32)
    pv_ok = nat.size == pv.size and nat.tobytes() == pv.astype(np.float32).ravel().tobytes()
    reply, href = j.get("text", ""), hf.get(n, {}).get("reply", "")
    same = 0
    while same < min(len(reply), len(href)) and reply[same] == href[same]: same += 1
    rows[n] = dict(prompt_equal=prompt_ok, pixel_values_equal=pv_ok, reply=reply, hf_fp32_reply=href,
                   same_prefix_chars=same, identical=reply == href, encode_seconds=j.get("encode_seconds"),
                   prefill_seconds=j.get("prefill_seconds"))
    print(f"== {n}: prompt {'EQUAL' if prompt_ok else 'DIFFER'}, pixel_values {'EQUAL' if pv_ok else 'DIFFER'}, "
          f"reply {'IDENTICAL' if reply == href else f'first {same} chars equal'}, encode {j.get('encode_seconds', 0):.3f} s")
    print("  native:", repr(reply))
    print("  HF f32:", repr(href))
(out / "summary.json").write_text(json.dumps(rows, indent=1))
PY
