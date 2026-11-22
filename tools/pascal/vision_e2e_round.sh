#!/usr/bin/env bash
# validation/images-v1 (this port): tools/thor/vision_e2e_round.sh in the card's container. The 11 stage-0
# conversations (tools/image_prompt_cases.py) through lfm3-speech, rendered natively from messages, text mode, greedy,
# 128 new tokens, with record_image. Per case: the native prompt against HF's input_ids, the native pixel_values
# against HF's (bytes), and the native reply next to HF FP32's (/workspace/data/vision-ref/hf-5.20.2/report.json).
# Its output directory is also tools/check_lfm3_image_server.py's ROUND_DIR.
# Usage (in the container): vision_e2e_round.sh BUILD_TAG OUT_DIR
set -euo pipefail
T=${1:?BUILD_TAG}; OUT=${2:?OUT_DIR}
B=/workspace/build; C=$B; O=$(realpath -m "$OUT"); M=/workspace/models/LFM3-Omni-2B
REF=/workspace/data/vision-ref/hf-5.20.2; PYB=$B/venv-asr/bin/python; TOOLS=/workspace/lfm3-CUDA-pascal/tools
mkdir -p $O/req $O/out
$PYB - $REF $O <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/lfm3-CUDA-pascal/tools")
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
[ -n "${COMPARE_ONLY:-}" ] || $C/$T/lfm3-speech $M --batch $O/list.txt > $O/run.log 2>&1 || { echo "RUN FAILED"; tail -5 $O/run.log; }
$PYB - $REF $O <<'PY'
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
