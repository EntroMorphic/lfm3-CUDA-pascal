# images-v1 development 1 (tools/pascal/remote.sh detach img-dev2 'bash validation/images-v1/dev/dev1.sh'): rule 7's
# image configuration peaked at 7,243 MiB, the audio-in configuration's figure, where the vision encoder's scratch at
# the largest image (about 436 MiB, kept) was expected on top. Here: a fresh server at the defaults, whole-card memory
# before and after one full-size noise image (speech_memory.py's), the reply's fields, then four more distinct ones.
set -x; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; M=/workspace/models/LFM3-Omni-2B; O=$B/img-dev2; mkdir -p $O
$B/venv-asr/bin/python - $B/text-v40/liquid-lfm3-server $M $O <<'PY'
import base64, json, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, "tools/pascal")
from check_lfm3_native_server import NativeServer
src = open("tools/pascal/speech_memory.py").read(); ns = {}
exec("import random, struct, zlib\n" + src[src.index("def noise_png"):src.index("IMAGES = ")], ns)
server = NativeServer(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), "server", None)
mem = lambda: int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout)
def post(seed):
    body = dict(messages=[{"role": "user", "content": [{"type": "image", "image": base64.b64encode(ns["noise_png"](seed)).decode()},
                {"type": "text", "text": "Describe the image."}]}], mode="text", max_new_tokens=16, seed=1)
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}/speech", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + server.token})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
try:
    time.sleep(2); print("idle", mem(), "MiB", flush=True)
    for seed in range(1, 6):
        t = time.monotonic(); r = post(seed)
        print(f"image {seed}: {time.monotonic() - t:.1f} s, after {mem()} MiB;", {k: r.get(k) for k in ("prompt_tokens", "consumed_tokens", "prefill_seconds", "generate_seconds", "truncated")},
              "text:", repr(r.get("text", ""))[:120], "keys:", sorted(r)[:20], flush=True)
finally:
    print("server exit", server.stop())
PY
