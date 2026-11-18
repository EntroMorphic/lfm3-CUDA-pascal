"""Host check of the engine's own tensor-core step code (native/include/lfm/tensor_core_model.hpp) against the Thor
pins, before it runs on the card: layer 0's conv.in_proj of each fixture, with the fast step and with the integer
reference step. The inputs are computed as in tcm.py (embedding rows, operator_norm).
    python3 host_kernel_check.py INPUT_IDS.json [CASE...]"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
from tcm import Weights, f32, sha, pins, norm_threads, norm_sums, norm, ROOT

def main():
    ids = json.load(open(sys.argv[1])); cases = sys.argv[2:] or ["arithmetic", "baseline", "window257"]
    W = Weights(); ok = True
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); exe = d / "check"
        subprocess.run(["g++", "-std=c++20", "-O2", "-fopenmp", "-ffp-contract=off", "-I", str(ROOT / "native/include"),
                        str(ROOT / "native/tests/tensor_core_model_check.cpp"), "-o", str(exe)], check=True)
        w = np.ascontiguousarray(W.u16("layers.0.conv.in_proj.weight")); w.astype("<u2").tofile(d / "w.bf16")
        for case in cases:
            x = np.array(W.u16("embed_tokens.weight")[ids[case]]); rows, width = x.shape
            sums = norm_sums(f32(x), norm_threads(rows, width))
            arg = ((sums * np.float32(1.0 / width)).astype(np.float32) + np.float32(1e-5)).astype(np.float32)
            n = norm(x, W.u16("layers.0.operator_norm.weight"), 1e-5, (1.0 / np.sqrt(arg.astype(np.float64))).astype(np.float32))
            assert sha(n) == pins(case)["0.layers.0.operator_norm.bf16"]
            n.astype("<u2").tofile(d / "x.bf16")
            for mode in ("fast", "reference"):
                t = time.time()
                subprocess.run([str(exe), mode, str(d / "x.bf16"), str(rows), str(d / "w.bf16"), str(w.shape[0]), str(width),
                                str(d / "out.bf16")], check=True)
                out = np.fromfile(d / "out.bf16", dtype="<u2")
                match = sha(out) == pins(case)["0.layers.0.conv.in_proj.bf16"]; ok &= match
                print(f"{case} ({rows} x {w.shape[0]}, K {width}) {mode:9s}: {'MATCH' if match else 'differ'} with the Thor pin  ({time.time() - t:.1f} s)", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
