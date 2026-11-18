"""Runs in the container: trace the four forward fixtures with build TAG's liquid-lfm3, teacher-forced as the Thor
round traced them, and compare the SHA-256 of every trace file with the Thor engine's pins
(validation/g4-gates-v2/pins/forward.json). Prints, per fixture, how many files are equal and the first that is
not, in the order the engine computes them. Exit 0 only if every file of every fixture is equal.

    python3 tools/pascal/compare_pins.py TAG [CASE...]
"""
import hashlib, json, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORDER = ["embed", "operator_norm", "conv.in_proj", "conv.product", "self_attn.q_rotary", "self_attn.k_rotary",
         "self_attn.probabilities", "self_attn.attended", "self_attn.gated", "operator", "residual", "ffn_norm",
         "feed_forward.w1", "feed_forward.w3", "feed_forward.product", "feed_forward", "output", "final_norm", "logits"]
TOKENS = dict(baseline=16, arithmetic=16, window257=4, single=4)  # tools/check_lfm3_forward.py


def key(name):  # STEP.[layers.N.]KIND.EXT -> (step, layer, position in the layer, ext)
    step, rest = name.split(".", 1)
    m = re.match(r"layers\.(\d+)\.(.*)\.(bf16|f32)$", rest)
    if m:
        return int(step), int(m.group(1)), ORDER.index(m.group(2)), m.group(3)
    kind, ext = rest.rsplit(".", 1)
    return int(step), -1 if kind == "embed" else 10**6, ORDER.index(kind), ext


def main():
    tag, cases = sys.argv[1], sys.argv[2:] or list(TOKENS)
    binary = Path("/workspace/build") / tag / "liquid-lfm3"
    pins = json.loads((REPO / "validation/g4-gates-v2/pins/forward.json").read_text())
    ids = json.loads((REPO / "validation/tensor-core-model-v1/input_ids.json").read_text())
    print(f"binary {hashlib.sha256(binary.read_bytes()).hexdigest()}  Thor pins from binary {pins['_provenance']['binary_sha256'][:16]}")
    ok, total = True, 0
    unknown = [c for c in cases if c not in TOKENS]
    if unknown:
        sys.exit(f"unknown fixture(s) {unknown}; the fixtures are {list(TOKENS)}")
    for case in cases:
        out = Path("/workspace/build") / tag / "pins" / case
        shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
        expected = pins[case]["generated_ids"]
        request = dict(input_ids=ids[case], teacher_ids=expected[:-1], trace_directory=str(out / "trace"),
                       capacity=min(4096, max(512, len(ids[case]) + TOKENS[case])))
        (out / "request.json").write_text(json.dumps(request))
        run = subprocess.run([str(binary), "--experimental", "/workspace/models/LFM3-Omni-2B", str(out / "request.json"),
                              str(out / "native.json")], capture_output=True, text=True)
        if run.returncode:
            print(f"{case}: engine exit {run.returncode}: {run.stderr[-300:]}"); ok = False; continue
        native = json.loads((out / "native.json").read_text())
        got = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (out / "trace").iterdir()}
        shutil.rmtree(out / "trace")
        want = pins[case]["files"]
        names = sorted(want, key=key)
        equal = [n for n in names if got.get(n) == want[n]]
        differ = [n for n in names if n not in equal]
        (out / "result.json").write_text(json.dumps(dict(equal=len(equal), total=len(names), differ=differ,
                                                         teacher_argmax=native["teacher_argmax"]), indent=1))
        argmax = native["teacher_argmax"] == expected
        total += len(equal)
        ok &= not differ and argmax and set(got) == set(want)
        print(f"{case}: {len(equal)} of {len(names)} trace files equal to the Thor pins; argmax ids {'equal' if argmax else 'DIFFER'}; "
              f"files traced here {len(got)}; prefill {native['prefill_seconds']:.2f} s")
        if differ:
            print(f"  first different: {differ[0]}   (before it, {names.index(differ[0])} files in order are equal)")
            kinds = {}
            for n in differ:
                kinds.setdefault(re.sub(r"^\d+\.(layers\.\d+\.)?", "", n), []).append(n)
            print("  different, by kind: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1]))))
    whole = sorted(cases) == sorted(TOKENS)
    ok &= total > 0
    print(f"PINS {'EQUAL' if ok else 'DIFFER'}: {total} files equal"
          + ("" if whole else f" (fixtures {cases} only: not the parity gate, which is all of {list(TOKENS)})"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
