#!/usr/bin/env python3
"""Compare the native front end (lfm3-image-decode-check [--preprocess]) with HF's (tools/hash_lfm3_image_front.py),
image by image (validation/images-v1 rule 1).
For each native row: the RGB size and sha256 (or both refused), and, when the native row has pixel_values, the grid,
the token count and the pixel_values sha256 (or both refused at processing). HF names "bench/images/x" match native
"bench/x" (the native check is run on the images directory).
Usage: compare_lfm3_image_front.py NATIVE.jsonl HF.jsonl... [--expect-differ NAME...]
Names after --expect-differ are deliberate departures (listed, not counted as failures). Exit status 1 on any other
difference or a native row HF lacks.
"""
import json
import sys


def rows(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "name" in r:
            out[r["name"]] = r
    return out


def main(argv):
    if "--expect-differ" in argv:
        i = argv.index("--expect-differ")
        args, expected = argv[1:i], set(argv[i + 1:])
    else:
        args, expected = argv[1:], set()
    if len(args) < 2:
        sys.exit(__doc__)
    nat = rows(args[0])
    hf = {}
    for p in args[1:]:
        for k, v in rows(p).items():
            hf[k.replace("bench/images/", "bench/", 1) if k.startswith("bench/images/") else k] = v
    same = pre = 0
    bad, deliberate, missing = [], [], []
    for k, n in nat.items():
        h = hf.get(k)
        if h is None:
            missing.append(k)
            continue
        why = None
        if bool(h.get("load_error")) != bool(n.get("load_error")):
            why = f"refusal differs: HF {h.get('load_error')!r} native {n.get('load_error')!r}"
        elif not n.get("load_error"):
            if h["rgb_size"] != n["rgb_size"] or h["rgb_sha256"] != n["rgb_sha256"]:
                why = f"rgb differs: HF {h['rgb_size']} native {n['rgb_size']}"
            elif "pixel_values_sha256" in n or "process_error" in n:
                pre += 1
                if bool(h.get("process_error")) != bool(n.get("process_error")):
                    why = f"processing refusal differs: HF {h.get('process_error')!r} native {n.get('process_error')!r}"
                elif not n.get("process_error"):
                    ht = h["tensors"]["pixel_values"]
                    if h["grid_thw"] != n["grid_thw"] or h["tokens"] != n["tokens"]:
                        why = f"grid differs: HF {h['grid_thw']} native {n['grid_thw']}"
                    elif ht["sha256"] != n["pixel_values_sha256"]:
                        why = f"pixel_values differ (grid {n['grid_thw']})"
        if why is None:
            same += 1
        elif k in expected:
            deliberate.append((k, why))
        else:
            bad.append((k, why))
    print(f"native rows {len(nat)} (with pixel_values compared: {pre}): identical {same}, differ {len(bad)}, "
          f"deliberate {len(deliberate)}, missing in HF {len(missing)}")
    for k, why in bad[:40]:
        print("  DIFFER", k, why)
    for k, why in deliberate:
        print("  deliberate", k, why)
    for k in missing[:10]:
        print("  MISSING", k)
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
