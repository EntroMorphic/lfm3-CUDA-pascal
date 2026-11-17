#!/usr/bin/env python3
"""Image input (#4): cases for native/tests/image_prompt_check.cpp (the native renderer against HF's processor), from
the reference dumps (tools/dump_lfm3_vision_reference.py): each case's conversation as the dump tool built it, and the
input_ids HF's processor gave it (input_ids.npy).
  stage-0 cases (REF_DIR/float32/<case>): the conversations of the dump tool's cases(), images in REF_DIR/images.
  held-out cases (--held-out NAME=PATH...): the DESCRIBE system prompt and one image each, from HELD_DIR/float32/<NAME>.
Usage: image_prompt_cases.py REF_DIR OUT.jsonl [--held-out HELD_DIR NAME=PATH...]
"""
import json
import sys
from pathlib import Path

import numpy as np

DESCRIBE = "Describe the provided image."
GROUNDED = "Answer questions about the provided image. Base every answer only on what can be seen."


def conversation(system, images, text=None):
    content = [{"type": "image", "image": str(p)} for p in images]
    if text:
        content.append({"type": "text", "text": text})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    ref, out = Path(argv[1]), argv[2]
    img = lambda n: (ref / "images" / f"{n}.png").resolve()
    stage0 = {
        "china": conversation(DESCRIBE, [img("china")]),
        "flower": conversation(GROUNDED, [img("flower")], "What kind of flower is this, and what color is it?"),
        "two": conversation(GROUNDED, [img("china"), img("flower")], "What is different between these two images?"),
    }
    for n in ("cherry", "mirror", "small", "wide", "tall", "noise", "gray", "rgba"):
        stage0[n] = conversation(DESCRIBE, [img(n)])
    rows = []
    for name, conv in stage0.items():
        ids = np.load(ref / "float32" / name / "input_ids.npy")[0].tolist()
        rows.append(dict(case=name, messages=conv, prompt_ids=ids))
    if len(argv) > 3:
        if argv[3] != "--held-out":
            sys.exit(__doc__)
        held = Path(argv[4])
        for a in argv[5:]:
            name, path = a.split("=", 1)
            ids = np.load(held / "float32" / name / "input_ids.npy")[0].tolist()
            rows.append(dict(case=name, messages=conversation(DESCRIBE, [Path(path).resolve()]), prompt_ids=ids))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "cases")


if __name__ == "__main__":
    main(sys.argv)
