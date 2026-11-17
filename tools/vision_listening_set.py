#!/usr/bin/env python3
"""validation/images-v1 rule 4: images described in speech, for Tripp's listening. Writes lfm3-speech requests
(interleaved mode, "Respond with interleaved text and audio.", seed 0, 1,024 new tokens) and their list:
  - the stage-0 photos (china, flower, cherry, mirror), the grayscale china, and two images in one prompt;
  - two held-out MMStar images (mm03, mm05: the first two of the held-out set above 300 pixels on both sides).
Usage: vision_listening_set.py STAGE0_IMAGES_DIR HELDOUT_IMAGES_JSON OUT_DIR
  HELDOUT_IMAGES_JSON: {"mm03": path, "mm05": path} (the held-out paths, as the container sees them)
"""
import json
import sys
from pathlib import Path

SYSTEM = "Respond with interleaved text and audio."
DESCRIBE = "Describe this image."


def request(images, text):
    content = [{"type": "image", "image": str(p)} for p in images] + [{"type": "text", "text": text}]
    return dict(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                mode="interleaved", max_new_tokens=1024, seed=0)


def main(stage0, heldout_json, out_dir):
    s0, out = Path(stage0), Path(out_dir)
    held = json.loads(Path(heldout_json).read_text())
    cases = {
        "01-china": request([s0 / "china.png"], DESCRIBE),
        "02-flower": request([s0 / "flower.png"], "What kind of flower is this, and what color is it?"),
        "03-cherry": request([s0 / "cherry.png"], DESCRIBE),
        "04-mirror": request([s0 / "mirror.png"], DESCRIBE),
        "05-gray": request([s0 / "gray.png"], DESCRIBE),
        "06-two": request([s0 / "china.png", s0 / "flower.png"], "What is different between these two images?"),
        "07-mm03": request([held["mm03"]], DESCRIBE),
        "08-mm05": request([held["mm05"]], DESCRIBE),
    }
    (out / "req").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)
    with open(out / "list.txt", "w") as lst:
        for name, req in cases.items():
            (out / "req" / f"{name}.json").write_text(json.dumps(req, indent=1))
            lst.write(f"{out / 'req' / name}.json {out / 'out' / name}\n")
    print(len(cases), "requests")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
