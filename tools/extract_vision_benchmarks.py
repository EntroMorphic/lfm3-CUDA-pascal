#!/usr/bin/env python3
"""Unpack MMStar and POPE (validation/images-v1) from their parquet files into each image's original bytes and a
JSONL manifest, so the HF run and the native engine read the same files and Thor needs no pyarrow.

Images are written byte for byte as stored (no decode, no re-encode), named by the sha256 of their bytes (POPE's
9,000 questions share 500 COCO images). Manifest rows: benchmark, split, id, question, answer, category (and MMStar's
l2_category), image (path relative to OUT_DIR), image_sha256.
Usage: extract_vision_benchmarks.py DATA_DIR OUT_DIR
  DATA_DIR holds mmstar/mmstar.parquet and pope/data/test-0000{0,1,2}-of-00003.parquet.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def image_bytes(cell):
    return cell if isinstance(cell, bytes) else cell["bytes"]


def write_image(out, data):
    digest = hashlib.sha256(data).hexdigest()
    path = out / "images" / f"{digest}.jpg"
    if not path.exists():
        path.write_bytes(data)
    elif path.read_bytes() != data:
        raise SystemExit(f"sha256 collision at {path}")
    return f"images/{digest}.jpg", digest


def main(data_dir, out_dir):
    data, out = Path(data_dir), Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in pq.read_table(data / "mmstar/mmstar.parquet").to_pylist():
        rel, digest = write_image(out, image_bytes(r["image"]))
        rows.append(dict(benchmark="mmstar", split="val", id=str(r["index"]), question=r["question"], answer=r["answer"],
                         category=r["category"], l2_category=r["l2_category"], image=rel, image_sha256=digest))
    for i in range(3):
        for r in pq.read_table(data / f"pope/data/test-0000{i}-of-00003.parquet").to_pylist():
            rel, digest = write_image(out, image_bytes(r["image"]))
            rows.append(dict(benchmark="pope", split=r["category"], id=str(r["question_id"]), question=r["question"],
                             answer=r["answer"], category=r["category"], image_source=r["image_source"], image=rel,
                             image_sha256=digest))
    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = {(r["benchmark"], r["split"], r["id"]) for r in rows}
    assert len(keys) == len(rows), "duplicate (benchmark, split, id)"
    print(json.dumps(dict(rows=len(rows), images=len(list((out / "images").iterdir())),
                          mmstar=sum(r["benchmark"] == "mmstar" for r in rows),
                          pope={s: sum(r["split"] == s for r in rows) for s in ("adversarial", "popular", "random")})))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
