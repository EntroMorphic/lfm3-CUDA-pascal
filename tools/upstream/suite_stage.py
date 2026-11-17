#!/usr/bin/env python3
"""Inventory official LFM2.5 BF16 text releases and optionally stage pinned files.

Run on the internet-connected workstation, never on the offline CUDA host.
Weights are checked against the upstream LFS SHA-256, including existing files.
No checkpoint code is executed. The manifest includes draft companions and
exclusions so that an architecture mismatch cannot disappear from coverage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def get_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(item, path):
    """Resume short HTTP bodies, publishing only a fully verified file."""
    partial = path.with_name(path.name + ".partial")
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["size"]:
            raise RuntimeError(f"oversized partial download: {partial}")
        if offset == item["size"]:
            if not partial.exists():
                partial.touch(exist_ok=False)
            break
        request = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (response.status != 206 or not match or int(match[1]) != offset or
                            int(match[3]) != item["size"]):
                        raise RuntimeError("server did not honor exact download-resume range")
                with partial.open("ab" if partial.exists() else "xb") as dest:
                    while block := response.read(8 * 1024 * 1024):
                        dest.write(block)
        except OSError:
            if attempt == 7:
                raise
    actual = digest(partial)
    expected = item["upstream_sha256"]
    if partial.stat().st_size != item["size"] or (expected and actual != expected):
        raise RuntimeError(f"download integrity mismatch: {partial}")
    partial.rename(path)
    return actual


def classify(name):
    suffix = name.removeprefix("LiquidAI/LFM2.5-")
    if any(x in suffix for x in ("GGUF", "MLX", "ONNX", "FP8", "LoRA")):
        return "alternate-format-or-adapter"
    if suffix.startswith(("VL-", "Audio-", "Embedding-", "Encoder-", "ColBERT-")):
        return "not-standalone-text-generation"
    return "draft-companion" if suffix.endswith("-DSpark") else "standalone"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--from-inventory", type=Path, help="Replay immutable revisions from a saved inventory")
    p.add_argument("--only", nargs="+", help="Download only these checkpoint basenames")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    releases = [] if args.from_inventory else get_json("https://huggingface.co/api/models?author=LiquidAI&search=LFM2.5&limit=1000")
    if len(releases) >= 1000:
        raise RuntimeError("inventory may be truncated; implement pagination")
    manifest = (json.loads(args.from_inventory.read_text()) if args.from_inventory else
                dict(source="LiquidAI Hugging Face API", models=[], exclusions=[]))
    for release in sorted(releases, key=lambda m: m["id"]):
        name = release["id"]
        category = classify(name)
        if category not in ("standalone", "draft-companion"):
            manifest["exclusions"].append(dict(id=name, reason=category))
            continue
        meta = get_json("https://huggingface.co/api/models/" + name + "?blobs=true")
        revision = meta["sha"]
        base = f"https://huggingface.co/{name}/resolve/{revision}/"
        config = get_json(base + "config.json")
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype != "bfloat16":
            raise RuntimeError(f"unexpected dtype for {name}: {dtype}")
        model = dict(id=name, revision=revision, category=category,
                     model_type=config["model_type"], config=config, files=[])
        for item in meta["siblings"]:
            filename = item["rfilename"]
            wanted = filename.endswith(".safetensors") or filename in (
                "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "added_tokens.json", "model.safetensors.index.json", "chat_template.jinja")
            if not wanted:
                continue
            if Path(filename).name != filename:
                raise RuntimeError("unexpected nested checkpoint file")
            lfs = item.get("lfs") or {}
            model["files"].append(dict(name=filename, size=item["size"],
                                       upstream_sha256=lfs.get("sha256"), url=base + filename))
        manifest["models"].append(model)
        print(name, category, config["model_type"], revision, flush=True)
    (args.output / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.download:
        return
    available = {m["id"].split("/")[1] for m in manifest["models"] if m["category"] == "standalone"}
    if args.only and not set(args.only) <= available:
        raise RuntimeError("requested checkpoint absent from standalone inventory")

    def stage(model):
        if model["category"] != "standalone" or (args.only and model["id"].split("/")[1] not in args.only):
            return
        directory = args.models / model["id"].split("/")[1]
        directory.mkdir(parents=True, exist_ok=True)
        checks = []
        for item in model["files"]:
            path = directory / item["name"]
            expected = item["upstream_sha256"]
            if path.suffix == ".safetensors" and not expected:
                raise RuntimeError("missing upstream weight digest")
            if path.exists():
                actual = digest(path)
                if path.stat().st_size != item["size"] or (expected and actual != expected):
                    raise RuntimeError(f"existing file conflicts with pinned release: {path}")
                if not expected:
                    with urllib.request.urlopen(item["url"], timeout=120) as response:
                        upstream = response.read()
                    if hashlib.sha256(upstream).hexdigest() != actual:
                        raise RuntimeError(f"existing metadata conflicts with pinned release: {path}")
            else:
                actual = download(item, path)
            checks.append(dict(name=item["name"], sha256=actual, size=item["size"]))
            item["sha256"] = actual
        (args.output / (directory.name + ".integrity.json")).write_text(json.dumps(checks, indent=2) + "\n")
        print("STAGED", directory, flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(stage, manifest["models"]))
    (args.output / "staged-inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
