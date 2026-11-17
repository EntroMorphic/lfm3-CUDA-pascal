"""Host-only, checkpoint-free evidence packet builder. Never executes a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

EVIDENCE = "validation/initcheck-version-shapes-v2/"
SHAPES = ((257, 2048), (1, 128), (257, 128), (1, 2048))
# Review anchor for the exact 80 existing files, not a vendor signature.
BASELINE_SHA256 = "65c11b8f41e97cb6cd2e9b2fe54949269002700fa0d241eb92fddd9fec0b1b38"
BUILD_RECIPE = """cmake_minimum_required(VERSION 3.24)
project(checkpoint_free_projection LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(lfm3-projection-probe native/tests/lfm3_projection_probe.cu)
target_link_libraries(lfm3-projection-probe PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(lfm3-projection-probe PROPERTIES CUDA_ARCHITECTURES 110)
"""
# The packet keeps the reviewed layout (its names are part of BASELINE_SHA256, and BUILD_RECIPE builds from them);
# a file the repository has since moved is read from where it is now (docs/moved-paths.md).
REPO_PATH = {"native/tests/lfm3_projection_probe.cu": "native/probes/lfm3_projection_probe.cu"}


def baseline_names():
    names = ["native/tests/lfm3_projection_probe.cu", "tools/check_lfm3_initcheck_matrix.py",
             "tools/lfm3_owned_process.py"]
    names += [EVIDENCE + name for name in ("summary.json", "probe.ldd.log", "run.log",
              "sanitizer-13.4.92-download.json", "sanitizer-13.5.25-download.json")]
    names += [EVIDENCE + f"tool-{tool}/{rows}x{width}-{mode}.{kind}.log"
              for tool in range(3) for rows, width in SHAPES
              for mode in ("kernel", "missing", "gemm") for kind in ("sanitizer", "application")]
    return sorted(names)


EXTRAS = ("CMakeLists.txt", "tools/check_lfm3_sanitizer_repro.py",
          "audit/SANITIZER_VENDOR_REPRO.md")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local_root(path):
    path = Path(path).absolute()
    if path != path.resolve(strict=True) or not path.is_dir():
        raise ValueError("root/parent must be a real local directory, without symlinks")
    return path


def read_local(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(p in (".", "..") for p in relative.parts) or "\\" in name:
        raise ValueError("unsafe relative path")
    path = root
    for component in relative.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("symlink evidence is forbidden")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024 * 1024:
        raise ValueError("evidence must be a bounded regular file")
    raw = path.read_bytes()
    # No environment dump or arbitrary file attachment is supported. Reject common
    # secret material even in the reviewed allowlist. Recorded absolute library
    # paths are inert provenance strings: they are never opened or executed.
    text = raw.decode("utf-8")
    if re.search(r"^-----BEGIN .*PRIVATE KEY|Authorization\s*:|Bearer\s+[A-Za-z0-9]|hf_[A-Za-z0-9]{20,}|(?:access_token|api_key|password)\s*[=:]\s*['\"]?[^\s'\"]+", text, re.I | re.M):
        raise ValueError("possible credential material")
    for url in re.findall(r"https?://[^\s\"<>]+", text):
        if "@" in url or "?" in url or "#" in url:
            raise ValueError("credential-bearing or parameterized URL forbidden")
    return raw


def hashes(files):
    return {name: digest(raw) for name, raw in sorted(files.items())}


def validate_evidence(files):
    baseline = {name: files[name] for name in baseline_names()}
    if digest(canonical(hashes(baseline))) != BASELINE_SHA256:
        raise ValueError("reviewed evidence/source hash mismatch (including controls)")
    summary = json.loads(files[EVIDENCE + "summary.json"])
    if summary.get("status") != "failed-controls-or-tool" or summary.get("integrated_initcheck_accepted") is not False:
        raise ValueError("failed status must be preserved")
    records = summary["records"]
    counts = {}
    for record in records:
        state = record["classification"]
        counts[state] = counts.get(state, 0) + 1
    if len(records) != 36 or counts != {"clean": 12, "negative-control-detected": 12,
                                      "tracking-reproduced": 9, "unexpected-failure": 3}:
        raise ValueError("incomplete control matrix")
    return summary


def build(repo, output):
    repo = local_root(repo)
    output = Path(output).absolute()
    local_root(output.parent)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory")
    files = {name: read_local(repo, REPO_PATH.get(name, name)) for name in baseline_names()}
    summary = validate_evidence(files)
    files["CMakeLists.txt"] = BUILD_RECIPE.encode()
    for name in EXTRAS[1:]:
        files[name] = read_local(repo, name)
    # Validate everything before creating the destination; no recursive copies.
    output.mkdir()
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = dict(schema=1, status=summary["status"], integrated_initcheck_accepted=False,
                    baseline_sha256=BASELINE_SHA256, files=hashes(files))
    (output / "packet-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(output)
    return manifest


def verify(packet):
    packet = local_root(packet)
    manifest = json.loads(read_local(packet, "packet-manifest.json"))
    names = set(baseline_names()) | set(EXTRAS)
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "status", "integrated_initcheck_accepted", "baseline_sha256", "files"}:
        raise ValueError("invalid manifest fields")
    if not isinstance(manifest["files"], dict):
        raise ValueError("invalid manifest file map")
    if set(manifest.get("files", {})) != names:
        raise ValueError("manifest must contain exactly the fixed allowlist")
    # Reject extra files/directories and all symlinks, including empty extras.
    allowed_dirs = {str(parent) for name in names for parent in PurePosixPath(name).parents if str(parent) != "."}
    for path in packet.rglob("*"):
        name = path.relative_to(packet).as_posix()
        if path.is_symlink() or (path.is_dir() and name not in allowed_dirs) or (not path.is_dir() and name not in names | {"packet-manifest.json"}):
            raise ValueError("unexpected packet entry")
    files = {name: read_local(packet, name) for name in names}
    if hashes(files) != manifest["files"]:
        raise ValueError("packet manifest hash mismatch")
    validate_evidence(files)
    if (manifest.get("schema") != 1 or manifest.get("baseline_sha256") != BASELINE_SHA256 or
            manifest.get("status") != "failed-controls-or-tool" or manifest.get("integrated_initcheck_accepted") is not False or
            files["CMakeLists.txt"] != BUILD_RECIPE.encode()):
        raise ValueError("invalid packet metadata/build recipe")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        manifest = verify(args.verify) if args.verify else build(args.repo, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"packet rejected: {exc}\n")
    print(json.dumps(dict(packet_verified=True, evidence_status=manifest["status"],
                          integrated_initcheck_accepted=False, files=len(manifest["files"]))))
    # Success means packaging integrity only, never sanitizer acceptance.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
