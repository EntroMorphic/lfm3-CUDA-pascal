#!/usr/bin/env python3
"""Generate an isolated, unaccepted PEDANTIC runtime mirror; never build or run it."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MAIN = "native/src/lfm3_main.cu"
BATCH = "native/src/lfm3_decode_batch.cuh"
PROBE = "native/probes/lfm3_decode_batch_probe.cu"
REPLACEMENTS = {
    MAIN: {"CUBLAS_COMPUTE_32F": ("CUBLAS_COMPUTE_32F_PEDANTIC", 3),
           "CUBLAS_GEMM_DEFAULT_TENSOR_OP": ("CUBLAS_GEMM_DEFAULT", 3),
           "CUBLAS_DEFAULT_MATH": ("CUBLAS_PEDANTIC_MATH", 1)},
    BATCH: {"CUBLAS_COMPUTE_32F": ("CUBLAS_COMPUTE_32F_PEDANTIC", 2),
            "CUBLAS_GEMM_DEFAULT_TENSOR_OP": ("CUBLAS_GEMM_DEFAULT", 2)},
}
BUILD_POLICY = {"cuda_architectures": "110", "cxx_standard": 20,
                "cuda_standard": 20, "cuda_flags": ["-O3", "--fmad=false"],
                "relaxed_constexpr": False, "production_modified": False}
CMAKE = '''cmake_minimum_required(VERSION 3.24)
set(CMAKE_SUPPRESS_REGENERATION ON)
project(lfm3_pedantic_diagnostic LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
function(diagnostic_target name source)
  add_executable(${name} ${source} native/src/safetensors.cpp)
  target_include_directories(${name} PRIVATE native/include)
  target_compile_features(${name} PRIVATE cxx_std_20 cuda_std_20)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(${name} PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES 110)
endfunction()
diagnostic_target(liquid-lfm3 native/src/lfm3_main.cu)
diagnostic_target(liquid-lfm3-worker native/src/lfm3_worker.cu)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(root, relative):
    path = root / relative
    for component in (path, *path.parents):
        if component == root.parent:
            break
        if component.is_symlink():
            raise ValueError(f"symlink source rejected: {component}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"escaped source rejected: {path}")
    return path


def prepare(root, output, with_probe=True):
    root, output = Path(root).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output must be a new directory: {output}")
    files = [MAIN, "native/src/lfm3_worker.cu", BATCH,
             "native/src/safetensors.cpp", "LICENSE", "native/third_party/PYTORCH_LICENSE"]
    if with_probe:
        files.append(PROBE)
    include = checked(root, "native/include")
    if not include.is_dir():
        raise ValueError("missing native/include")
    for path in sorted(include.rglob("*")):
        checked(root, path.relative_to(root))
        if path.is_file() and path.suffix in {".hpp", ".cuh"}:
            files.append(path.relative_to(root).as_posix())
    prepared, records = {}, {}
    for relative in files:
        original = checked(root, relative).read_bytes()
        modified, counts = original, {}
        if relative in REPLACEMENTS:
            text = original.decode("utf-8")
            for before, (after, expected) in REPLACEMENTS[relative].items():
                pattern = rf"\b{before}\b"
                actual = len(re.findall(pattern, text))
                if actual != expected:
                    raise ValueError(f"{relative}: {before}: expected {expected}, found {actual}")
                text = re.sub(pattern, after, text)
                counts[before] = {"replacement": after, "count": actual}
            modified = text.encode("utf-8")
        prepared[relative] = modified
        records[relative] = {"source_origin": str(root / relative),
                             "original_sha256": digest(original),
                             "copied_sha256": digest(original),
                             "modified_sha256": digest(modified), "replacements": counts}
    cmake = CMAKE + ("diagnostic_target(lfm3-decode-batch-probe native/probes/lfm3_decode_batch_probe.cu)\n"
                     if with_probe else "")
    manifest = {"schema_version": 1, "diagnostic_only": True, "full_acceptance": False,
                "status": "generated_not_built_or_validated", "source_origin": str(root),
                "required_before_runtime_use": "All 6890 eager checks, unchanged original exact gates",
                "expected_eager_checks": 6890, "goldens_may_change": False,
                "failure_policy": "Keep failed evidence; unsupported math API or any bitwise failure is failure",
                "build_policy": BUILD_POLICY, "files": records,
                "original_cmake_sha256": digest(checked(root, "native/CMakeLists.txt").read_bytes()),
                "generated_files": {"CMakeLists.txt": {"sha256": digest(cmake.encode())}}}
    # Preflight every input before exclusively creating the destination. No overwrite/cleanup.
    output.mkdir(parents=True, exist_ok=False)
    for relative, data in prepared.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
    (output / "CMakeLists.txt").write_text(cmake)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent directory")
    parser.add_argument("--without-probe", action="store_true")
    args = parser.parse_args()
    prepare(ROOT, args.output, not args.without_probe)
    print(f"Diagnostic mirror only (NOT accepted): {args.output}")


if __name__ == "__main__":
    main()
