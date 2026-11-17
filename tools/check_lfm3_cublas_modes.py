"""Exclusive-owner, checkpoint-free cuBLAS mode diagnostic; never acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

try:
    from lfm3_owned_process import OwnedJob, owned_main
except ModuleNotFoundError:
    from tools.lfm3_owned_process import OwnedJob, owned_main

MODES = ("default", "pedantic", "default-algo")
SHAPES = ((257, 2048), (1, 2048))
SOURCE = Path(__file__).resolve().parents[1]/"native/probes/lfm3_projection_modes_probe.cu"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment(root, fallback):
    env = dict(os.environ)
    if env.get("LD_PRELOAD") or env.get("LD_AUDIT"):
        raise ValueError("preload/audit overrides are forbidden")
    paths = [str(root), str(fallback)] + env.get("LD_LIBRARY_PATH", "").split(":")
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
    return env


def libraries(text, root):
    if "not found" in text:
        raise ValueError("unresolved ldd dependency")
    found = {}
    for line in text.splitlines():
        match = re.search(r"(?:(\S+)\s+=>\s+)?(/\S+)\s+\(", line)
        if match:
            path = Path(match[2]).resolve(strict=True)
            found[match[1] or path.name] = dict(path=str(path), sha256=digest(path))
    for name in ("libcublas.so.13", "libcublasLt.so.13"):
        if name not in found or not Path(found[name]["path"]).is_relative_to(root):
            raise ValueError(f"{name} must resolve within selected root {root}")
    if not any(name.startswith("libcudart.so.") for name in found):
        raise ValueError("shared current cudart must resolve")
    return found


def library_version(found, expected=None):
    name = Path(found["libcublas.so.13"]["path"]).name
    if expected is not None and (type(expected) is not int or not 130000 <= expected <= 139999):
        raise ValueError("expected version must be a CUDA 13 cuBLAS API integer (130000..139999)")
    match = re.fullmatch(r"libcublas\.so\.(13)\.(\d+)\.(\d+)(?:\.\d+)?", name)
    if not match:
        if name == "libcublas.so.13" and expected is not None:
            return expected  # Configuration expectation only; actual API checked in every case.
        raise ValueError("cannot establish exact cuBLAS API version from resolved soname file")
    major, minor, patch = map(int, match.groups())
    derived = major*10000 + minor*100 + patch
    if expected is not None and expected != derived:
        raise ValueError("expected version disagrees with versioned cuBLAS filename")
    return derived


def classify(producer, mode, rows, width, version, code, log, app, timed_out=False):
    if timed_out:
        return "timeout"
    lines = app.splitlines()
    identity = (any(line.startswith(f"CONFIG producer={producer}; rows={rows}; width={width}; mode={mode};") for line in lines) and
                f"LIBRARY cublas={version}; mode={mode}" in lines and
                f"MATH actual={2 if mode == 'pedantic' else 0}; mode={mode}" in lines)
    if not identity or "FAIL:" in app:
        return "unsupported-or-error" if "cuBLAS status=" in app else "unexpected-failure"
    prefix = "INTENTIONAL UNINITIALIZED CONTROL: " if producer == "missing" else "PASS: "
    correct = (f"{prefix}128 sampled outputs of {rows*3*width}; producer={producer}; mode={mode}"
               f"; cublas={version}; numerical-only (not sanitizer acceptance)" in lines)
    summaries = re.findall(r"(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$", log)
    reported = len(summaries) == 1 and int(summaries[0]) > 0
    uninit = "Uninitialized __global__ memory read" in log
    downstream = uninit and "read_output(" in log
    internal = uninit and "splitKreduce_kernel" in log
    if not correct:
        return "unexpected-failure"
    if producer == "missing":
        return "negative-control-detected" if code == 86 and reported and downstream and not internal else "unexpected-failure"
    if code == 0 and summaries == ["0"] and not uninit:
        return "clean"
    if producer == "gemm" and code == 86 and reported and internal:
        return "internal-splitK"
    if producer == "gemm" and code == 86 and reported and downstream:
        return "downstream-tracking"
    return "unexpected-failure"


def run_owned(command, env, destination, timeout):
    owner = OwnedJob()
    with destination.open("w") as stream:
        try:
            owner.start(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            return owner.wait(timeout)
        finally:
            owner.close()


@owned_main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("sanitizer", "probe", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, action="append", required=True)
    parser.add_argument("--expected-version", type=int, action="append")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.timeout <= 1200 or len(args.library_dir) != 2:
        raise ValueError("require two library roots and finite timeout in (0,1200]")
    if args.expected_version is not None and (len(args.expected_version) != 2 or
            any(not 130000 <= v <= 139999 for v in args.expected_version)):
        raise ValueError("provide exactly two expected API versions in 130000..139999")
    roots = [p.resolve(strict=True) for p in args.library_dir]
    if roots[0] == roots[1] or not all(p.is_dir() for p in roots):
        raise ValueError("require two distinct library directories")
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    args.probe = args.probe.resolve(strict=True); args.sanitizer = args.sanitizer.resolve(strict=True)
    report = dict(status="running", diagnostic_completed=False, integrated_initcheck_accepted=False,
                  records=[], libraries=[], hashes={}, scope="128 samples; checkpoint-free, not runtime acceptance")
    def save():
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    def pin(path):
        report["hashes"][str(path)] = digest(path)
    try:
        for path in (Path(__file__).resolve(), SOURCE, args.probe, args.sanitizer):
            pin(path)
        save()
        for index, root in enumerate(roots):
            env = environment(root, roots[0]); directory = args.output/f"library-{index}"; directory.mkdir()
            code, timeout = run_owned([shutil.which("ldd"), str(args.probe)], env, directory/"ldd.log", 10)
            if code != 0 or timeout:
                raise ValueError("ldd preflight failed")
            found = libraries((directory/"ldd.log").read_text(), root)
            for entry in found.values():
                pin(Path(entry["path"]))
            expected = args.expected_version[index] if args.expected_version is not None else None
            provenance = "expected-config-verified-per-case" if Path(found["libcublas.so.13"]["path"]).name == "libcublas.so.13" else "versioned-filename"
            report["libraries"].append(dict(root=str(root), resolved=found, version=library_version(found, expected),
                                             expected_version=expected, version_provenance=provenance,
                                             ld_library_path=env["LD_LIBRARY_PATH"]))
            code, timeout = run_owned([str(args.sanitizer), "--version"], env, directory/"tool-version.log", 10)
            if code != 0 or timeout:
                raise ValueError("sanitizer version preflight failed")
            save()
        cudarts = [{k: v for k, v in lib["resolved"].items() if k.startswith("libcudart.so.")} for lib in report["libraries"]]
        if cudarts[0] != cudarts[1]:
            raise ValueError("selected roots changed current cudart")
        stop = False
        for index, lib in enumerate(report["libraries"]):
            env = environment(roots[index], roots[0]); directory = args.output/f"library-{index}"
            for mode in MODES:
                for rows, width in SHAPES:
                    for producer in ("kernel", "missing", "gemm"):
                        stem = directory/f"{mode}-{rows}x{width}-{producer}"
                        log, app = Path(str(stem)+".sanitizer.log"), Path(str(stem)+".application.log")
                        command = [str(args.sanitizer), "--tool", "initcheck", "--error-exitcode", "86",
                                   "--print-limit", "4", "--log-file", str(log), str(args.probe), producer, str(rows), str(width), mode]
                        record = dict(library=index, mode=mode, rows=rows, width=width, producer=producer,
                                      command=command, classification="incomplete")
                        report["records"].append(record); save(); start = time.monotonic()
                        code, timeout = run_owned(command, env, app, args.timeout)
                        state = classify(producer, mode, rows, width, lib["version"], code,
                                         log.read_text(errors="replace") if log.exists() else "", app.read_text(errors="replace"), timeout)
                        record.update(returncode=code, timed_out=timeout, seconds=time.monotonic()-start,
                                      classification=state, group_cleanup_complete=True)
                        save()
                        stop = timeout or (producer != "gemm" and state not in ("clean", "negative-control-detected"))
                        if stop: break
                    if stop: break
                if stop: break
            if stop: break
        report["diagnostic_completed"] = not stop and len(report["records"]) == 36
        blocked = any(r["classification"] not in ("clean", "negative-control-detected") for r in report["records"])
        report["status"] = "failed-controls-or-timeout" if stop else "blocked" if blocked else "diagnostic-completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            report["hashes_stable"] = all(digest(p) == h for p, h in report["hashes"].items())
            for index, lib in enumerate(report["libraries"]):
                destination = args.output/f"library-{index}"/"ldd.final.log"
                code, timeout = run_owned([shutil.which("ldd"), str(args.probe)], environment(roots[index], roots[0]), destination, 10)
                report["hashes_stable"] &= code == 0 and not timeout and libraries(destination.read_text(), roots[index]) == lib["resolved"]
        except (OSError, ValueError, RuntimeError):
            report["hashes_stable"] = False
        if not report["hashes_stable"]:
            report.update(status="failed-identity-change", diagnostic_completed=False)
        save()
    print(json.dumps(dict(status=report["status"], records=len(report["records"]), integrated_initcheck_accepted=False)))
    return 0 if report["status"] == "diagnostic-completed" else 86 if report["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
