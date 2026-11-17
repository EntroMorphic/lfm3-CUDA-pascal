"""Checkpoint-free sanitizer-only version/shape matrix. Never waives initcheck."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

try:
    from lfm3_owned_process import OwnedJob, owned_main
except ModuleNotFoundError:
    from tools.lfm3_owned_process import OwnedJob, owned_main


def classify(mode, returncode, sanitizer_log, application_log, timed_out=False):
    if timed_out:
        return "timeout"
    uninitialized = ("Uninitialized __global__ memory read" in sanitizer_log and
                     "read_output(" in sanitizer_log)
    clean = returncode == 0 and "ERROR SUMMARY: 0 errors" in sanitizer_log
    if mode == "missing":
        return "negative-control-detected" if returncode == 86 and uninitialized and "INTENTIONAL UNINITIALIZED CONTROL: 128 sampled outputs" in application_log else "unexpected-failure"
    correct = "PASS: 128 sampled outputs" in application_log
    if clean and correct:
        return "clean"
    if mode == "gemm" and correct and returncode == 86 and uninitialized:
        return "tracking-reproduced"
    return "unexpected-failure"


@owned_main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitizer", type=Path, action="append", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.timeout <= 1200:
        raise ValueError("timeout must be finite in(0,1200]")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    probe_hash = digest(args.probe)
    ldd = subprocess.run(["ldd", str(args.probe)], env=env, text=True, capture_output=True, check=True)
    (args.output/"probe.ldd.log").write_text(ldd.stdout+ldd.stderr)
    libraries = {}
    for line in ldd.stdout.splitlines():
        match = re.search(r"(lib(?:cublas|cudart)[^ ]*)\s+=>\s+(/\S+)", line)
        if match:
            path = Path(match[2]).resolve(strict=True)
            libraries[match[1]] = dict(path=str(path), sha256=digest(path))
    if not any(name.startswith("libcublas.so") for name in libraries):
        raise ValueError("could not resolve loaded cuBLAS")
    report = dict(status="running", probe_sha256=probe_hash, libraries=libraries, versions=[], records=[],
                  scope="same checkpoint-free binary and resolved libraries; sanitizer version/shape diagnostic only; each probe numerically checks128 sampled outputs, not entire GEMM output; not integrated initcheck acceptance")
    def save():
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    for index, sanitizer in enumerate(args.sanitizer):
        directory = args.output/f"tool-{index}"
        directory.mkdir()
        tool_hash = digest(sanitizer)
        version = subprocess.run([str(sanitizer), "--version"], env=env, capture_output=True, text=True, check=True, timeout=10)
        report["versions"].append(dict(path=str(sanitizer), sha256=tool_hash, version=version.stdout+version.stderr))
        save()
        for rows, width in ((257,2048), (1,128), (257,128), (1,2048)):
            for mode in ("kernel", "missing", "gemm"):
                name = f"{rows}x{width}-{mode}"
                log, app = directory/(name+".sanitizer.log"), directory/(name+".application.log")
                command = [str(sanitizer), "--tool", "initcheck", "--error-exitcode", "86", "--log-file", str(log),
                           str(args.probe), mode, str(rows), str(width)]
                start = time.monotonic()
                timed_out = False
                with app.open("w") as stream:
                    owner = OwnedJob()
                    try:
                        owner.start(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        code, timed_out = owner.wait(args.timeout)
                    finally:
                        owner.close()
                text = log.read_text(errors="replace") if log.exists() else ""
                state = classify(mode, code, text, app.read_text(errors="replace"), timed_out)
                report["records"].append(dict(tool=index, rows=rows, width=width, mode=mode, command=command,
                    returncode=code, timed_out=timed_out, seconds=time.monotonic()-start, classification=state))
                save()
        if digest(sanitizer) != tool_hash:
            raise ValueError("sanitizer changed during matrix")
    if digest(args.probe) != probe_hash:
        raise ValueError("probe changed during matrix")
    records = report["records"]
    valid = all(r["classification"] in ("clean", "negative-control-detected", "tracking-reproduced") for r in records)
    if not valid:
        report["status"] = "failed-controls-or-tool"
    elif any(r["classification"] == "tracking-reproduced" for r in records):
        report["status"] = "blocked-cuBLAS-output-tracking"
    else:
        report["status"] = "control-matrix-passed"
    report["integrated_initcheck_accepted"] = False
    save()
    print(json.dumps(dict(status=report["status"], records=len(records), integrated_initcheck_accepted=False), indent=2))
    return 0 if report["status"] == "control-matrix-passed" else 86 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
