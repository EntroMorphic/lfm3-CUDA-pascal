"""Fail-closed sanitizer preflight: never credit a library tracking failure as clean."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from lfm3_owned_process import OwnedJob, owned_main
except ModuleNotFoundError:
    from tools.lfm3_owned_process import OwnedJob, owned_main

TOOLS = ("memcheck", "racecheck", "synccheck")


class IdentityChanged(RuntimeError):
    """Pinned execution artifacts are missing or no longer match."""


def artifact_hashes(paths):
    hashes, errors = {}, {}
    for name, path in paths.items():
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            hashes[name] = None
            errors[name] = str(exc)
    return hashes, errors


def native_result(path, request, expected=None):
    """Validate native evidence, never treating a clean sanitizer log as output."""
    result = json.loads(path.read_text())
    def ids(value):
        return isinstance(value, list) and all(type(x) is int and 0 <= x < 128000 for x in value)
    if not isinstance(request, dict) or not ids(request.get("input_ids")) or not request["input_ids"]:
        raise ValueError("invalid request input_ids")
    teacher = request.get("teacher_ids", [])
    count = request.get("max_new_tokens", 0)
    if not ids(teacher) or type(count) is not int or count < 0 or (teacher and count):
        raise ValueError("invalid teacher/decode request")
    if not isinstance(result, dict) or result.get("backend") != "native-cuda-experimental":
        raise ValueError("missing native backend")
    if result.get("input_ids") != request["input_ids"]:
        raise ValueError("native input_ids mismatch")
    generated, argmax = result.get("generated_ids"), result.get("teacher_argmax")
    if not ids(generated) or not ids(argmax):
        raise ValueError("invalid native token IDs")
    if count:
        if len(argmax) != 1 or not 0 < len(generated) <= count:
            raise ValueError("invalid generation lengths")
        if len(generated) != count and (not generated or generated[-1] not in request.get("stop_tokens", [])):
            raise ValueError("unexplained early generation stop")
        if set(generated) & set(request.get("suppress_tokens", [])):
            raise ValueError("suppressed generated token")
    elif generated or len(argmax) != len(teacher) + 1:
        raise ValueError("invalid teacher lengths")
    if request.get("check_reset") and result.get("reset_exact_logits_ok") is not True:
        raise ValueError("reset logits check failed")
    keys = ("input_ids", "generated_ids", "teacher_argmax")
    if expected is not None and (not isinstance(expected, dict) or any(result[k] != expected.get(k) for k in keys)):
        raise ValueError("native numeric result differs from expected result")
    return {k: result[k] for k in keys}


@owned_main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sanitizer", "probe", "binary", "checkpoint", "request", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--exclude-nvjet-synccheck", action="store_true")
    p.add_argument("--tools", nargs="+", choices=TOOLS)
    p.add_argument("--diagnostic-only", action="store_true")
    p.add_argument("--expected-result", type=Path)
    args = p.parse_args()
    if not 0 < args.timeout <= 3600:
        p.error("timeout must be in (0,3600]")
    if args.tools is not None and not args.diagnostic_only:
        p.error("--tools requires --diagnostic-only; scoped runs cannot claim full acceptance")
    selected = list(dict.fromkeys(args.tools or TOOLS))
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = {name: getattr(args, name) for name in ("binary", "probe", "sanitizer", "request")}
    if args.expected_result:
        artifacts["expected_result"] = args.expected_result
    start_hashes, identity_errors = artifact_hashes(artifacts)

    def verify_identity():
        current, errors = artifact_hashes(artifacts)
        identity_errors.update(errors)
        for name in artifacts:
            if current[name] != start_hashes[name]:
                identity_errors.setdefault(name, "SHA256 changed from pinned start identity")
        if identity_errors:
            raise IdentityChanged("execution artifact identity changed or missing")

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    records = []
    try:
        request = json.loads(args.request.read_text())
        expected = json.loads(args.expected_result.read_text()) if args.expected_result else None
        fixture_error = None
        if args.expected_result:
            native_result(args.expected_result, request)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        request, expected, fixture_error = None, None, str(exc)
    reference = expected

    def run(name, tool, command, extra=(), application=None):
        nonlocal reference
        verify_identity()
        log = args.output / (name + ".log")
        cmd = [str(args.sanitizer), "--tool", tool, "--error-exitcode", "86", "--print-limit", "4",
               "--log-file", str(log), *extra, *map(str, command)]
        timed_out = False
        with (args.output / (name + ".stdout.log")).open("w") as stream:
            owner = OwnedJob()
            try:
                owner.start(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                code, timed_out = owner.wait(args.timeout)
            finally:
                owner.close()
        text = log.read_text() if log.exists() else ""
        clean = code == 0 and not timed_out and "Internal Sanitizer Error" not in text and (
            "ERROR SUMMARY: 0 errors" in text or "0 errors, 0 warnings" in text)
        application_ok, application_error = None, None
        if application is not None:
            try:
                if fixture_error:
                    raise ValueError(fixture_error)
                result = native_result(application, request, reference)
                if reference is None:
                    reference = result
                application_ok = True
            except (OSError, ValueError, TypeError, KeyError) as exc:
                application_ok, application_error = False, str(exc)
            clean = clean and application_ok
        record = dict(name=name, tool=tool, command=cmd, returncode=code, timed_out=timed_out, clean=clean,
                      application_ok=application_ok, application_error=application_error, group_cleanup_complete=owner.closed)
        records.append(record)
        verify_identity()
        return record

    native = [args.binary, "--experimental", args.checkpoint, args.request]
    try:
        for tool in selected:
            extra = ["--kernel-name-exclude", "kns=nvjet"] if tool == "synccheck" and args.exclude_nvjet_synccheck else []
            destination = args.output / (tool + ".json")
            run(tool, tool, [*native, destination], extra, application=destination)
        if args.diagnostic_only:
            status, controls_ok = "not-run-diagnostic-only", False
        else:
            status, controls_ok = initcheck_controls(run, args, native)
        verify_identity()
    except IdentityChanged:
        status, controls_ok = "failed-identity-change", False
    end_hashes, end_errors = artifact_hashes(artifacts)
    identity_errors.update(end_errors)
    for name in artifacts:
        if end_hashes[name] != start_hashes[name]:
            identity_errors.setdefault(name, "SHA256 changed from pinned start identity")
    hashes_stable = not identity_errors
    if not hashes_stable:
        status, controls_ok = "failed-identity-change", False
    scoped_passed = (hashes_stable and len(records) >= len(selected)
                     and all(r["clean"] for r in records[:len(selected)]))
    full_acceptance = (not args.diagnostic_only and not args.exclude_nvjet_synccheck
                       and scoped_passed and status == "passed")
    passed = full_acceptance
    report = dict(passed=passed, full_acceptance=full_acceptance, scoped_passed=scoped_passed,
                  diagnostic_only=args.diagnostic_only, selected_tools=selected,
                  numeric_reference="expected-result" if args.expected_result else "cross-tool-consistency",
                  initcheck_status=status, controls_ok=controls_ok,
                  synccheck_nvjet_excluded=args.exclude_nvjet_synccheck, records=records,
                  start_hashes=start_hashes, end_hashes=end_hashes, hashes_stable=hashes_stable,
                  identity_errors=identity_errors,
                  binary_sha256=start_hashes["binary"], probe_sha256=start_hashes["probe"])
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0 if (scoped_passed if args.diagnostic_only else passed) else 86


def initcheck_controls(run, args, native):
    control = run("initcheck-kernel-control", "initcheck", [args.probe, "kernel", "257", "2048"])
    missing = run("initcheck-missing-control", "initcheck", [args.probe, "missing", "257", "2048"])
    producer = run("initcheck-cublas-producer", "initcheck", [args.probe, "gemm", "257", "2048"])
    missing_path = args.output / "initcheck-missing-control.log"
    missing_log = missing_path.read_text() if missing_path.exists() else ""
    controls_ok = control["clean"] and missing["returncode"] == 86 and "Uninitialized __global__ memory read" in missing_log
    if controls_ok and producer["clean"]:
        init = run("initcheck", "initcheck", [*native, args.output / "initcheck.json"], application=args.output / "initcheck.json")
        status = "passed" if init["clean"] else "failed"
    else:
        producer_log = args.output / "initcheck-cublas-producer.log"
        text = producer_log.read_text() if producer_log.exists() else ""
        stdout = (args.output / "initcheck-cublas-producer.stdout.log").read_text()
        reproduced = producer["returncode"] == 86 and "PASS: 128 sampled outputs" in stdout and "Uninitialized __global__ memory read" in text
        status = "blocked-cuBLAS-output-tracking" if controls_ok and reproduced else "failed-preflight-controls"
    return status, controls_ok


if __name__ == "__main__":
    raise SystemExit(main())
