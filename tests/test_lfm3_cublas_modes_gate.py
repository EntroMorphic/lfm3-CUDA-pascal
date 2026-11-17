"""Host-only classification, resolver and owned-process tests; no CUDA launches."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
SPEC = importlib.util.spec_from_file_location("cublas_modes", ROOT/"tools/check_lfm3_cublas_modes.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def application(producer="gemm", mode="default", version=130100):
    prefix = "INTENTIONAL UNINITIALIZED CONTROL: " if producer == "missing" else "PASS: "
    return (f"CONFIG producer={producer}; rows=1; width=2048; mode={mode}; compute=68\n"
            f"LIBRARY cublas={version}; mode={mode}\nMATH actual=0; mode={mode}\n"
            f"{prefix}128 sampled outputs of 6144; producer={producer}; mode={mode}; cublas={version}; "
            "numerical-only (not sanitizer acceptance)\n")


CLEAN = "========= ERROR SUMMARY: 0 errors\n"
DOWN = "Uninitialized __global__ memory read at read_output(\n========= ERROR SUMMARY: 1 errors\n"
SPLIT = "Uninitialized __global__ memory read at cublasLt::splitKreduce_kernel(\n========= ERROR SUMMARY: 1 errors\n"


@pytest.mark.parametrize("producer,code,log,expected", [
    ("kernel", 0, CLEAN, "clean"), ("gemm", 0, CLEAN, "clean"),
    ("missing", 86, DOWN, "negative-control-detected"),
    ("missing", 0, CLEAN, "unexpected-failure"), ("kernel", 86, DOWN, "unexpected-failure"),
    ("gemm", 86, DOWN, "downstream-tracking"), ("gemm", 86, SPLIT, "internal-splitK"),
    ("gemm", 86, DOWN+SPLIT.replace("========= ERROR SUMMARY: 1 errors", ""), "internal-splitK"), ("gemm", 1, DOWN, "unexpected-failure"),
    ("gemm", 0, "", "unexpected-failure"),
])
def test_classification(producer, code, log, expected):
    assert gate.classify(producer, "default", 1, 2048, 130100, code, log, application(producer)) == expected


@pytest.mark.parametrize("change", [
    lambda s: s.replace("128 sampled", "127 sampled"),
    lambda s: s.replace("default", "pedantic"),
    lambda s: s.replace("130100", "131000"),
    lambda s: s.replace("rows=1;", "rows=257;"),
    lambda s: s.replace("MATH actual=0", "MATH actual=2"),
    lambda s: s.replace("PASS:", "incorrect projection output"),
    lambda s: s + "FAIL: incorrect projection output\n",
    lambda s: s.replace("LIBRARY", "NOT_LIBRARY", 1).replace("NOT_LIBRARY cublas", "MISSING cublas"),
])
def test_wrong_output_identity_or_error_never_clean(change):
    assert gate.classify("gemm", "default", 1, 2048, 130100, 0, CLEAN, change(application())) == "unexpected-failure"


def test_unsupported_and_timeout_are_distinct():
    assert gate.classify("gemm", "pedantic", 1, 2048, 130100, 1, "", "FAIL: cuBLAS status=15") == "unsupported-or-error"
    assert gate.classify("gemm", "default", 1, 2048, 130100, None, CLEAN, application(), True) == "timeout"


def fake_ldd(root):
    root.mkdir(exist_ok=True)
    result = []
    for name, suffix in (("libcublas.so.13", ".1.0.3"), ("libcublasLt.so.13", ".1.0.3"), ("libcudart.so.13", ".0.88")):
        path = root/(name+suffix); path.write_bytes(b"host-only fake library")
        (root/name).symlink_to(path.name)
        result.append(f" {name} => {root/name} (0x1234)")
    return "\n".join(result)


def test_resolution_hashes_real_files_and_rejects_alias_escape(tmp_path):
    selected = tmp_path/"selected"; outside = tmp_path/"outside"
    text = fake_ldd(selected)
    found = gate.libraries(text, selected)
    assert gate.library_version(found) == 130100
    assert all(len(v["sha256"]) == 64 for v in found.values())
    fake_ldd(outside)
    (selected/"libcublasLt.so.13").unlink()
    (selected/"libcublasLt.so.13").symlink_to(outside/"libcublasLt.so.13")
    with pytest.raises(ValueError, match="within selected root"):
        gate.libraries(text, selected)
    with pytest.raises(ValueError, match="within selected root"):
        gate.libraries(text, tmp_path/"wrong")
    with pytest.raises(ValueError, match="unresolved"):
        gate.libraries(text+"\nlibfoo => not found", selected)


def test_explicit_root_precedes_serving_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited::/fallback")
    monkeypatch.delenv("LD_PRELOAD", raising=False); monkeypatch.delenv("LD_AUDIT", raising=False)
    assert gate.environment(tmp_path, Path("/fallback"))["LD_LIBRARY_PATH"].split(":") == [str(tmp_path), "/fallback", "/inherited"]
    monkeypatch.setenv("LD_PRELOAD", "/unofficial.so")
    with pytest.raises(ValueError): gate.environment(tmp_path, tmp_path)


@pytest.mark.parametrize("behavior", ["exit", "timeout", "wait-error", "start-error"])
def test_owned_cleanup_for_fake_helper(tmp_path, monkeypatch, behavior):
    owners = []
    original = gate.OwnedJob
    class Observed(original):
        def __init__(self):
            super().__init__(); owners.append(self)
        def start(self, *args, **kwargs):
            result = super().start(*args, **kwargs)
            if behavior == "start-error": raise RuntimeError("injected start error")
            return result
        def wait(self, timeout):
            if behavior == "wait-error": raise RuntimeError("injected wait error")
            return super().wait(timeout)
    monkeypatch.setattr(gate, "OwnedJob", Observed)
    helper = "import time;time.sleep(30)"
    script = ("import subprocess,sys,time\n"
              f"subprocess.Popen([sys.executable,'-c',{helper!r}])\n" +
              ("sys.exit(0)" if behavior == "exit" else "time.sleep(30)"))
    command = [sys.executable, "-c", script]
    if behavior.endswith("error"):
        with pytest.raises(RuntimeError): gate.run_owned(command, dict(), tmp_path/"log", 0.3)
    else:
        code, timeout = gate.run_owned(command, dict(), tmp_path/"log", 0.3)
        assert timeout == (behavior == "timeout")
        assert code == (0 if behavior == "exit" else None)
    import lfm3_owned_process
    assert owners[0].closed
    assert owners[0].process.poll() is not None
    assert lfm3_owned_process.live_group(owners[0].process.pid) == []


def test_matrix_source_bounds_and_never_accepts_runtime():
    source = Path(gate.__file__).read_text()
    assert gate.MODES == ("default", "pedantic", "default-algo")
    assert gate.SHAPES == ((257, 2048), (1, 2048))
    assert '"--print-limit", "4"' in source
    assert '"--error-exitcode", "86"' in source
    assert "integrated_initcheck_accepted=False" in source
    assert "hashes_stable" in source and "@owned_main" in source
    for forbidden in ("--dump", "--record", "--kernel-name", "--suppress", "full_acceptance"):
        assert forbidden not in source


@pytest.mark.parametrize("failure,count,status", [("clean", 36, "diagnostic-completed"), ("abi-clean", 36, "diagnostic-completed"), ("version", 1, "failed-controls-or-timeout"), ("gemm", 36, "blocked"), ("control", 1, "failed-controls-or-timeout"), ("timeout", 3, "failed-controls-or-timeout")])
def test_main_preflights_both_roots_and_failstops(tmp_path, monkeypatch, failure, count, status):
    import json
    serving, isolated = tmp_path/"serving", tmp_path/"isolated"
    texts = {str(p): fake_ldd(p) for p in (serving, isolated)}
    if failure in ("abi-clean", "version"):
        for root in (serving, isolated):
            (root/"libcublas.so.13").unlink()
            (root/"libcublas.so.13.1.0.3").rename(root/"libcublas.so.13")
    (isolated/"libcudart.so.13").unlink()
    (isolated/"libcudart.so.13").symlink_to(serving/"libcudart.so.13")
    probe, tool = tmp_path/"probe", tmp_path/"sanitizer"
    probe.write_bytes(b"never executed"); tool.write_bytes(b"never executed")
    output = tmp_path/"result"; events = []
    def fake_run(command, env, destination, timeout):
        root = env["LD_LIBRARY_PATH"].split(":")[0]
        if command[0].endswith("ldd"):
            events.append("ldd"); destination.write_text(texts[root]); return 0, False
        if command[-1] == "--version":
            destination.write_text("fake host-only tool"); return 0, False
        assert events[:2] == ["ldd", "ldd"]
        events.append("case")
        producer, rows, width, mode = command[-4:]
        app = application(producer, mode).replace("rows=1;", f"rows={rows};").replace("outputs of 6144;", f"outputs of {int(rows)*3*int(width)};")
        app = app.replace("MATH actual=0;", f"MATH actual={2 if mode == 'pedantic' else 0};")
        if failure == "version": app = app.replace("130100", "131000")
        destination.write_text(app)
        log = Path(command[command.index("--log-file")+1])
        if producer == "gemm" and failure == "timeout": return None, True
        if producer == "kernel" and failure == "control": log.write_text(DOWN); return 86, False
        if producer == "missing" or (producer == "gemm" and failure not in ("clean", "abi-clean")): log.write_text(DOWN); return 86, False
        log.write_text(CLEAN); return 0, False
    monkeypatch.setattr(gate, "run_owned", fake_run)
    monkeypatch.delenv("LD_PRELOAD", raising=False); monkeypatch.delenv("LD_AUDIT", raising=False)
    monkeypatch.setattr(sys, "argv", ["runner", "--sanitizer", str(tool), "--probe", str(probe),
                        "--library-dir", str(serving), "--library-dir", str(isolated), "--output", str(output)] +
                        (["--expected-version", "130100", "--expected-version", "130100"] if failure in ("abi-clean", "version") else []))
    assert gate.main() == (0 if failure in ("clean", "abi-clean") else 86 if failure == "gemm" else 1)
    report = json.loads((output/"summary.json").read_text())
    assert report["status"] == status and len(report["records"]) == count
    assert report["hashes_stable"] and report["integrated_initcheck_accepted"] is False
    assert report["diagnostic_completed"] == (count == 36)
    if failure in ("abi-clean", "version"):
        assert report["libraries"][0]["version_provenance"] == "expected-config-verified-per-case"
        assert report["libraries"][0]["expected_version"] == 130100
    if failure == "version":
        assert report["records"][0]["producer"] == "kernel"
        assert report["records"][0]["classification"] == "unexpected-failure"


def test_version_expectation_never_invents_or_overrides_filename():
    def found(name): return {"libcublas.so.13": {"path": "/fake/"+name}}
    abi = found("libcublas.so.13")
    with pytest.raises(ValueError, match="cannot establish"): gate.library_version(abi)
    assert gate.library_version(abi, 130100) == 130100
    assert gate.library_version(found("libcublas.so.13.10.0.20"), 131000) == 131000
    with pytest.raises(ValueError, match="disagrees"):
        gate.library_version(found("libcublas.so.13.10.0.20"), 130100)
    with pytest.raises(ValueError, match="cannot establish"):
        gate.library_version(found("arbitrary-name"), 130100)


@pytest.mark.parametrize("values", [["130100"], ["130100", "131000", "131000"],
                                  ["129999", "131000"], ["130100", "140000"],
                                  ["13.1", "131000"], ["bad", "131000"]])
def test_cli_expected_versions_strict_count_range_and_integer(monkeypatch, values):
    argv = ["runner", "--sanitizer", "/unused", "--probe", "/unused", "--output", "/unused",
            "--library-dir", "/unused-a", "--library-dir", "/unused-b"]
    for value in values: argv += ["--expected-version", value]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((ValueError, SystemExit)): gate.main()


def test_actual_v2_missing_control_with_print_limit_footer():
    stem = ROOT/"validation/cublas-modes-13.1-vs-13.10-v2/library-0/default-257x2048-missing"
    log = Path(str(stem)+".sanitizer.log").read_text()
    app = Path(str(stem)+".application.log").read_text()
    assert "ERROR SUMMARY: 128 errors\n" in log
    assert "ERROR SUMMARY: 124 errors were not printed." in log
    assert gate.classify("missing", "default", 257, 2048, 130100, 86, log, app) == "negative-control-detected"
    for broken in (log.replace("========= ERROR SUMMARY: 128 errors\n", ""),
                   log.replace("128 errors\n", "128 err"),
                   log+"\n========= ERROR SUMMARY: 128 errors\n"):
        assert gate.classify("missing", "default", 257, 2048, 130100, 86, broken, app) == "unexpected-failure"


@pytest.mark.parametrize("producer", ["kernel", "gemm"])
def test_print_limit_zero_footer_is_not_a_clean_total(producer):
    footer = "========= ERROR SUMMARY: 0 errors were not printed. Use --print-limit option to adjust the number of printed errors\n"
    assert gate.classify(producer, "default", 1, 2048, 130100, 0, footer, application(producer)) == "unexpected-failure"
    assert gate.classify(producer, "default", 1, 2048, 130100, 0, CLEAN+CLEAN, application(producer)) == "unexpected-failure"
