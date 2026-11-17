"""CPU-only integrated sanitizer ownership and fail-closed evidence tests."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools.lfm3_owned_process import live_group

TOOL = Path(__file__).resolve().parents[1] / "tools/check_lfm3_sanitizer.py"
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, sys, time
a=sys.argv; log=pathlib.Path(a[a.index('--log-file')+1]); name=log.stem
with (log.parent/'invocations').open('a') as counter: counter.write(name+chr(10))
mode=os.environ.get('INTEGRATED_FAKE', 'clean')
if mode in ('descendant', 'timeout', 'signal'):
    child=subprocess.Popen([sys.executable,'-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
    pathlib.Path(os.environ['PID_FILE']).write_text(str(os.getpgrp()))
    if mode in ('timeout', 'signal'): time.sleep(60)
text='ERROR SUMMARY: 0 errors'; code=0
if name=='initcheck-missing-control':
    text='Uninitialized __global__ memory read'; code=86
log.write_text(text)
if name in ('memcheck','racecheck','synccheck','initcheck') and mode!='missing':
    request=json.loads(pathlib.Path(a[-2]).read_text())
    result=dict(backend='native-cuda-experimental', input_ids=request['input_ids'],
                generated_ids=[], teacher_argmax=[7], reset_exact_logits_ok=True)
    if mode=='wrong': result['input_ids']=[9]
    if mode=='mismatch': result['teacher_argmax']=[8]
    pathlib.Path(a[-1]).write_text('{' if mode=='bad-json' else json.dumps(result))
    targets=dict(binary=pathlib.Path(a[a.index('--experimental')-1]),
                 probe=pathlib.Path(a[-2]).parent/'probe', sanitizer=pathlib.Path(a[0]),
                 request=pathlib.Path(a[-2]), expected_result=pathlib.Path(a[-2]).parent/'expected.json')
    if mode.startswith('mutate-'):
        target=targets[mode.removeprefix('mutate-')]
        target.write_bytes(target.read_bytes()+b' changed')
    if mode.startswith('remove-'):
        targets[mode.removeprefix('remove-')].unlink()
sys.exit(code)
'''


def fixture(tmp_path, monkeypatch, mode="clean", extra=()):
    sanitizer = tmp_path / "sanitizer"
    sanitizer.write_text(FAKE)
    sanitizer.chmod(0o755)
    request = dict(input_ids=[1, 2], max_new_tokens=0)
    (tmp_path / "request").write_text(json.dumps(request))
    for name in ("binary", "probe"):
        (tmp_path / name).write_text("not executed by fake")
    monkeypatch.setenv("INTEGRATED_FAKE", mode)
    monkeypatch.setenv("PID_FILE", str(tmp_path / "pgid"))
    command = [sys.executable, str(TOOL)]
    for name in ("sanitizer", "binary", "probe", "request"):
        command += ["--" + name, str(tmp_path / name)]
    return command + ["--checkpoint", str(tmp_path), "--output", str(tmp_path / "out"), *extra]


def report(tmp_path):
    return json.loads((tmp_path / "out/summary.json").read_text())


def test_default_full_gate(tmp_path, monkeypatch):
    result = subprocess.run(fixture(tmp_path, monkeypatch), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    evidence = report(tmp_path)
    assert evidence["full_acceptance"] and evidence["passed"]
    assert len(evidence["records"]) == 7
    assert all(r["group_cleanup_complete"] for r in evidence["records"])
    assert evidence["hashes_stable"] is True
    assert evidence["start_hashes"] == evidence["end_hashes"]
    assert set(evidence["start_hashes"]) == {"binary", "probe", "sanitizer", "request"}


@pytest.mark.parametrize("mode", ["missing", "bad-json", "wrong"])
def test_invalid_application_cannot_pass(tmp_path, monkeypatch, mode):
    result = subprocess.run(fixture(tmp_path, monkeypatch, mode), timeout=20)
    assert result.returncode == 86
    assert report(tmp_path)["full_acceptance"] is False


def test_scoped_pass_is_never_full_acceptance(tmp_path, monkeypatch):
    result = subprocess.run(fixture(tmp_path, monkeypatch, extra=("--tools", "synccheck", "--diagnostic-only")), timeout=20)
    assert result.returncode == 0
    evidence = report(tmp_path)
    assert evidence["scoped_passed"] is True
    assert evidence["passed"] is evidence["full_acceptance"] is False
    assert len(evidence["records"]) == 1
    assert evidence["controls_ok"] is False


def test_selector_requires_explicit_diagnostic(tmp_path, monkeypatch):
    result = subprocess.run(fixture(tmp_path, monkeypatch, extra=("--tools", "synccheck")), timeout=20)
    assert result.returncode == 2
    assert not (tmp_path / "out").exists()


def test_expected_numeric_mismatch(tmp_path, monkeypatch):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(dict(backend="native-cuda-experimental", input_ids=[1, 2], generated_ids=[], teacher_argmax=[7])))
    result = subprocess.run(fixture(tmp_path, monkeypatch, "mismatch", ("--diagnostic-only", "--tools", "memcheck", "--expected-result", str(expected))), timeout=20)
    assert result.returncode == 86
    assert report(tmp_path)["records"][0]["application_ok"] is False


@pytest.mark.parametrize("mode", ["descendant", "timeout", "signal"])
def test_group_descendants_quiesced(tmp_path, monkeypatch, mode):
    command = fixture(tmp_path, monkeypatch, mode, ("--diagnostic-only", "--tools", "synccheck", "--timeout", "0.5"))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "pgid").exists():
            assert time.monotonic() < deadline
            assert process.poll() is None
            time.sleep(0.01)
        pgid = int((tmp_path / "pgid").read_text())
        if mode == "signal":
            process.send_signal(signal.SIGTERM)
        process.communicate(timeout=15)
        assert not live_group(pgid)
        if mode == "timeout":
            assert process.returncode == 86
            assert report(tmp_path)["records"][0]["timed_out"]
        elif mode == "descendant":
            assert process.returncode == 0
        else:
            assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_exclusion_cannot_be_full_clean(tmp_path, monkeypatch):
    result = subprocess.run(fixture(tmp_path, monkeypatch, extra=("--exclude-nvjet-synccheck",)), timeout=20)
    assert result.returncode == 86
    assert report(tmp_path)["full_acceptance"] is False


def test_postspawn_exception_still_closes_group(tmp_path, monkeypatch):
    from tools import check_lfm3_sanitizer as runner
    command = fixture(tmp_path, monkeypatch, "signal", ("--diagnostic-only", "--tools", "synccheck"))
    groups = []
    def fail_after_start(owner, timeout):
        deadline = time.monotonic() + 10
        while not (tmp_path / "pgid").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        groups.append(owner.process.pid)
        raise RuntimeError("injected postspawn setup failure")
    monkeypatch.setattr(runner.OwnedJob, "wait", fail_after_start)
    monkeypatch.setattr(sys, "argv", command[1:])
    with pytest.raises(RuntimeError, match="injected postspawn"):
        runner.main()
    assert groups and not live_group(groups[0])


@pytest.mark.parametrize("source,generated,argmax,valid", [
    ({"input_ids": [1], "teacher_ids": [3, 4]}, [], [2, 3, 4], True),
    ({"input_ids": [1], "teacher_ids": [3, 4]}, [], [2], False),
    ({"input_ids": [1], "max_new_tokens": 2}, [3, 4], [3], True),
    ({"input_ids": [1], "max_new_tokens": 2}, [3], [3], False),
    ({"input_ids": [1], "max_new_tokens": 2, "stop_tokens": [3]}, [3], [3], True),
    ({"input_ids": [1], "max_new_tokens": 1}, [True], [3], False),
])
def test_native_result_lengths_and_types(tmp_path, source, generated, argmax, valid):
    from tools.check_lfm3_sanitizer import native_result
    path = tmp_path / "result.json"
    path.write_text(json.dumps(dict(backend="native-cuda-experimental", input_ids=[1],
                                    generated_ids=generated, teacher_argmax=argmax)))
    if valid:
        native_result(path, source)
    else:
        with pytest.raises(ValueError):
            native_result(path, source)


@pytest.mark.parametrize("tool", ["memcheck", "synccheck"])
@pytest.mark.parametrize("explicit_count", [False, True])
def test_full32k_prefill_scoped_with_default_teacher_shape(tmp_path, monkeypatch, tool, explicit_count):
    expected = tmp_path / "expected.json"
    command = fixture(tmp_path, monkeypatch, extra=(
        "--diagnostic-only", "--tools", tool, "--expected-result", str(expected)))
    source = dict(input_ids=[1] * 32768, capacity=32768, experimental_long_context=True)
    if explicit_count:
        source["max_new_tokens"] = 0
    (tmp_path / "request").write_text(json.dumps(source))
    expected.write_text(json.dumps(dict(backend="native-cuda-experimental",
                                       input_ids=source["input_ids"], generated_ids=[], teacher_argmax=[7])))
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    evidence = report(tmp_path)
    assert evidence["selected_tools"] == [tool]
    assert evidence["scoped_passed"] is True
    assert evidence["passed"] is evidence["full_acceptance"] is False
    assert evidence["controls_ok"] is False
    assert evidence["initcheck_status"] == "not-run-diagnostic-only"
    assert evidence["records"][0]["application_ok"] is True
    assert len(evidence["records"]) == 1
    assert evidence["hashes_stable"] is True
    assert "expected_result" in evidence["start_hashes"]


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("mode,artifact", [
    ("mutate-binary", "binary"),
    ("mutate-probe", "probe"),
    ("mutate-sanitizer", "sanitizer"),
    ("mutate-request", "request"),
    ("mutate-expected_result", "expected_result"),
    ("remove-sanitizer", "sanitizer"),
    ("remove-request", "request"),
])
def test_artifact_identity_failure_stops_further_jobs(tmp_path, monkeypatch, diagnostic, mode, artifact):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(dict(backend="native-cuda-experimental", input_ids=[1, 2],
                                       generated_ids=[], teacher_argmax=[7])))
    extra = ["--expected-result", str(expected)]
    if diagnostic:
        extra += ["--diagnostic-only", "--tools", "memcheck", "synccheck"]
    command = fixture(tmp_path, monkeypatch, mode, extra)
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 86, result.stderr
    evidence = report(tmp_path)
    assert evidence["hashes_stable"] is False
    assert evidence["start_hashes"][artifact] != evidence["end_hashes"][artifact]
    assert artifact in evidence["identity_errors"]
    assert evidence["initcheck_status"] == "failed-identity-change"
    assert evidence["scoped_passed"] is evidence["passed"] is evidence["full_acceptance"] is False
    assert evidence["controls_ok"] is False
    # The counterexample has both valid application evidence and a clean log.
    assert len(evidence["records"]) == 1
    assert evidence["records"][0]["clean"] is True
    assert evidence["records"][0]["application_ok"] is True
    assert evidence["records"][0]["returncode"] == 0
    assert (tmp_path / "out/invocations").read_text().splitlines() == ["memcheck"]
