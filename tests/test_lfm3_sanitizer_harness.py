"""Fail-closed sanitizer-report tests using a fake tool, not GPU coverage."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/check_lfm3_sanitizer.py"
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
a=sys.argv[1:]; log=pathlib.Path(a[a.index('--log-file')+1]); mode=os.environ['FAKE_MODE']
name=log.stem; code=0; text='ERROR SUMMARY: 0 errors'; stdout=''
if name=='initcheck-missing-control':
    text='Uninitialized __global__ memory read'; code=86
    if mode=='bad-control': code=0
if name=='initcheck-cublas-producer' and mode in ('blocked','bad-numerics'):
    text='Uninitialized __global__ memory read'; code=86
    stdout='PASS: 128 sampled outputs' if mode=='blocked' else 'incorrect projection output'
if name=='memcheck' and mode=='mem-error': code=86
if name=='initcheck-cublas-producer' and mode=='internal': text='Internal Sanitizer Error'
if name in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
    request=json.loads(pathlib.Path(a[-2]).read_text())
    result=dict(backend='native-cuda-experimental', input_ids=request['input_ids'],
                generated_ids=[], teacher_argmax=[7], reset_exact_logits_ok=True)
    if name=='memcheck' and mode=='wrong-input': result['input_ids']=[9]
    if name=='memcheck' and mode=='wrong-teacher': result['teacher_argmax']=[8]
    if name=='memcheck' and mode=='wrong-count': result['generated_ids']=[7]
    if not (name=='memcheck' and mode=='missing-app'):
        pathlib.Path(a[-1]).write_text('{' if name=='memcheck' and mode=='bad-app-json' else json.dumps(result))
else:
    # Probe controls take producer/ROWS/WIDTH, not a native request/output pair.
    assert a[-3:] == [{'initcheck-kernel-control':'kernel',
                      'initcheck-missing-control':'missing',
                      'initcheck-cublas-producer':'gemm'}[name], '257', '2048']
    if code==0: stdout='PASS: 128 sampled outputs'
log.write_text(text); print(stdout); sys.exit(code)
'''


@pytest.mark.parametrize("mode,status,passed", [
    ("clean", "passed", True),
    ("blocked", "blocked-cuBLAS-output-tracking", False),
    ("bad-control", "failed-preflight-controls", False),
    ("bad-numerics", "failed-preflight-controls", False),
    ("internal", "failed-preflight-controls", False),
    ("mem-error", "passed", False),
    ("missing-app", "passed", False),
    ("bad-app-json", "passed", False),
    ("wrong-input", "passed", False),
    ("wrong-teacher", "passed", False),
    ("wrong-count", "passed", False),
])
def test_cannot_credit_incomplete_or_failed_checks(tmp_path, monkeypatch, mode, status, passed):
    sanitizer = tmp_path / "fake-sanitizer"
    sanitizer.write_text(FAKE)
    sanitizer.chmod(0o755)
    for name in ("probe", "binary"):
        (tmp_path / name).write_text("fixture")
    (tmp_path / "request").write_text(json.dumps(dict(
        input_ids=[1, 2], teacher_ids=[], max_new_tokens=0, capacity=32768,
        experimental_long_context=True, check_reset=True)))
    (tmp_path / "expected.json").write_text(json.dumps(dict(
        backend="native-cuda-experimental", input_ids=[1, 2],
        generated_ids=[], teacher_argmax=[7], reset_exact_logits_ok=True)))
    monkeypatch.setenv("FAKE_MODE", mode)
    output = tmp_path / "evidence"
    command = [sys.executable, str(TOOL), "--sanitizer", str(sanitizer),
               "--probe", str(tmp_path / "probe"), "--binary", str(tmp_path / "binary"),
               "--checkpoint", str(tmp_path), "--request", str(tmp_path / "request"), "--output", str(output),
               "--expected-result", str(tmp_path / "expected.json")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if passed else 86), result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["passed"] is passed
    assert report["full_acceptance"] is passed
    assert report["initcheck_status"] == status
    assert report["synccheck_nvjet_excluded"] is False
    if mode in ("missing-app", "bad-app-json", "wrong-input", "wrong-teacher", "wrong-count"):
        assert report["records"][0]["application_ok"] is False
        assert report["records"][0]["clean"] is False
