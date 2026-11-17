"""Host checks for the silent-failure guards in tools/thor_remote.py and the gate runner."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import thor_remote as tr

RUNNER = 'a' * 64


def live(**kw):
    return dict(dict(runner_sha256=RUNNER, status='running', heartbeat_unix=1000.0), **kw)


def test_live_runner_with_fresh_heartbeat_is_accepted():
    assert tr.check_live(live(), RUNNER, now=1030.0, stale_after=120) is None


def test_wrong_runner_version_is_rejected():
    # The 2026-11-06 failure: an old runner was still on Thor.
    problem = tr.check_live(live(runner_sha256='b' * 64), RUNNER, now=1000.0, stale_after=120)
    assert problem and 'identity' in problem


def test_missing_identity_is_rejected():
    # Status files from runners that predate identity reporting can never pass as live.
    st = live(); del st['runner_sha256']
    assert tr.check_live(st, RUNNER, now=1000.0, stale_after=120)


def test_stale_heartbeat_means_dead_or_hung():
    problem = tr.check_live(live(heartbeat_unix=1000.0), RUNNER, now=1200.0, stale_after=120)
    assert problem and 'stale' in problem


def test_finished_run_is_not_judged_by_heartbeat():
    st = live(status='stopped-at-failure', heartbeat_unix=0.0)
    assert tr.check_live(st, RUNNER, now=10**9, stale_after=120) is None


def test_sync_refuses_paths_outside_repo(tmp_path):
    outside = tmp_path / 'x.py'; outside.write_text('x')
    with pytest.raises(tr.Verification):
        tr.sync([outside])


def test_sync_refuses_missing_files():
    with pytest.raises(tr.Verification):
        tr.sync([ROOT / 'tools' / 'does_not_exist.py'])


def test_remote_hash_parse_requires_every_path(monkeypatch):
    monkeypatch.setattr(tr, 'remote', lambda cmd, timeout=60: 'abc  /p/one\n')
    with pytest.raises(tr.Verification):
        tr.remote_sha256(['/p/one', '/p/two'])
    assert tr.remote_sha256(['/p/one']) == {'/p/one': 'abc'}


def test_remote_needs_thor_host(monkeypatch):
    monkeypatch.setattr(tr, 'HOST', '')
    monkeypatch.setattr(tr.subprocess, 'run', lambda *a, **k: pytest.fail('ssh ran without a host'))
    with pytest.raises(tr.Unreachable, match='THOR_HOST'):
        tr.remote('true')


def test_remote_timeout_and_ssh_failure_raise(monkeypatch):
    monkeypatch.setattr(tr, 'HOST', 'user@thor.example')
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd='ssh', timeout=1)
    monkeypatch.setattr(tr.subprocess, 'run', timeout)
    with pytest.raises(tr.Unreachable):
        tr.remote('true', timeout=1)
    monkeypatch.setattr(tr.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 255, '', 'no route'))
    with pytest.raises(tr.Unreachable):
        tr.remote('true')
    monkeypatch.setattr(tr.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, '', 'boom'))
    with pytest.raises(tr.Verification):
        tr.remote('false')


def test_watch_fails_after_repeated_unreachable(monkeypatch, tmp_path):
    runner = tmp_path / 'r.py'; runner.write_text('x')
    def down(path):
        raise tr.Unreachable('down')
    monkeypatch.setattr(tr, 'read_status', down)
    monkeypatch.setattr(tr.time, 'sleep', lambda s: None)
    with pytest.raises(tr.Unreachable):
        tr.watch('/x/gates.json', runner, max_unreachable=3)


def test_watch_fails_on_stale_heartbeat(monkeypatch, tmp_path):
    runner = tmp_path / 'r.py'; runner.write_text('x')
    st = dict(runner_sha256=tr.sha256(runner), status='running', heartbeat_unix=0.0)
    monkeypatch.setattr(tr, 'read_status', lambda path: st)
    with pytest.raises(RuntimeError):
        tr.watch('/x/gates.json', runner, stale_after=120)


def test_gate_runner_reports_identity_and_heartbeat():
    source = (ROOT / 'tools' / 'run_lfm3_thin_hybrid_gates.py').read_text()
    for field in ("runner_sha256=digest(__file__)", 'heartbeat_unix', "'runner-error'", 'os.replace(tmp, summary)'):
        assert field in source


def test_probe_rejects_timings_above_dram_peak():
    source = (ROOT / 'native' / 'probes' / 'lfm3_thin_v5_probe.cu').read_text()
    assert 'bytes / t / 1e9 <= peak' in source and 'return exact && timing_valid ? 0 : 1;' in source
