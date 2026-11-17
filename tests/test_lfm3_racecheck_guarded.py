import json
import os
import signal
import sys

import pytest

from tools import check_lfm3_racecheck_guarded as gate
from tools.lfm3_owned_process import live_group


@pytest.fixture
def run(tmp_path, monkeypatch):
    binary = tmp_path / 'app'
    binary.write_text(f'#!{sys.executable}\nimport json,sys\n'
                      'r=json.load(open(sys.argv[-2]))\n'
                      'json.dump(dict(backend="native-cuda-experimental",input_ids=r["input_ids"],'
                      'generated_ids=[],teacher_argmax=[7]),open(sys.argv[-1],"w"))\n')
    binary.chmod(0o700)
    sanitizer = tmp_path / 'sanitizer'
    sanitizer.write_text(f'#!{sys.executable}\nimport os,sys,subprocess,time\n'
                         'log=sys.argv[sys.argv.index("--log-file")+1]\n'
                         'open(log,"w").write("========= RACECHECK SUMMARY: 0 hazards displayed '
                         '(0 errors, 0 warnings)\\n")\n'
                         'mode=os.environ.get("FAKE_MODE", "valid")\n'
                         'if mode=="sleep":time.sleep(30)\n'
                         'if mode=="missing":sys.exit(0)\n'
                         'if mode=="death9":sys.exit(9)\n'
                         'if mode=="flagged":sys.exit(86)\n'
                         'subprocess.run(sys.argv[-5:],check=True)\n')
    sanitizer.chmod(0o700)
    request, expected = tmp_path / 'request.json', tmp_path / 'expected.json'
    ids = [1] * 32768
    request.write_text(json.dumps(dict(capacity=32768, input_ids=ids, max_new_tokens=0)))
    expected.write_text(json.dumps(dict(backend='native-cuda-experimental', input_ids=ids,
                                        generated_ids=[], teacher_argmax=[7])))
    monkeypatch.setattr(gate, 'cgroup_snapshot', lambda: dict(
        events=dict(oom=0, oom_kill=0), current='1', max='max', peak='1', path='fake'))
    monkeypatch.setattr(gate, 'resource_snapshot', lambda pgid: dict(
        processes=[], system_memory=dict(MemAvailable_bytes=100 * gate.GIB)))
    output = tmp_path / 'evidence'

    def invoke(extra=()):
        monkeypatch.setattr(sys, 'argv', ['guarded', '--sanitizer', str(sanitizer),
            '--binary', str(binary), '--checkpoint', str(tmp_path), '--request', str(request),
            '--expected-result', str(expected), '--output', str(output), *extra])
        code = gate.main()
        return code, json.loads((output / 'summary.json').read_text())
    invoke.output, invoke.binary = output, binary
    return invoke


@pytest.mark.parametrize('mode', ['valid', 'missing', 'death9', 'flagged'])
def test_fake_host_native_execution(run, monkeypatch, mode):
    monkeypatch.setenv('FAKE_MODE', mode)
    code, report = run()
    assert (code == 0) == (mode == 'valid')
    assert report['scoped_passed'] == (mode == 'valid')
    assert report['full_acceptance'] is False
    assert report['group_cleanup_complete'] is True
    assert report['hashes_stable'] is True
    command = report['command']
    assert command[command.index('--racecheck-num-workers') + 1] == '1'
    assert command[command.index('--print-limit') + 1] == '0'
    assert not any(x in command for x in ('--kernel-name', '--kernel-name-exclude',
        '--launch-skip', '--launch-count', '--force-blocking-launches', '--racecheck-hazard-limit',
        '--racecheck-trace-sync', '--racecheck-report'))
    assert report['cgroup_before']['events'] == report['cgroup_after']['events']


@pytest.mark.parametrize('kind', ['rss', 'available', 'oom'])
def test_resource_guards_kill_only_owned_group(run, monkeypatch, kind):
    monkeypatch.setenv('FAKE_MODE', 'sleep')
    running = [False]
    def snapshot(pgid):
        running[0] = pgid > 0
        return dict(processes=[dict(rss_bytes=33 * gate.GIB)] if running[0] and kind == 'rss' else [],
                    system_memory=dict(MemAvailable_bytes=(1 if running[0] and kind == 'available' else 100) * gate.GIB))
    monkeypatch.setattr(gate, 'resource_snapshot', snapshot)
    monkeypatch.setattr(gate, 'cgroup_snapshot', lambda: dict(
        events=dict(oom=1 if running[0] and kind == 'oom' else 0, oom_kill=0),
        current='1', max='max', peak='1', path='fake'))
    code, report = run()
    assert code == 86 and report['status'] == 'resource_stopped'
    assert not report['scoped_passed'] and not report['full_acceptance']
    assert report['group_cleanup_complete']
    assert not live_group(report['owned_process_group'])
    samples = [json.loads(line) for line in (run.output / 'resources.jsonl').read_text().splitlines()]
    assert len(samples) >= 2 and all('cgroup' in s and 'owned_process_group' in s for s in samples)


def test_timeout(run, monkeypatch):
    monkeypatch.setenv('FAKE_MODE', 'sleep')
    code, report = run(['--timeout', '0.05'])
    assert code == 86 and report['timed_out'] and report['group_cleanup_complete']
    assert not live_group(report['owned_process_group'])


@pytest.mark.parametrize('failure', ['setup', 'signal'])
def test_setup_and_deferred_signal_unwind_owner(run, monkeypatch, failure):
    groups = []
    original = gate.OwnedJob.start
    if failure == 'setup':
        monkeypatch.setattr(gate, 'cgroup_snapshot', lambda: (_ for _ in ()).throw(OSError('denied')))
    else:
        monkeypatch.setenv('FAKE_MODE', 'sleep')
        def interrupted(owner, *args, **kwargs):
            process = original(owner, *args, **kwargs)
            groups.append(process.pid)
            os.kill(os.getpid(), signal.SIGTERM)
        monkeypatch.setattr(gate.OwnedJob, 'start', interrupted)
    code, report = run()
    assert code == 86 and report['group_cleanup_complete'] and not report['scoped_passed']
    assert all(not live_group(group) for group in groups)


def test_hash_change_is_failure(run, monkeypatch):
    original = gate.native_result
    def changed(path, request, expected=None):
        result = original(path, request, expected)
        if expected is not None:
            run.binary.write_text('changed')
        return result
    monkeypatch.setattr(gate, 'native_result', changed)
    code, report = run()
    assert code == 86 and not report['hashes_stable'] and not report['scoped_passed']


@pytest.mark.parametrize('damage', ['wrong-input', 'wrong-argmax', 'bool-argmax', 'wrong-backend'])
def test_native_must_match_exactly(run, monkeypatch, damage):
    original = gate.native_result
    def damaged(path, request, expected=None):
        if expected is not None:
            result = json.loads(path.read_text())
            if damage == 'wrong-input':
                result['input_ids'][0] = 2
            elif damage == 'wrong-argmax':
                result['teacher_argmax'] = [8]
            elif damage == 'bool-argmax':
                result['teacher_argmax'] = [True]
            else:
                result['backend'] = 'fake'
            path.write_text(json.dumps(result))
        return original(path, request, expected)
    monkeypatch.setattr(gate, 'native_result', damaged)
    code, report = run()
    assert code == 86 and not report['application_ok'] and not report['scoped_passed']


@pytest.mark.parametrize('text,ok', [
    ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n', True),
    ('0 hazards displayed', False), ('0 errors, 0 warnings', False),
    ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 1 warnings)', False)])
def test_canonical_summary(text, ok):
    assert gate.clean_log(text) is ok
