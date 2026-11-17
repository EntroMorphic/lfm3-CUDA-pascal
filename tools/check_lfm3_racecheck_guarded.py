"""Resource-bounded, all-launch 32768-token racecheck diagnostic (not acceptance)."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

try:
    from lfm3_owned_process import OwnedJob, owned_main, resource_snapshot
    from check_lfm3_sanitizer import artifact_hashes, native_result
except ModuleNotFoundError:
    from tools.lfm3_owned_process import OwnedJob, owned_main, resource_snapshot
    from tools.check_lfm3_sanitizer import artifact_hashes, native_result

GIB = 1024 ** 3


def cgroup_snapshot():
    """Resolve this process's v2 cgroup, including a namespaced cgroup root."""
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    root = Path('/sys/fs/cgroup')
    candidate = root / relative.lstrip('/')
    directory = candidate if (candidate / 'memory.events').is_file() else root
    values = {name: (directory / ('memory.' + name)).read_text().strip()
              for name in ('events', 'current', 'max', 'peak')}
    values['events'] = {key: int(value) for key, value in
                        (line.split() for line in values['events'].splitlines())}
    return dict(path=str(directory), **values)


def guard_reason(snapshot, before, rss_budget, min_available):
    if sum(p['rss_bytes'] for p in snapshot['processes']) > rss_budget:
        return 'owned-group-rss-budget'
    if snapshot['system_memory']['MemAvailable_bytes'] < min_available:
        return 'host-available-memory'
    events = snapshot['cgroup']['events']
    if any(events.get(key, 0) > before['events'].get(key, 0)
           for key in ('oom', 'oom_kill', 'oom_group_kill')):
        return 'new-cgroup-oom-event'
    return None


def clean_log(text):
    summaries = re.findall(r'^========= RACECHECK SUMMARY: (.+)$', text, re.M)
    return (bool(summaries) and all(s == '0 hazards displayed (0 errors, 0 warnings)'
                                 for s in summaries)
            and 'Internal Sanitizer Error' not in text)


@owned_main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sanitizer', 'binary', 'checkpoint', 'request', 'expected-result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--rss-budget-gib', type=float, default=32)
    parser.add_argument('--min-host-available-gib', type=float, default=16)
    args = parser.parse_args()
    if (not 0 < args.timeout <= 1800 or not 1 <= args.num_workers <= 64 or
            not 0 < args.rss_budget_gib <= 64 or
            not math.isfinite(args.min_host_available_gib) or args.min_host_available_gib <= 0):
        parser.error('invalid timeout, worker count or memory guard')
    for name in ('sanitizer', 'binary', 'checkpoint', 'request', 'expected_result', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    owner = OwnedJob()
    report = dict(status='failed', full_acceptance=False, scoped_passed=False,
                  resource_stopped=False, timed_out=False, returncode=None,
                  owned_process_group=None, no_launch_filters=True,
                  rss_budget_gib=args.rss_budget_gib,
                  min_host_available_gib=args.min_host_available_gib,
                  scope='all launches; full32768 prefill only; no broader acceptance')
    created = False
    paths = {name: getattr(args, name) for name in
             ('sanitizer', 'binary', 'request', 'expected_result')}
    start = time.monotonic()
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        created = True
        report['start_hashes'], errors = artifact_hashes(paths)
        if errors:
            raise ValueError(str(errors))
        request = json.loads(args.request.read_text())
        expected = json.loads(args.expected_result.read_text())
        if (request.get('capacity') != 32768 or len(request.get('input_ids', [])) != 32768 or
                request.get('max_new_tokens', 0) != 0 or request.get('teacher_ids', []) or
                request.get('trace_directory')):
            raise ValueError('requires identical full32768 prefill fixture without traces')
        native_result(args.expected_result, request)
        fixture = args.output / 'request.json'
        fixture.write_bytes(args.request.read_bytes())
        if artifact_hashes({'request': fixture})[0]['request'] != report['start_hashes']['request']:
            raise ValueError('request changed during setup')
        destination, log = args.output / 'native.json', args.output / 'racecheck.log'
        command = [str(args.sanitizer), '--tool', 'racecheck', '--error-exitcode', '86',
                   '--print-limit', '0', '--racecheck-num-workers', str(args.num_workers),
                   '--log-file', str(log), str(args.binary), '--experimental',
                   str(args.checkpoint), str(fixture), str(destination)]
        report['command'] = command
        report['cgroup_before'] = before = cgroup_snapshot()
        with (args.output / 'resources.jsonl').open('w') as resources:
            def sample(pgid):
                snapshot = resource_snapshot(pgid)
                snapshot.update(cgroup=cgroup_snapshot(), owned_process_group=pgid,
                                seconds=time.monotonic() - start)
                resources.write(json.dumps(snapshot) + '\n')
                resources.flush()
                return guard_reason(snapshot, before, args.rss_budget_gib * GIB,
                                    args.min_host_available_gib * GIB)
            # Preflight uses no existing process group and cannot touch unrelated jobs.
            reason = sample(-1)
            with (args.output / 'application.log').open('w') as stream:
                if not reason:
                    process = owner.start(command, stdout=stream, stderr=subprocess.STDOUT)
                    report['owned_process_group'] = process.pid
                    deadline = time.monotonic() + args.timeout
                    while True:
                        reason = sample(process.pid)
                        if reason:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            report['timed_out'] = True
                            break
                        try:
                            report['returncode'] = process.wait(timeout=min(1, remaining))
                            reason = sample(process.pid)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                if reason:
                    report.update(resource_stopped=True, guard_reason=reason)
        owner.close()
        report['application_ok'] = False
        try:
            native_result(destination, request, expected)
            report['application_ok'] = True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report['application_error'] = str(exc)
        report['canonical_zero_summary'] = clean_log(log.read_text(errors='replace')) if log.exists() else False
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            report['interrupted'] = True
    finally:
        try:
            owner.close()
            report['group_cleanup_complete'] = owner.closed
        except BaseException as exc:
            report['cleanup_error'] = str(exc)
        if created:
            try:
                report['cgroup_after'] = cgroup_snapshot()
                before = report.get('cgroup_before')
                if before and any(report['cgroup_after']['events'].get(k, 0) > before['events'].get(k, 0)
                                  for k in ('oom', 'oom_kill', 'oom_group_kill')):
                    report.update(resource_stopped=True, guard_reason='new-cgroup-oom-event')
            except Exception as exc:
                report['error'] = f'final cgroup observation failed: {exc}'
            report['end_hashes'], errors = artifact_hashes(paths)
            report['hashes_stable'] = not errors and report.get('start_hashes') == report['end_hashes']
            report['scoped_passed'] = (report['returncode'] == 0 and not report['timed_out'] and
                not report['resource_stopped'] and not report.get('error') and
                not report.get('cleanup_error') and report.get('application_ok', False) and
                report.get('canonical_zero_summary', False) and report['hashes_stable'])
            report['status'] = ('scoped-passed' if report['scoped_passed'] else 'resource_stopped'
                                if report['resource_stopped'] else 'timeout' if report['timed_out'] else 'failed')
            report['seconds'] = time.monotonic() - start
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['scoped_passed'] else 86


if __name__ == '__main__':
    raise SystemExit(main())
