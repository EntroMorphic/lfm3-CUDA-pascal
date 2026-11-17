"""Workstation-side Thor operations that fail loudly instead of silently.

Guards added after silent failures on 2026-11-05/06:
- a copy that hung left an old runner in place; delivery was "checked" with a grep that the old
  file also satisfied  -> `sync` verifies SHA-256 on both ends, every file, every time.
- `docker exec -d` returned success for a runner that died at argument parsing  -> `launch`
  waits for the runner itself to report the expected runner hash and a fresh heartbeat.
- a watcher would have looped for hours on a status file that never appeared  -> `watch`
  fails on unreachable host, missing status, wrong runner identity or stale heartbeat.
Every remote call has a timeout and a checked exit status. Exit codes: 0 success,
2 verification/identity failure, 3 unreachable/timeout, 4 runner died or stalled, 5 gates failed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HOST = os.environ.get('THOR_HOST', '')  # user@host of the target machine (ssh, key-based); required for remote operations
REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = os.environ.get('THOR_REPO', str(REPO))  # same absolute path on Thor
CONTAINER = os.environ.get('THOR_CONTAINER', 'lfm3-profile-v1')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']


class Unreachable(Exception):
    pass


class Verification(Exception):
    pass


def host():
    """THOR_HOST, required: this tool never guesses which machine to run on."""
    if not HOST:
        raise Unreachable('set THOR_HOST to the target machine (user@host) for remote operations')
    return HOST


def remote(command, timeout=60):
    """Run a shell command on Thor; raise instead of returning partial or empty output."""
    try:
        r = subprocess.run(SSH + [host(), command], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise Unreachable(f'remote command timed out after {timeout}s: {command[:120]}') from e
    if r.returncode == 255:
        raise Unreachable(f'ssh failed: {r.stderr.strip()[:300]}')
    if r.returncode != 0:
        raise Verification(f'remote command exited {r.returncode}: {command[:120]}\n{r.stderr.strip()[:600]}')
    return r.stdout


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_sha256(paths):
    out = remote('sha256sum ' + ' '.join(shlex.quote(p) for p in paths))
    found = {}
    for line in out.splitlines():
        digest, _, name = line.partition('  ')
        found[name] = digest
    missing = [p for p in paths if p not in found]
    if missing:
        raise Verification(f'remote hash missing for {missing}')
    return found


def sync(paths, timeout=120):
    """Copy repo files to the same paths on Thor and prove byte identity."""
    local = [Path(p).resolve() for p in paths]
    for p in local:
        if REPO not in p.parents:
            raise Verification(f'refusing to sync outside the repo: {p}')
        if not p.is_file():
            raise Verification(f'not a file: {p}')
    dests = [str(Path(REMOTE_REPO) / p.relative_to(REPO)) for p in local]
    # Creating parents over ssh also separates "Thor unreachable" (ssh 255 -> Unreachable)
    # from a copy that fails on a reachable host (-> Verification below).
    remote('mkdir -p ' + ' '.join(sorted({shlex.quote(str(Path(d).parent)) for d in dests})))
    for src, dst in zip(local, dests):
        try:
            r = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', str(src), f'{host()}:{dst}'],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise Unreachable(f'scp timed out after {timeout}s: {src}') from e
        if r.returncode:
            raise Verification(f'scp exited {r.returncode} for {src}: {r.stderr.strip()[:300]}')
    theirs = remote_sha256(dests)
    report = []
    for src, dst in zip(local, dests):
        mine = sha256(src)
        if theirs[dst] != mine:
            raise Verification(f'hash mismatch after copy: {src.relative_to(REPO)} local {mine[:16]} thor {theirs[dst][:16]}')
        report.append((str(src.relative_to(REPO)), mine))
    return report


def read_status(remote_status):
    out = remote(f'cat {shlex.quote(remote_status)}', timeout=30)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise Verification(f'status file is not valid JSON ({len(out)} bytes)') from e


def check_live(status, expected_runner, now, stale_after):
    """Return a problem description, or None when the status shows a live, expected runner."""
    if status.get('runner_sha256') != expected_runner:
        return f"runner identity {str(status.get('runner_sha256'))[:16]} != expected {expected_runner[:16]}"
    if status.get('status') == 'running':
        age = now - float(status.get('heartbeat_unix', 0))
        if age > stale_after:
            return f'heartbeat stale for {age:.0f}s (limit {stale_after}s): runner died or hung'
    return None


def launch(runner, args, output, start_timeout=90, stale_after=120):
    """Start the runner detached in the container, then prove it is the intended version and alive."""
    runner = Path(runner).resolve()
    expected = sha256(runner)
    remote_runner = str(Path(REMOTE_REPO) / runner.relative_to(REPO))
    if remote_sha256([remote_runner])[remote_runner] != expected:
        raise Verification('runner on Thor differs from local; sync it first')
    status = f'{output}/gates.json'
    log = f'{output}.runner.log'
    container_runner = '/workspace/liquid-CUDA/' + str(runner.relative_to(REPO))
    host_output = output.replace('/workspace/', str(Path(REMOTE_REPO).parent) + '/', 1)
    host_log = host_output + '.runner.log'
    for path in (host_output, host_log):  # a stale log would be mistaken for this run's output
        remote(f'test ! -e {shlex.quote(path)} || {{ echo "already exists: {path}" >&2; exit 1; }}')
    inner = f'/opt/venv/bin/python3 {shlex.quote(container_runner)} {" ".join(shlex.quote(x) for x in args)} --output {shlex.quote(output)} > {shlex.quote(log)} 2>&1'
    remote(f'docker exec -d {CONTAINER} bash -c {shlex.quote(inner)}')
    deadline = time.time() + start_timeout
    while time.time() < deadline:
        time.sleep(5)
        try:
            st = read_status(host_output + '/gates.json')
        except Verification:
            # Not written yet, or the runner exited before writing: look at its log.
            tail = remote(f'tail -c 2000 {shlex.quote(host_log)} 2>/dev/null || true', timeout=30)
            if tail.strip():
                raise Verification(f'runner exited before reporting status:\n{tail}')
            continue
        problem = check_live(st, expected, time.time(), stale_after)
        if problem:
            raise Verification(problem)
        return st
    raise Verification(f'runner wrote no status within {start_timeout}s')


def watch(host_status, runner, poll=30, stale_after=120, max_unreachable=10):
    expected = sha256(Path(runner).resolve())
    misses = 0
    while True:
        try:
            st = read_status(host_status)
            misses = 0
        except Unreachable as e:
            misses += 1
            print(f'warning: Thor unreachable ({misses}/{max_unreachable}): {e}', file=sys.stderr, flush=True)
            if misses >= max_unreachable:
                raise
            time.sleep(poll)
            continue
        problem = check_live(st, expected, time.time(), stale_after)
        if problem:
            raise RuntimeError(problem)
        if st.get('status') != 'running':
            return st
        time.sleep(poll)


def summarize(st):
    lines = [f"status={st.get('status')} hashes_stable={st.get('hashes_stable')}"]
    for r in st.get('records', []):
        lines.append(f"  {r['name']:36s} {r['status']:8s} {r.get('seconds', 0):8.1f}s rc={r.get('returncode')} timeout={r.get('timed_out')}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sync'); s.add_argument('paths', nargs='+')
    l = sub.add_parser('launch'); l.add_argument('--runner', required=True); l.add_argument('--output', required=True,
                        help='container path, e.g. /workspace/liquid-CUDA/build/NAME'); l.add_argument('args', nargs=argparse.REMAINDER)
    w = sub.add_parser('watch'); w.add_argument('--runner', required=True); w.add_argument('--status', required=True,
                        help='host path of gates.json on Thor')
    a = p.parse_args()
    try:
        if a.cmd == 'sync':
            for name, digest in sync(a.paths):
                print(f'verified {digest[:16]}  {name}')
        elif a.cmd == 'launch':
            args = a.args[1:] if a.args[:1] == ['--'] else a.args
            st = launch(a.runner, args, a.output)
            print(f"live: runner {st['runner_sha256'][:16]} pid {st['pid']} status {st['status']} "
                  f"current step {st['records'][-1]['name'] if st['records'] else '(starting)'}")
        else:
            st = watch(a.status, a.runner)
            print(summarize(st))
            return 0 if st.get('status') == 'all-steps-passed' else 5
    except Unreachable as e:
        print(f'UNREACHABLE: {e}', file=sys.stderr); return 3
    except Verification as e:
        print(f'VERIFICATION FAILED: {e}', file=sys.stderr); return 2
    except RuntimeError as e:
        print(f'RUNNER FAILURE: {e}', file=sys.stderr); return 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
