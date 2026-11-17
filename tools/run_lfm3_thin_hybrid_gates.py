"""Run the thin-hybrid candidate gate matrix sequentially; stop at the first failure.

Mirrors the v4 gate list (validation/wmma-v4-gates.json, wmma-v4-upper-gates.json) with the
same checker tools and arguments, pointed at the hybrid candidate, plus tail-geometry runs of
the v5/v6 probe under all four compute-sanitizer tools. Records every step; never waives one.
Not release acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

W = '/workspace/liquid-CUDA'
HEARTBEAT = 15  # seconds between liveness updates in gates.json
B = W + '/build'
SAN = B + '/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer'
CK = '/workspace/models/LFM3-Omni-2B'
PY = '/opt/venv/bin/python3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DEFAULT_TAILS = ('thin-v6-s8', 'thin-v5-u4', 'thin-v5-u16')


def steps(candidate, tails, tag, tail_binaries=DEFAULT_TAILS):
    c = f'{B}/{candidate}/build'
    out = lambda name: f'{B}/thin-hybrid-{name}-{tag}'
    s = []
    for binary in tail_binaries:
        for tool in ('initcheck', 'synccheck', 'memcheck', 'racecheck'):
            s.append(dict(name=f'tails-{binary}-{tool}', kind='sanitizer', tool=tool, timeout=900,
                          command=[SAN, '--tool', tool, '--error-exitcode', '99', f'{B}/{tails}/{binary}', '--tails']))
    forward = out('forward')
    s += [
        dict(name='forward', timeout=900, command=[PY, f'{W}/tools/check_lfm3_forward.py', '--checkpoint', CK,
                                                   '--binary', f'{c}/liquid-lfm3', '--output', forward]),
        dict(name='sanitizers', timeout=1800, command=[PY, f'{W}/tools/check_lfm3_candidate_sanitizers.py', '--checkpoint', CK,
             '--binary', f'{c}/liquid-lfm3', '--sanitizer', SAN, '--library-dir', '/pinned-cuda',
             '--request', f'{B}/forward-bounded-prefill-v1/single/generation-request.json',
             '--expected-result', f'{B}/forward-bounded-prefill-v1/single/generation.json',
             '--tools', 'initcheck', 'synccheck', 'memcheck', 'racecheck', '--timeout', '300', '--output', out('sanitizers')]),
        dict(name='benchmark', timeout=1800, command=[PY, f'{W}/tools/check_lfm3_native_policy_benchmark.py',
             '--baseline', f'{B}/cmake-bounded-prefill-v1/liquid-lfm3', '--candidate', f'{c}/liquid-lfm3',
             '--baseline-evidence', f'{B}/forward-bounded-prefill-v1/summary.json', '--candidate-evidence', f'{forward}/summary.json',
             '--checkpoint', CK, '--request', f'{B}/forward-bounded-prefill-v1/baseline/generation-request.json',
             '--expected-result', f'{B}/forward-bounded-prefill-v1/baseline/generation.json', '--library-dir', '/pinned-cuda',
             '--samples', '20', '--output', out('benchmark')]),
        dict(name='context', timeout=1200, command=[PY, f'{W}/tools/check_lfm3_context.py', '--checkpoint', CK,
             '--binary', f'{c}/liquid-lfm3', '--suite', 'full', '--reuse-reference', f'{B}/context-full-bounded-release-v2',
             '--timeout', '180', '--output', out('context')]),
        dict(name='paired-normal', timeout=1200, command=[PY, f'{W}/tools/check_lfm3_decode_batch.py', '--checkpoint', CK,
             '--binary', f'{c}/lfm3-decode-batch-probe', '--reference-mode', 'normal', '--timeout', '300', '--output', out('paired-normal')]),
        dict(name='paired-compact', timeout=1200, command=[PY, f'{W}/tools/check_lfm3_decode_batch.py', '--checkpoint', CK,
             '--binary', f'{c}/lfm3-decode-batch-probe', '--reference-mode', 'compact', '--timeout', '300', '--output', out('paired-compact')]),
        dict(name='grouped-worker', timeout=1200, command=[PY, f'{W}/tools/check_lfm3_batch_worker.py', '--checkpoint', CK,
             '--worker', f'{c}/liquid-lfm3-worker', '--request', f'{B}/forward-bounded-prefill-v1/baseline/generation-request.json',
             '--oracle', f'{B}/cmake/text-sampling-oracle', '--long-request', f'{B}/context32k-bounded-prefill-v1/decode_into32768/request.json',
             '--output', out('grouped-worker')]),
        dict(name='32k', timeout=1800, command=[PY, f'{W}/tools/check_lfm3_32k.py', '--checkpoint', CK,
             '--binary', f'{c}/liquid-lfm3', '--output', out('32k')]),
        dict(name='upper-racecheck', timeout=1200, command=[PY, f'{W}/tools/check_lfm3_racecheck_guarded.py', '--checkpoint', CK,
             '--binary', f'{c}/liquid-lfm3', '--sanitizer', SAN, '--request', f'{B}/wmma-upper-fixture-v4/request.json',
             '--expected-result', f'{B}/wmma-upper-fixture-v4/expected.json', '--timeout', '900', '--num-workers', '1',
             '--rss-budget-gib', '32', '--min-host-available-gib', '16', '--output', out('upper-racecheck')]),
    ]
    return s


def sanitizer_clean(tool, log):
    # Canonical footers only; a missing footer is never treated as clean.
    if tool == 'racecheck':
        m = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors?, (\d+) warnings?\)', log)
        return bool(m) and all(x == ('0', '0', '0') for x in m)
    m = re.findall(r'ERROR SUMMARY: (\d+) errors?', log)
    return bool(m) and all(x == '0' for x in m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--tails', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--tail-binaries', nargs='+', default=list(DEFAULT_TAILS), help='probe binaries in --tails to run under the four sanitizers')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', nargs='+', help='run only these named steps, in plan order (earlier records stand as they are)')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, LD_LIBRARY_PATH='/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia',
               TRITON_PTXAS_PATH='/workspace/tools/ptxas', TRITON_PTXAS_BLACKWELL_PATH='/workspace/tools/ptxas',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    plan = steps(a.candidate, a.tails, a.tag, a.tail_binaries)
    if a.steps:
        unknown = set(a.steps) - {x['name'] for x in plan}
        if unknown: raise SystemExit(f'unknown steps: {sorted(unknown)}')
        plan = [x for x in plan if x['name'] in a.steps]
    pinned = [f'{B}/{a.candidate}/build/{n}' for n in ('liquid-lfm3', 'liquid-lfm3-worker', 'lfm3-decode-batch-probe')]
    pinned += [f'{B}/{a.tails}/{n}' for n in a.tail_binaries]
    pinned += sorted(str(x) for x in Path(f'{W}/tools').glob('*.py')) + ['/workspace/tools/ptxas']
    pinned += sorted(str(x) for x in Path('/pinned-cuda').glob('*.so*'))
    # Identity and liveness, so a launcher/watcher can tell a live run of the intended runner
    # from a stale file, a different runner version, or a process that died silently.
    report = dict(full_acceptance=False, status='running', selected_steps=a.steps,
                  runner_sha256=digest(__file__), argv=sys.argv, pid=os.getpid(),
                  started_unix=time.time(), heartbeat_unix=time.time(), heartbeat_interval_seconds=HEARTBEAT,
                  start_sha256={x: digest(x) for x in pinned}, records=[])
    summary = a.output / 'gates.json'
    lock = threading.Lock()

    def save():
        # Atomic replace: readers never see a partially written file.
        with lock:
            tmp = summary.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            os.replace(tmp, summary)

    save()
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT):
            report['heartbeat_unix'] = time.time(); save()

    threading.Thread(target=beat, daemon=True).start()
    try:
        return run_plan(plan, a, env, report, save, pinned)
    except BaseException:
        report['status'] = 'runner-error'; report['error'] = traceback.format_exc(); raise
    finally:
        stop.set(); report['heartbeat_unix'] = time.time(); report['finished_unix'] = time.time(); save()


def run_plan(plan, a, env, report, save, pinned):
    for step in plan:
        r = dict(name=step['name'], command=step['command'], status='running')
        report['records'].append(r); save()
        log = a.output / f"{step['name']}.log"
        start = time.monotonic()
        with open(log, 'w') as f:
            proc = subprocess.Popen(step['command'], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=step['timeout']); timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); code = proc.wait(); timed_out = True
        r.update(returncode=code, timed_out=timed_out, seconds=time.monotonic() - start)
        ok = code == 0 and not timed_out
        if ok and step.get('kind') == 'sanitizer':
            r['canonical_clean'] = sanitizer_clean(step['tool'], log.read_text(errors='replace'))
            ok = r['canonical_clean']
        r['status'] = 'passed' if ok else 'failed'
        save()
        if not ok:
            report['status'] = 'stopped-at-failure'
            break
    else:
        report['status'] = 'all-steps-passed'
    report['end_sha256'] = {x: digest(x) for x in pinned}
    report['hashes_stable'] = report['end_sha256'] == report['start_sha256']
    if not report['hashes_stable']:
        report['status'] = 'failed-identity-change'
    save()
    return 0 if report['status'] == 'all-steps-passed' else 86


if __name__ == '__main__':
    raise SystemExit(main())
