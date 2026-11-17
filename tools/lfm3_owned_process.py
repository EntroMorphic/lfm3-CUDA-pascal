"""Bounded Linux process-group cleanup for GPU acceptance jobs.

Callers must spawn with start_new_session=True. Waiting for only the sanitizer
leader is insufficient: TreeLauncherSubreaper and its CUDA child can outlive it.
"""
import os
from functools import wraps
from pathlib import Path
import signal
import subprocess
import threading
import time


def live_group(pgid):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            # Zombie processes cannot execute CUDA work. Their reaping parent
            # may be container init rather than this direct-child owner.
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def resource_snapshot(pgid):
    processes = []
    for pid in live_group(pgid):
        try:
            root = Path(f"/proc/{pid}")
            fields = (root/"stat").read_text().rsplit(") ",1)[1].split()
            status = (root/"status").read_text().splitlines()
            rss = next(int(line.split()[1])*1024 for line in status if line.startswith("VmRSS:"))
            processes.append(dict(pid=pid,state=fields[0],user_ticks=int(fields[11]),system_ticks=int(fields[12]),rss_bytes=rss))
        except (OSError,IndexError,ValueError,StopIteration):
            continue
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:","SwapFree:")):
            memory[line.split(":",1)[0]+"_bytes"] = int(line.split()[1])*1024
    return dict(processes=processes,system_memory=memory)


def _quiesce(process, cleanup_timeout):
    group = process.pid
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=cleanup_timeout)
    deadline = time.monotonic() + cleanup_timeout
    while True:
        remaining = live_group(group)
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"owned GPU job cleanup incomplete; live PIDs {remaining}")
        time.sleep(0.01)


def owned_main(function):
    """CLI main-thread guard: one termination unwinds; repeats cannot abort cleanup."""
    @wraps(function)
    def guarded(*args, **kwargs):
        interrupted = False
        def terminate(signum, frame):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for sig in previous:
                signal.signal(sig, terminate)
            return function(*args, **kwargs)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return guarded


class OwnedJob:
    """Create before try/finally; call close unconditionally, even if start fails."""
    def __init__(self, cleanup_timeout=10):
        if not 0 < cleanup_timeout <= 60:
            raise ValueError("invalid cleanup deadline")
        self.process = None
        self.closed = False
        self.cleanup_timeout = cleanup_timeout
        self.lock = threading.Lock()

    def start(self, command, **kwargs):
        if self.process is not None or self.closed or "start_new_session" in kwargs:
            raise ValueError("owned job is single-use and always starts a new session")
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            # Record ownership before deferred Python termination is delivered.
            # exec preserves masks, so restore the original mask in the child.
            import sys
            launcher = [sys.executable, "-c", "import os,signal,sys;signal.pthread_sigmask(signal.SIG_SETMASK,"
                        + repr(sorted(int(sig) for sig in blocked)) + ");os.execv(sys.argv[1],sys.argv[1:])"]
            self.process = subprocess.Popen(launcher + list(command), start_new_session=True, **kwargs)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        return self.process

    def wait(self, timeout):
        if self.process is None or not 0 < timeout <= 86400:
            raise ValueError("owned process/deadline required")
        try:
            _check_group(self.process)
            try:
                return self.process.wait(timeout=timeout), False
            except subprocess.TimeoutExpired:
                return None, True
        finally:
            self.close()

    def close(self):
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            with self.lock:
                if self.closed:
                    return
                if self.process is not None:
                    _check_group(self.process)
                    _quiesce(self.process, self.cleanup_timeout)
                self.closed = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def _check_group(process):
    group = process.pid
    try:
        if os.getpgid(process.pid) != group or group == os.getpgrp():
            raise ValueError("GPU job must own a new process session")
    except ProcessLookupError:
        # A caller may have already observed/reaped a very short-lived leader;
        # the explicit new-session precondition still applies to its children.
        pass
def wait_owned(process, timeout, cleanup_timeout=10):
    """Compatibility for already-owned new-session children; prefer OwnedJob."""
    if not 0 < timeout <= 86400 or not 0 < cleanup_timeout <= 60:
        raise ValueError("invalid owned-process deadline")
    _check_group(process) # Never kill a caller's shared group after rejection.
    owner = OwnedJob(cleanup_timeout)
    owner.process = process
    try:
        return owner.wait(timeout)
    finally:
        owner.close()
