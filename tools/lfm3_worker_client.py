"""Bounded local resident-worker transport, independent of the GPU gate tool."""
import json
from contextlib import nullcontext
import math
import os
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time

try:
    from lfm3_owned_process import _check_group, live_group
except ModuleNotFoundError:
    from tools.lfm3_owned_process import _check_group, live_group


MAX_FRAME = 4 * 1024 * 1024


def strict_json(raw):
    text = raw.decode("utf-8", errors="strict")
    quoted = escaped = False
    depth = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds limit")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        raise ValueError("nonfinite JSON number")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=finite, parse_float=number)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds limit") from exc
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


class WorkerFailure(RuntimeError):
    """Pipe/process/protocol failure: no cache state remains trustworthy."""


class WorkerClient:
    def __init__(self, binary, checkpoint, *, timeout=300, env=None, cancellation=False, sampling_trace_root=None, batching=False):
        if not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError("invalid worker timeout")
        if type(cancellation) is not bool:
            raise ValueError("cancellation must be boolean")
        if type(batching) is not bool:
            raise ValueError("batching must be boolean")
        self.supports_batching = False
        self.timeout = timeout
        self.closed = False
        self.cleanup_failed = False
        self.cleanup_error = None
        self.lock = threading.Lock()
        self.buffer = bytearray()
        self.sequence = 0
        self.supports_cancellation = cancellation
        self.active_lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.active_generation = None
        self.active_generations = {}
        self.cancel_read_fd = self.cancel_write_fd = None
        # No checkpoint writes, shell, inherited stdout or unbounded stderr pipe.
        self.process = None
        try:
            worker_command = [str(binary), "--experimental", str(checkpoint)]
            if sampling_trace_root is not None:
                worker_command += ["--sampling-trace-root", str(sampling_trace_root)]
            if cancellation:
                self.cancel_read_fd, self.cancel_write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
                worker_command += ["--cancel-fd", str(self.cancel_read_fd)]
            # Defer interruption until Popen's child is recorded; otherwise a
            # signal between spawn and assignment can orphan its separate group.
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                self.process = subprocess.Popen(
                    # exec preserves blocked signals. A tiny exec-only launcher
                    # restores the original mask in the child without unsafe
                    # preexec_fn use in this multithreaded serving process.
                    [sys.executable, "-c",
                     "import os,signal,sys;signal.pthread_sigmask(signal.SIG_SETMASK,"
                     + repr(sorted(int(sig) for sig in blocked))
                     + ");os.execv(sys.argv[1],sys.argv[1:])",
                     *worker_command], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    restore_signals=True,
                    pass_fds=() if self.cancel_read_fd is None else (self.cancel_read_fd,),
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            if self.cancel_read_fd is not None:
                self._retire_cancel_fd("cancel_read_fd")
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            ready = self._read(time.monotonic() + timeout)
            if (ready.get("ready") is not True or type(ready.get("protocol")) is not int
                    or ready["protocol"] != 2 or ready.get("gpu_batching") is not False
                    or type(ready.get("max_sessions")) is not int or ready["max_sessions"] != 2
                    or type(ready.get("aggregate_capacity")) is not int
                    or ready["aggregate_capacity"] != 32768
                    or (cancellation and ready.get("cancellation") is not True)
                    or ("decode_batching" in ready and type(ready["decode_batching"]) is not bool)
                    or (batching and ready.get("decode_batching") is not True)):
                raise WorkerFailure("incompatible worker handshake")
            self.supports_batching = ready.get("decode_batching") is True
        except BaseException:
            self.close()
            raise

    def _read(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                end = self.buffer.find(b"\n")
                if end >= 0:
                    if end > MAX_FRAME:
                        raise WorkerFailure("worker frame exceeds limit")
                    raw = bytes(self.buffer[:end])
                    del self.buffer[:end + 1]
                    return strict_json(raw)
                if len(self.buffer) > MAX_FRAME:
                    raise WorkerFailure("worker frame exceeds limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("worker response deadline")
                try:
                    block = os.read(self.process.stdout.fileno(), min(65536, MAX_FRAME + 1 - len(self.buffer)))
                except BlockingIOError:
                    continue
                if not block:
                    raise WorkerFailure("worker exited before reply")
                self.buffer.extend(block)

    def call_cancellable(self, request, cancel_event):
        return self.call(request, cancel_event=cancel_event)

    def call_batch_cancellable(self, lanes, events):
        if not self.supports_batching:
            raise WorkerFailure("worker decode batching unavailable")
        if (not isinstance(lanes, list) or len(lanes) != 2 or len(events) != 2
                or any(not isinstance(lane, dict) or "op" in lane or "rpc_id" in lane for lane in lanes)
                or any(not isinstance(lane.get("session_id"), str) for lane in lanes)
                or lanes[0]["session_id"] == lanes[1]["session_id"]):
            raise ValueError("batch requires two distinct generation lanes")
        return self.call(dict(op="generate_batch", lanes=lanes), cancel_event=events)

    def cancel(self, session_id):
        """Request only: terminal cancellation is acknowledged by the RPC reply."""
        with self.active_lock:
            active = getattr(self, "active_generations", {}).get(session_id)
            if active is None:
                single = getattr(self, "active_generation", None)
                if single is not None and single["session_id"] == session_id:
                    active = single
            if self.closed or self.cancel_write_fd is None or active is None:
                return False
            if active["sent"]:
                return True
            try:
                written = os.write(self.cancel_write_fd, struct.pack("<Q", active["rpc_id"]))
            except BlockingIOError:
                raise WorkerFailure("cancellation control queue full") from None
            except OSError as exc:
                raise WorkerFailure("cancellation control pipe failed") from exc
            if written != 8:
                raise WorkerFailure("partial cancellation control write")
            active["sent"] = True
            return True

    def call(self, request, *, cancel_event=None):
        if "rpc_id" in request:
            raise ValueError("rpc_id is owned by the transport")
        # Validate size before admission without consuming an RPC sequence.
        raw = json.dumps(request, allow_nan=False, separators=(",", ":"))
        if len(raw.encode("utf-8")) + 32 > MAX_FRAME:
            raise ValueError("request frame exceeds limit")
        if not self.lock.acquire(blocking=False):
            raise WorkerFailure("concurrent worker pipe access is forbidden")
        try:
            if self.closed:
                raise WorkerFailure("worker closed")
            if self.sequence >= 2147483647:
                raise WorkerFailure("RPC sequence exhausted; restart required")
            self.sequence += 1
            sequence = self.sequence
            payload = (json.dumps(dict(request, rpc_id=sequence), allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if self.buffer:
                raise WorkerFailure("unsolicited trailing worker data")
            if request.get("op") == "generate":
                with self.active_lock:
                    self.active_generation = dict(session_id=request.get("session_id"), rpc_id=sequence, sent=False)
                    self.active_generations = {request.get("session_id"): self.active_generation}
                if self.supports_cancellation and cancel_event is not None and cancel_event.is_set():
                    if not self.cancel(request.get("session_id")):
                        raise WorkerFailure("pre-dispatch cancellation was not admitted")
            elif request.get("op") == "generate_batch":
                lanes = request["lanes"]
                with self.active_lock:
                    self.active_generations = {
                        lane["session_id"]: dict(session_id=lane["session_id"],
                                                 rpc_id=((index + 1) << 32) | sequence, sent=False)
                        for index, lane in enumerate(lanes)}
                if self.supports_cancellation and cancel_event is not None:
                    for lane, event in zip(lanes, cancel_event):
                        if event is not None and event.is_set() and not self.cancel(lane["session_id"]):
                            raise WorkerFailure("pre-dispatch cancellation was not admitted")
            deadline = time.monotonic() + self.timeout
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                offset = 0
                while offset < len(payload):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("worker write deadline")
                    try:
                        written = os.write(self.process.stdin.fileno(), payload[offset:])
                    except BlockingIOError:
                        continue
                    if written <= 0:
                        raise WorkerFailure("worker pipe closed")
                    offset += written
            result = self._read(deadline)
            if (type(result.get("ok")) is not bool or type(result.get("rpc_id")) is not int
                    or result["rpc_id"] != sequence or self.buffer):
                raise WorkerFailure("invalid worker reply")
            del result["rpc_id"]
            return result
        except (OSError, ValueError, TimeoutError, WorkerFailure) as exc:
            self.close()
            raise WorkerFailure("worker RPC failed; all caches invalid") from exc
        finally:
            with self.active_lock:
                self.active_generation = None
                self.active_generations = {}
            self.lock.release()

    def close(self):
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            # Concurrent callers must wait for the in-progress cleanup rather
            # than returning merely because requests have been marked closed.
            with getattr(self, "cleanup_lock", None) or nullcontext():
                if getattr(self, "closed", False):
                    if getattr(self, "cleanup_failed", False):
                        raise WorkerFailure("worker cleanup previously failed") from self.cleanup_error
                    return
                self.closed = True
                try:
                    self._close_resources()
                except BaseException as exc:
                    # closed rejects new RPCs; it is not proof of quiescence.
                    # Concurrent/repeated close callers must observe failure too.
                    self.cleanup_failed = True
                    self.cleanup_error = exc
                    raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)

    def _close_resources(self):
        descriptor_error = None
        with getattr(self, "active_lock", None) or nullcontext():
            for name in ("cancel_read_fd", "cancel_write_fd"):
                try:
                    self._retire_cancel_fd(name)
                except OSError as exc:
                    descriptor_error = exc
        if getattr(self, "process", None) is None:
            if descriptor_error is not None:
                raise descriptor_error
            return
        # All real children start in a new session. Reject any observable group
        # mismatch before signaling; a reaped leader may leave owned helpers.
        # Descriptor cleanup failure must never skip child termination.
        try:
            _check_group(self.process)
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=2)
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while True:
                remaining = live_group(self.process.pid)
                if not remaining:
                    break
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise WorkerFailure(f"worker group cleanup incomplete; live PIDs {remaining}")
                time.sleep(min(0.01, budget))
        finally:
            try:
                self.process.stdin.close()
            finally:
                self.process.stdout.close()
        if descriptor_error is not None:
            raise descriptor_error

    def _retire_cancel_fd(self, name):
        # exec/spawn ownership is interruption-safe; retiring its inherited
        # descriptor must be too. In particular, don't double-close a stale fd
        # if SIGINT arrives after close but before clearing the field.
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            fd = getattr(self, name, None)
            setattr(self, name, None)
            if fd is not None:
                os.close(fd)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
