"""Experimental loopback-only HTTP text serving with explicit two-lane decode batches."""
import argparse
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import socket
import stat
import threading
import time

try:
    from lfm3_chat import integer, make_request, policy, read_json, token_ids, sampling_options, validate_messages
    from lfm3_text_frontend import load_frontend
    from lfm3_worker_client import WorkerClient, WorkerFailure, strict_json
except ModuleNotFoundError:
    from tools.lfm3_chat import integer, make_request, policy, read_json, token_ids, sampling_options, validate_messages
    from tools.lfm3_text_frontend import load_frontend
    from tools.lfm3_worker_client import WorkerClient, WorkerFailure, strict_json


MAX_BODY = 4 * 1024 * 1024
MAX_RESPONSE = 4 * 1024 * 1024
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{32}\Z")


class APIError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


@dataclass(frozen=True)
class Settings:
    max_sessions: int = 2
    total_capacity: int = 32768
    queue_size: int = 4
    handlers: int = 8
    request_timeout: float = 330
    socket_timeout: float = 10
    session_idle_timeout: float = 300

    def __post_init__(self):
        integer(self.max_sessions, "max_sessions", 1, 2)
        integer(self.total_capacity, "total_capacity", 2, 32768)
        integer(self.queue_size, "queue_size", 1, 64)
        integer(self.handlers, "handlers", 1, 128)
        for value in (self.request_timeout, self.socket_timeout, self.session_idle_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
                raise ValueError("invalid timeout")


class Service:
    """One owner thread for every native RPC; per-session lease includes queued work.

    One daemon executes serialized frontend work; callers wait only until their
    absolute deadline. CPU work is not preemptible. No history stored.
    """
    def __init__(self, worker, frontend, config, settings=Settings()):
        self.worker, self.frontend, self.config, self.settings = worker, frontend, config, settings
        self.vocabulary, self.suppress, self.stops = policy(config)
        self.sessions = {}
        self.lock = threading.Lock()
        self.frontend_lock = threading.Lock()
        self.jobs = queue.Queue(settings.queue_size)
        self.failed = False
        self.cleanup_failed = False
        self.stopping = threading.Event()
        self.frontend_jobs = queue.Queue(1)
        self.frontend_owner = threading.Thread(target=self._run_frontend,
                                               name="lfm3-frontend-owner", daemon=True)
        self.frontend_owner.start()
        self.owner = threading.Thread(target=self._run, name="lfm3-worker-owner", daemon=True)
        self.owner.start()

    def _run_frontend(self):
        while not self.stopping.is_set() or not self.frontend_jobs.empty():
            try:
                function, future, deadline = self.frontend_jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            error = None
            try:
                if self.stopping.is_set() or time.monotonic() >= deadline:
                    raise APIError(504, "frontend_deadline")
                result = function()
                if time.monotonic() >= deadline:
                    raise APIError(504, "frontend_deadline")
            except BaseException as exc:
                error = exc
            finally:
                # This worker never touches session leases or submits native
                # work. Late CPU completion cannot release a newer lease.
                self.frontend_lock.release()
                self.frontend_jobs.task_done()
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

    def _frontend_call(self, function, deadline, *, wait=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIError(504, "frontend_deadline")
        acquired = (self.frontend_lock.acquire(timeout=remaining) if wait
                    else self.frontend_lock.acquire(blocking=False))
        if not acquired:
            raise APIError(504 if wait else 429, "frontend_deadline" if wait else "frontend_busy")
        future = Future()
        try:
            if self.stopping.is_set():
                raise APIError(503, "worker_unavailable")
            if time.monotonic() >= deadline:
                raise APIError(504, "frontend_deadline")
            self.frontend_jobs.put_nowait((function, future, deadline))
        except BaseException:
            self.frontend_lock.release()
            raise
        try:
            return future.result(timeout=max(0, deadline - time.monotonic()))
        except FutureTimeout:
            if future.done():
                return future.result()
            # No cancellation: the sole daemon retains the frontend lock until
            # completion, discarding expired results. It cannot reach the GPU.
            raise APIError(504, "frontend_deadline") from None

    def _fatal(self):
        with self.lock:
            self.failed = True
            self.sessions.clear()
        try:
            self.worker.close()
        except Exception:
            # Reap timeout is not permission to resume service, nor should it
            # kill the owner thread and leave queued callers without their503.
            with self.lock:
                self.cleanup_failed = True

    def _rpc(self, request, cancel_event=None):
        cancellable = getattr(self.worker, "call_cancellable", None)
        result = (cancellable(request, cancel_event) if cancel_event is not None and cancellable is not None
                  else self.worker.call(request))
        # Even native execution errors are fail-closed, rather than guessing
        # which CUDA errors might leave the process or another cache usable.
        if result.get("ok") is not True:
            raise WorkerFailure("native operation rejected")
        return result

    def _run(self):
        while not self.stopping.is_set() or not self.jobs.empty():
            try:
                self._reap_idle()
            except Exception:
                self._fatal()
            try:
                function, future, deadline, cleanup, state, completion_lock, discard = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if self.failed:
                    raise APIError(503, "worker_unavailable")
                if time.monotonic() >= deadline:
                    raise APIError(504, "queue_deadline")
                result = function()
            except APIError as exc:
                cleanup()
                future.set_exception(exc)
            except Exception:
                self._fatal()
                cleanup()
                future.set_exception(APIError(503, "worker_unavailable"))
            else:
                # Publish completion and observe abandonment atomically. Only
                # this owner thread may release an abandoned native allocation.
                with completion_lock:
                    expired = state["abandoned"] or time.monotonic() >= deadline
                    abandoned = discard and expired
                    if expired:
                        if not abandoned:
                            cleanup()
                        future.set_exception(APIError(504, "request_deadline_outcome_unknown"))
                    else:
                        cleanup()
                        future.set_result(result)
                if abandoned:
                    try:
                        discard()
                    except Exception:
                        self._fatal()
                    finally:
                        cleanup()
            finally:
                self.jobs.task_done()

    def _reap_idle(self):
        with self.lock:
            if self.failed or self.stopping.is_set():
                return
            expired = [sid for sid, session in self.sessions.items()
                       if not session["busy"] and time.monotonic() - session["last_used"] >= self.settings.session_idle_timeout]
            for sid in expired:
                self.sessions[sid]["busy"] = True
        for sid in expired:
            self._close_allocation(sid)

    def _close_allocation(self, sid):
        result = self._rpc(dict(op="close", session_id=sid))
        if result.get("closed") is not True or result.get("session_id", sid) != sid:
            raise WorkerFailure("invalid close reply")
        with self.lock:
            self.sessions.pop(sid, None)

    def _submit(self, function, cleanup, discard=None, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + self.settings.request_timeout
        future = Future()
        state = {"abandoned": False}
        completion_lock = threading.Lock()
        try:
            with self.lock:
                if self.failed or self.stopping.is_set():
                    raise APIError(503, "worker_unavailable")
                if time.monotonic() >= deadline:
                    raise APIError(504, "queue_deadline")
                try:
                    self.jobs.put_nowait((function, future, deadline,
                                         cleanup, state, completion_lock, discard))
                except queue.Full:
                    raise APIError(429, "queue_full") from None
        except BaseException:
            cleanup()
            raise
        try:
            return future.result(timeout=max(0, deadline - time.monotonic()))
        except FutureTimeout:
            with completion_lock:
                if future.done():
                    return future.result()
                state["abandoned"] = True
            # Do not release a running generation lease or pretend cancellation.
            raise APIError(504, "request_deadline_outcome_unknown") from None

    def create(self, body):
        if set(body) != {"capacity"}:
            raise ValueError("create requires capacity only")
        capacity = integer(body["capacity"], "capacity", 2, self.settings.total_capacity)
        sid = secrets.token_urlsafe(24)
        with self.lock:
            if self.failed or self.stopping.is_set():
                raise APIError(503, "worker_unavailable")
            if (len(self.sessions) >= self.settings.max_sessions
                    or sum(s["capacity"] for s in self.sessions.values()) + capacity > self.settings.total_capacity):
                raise APIError(429, "session_quota")
            self.sessions[sid] = {"capacity": capacity, "busy": True, "operation": "create",
                                  "cancel_event": None, "last_used": time.monotonic()}
        created = False

        def run():
            nonlocal created
            result = self._rpc(dict(op="create", session_id=sid, capacity=capacity))
            if result.get("session_id") != sid:
                raise WorkerFailure("session mismatch")
            created = True
            return dict(session_id=sid, capacity=capacity, gpu_batching=False)

        def cleanup():
            with self.lock:
                if sid in self.sessions:
                    if created:
                        self.sessions[sid]["busy"] = False
                        self.sessions[sid]["operation"] = None
                        self.sessions[sid]["last_used"] = time.monotonic()
                    else:
                        del self.sessions[sid]
        return self._submit(run, cleanup, lambda: self._close_allocation(sid))

    def _lease(self, sid, operation=None, cancel_event=None):
        if not isinstance(sid, str) or not SESSION_ID.fullmatch(sid):
            raise ValueError("invalid session ID")
        with self.lock:
            if self.failed or self.stopping.is_set():
                raise APIError(503, "worker_unavailable")
            session = self.sessions.get(sid)
            if session is None:
                raise APIError(404, "unknown_session")
            if session["busy"]:
                raise APIError(409, "session_busy")
            session["busy"] = True
            session["operation"] = operation
            session["cancel_event"] = cancel_event
            return session["capacity"]

    def _release(self, sid, cancel_event=None):
        with self.lock:
            session = self.sessions.get(sid)
            if session is not None and (cancel_event is None or session.get("cancel_event") is cancel_event):
                session["busy"] = False
                session["operation"] = None
                session["cancel_event"] = None
                session["last_used"] = time.monotonic()

    def cancel(self, sid, body):
        if body:
            raise ValueError("cancel requires an empty object")
        if not isinstance(sid, str) or not SESSION_ID.fullmatch(sid):
            raise ValueError("invalid session ID")
        with self.lock:
            if self.failed or self.stopping.is_set():
                raise APIError(503, "worker_unavailable")
            session = self.sessions.get(sid)
            if session is None:
                raise APIError(404, "unknown_session")
            event = session.get("cancel_event")
            if not session["busy"] or session.get("operation") != "generate" or event is None:
                raise APIError(409, "generation_not_active")
            if getattr(self.worker, "supports_cancellation", False) is not True:
                raise APIError(501, "cancellation_unavailable")
            event.set()
            send = getattr(self.worker, "cancel", None)
            if send is not None:
                # Nonblocking targeted pipe notification only, never a runner
                # mutation. Keep the lease identity stable until it is sent:
                # an old cancellation must not target a subsequent generation.
                try:
                    send(sid)
                except WorkerFailure as exc:
                    # A failed notification cannot be acknowledged as 202.
                    # The event remains set; only the owner may complete the
                    # generation/reset, and completion may still win.
                    raise APIError(503, "cancellation_transport_failed") from exc
        return dict(session_id=sid, cancel_requested=True, gpu_batching=False)

    def lifecycle(self, sid, op, body):
        if op not in ("reset", "close") or body:
            raise ValueError("invalid lifecycle request")
        self._lease(sid, op)

        def run():
            result = self._rpc(dict(op=op, session_id=sid))
            if (result.get("reset" if op == "reset" else "closed") is not True
                    or result.get("session_id", sid) != sid):
                raise WorkerFailure("invalid lifecycle reply")
            if op == "close":
                with self.lock:
                    self.sessions.pop(sid, None)
            return {"ok": True}
        return self._submit(run, lambda: self._release(sid))

    def _validate_generation(self, body):
        if not isinstance(body, dict) or set(body) - {"session_id", "messages", "max_new_tokens", "sampling"} or not {
                "session_id", "messages", "max_new_tokens"} <= set(body):
            raise ValueError("invalid generation fields")
        if "sampling" in body and not isinstance(body["sampling"], dict):
            raise ValueError("sampling must be an object")
        if not isinstance(body["messages"], list):
            raise ValueError("messages must be a list")
        if not isinstance(body["session_id"], str) or not SESSION_ID.fullmatch(body["session_id"]):
            raise ValueError("invalid session ID")
        integer(body["max_new_tokens"], "max_new_tokens", 1, 32768)
        validate_messages(body["messages"])
        sampling_options(body.get("sampling"), self.vocabulary)

    def generate(self, body):
        deadline = time.monotonic() + self.settings.request_timeout
        self._validate_generation(body)
        sid = body["session_id"]
        cancel_event = threading.Event()
        capacity = self._lease(sid, "generate", cancel_event)
        try:
            request = self._frontend_call(
                lambda: make_request(self.frontend, None, None, self.config, capacity,
                                     body["max_new_tokens"], messages=body["messages"],
                                     sampling=body.get("sampling")), deadline)
            native = {key: value for key, value in request.items()
                      if key not in ("capacity", "suppress_tokens", "stop_tokens", "experimental_long_context")}
            native.update(op="generate", session_id=sid)
        except BaseException:
            self._release(sid, cancel_event)
            raise

        def run():
            if time.monotonic() >= deadline:
                raise APIError(504, "queue_deadline")
            if cancel_event.is_set():
                reset = self._rpc(dict(op="reset", session_id=sid))
                if reset.get("reset") is not True or reset.get("session_id", sid) != sid:
                    raise WorkerFailure("invalid cancellation reset reply")
                return dict(session_id=sid, text="", generated_ids=[], consumed_tokens=0,
                            reused_tokens=0, cancelled=True, finish_reason="cancelled", cache_valid=False,
                            do_sample=request["do_sample"], seed=request["seed"], gpu_batching=False)
            result = self._rpc(native, cancel_event)
            response = self._generation_response(sid, request, result, cancel_event)
            self._decode_responses([response], deadline)
            return response
        return self._submit(run, lambda: self._release(sid, cancel_event), deadline=deadline)

    def _generation_response(self, sid, request, result, cancel_event, *, gpu_batching=False):
        if not isinstance(result, dict) or any(key in result for key in ("sampling_policy", "sampled_policy", "sampling")):
            raise WorkerFailure("ambiguous generation metadata")
        if "cancelled" in result and type(result["cancelled"]) is not bool:
            raise WorkerFailure("invalid cancellation metadata")
        cancelled = result.get("cancelled", False)
        if cancelled:
            if not cancel_event.is_set() or result.get("cache_valid") is not False:
                raise WorkerFailure("unsolicited or invalid cancellation")
        elif (("cache_valid" in result and result["cache_valid"] is not True)
              or result.get("finish_reason") == "cancelled"):
            raise WorkerFailure("unsolicited cache invalidation")
        generated = token_ids(result.get("generated_ids"), "generated_ids", self.vocabulary, request["max_new_tokens"])
        if (result.get("session_id") != sid or (not generated and not cancelled) or set(generated) & set(self.suppress)
                or any(t in self.stops for t in generated[:-1])
                or (not cancelled and len(generated) < request["max_new_tokens"] and generated[-1] not in self.stops)):
            raise WorkerFailure("invalid generation reply")
        for key in ("do_sample", "seed"):
            if type(result.get(key)) is not type(request[key]) or result[key] != request[key]:
                raise WorkerFailure("sampling metadata mismatch")
        consumed = 0 if cancelled else len(request["input_ids"]) + len(generated) - 1
        integer(result.get("consumed_tokens"), "consumed_tokens", consumed, consumed)
        integer(result.get("reused_tokens"), "reused_tokens", 0, 0 if cancelled else len(request["input_ids"]))
        return dict(session_id=sid, text="", generated_ids=generated,
                    reused_tokens=result["reused_tokens"], consumed_tokens=consumed,
                    do_sample=result["do_sample"], seed=result["seed"], gpu_batching=gpu_batching,
                    cancelled=cancelled, cache_valid=not cancelled,
                    finish_reason="cancelled" if cancelled else "stop" if generated[-1] in self.stops else "length")

    def _decode_responses(self, responses, deadline):
        def decode():
            texts = []
            for response in responses:
                if time.monotonic() >= deadline:
                    raise APIError(504, "frontend_deadline")
                ids = response["generated_ids"]
                texts.append("" if not ids else self.frontend.decode(ids[:-1] if ids[-1] in self.stops else ids))
            return texts
        texts = self._frontend_call(decode, deadline, wait=True) if any(r["generated_ids"] for r in responses) else [""] * len(responses)
        for response, text in zip(responses, texts):
            response["text"] = text
            if len(encode_json(response)) > MAX_RESPONSE:
                raise WorkerFailure("response exceeds limit")

    def generate_batch(self, body):
        deadline = time.monotonic() + self.settings.request_timeout
        if not isinstance(body, dict) or set(body) != {"requests"} or not isinstance(body["requests"], list) or len(body["requests"]) != 2:
            raise ValueError("batch requires exactly two requests")
        bodies = body["requests"]
        for item in bodies:
            self._validate_generation(item)
        sids = [item["session_id"] for item in bodies]
        if sids[0] == sids[1]:
            raise ValueError("duplicate batch session")
        if getattr(self.worker, "supports_batching", False) is not True:
            raise APIError(501, "batching_unavailable")
        events = [threading.Event(), threading.Event()]
        # Check both before mutating either lease. Never release somebody else's
        # busy lease when the second lane cannot be admitted.
        with self.lock:
            if self.failed or self.stopping.is_set():
                raise APIError(503, "worker_unavailable")
            sessions = [self.sessions.get(sid) for sid in sids]
            if any(session is None for session in sessions):
                raise APIError(404, "unknown_session")
            if any(session["busy"] for session in sessions):
                raise APIError(409, "session_busy")
            capacities = [session["capacity"] for session in sessions]
            for session, event in zip(sessions, events):
                session.update(busy=True, operation="generate", cancel_event=event)

        def cleanup():
            for sid, event in zip(sids, events):
                self._release(sid, event)

        def encode():
            requests = []
            for item, capacity in zip(bodies, capacities):
                if time.monotonic() >= deadline:
                    raise APIError(504, "frontend_deadline")
                requests.append(make_request(self.frontend, None, None, self.config, capacity,
                                             item["max_new_tokens"], messages=item["messages"], sampling=item.get("sampling")))
            return requests
        try:
            requests = self._frontend_call(encode, deadline)
            lanes = [{key: value for key, value in request.items()
                      if key not in ("capacity", "suppress_tokens", "stop_tokens", "experimental_long_context")}
                     for request in requests]
            for lane, sid in zip(lanes, sids):
                lane["session_id"] = sid
        except BaseException:
            cleanup()
            raise

        def run():
            if time.monotonic() >= deadline:
                raise APIError(504, "queue_deadline")
            # The transport bridges pre-set events before its single group write;
            # native owns lane-local abort/reset and peer continuation.
            result = self.worker.call_batch_cancellable(lanes, events)
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise WorkerFailure("native batch rejected")
            paired = result.get("paired_decode_calls")
            if type(paired) is not int or paired < 0 or type(result.get("gpu_batching")) is not bool or result["gpu_batching"] != (paired > 0):
                raise WorkerFailure("invalid batch accounting")
            results = result.get("results")
            if not isinstance(results, list) or len(results) != 2:
                raise WorkerFailure("invalid batch results")
            responses = []
            for sid, request, event, lane in zip(sids, requests, events, results):
                if (not isinstance(lane, dict) or type(lane.get("gpu_batching")) is not bool
                        or ("ok" in lane and lane["ok"] is not True)
                        or lane["gpu_batching"] != result["gpu_batching"]
                        or type(lane.get("paired_decode_calls")) is not int or lane["paired_decode_calls"] != paired
                        or type(lane.get("cancelled")) is not bool or type(lane.get("cache_valid")) is not bool):
                    raise WorkerFailure("invalid lane accounting")
                response = self._generation_response(sid, request, lane, event, gpu_batching=result["gpu_batching"])
                response["paired_decode_calls"] = paired
                responses.append(response)
            # Prefill samples the first token without a decode forward. Each
            # paired forward therefore needs another output slot in both lanes.
            # Cancellation may discard the token sampled after that forward,
            # leaving one fewer reported token than normal completion would.
            if paired > min(request["max_new_tokens"] - 1 for request in requests):
                raise WorkerFailure("paired calls exceed requested decode budget")
            for response in responses:
                maximum = len(response["generated_ids"]) - (0 if response["cancelled"] else 1)
                if paired > maximum:
                    raise WorkerFailure("paired calls exceed lane decode work")
            if (not any(response["cancelled"] for response in responses)
                    and paired != min(len(response["generated_ids"]) - 1 for response in responses)):
                raise WorkerFailure("paired calls do not match completed group")
            self._decode_responses(responses, deadline)
            response = dict(results=responses, gpu_batching=result["gpu_batching"], paired_decode_calls=paired)
            if len(encode_json(response)) > MAX_RESPONSE:
                raise WorkerFailure("response exceeds limit")
            return response
        return self._submit(run, cleanup, deadline=deadline)

    def close(self):
        self.stopping.set()
        with self.lock:
            self.failed = True
            self.sessions.clear()
        self.owner.join(timeout=getattr(self.worker, "timeout", 300) + self.settings.request_timeout + 5)
        self._fatal()


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def validate_token(token):
    if not isinstance(token, str) or not 32 <= len(token) <= 256 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError("bearer token must be 32..256 URL-safe characters")
    return token


def token_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > 257):
            raise ValueError("token file must be owned regular mode-0600 file")
        return validate_token(stream.read(258).decode("ascii").removesuffix("\n"))


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(self, address, service, token):
        # Literal IPv4 loopback only: no hostname/DNS or wildcard ambiguity.
        if address[0] != "127.0.0.1":
            raise ValueError("only 127.0.0.1 binding is supported")
        self.service, self.token = service, validate_token(token)
        self.slots = threading.BoundedSemaphore(service.settings.handlers)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            # No new thread, parsing, auth work, or GPU admission on saturation.
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "LFM3Local"
    sys_version = ""
    protocol_version = "HTTP/1.0"  # one request per socket, no pipelining/keepalive

    def setup(self):
        self.request.settimeout(self.server.service.settings.socket_timeout)
        super().setup()
        # Bound total header time as well as individual socket reads; trickling
        # one byte per timeout must not retain a handler forever.
        self.header_timer = threading.Timer(self.server.service.settings.socket_timeout, self._expire_headers)
        self.header_timer.daemon = True
        self.header_timer.start()

    def _expire_headers(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def parse_request(self):
        try:
            return super().parse_request()
        finally:
            self.header_timer.cancel()

    def finish(self):
        self.header_timer.cancel()
        super().finish()

    def log_message(self, *args):
        pass  # Never log request paths, prompts, session IDs, or credentials.

    def _reply(self, status, body):
        payload = encode_json(body)
        if len(payload) > MAX_RESPONSE:
            status, payload = 500, b'{"error":"response_limit"}'
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except OSError:
            # A disconnected/non-reading peer cannot receive a second error
            # reply. Close after the existing socket write deadline instead.
            self.close_connection = True

    def do_GET(self):
        if self.path != "/health":
            self._reply(404, {"error": "not_found"})
        else:
            ready = not self.server.service.failed and not self.server.service.stopping.is_set()
            self._reply(200 if ready else 503, {"ready": ready, "gpu_batching": False})

    def do_POST(self):
        try:
            auth = self.headers.get_all("Authorization", [])
            expected = "Bearer " + self.server.token
            if (len(auth) != 1 or len(auth[0]) > 263 or not auth[0].isascii()
                    or not hmac.compare_digest(auth[0], expected)):
                raise APIError(401, "unauthorized")
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1:
                raise APIError(400, "content_length_required")
            if not re.fullmatch(r"[0-9]{1,8}", lengths[0]):
                raise APIError(400, "invalid_content_length")
            length = int(lengths[0])
            if not 1 <= length <= MAX_BODY:
                raise APIError(413, "body_limit")
            types = self.headers.get_all("Content-Type", [])
            if len(types) != 1 or types[0].lower() != "application/json":
                raise APIError(415, "application_json_required")
            deadline = time.monotonic() + self.server.service.settings.socket_timeout
            body = bytearray()
            while len(body) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIError(408, "body_deadline")
                self.connection.settimeout(remaining)
                block = self.rfile.read1(min(65536, length - len(body)))
                if not block:
                    raise APIError(400, "incomplete_body")
                body.extend(block)
            self.connection.settimeout(self.server.service.settings.socket_timeout)
            request = strict_json(bytes(body))
            service = self.server.service
            if self.path == "/sessions":
                result = service.create(request)
            elif self.path == "/generate":
                result = service.generate(request)
            elif self.path == "/generate_batch":
                result = service.generate_batch(request)
            else:
                match = re.fullmatch(r"/sessions/([A-Za-z0-9_-]{32})/(reset|close|cancel)", self.path)
                if not match:
                    raise APIError(404, "not_found")
                if match[2] == "cancel":
                    result = service.cancel(match[1], request)
                    self._reply(202, result)
                    return
                result = service.lifecycle(match[1], match[2], request)
            self._reply(200, result)
        except APIError as exc:
            self._reply(exc.status, {"error": exc.code})
        except (ValueError, UnicodeError):
            self._reply(400, {"error": "invalid_request"})
        except (TimeoutError, socket.timeout):
            self._reply(408, {"error": "socket_deadline"})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, help="otherwise use LFM3_SERVER_TOKEN")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--worker-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=330)
    parser.add_argument("--total-capacity", type=int, default=32768)
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--session-idle-timeout", type=float, default=300)
    args = parser.parse_args()
    token = token_file(args.token_file) if args.token_file else validate_token(os.environ.get("LFM3_SERVER_TOKEN"))
    integer(args.port, "port", 1, 65535)
    settings = Settings(total_capacity=args.total_capacity, max_sessions=args.max_sessions,
                        queue_size=args.queue_size, request_timeout=args.request_timeout,
                        session_idle_timeout=args.session_idle_timeout)
    checkpoint = args.checkpoint.resolve(strict=True)
    config = read_json(checkpoint / "config.json")
    policy(config)
    frontend = load_frontend(checkpoint)
    worker_env = dict(os.environ)
    worker_env.pop("LFM3_SERVER_TOKEN", None)
    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, terminate)
    worker = service = None
    try:
        # Retain ownership before initialization, including interruption between
        # a successful constructor return and the caller's assignment.
        worker = WorkerClient.__new__(WorkerClient)
        worker.__init__(args.worker.resolve(strict=True), checkpoint, timeout=args.worker_timeout,
                        env=worker_env, cancellation=True, batching=True)
        service = Service(worker, frontend, config, settings)
        with LocalHTTPServer(("127.0.0.1", args.port), service, token) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if service is not None:
                service.close()
            elif worker is not None:
                worker.close()
        finally:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
