"""Request-validation parity: tools/lfm3_server.py against the native HTTP layer (lfm/lfm3_http.hpp).

Both sides answer the same raw request bodies with a fake engine, so every difference is in admission,
validation, prompt construction or response encoding, never in the model:
  cases      build the corpus: the frontend chat cases plus raw-JSON, number-literal, sampling, field
             and session-ID mutations (DIR/http-cases.jsonl)
  reference  Python: Service + strict_json in-process with Handler.do_POST's exception mapping and a
             fake worker that answers every generation with one stop token (needs transformers)
  native     the same bodies over HTTP to native/tests/lfm3_http_harness.cpp --mode stop
  compare    status and response bytes per request, session IDs normalized
Each case is a session lifecycle: create, then generate (if the case has one and creation succeeded),
then close.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys

SID = b"@SID@"
GOOD_MESSAGES = [{"role": "system", "content": "Respond with text."}, {"role": "user", "content": "Hi there"}]
# Literals whose value underflows a double: Python reads 0.0 or a subnormal, lfm::Json rejects them.
# A documented boundary difference (lfm3_http.hpp); reported, not failed.
UNDERFLOW = ("1e-400", "5e-324", "1e-310")


def b64(raw):
    return base64.b64encode(raw).decode()


def case(create, generate=None, *, kind, expect_difference=False):
    return dict(kind=kind, create=b64(create), generate=None if generate is None else b64(generate),
                expect_difference=expect_difference)


def body(capacity=b"256", generate=None):
    return b'{"capacity":' + capacity + b"}"


def generation(messages=None, count=b"8", sampling=None, sid=b'"' + SID + b'"', extra=b""):
    messages = json.dumps(GOOD_MESSAGES, ensure_ascii=False).encode() if messages is None else messages
    raw = b'{"session_id":' + sid + b',"messages":' + messages + b',"max_new_tokens":' + count
    if sampling is not None:
        raw += b',"sampling":' + sampling
    return raw + extra + b"}"


def build_cases(chat):
    cases = []
    for line in chat.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        capacity = row.get("capacity", 256)
        count = row.get("max_new_tokens", 8)
        create = json.dumps({"capacity": capacity}).encode()
        generate = (b'{"session_id":"' + SID + b'","messages":' + json.dumps(row["messages"], ensure_ascii=False).encode()
                    + b',"max_new_tokens":' + json.dumps(count).encode() + b"}")
        cases.append(case(create, generate, kind="chat"))
    numbers = ["0", "1", "2", "8", "256", "-0", "0.0", "8.0", "8e0", "8E+0", "0.5", "1.5", "1e-300", "1e300",
               "1.7976931348623157e308", "2.2250738585072014e-308", "-1", "-0.0", "32768", "32769", "128000", "128001",
               "2147483647", "2147483648", "1e2", "99999999999999999999999", "true", "false", "null", '"8"', "[]", "{}",
               "08", "0x8", "+8", ".5", "5.", "1e400", "-1e400", "NaN", "Infinity", "-Infinity", *UNDERFLOW]
    for literal in numbers:
        raw = literal.encode()
        difference = literal in UNDERFLOW
        cases.append(case(body(raw), kind="capacity-literal", expect_difference=difference))
        cases.append(case(body(), generation(count=raw), kind="count-literal", expect_difference=difference))
        for key in ("do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed"):
            cases.append(case(body(), generation(sampling=b'{"' + key.encode() + b'":' + raw + b"}"),
                              kind="sampling-literal", expect_difference=difference))
    for raw in (b"{}", b'{"capacity":256,"x":1}', b'{"capacity":256,"capacity":256}', b'{"Capacity":256}', b"[]", b"256",
                b'{"capacity":256', b'{"capacity":256}x', b'\xef\xbb\xbf{"capacity":256}', b' {"capacity" : 256 } ',
                b'{"capacity":256,"x":"\xff"}', b'{"capacity":256,"x":"\\ud800"}', b'{"capacity":256,"x":"\\u0000"}',
                b'{"capacity":256,"x":"a\tb"}', b'{"capacity":256,"x":"\xed\xa0\x80"}', b'{"capacity":256,"x":"\xc0\xaf"}',
                b'{"capacity":256,"x":' + b"[" * 63 + b"]" * 63 + b"}", b'{"capacity":256,"x":' + b"[" * 64 + b"]" * 64 + b"}"):
        cases.append(case(raw, kind="create-json"))
    sampling_objects = [b"{}", b"null", b"[]", b'{"top_p":0.9,"top_p":0.9}', b'{"beam":1}', b'{"do_sample":true,"seed":7}',
                        b'{"temperature":0.7,"top_k":42,"top_p":0.963,"min_p":0.05,"repetition_penalty":1.1,"seed":42,"do_sample":true}',
                        b'{"top_k":0,"top_p":1,"min_p":0,"repetition_penalty":1,"temperature":1,"seed":0,"do_sample":false}']
    for s in sampling_objects:
        cases.append(case(body(), generation(sampling=s), kind="sampling-object"))
    good = json.dumps(GOOD_MESSAGES).encode()
    message_variants = [b"[]", b"{}", b"null", b'"hi"', b'[{"role":"user"}]', b'[{"role":"user","content":null}]',
                        b'[{"role":"user","content":"a","content":"b"}]', b'[{"role":"user","content":"\\ud83d\\ude00"}]',
                        b'[{"role":"user","content":"\\ud800"}]', b'[{"role":"user","content":"\\u0000x"}]',
                        b'[{"role":"user","content":"\xff"}]', b'[{"role":"user","content":"caf\xc3\xa9 \xe2\x80\xa8"}]',
                        b'[{"role":"user","content":[' + b"[" * 62 + b"]" * 62 + b"]}]",
                        b'[{"role":"user","content":[' + b"[" * 63 + b"]" * 63 + b"]}]",
                        b'[{"role":"user","content":"<image>"}]', b'[{"role":"user","content":"<|audio_start|>"}]',
                        b'[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]',
                        b'[{"role":"user","content":[{"type":"text","text":"a","x":1}]}]',
                        b'[{"role":"tool","content":"x"}]', b'[{"role":"user","content":"x","tool_calls":[]}]']
    for m in message_variants:
        cases.append(case(body(), generation(messages=m), kind="messages"))
    for sid in (b'"' + b"A" * 31 + b'"', b'"' + b"A" * 33 + b'"', b'"' + b"A" * 31 + b'!"', b"1", b"null", b'"' + b"Z" * 32 + b'"'):
        cases.append(case(body(), generation(sid=sid), kind="session-id"))
    for raw in (generation(extra=b',"x":1'), b'{"session_id":"' + SID + b'","max_new_tokens":8}',
                b'{"session_id":"' + SID + b'","messages":' + good + b"}", b'{"messages":' + good + b',"max_new_tokens":8}',
                generation(extra=b',"max_new_tokens":8'), b"[]", b"null", generation()[:-1]):
        cases.append(case(body(), raw, kind="generate-fields"))
    # Capacity fit: prompt plus output against the session's capacity.
    for capacity, count in ((32, 1), (32, 8), (32, 31), (64, 40), (2, 1), (32768, 32767), (32768, 32768)):
        cases.append(case(body(str(capacity).encode()), generation(count=str(count).encode()), kind="fit"))
    return cases


def normalize(raw, sids):
    for sid in sids:
        raw = raw.replace(sid.encode(), SID)
    return raw


def reference(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lfm3_chat import policy, read_json
    from lfm3_server import APIError, Service, Settings, encode_json
    from lfm3_text_frontend import load_frontend
    from lfm3_worker_client import strict_json

    config = read_json(args.checkpoint / "config.json")
    stop = policy(config)[2][0]

    class FakeWorker:
        supports_cancellation = supports_batching = True
        timeout = 10

        def call(self, request):
            op = request["op"]
            if op == "create":
                return {"ok": True, "session_id": request["session_id"]}
            if op == "close":
                return {"ok": True, "closed": True}
            if op == "reset":
                return {"ok": True, "reset": True}
            return {"ok": True, "session_id": request["session_id"], "generated_ids": [stop], "reused_tokens": 0,
                    "consumed_tokens": len(request["input_ids"]), "do_sample": request["do_sample"], "seed": request["seed"]}

        def call_cancellable(self, request, event):
            return self.call(request)

        def cancel(self, sid):
            return False

        def close(self):
            pass

    service = Service(FakeWorker(), load_frontend(args.checkpoint), config, Settings())

    def post(kind, raw, sid=None):
        try:
            request = strict_json(raw)
            if kind == "create":
                result = service.create(request)
            elif kind == "generate":
                result = service.generate(request)
            else:
                result = service.lifecycle(sid, "close", request)
            return 200, encode_json(result)
        except APIError as exc:
            return exc.status, encode_json({"error": exc.code})
        except (ValueError, UnicodeError):
            return 400, encode_json({"error": "invalid_request"})
        except Exception as exc:  # http.server would drop the connection without a reply
            return "dropped", repr(exc).encode()

    run_cases(args.dir / "http-cases.jsonl", args.dir / "http.reference.jsonl", post)
    service.close()


def run_cases(cases_path, out_path, post):
    with out_path.open("w", encoding="utf-8") as out:
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            row = {}
            status, raw = post("create", base64.b64decode(c["create"]))
            sids = []
            if status == 200:
                sids.append(json.loads(raw)["session_id"])
            row["create"] = [status, normalize(raw, sids).decode("utf-8", "replace")]
            if c["generate"] is not None and sids:
                status, raw = post("generate", base64.b64decode(c["generate"]).replace(SID, sids[0].encode()))
                row["generate"] = [status, normalize(raw, sids).decode("utf-8", "replace")]
            if sids:
                status, raw = post("close", b"{}", sids[0])
                row["close"] = [status, normalize(raw, sids).decode("utf-8", "replace")]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def native(args):
    token = secrets.token_urlsafe(32)
    process = subprocess.Popen([str(args.harness), str(args.checkpoint), "--mode", "stop"], stdout=subprocess.PIPE,
                               env=dict(os.environ, LFM3_SERVER_TOKEN=token))
    try:
        port = json.loads(process.stdout.readline())["port"]

        def post(kind, raw, sid=None):
            path = {"create": "/sessions", "generate": "/generate"}.get(kind) or f"/sessions/{sid}/close"
            head = (f"POST {path} HTTP/1.0\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(raw)}\r\n\r\n").encode()
            with socket.create_connection(("127.0.0.1", port), timeout=120) as s:
                s.sendall(head + raw)
                chunks = []
                while block := s.recv(65536):
                    chunks.append(block)
            response = b"".join(chunks)
            if not response:
                return "dropped", b""
            status_line, _, rest = response.partition(b"\r\n")
            return int(status_line.split()[1]), rest.partition(b"\r\n\r\n")[2]

        run_cases(args.dir / "http-cases.jsonl", args.dir / "http.native.jsonl", post)
    finally:
        process.send_signal(signal.SIGTERM)
        if process.wait(timeout=30) != 0:
            raise SystemExit("harness exit failure")


def compare(args):
    cases = [json.loads(line) for line in (args.dir / "http-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    ref = (args.dir / "http.reference.jsonl").read_text(encoding="utf-8").splitlines()
    nat = (args.dir / "http.native.jsonl").read_text(encoding="utf-8").splitlines()
    if not len(cases) == len(ref) == len(nat):
        raise SystemExit("row count mismatch")
    report = dict(cases=len(cases), cases_sha256=hashlib.sha256((args.dir / "http-cases.jsonl").read_bytes()).hexdigest(),
                  mismatches=[], expected_differences=[], dropped_by_python=[], statuses={}, kinds={})
    for i, (c, r, n) in enumerate(zip(cases, ref, nat)):
        r, n = json.loads(r), json.loads(n)
        report["kinds"][c["kind"]] = report["kinds"].get(c["kind"], 0) + 1
        for step in ("create", "generate", "close"):
            if step in r:
                key = f"{step}:{r[step][0]}"
                report["statuses"][key] = report["statuses"].get(key, 0) + 1
                if r[step][0] == "dropped":
                    report["dropped_by_python"].append(dict(case=i, step=step, python=r[step], native=n.get(step)))
        if r != n and not any(r.get(s, [0])[0] == "dropped" for s in r):
            entry = dict(case=i, kind=c["kind"], python=r, native=n)
            (report["expected_differences"] if c["expect_difference"] else report["mismatches"]).append(entry)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(cases=report["cases"], mismatches=len(report["mismatches"]),
                          expected_differences=len(report["expected_differences"]),
                          dropped_by_python=len(report["dropped_by_python"]), statuses=report["statuses"]), indent=1))
    for entry in report["mismatches"][:10]:
        print(json.dumps(entry, ensure_ascii=False)[:600])
    print("PASS" if not report["mismatches"] else "FAIL")
    return 0 if not report["mismatches"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cases"); c.add_argument("--chat", type=Path, required=True); c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("reference"); r.add_argument("--dir", type=Path, required=True); r.add_argument("--checkpoint", type=Path, required=True)
    n = sub.add_parser("native"); n.add_argument("--dir", type=Path, required=True); n.add_argument("--checkpoint", type=Path, required=True)
    n.add_argument("--harness", type=Path, required=True)
    m = sub.add_parser("compare"); m.add_argument("--dir", type=Path, required=True); m.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.command == "cases":
        args.out.mkdir(parents=True, exist_ok=False)
        rows = build_cases(args.chat)
        (args.out / "http-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps(dict(cases=len(rows), sha256=hashlib.sha256((args.out / "http-cases.jsonl").read_bytes()).hexdigest())))
        return 0
    if args.command == "reference":
        return reference(args)
    if args.command == "native":
        return native(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
