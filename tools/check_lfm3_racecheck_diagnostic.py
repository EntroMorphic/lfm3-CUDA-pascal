"""All-launch32K racecheck with launch logs and bounded CPU/RSS observations.

No launch/kernel filters, suppressed instructions, force-blocking launches or
reduced hazard tracking. A timed-out diagnostic never becomes a clean gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

try:
    from lfm3_owned_process import OwnedJob, owned_main, resource_snapshot
except ModuleNotFoundError:
    from tools.lfm3_owned_process import OwnedJob, owned_main, resource_snapshot


def clean_result(code, timed_out, log, result, request, expected):
    return (not timed_out and code == 0 and "0 errors, 0 warnings" in log and
            isinstance(result, dict) and result.get("backend") == "native-cuda-experimental" and
            result.get("input_ids") == request.get("input_ids") and result.get("generated_ids") == [] and
            isinstance(result.get("teacher_argmax"), list) and len(result["teacher_argmax"]) == 1 and
            type(result["teacher_argmax"][0]) is int and result["teacher_argmax"] == expected.get("teacher_argmax"))


@owned_main
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitizer",type=Path,required=True)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--expected-result",type=Path,required=True,help="Accepted output for the identical32K prefill fixture")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout",type=float,default=1200)
    parser.add_argument("--num-workers",type=int,default=0)
    args=parser.parse_args()
    if not 0<args.timeout<=7200 or not 0<=args.num_workers<=64:raise ValueError("invalid diagnostic limits")
    args.output.mkdir(parents=True,exist_ok=False)
    raw=args.request.read_bytes();request=json.loads(raw)
    expected=json.loads(args.expected_result.read_text())
    if (request.get("capacity")!=32768 or len(request.get("input_ids",[]))!=32768 or
        request.get("max_new_tokens",0)!=0 or request.get("teacher_ids",[]) or request.get("trace_directory")):
        raise ValueError("diagnostic must preserve the original full32K prefill shape without extra trace paths")
    if (expected.get("input_ids") != request["input_ids"] or expected.get("generated_ids") != [] or
        not isinstance(expected.get("teacher_argmax"),list) or len(expected["teacher_argmax"])!=1 or
        type(expected["teacher_argmax"][0]) is not int):raise ValueError("invalid matching reference fixture")
    (args.output/"request.json").write_bytes(raw)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes=dict(binary=digest(args.binary),sanitizer=digest(args.sanitizer),request=hashlib.sha256(raw).hexdigest(),expected=digest(args.expected_result))
    env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH","")
    log=args.output/"racecheck.log";destination=args.output/"native.json"
    command=[str(args.sanitizer),"--tool","racecheck","--error-exitcode","86","--dump-kernel-launches",
        "--print-session-details","--print-limit","0","--racecheck-num-workers",str(args.num_workers),"--log-file",str(log),
        str(args.binary),"--experimental",str(args.checkpoint),str(args.output/"request.json"),str(destination)]
    version=subprocess.run([str(args.sanitizer),"--version"],env=env,capture_output=True,text=True,check=True,timeout=10)
    report=dict(status="running",hashes=hashes,command=command,sanitizer_version=version.stdout+version.stderr,
                scope="full32768-token prefill; all kernel launches instrumented; no decode, initcheck or batching acceptance inferred")
    def save():(args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    save();start=time.monotonic();stop=threading.Event();timed_out=False;observer_errors=[]
    with (args.output/"application.log").open("w") as stream:
        owner=OwnedJob();observer=None
        try:
            process=owner.start(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            report["owned_process_group"]=process.pid;save()
            def observe():
                try:
                    with (args.output/"resources.jsonl").open("w") as resources:
                        while True:
                            resources.write(json.dumps(dict(seconds=time.monotonic()-start,**resource_snapshot(process.pid)))+"\n")
                            resources.flush()
                            if stop.wait(2):break
                except BaseException as exc:
                    observer_errors.append(str(exc))
            observer=threading.Thread(target=observe,daemon=True);observer.start()
            code,timed_out=owner.wait(args.timeout)
        finally:
            try:owner.close()
            finally:
                stop.set()
                if observer is not None and observer.ident is not None:observer.join(timeout=5)
        if observer is None or observer.ident is None or observer.is_alive() or observer_errors:raise RuntimeError("racecheck resource observer incomplete")
    result=json.loads(destination.read_text()) if destination.exists() else None
    log_text=log.read_text(errors="replace") if log.exists() else ""
    passed=clean_result(code,timed_out,log_text,result,request,expected)
    stable=(digest(args.binary)==hashes["binary"] and digest(args.sanitizer)==hashes["sanitizer"] and digest(args.request)==hashes["request"] and digest(args.expected_result)==hashes["expected"])
    report.update(status="passed" if passed and stable else "timeout" if timed_out else "failed",returncode=code,
                  timed_out=timed_out,seconds=time.monotonic()-start,hashes_stable=stable,group_cleanup_complete=owner.closed)
    save();print(json.dumps(report,indent=2))
    return 0 if report["status"]=="passed" else 86


if __name__=="__main__":raise SystemExit(main())
