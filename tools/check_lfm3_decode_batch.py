"""Fail-fast standalone paired-decode fixture runner with owned-group cleanup."""
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


FIXTURES = dict(scratch=(59,10,35,2),sliding=(47,17,9,2),softmax=(49,17,11,2),
                tiled=(49,17,11,2),upper=(5,1,1,0))
GEOMETRY = dict(scratch=([12,300],[3,257]),sliding=([270,280],[254,256]),
                softmax=([2062,2074],[2046,2048]),tiled=([4112,4124],[4096,4098]),upper=([2,32766],[1,32764]))


def validate_case(fixture, summary, rows, reference_mode="normal"):
    if reference_mode not in ("normal","compact"):raise AssertionError("unknown reference mode")
    comparisons, pairs, retired, resets = FIXTURES[fixture]
    owners=2 if reference_mode=="compact" else 4
    if (summary.get("reference_mode")!=reference_mode or type(summary.get("gpu_runner_owners")) is not int or
        summary["gpu_runner_owners"]!=owners or summary.get("worker_integration") is not False):
        raise AssertionError("incorrect reference mode/owner/integration metadata")
    expected = dict(comparisons=comparisons,expected_comparisons=comparisons,
        paired_calls=pairs,expected_paired_calls=pairs,retired_checks=retired,
        expected_retired_checks=retired,reset_checks=resets,rejections=11,dense_batch_rows=2)
    if (summary.get("status")!="passed" or summary.get("fixture")!=fixture or
        any(type(summary.get(k)) is not int or summary[k]!=value for k,value in expected.items())):
        raise AssertionError("paired-decode fixture/count mismatch")
    checked=[r for r in rows if "exact_logits" in r]
    rejected=[r for r in rows if "rejected" in r]
    inactive=[r for r in rows if "retired_lane" in r]
    reset=[r for r in rows if "exact_full_cache" in r and "exact_logits" not in r]
    if len(checked)!=comparisons or len(rejected)!=11 or len(inactive)!=retired or len(reset)!=resets:
        raise AssertionError("incomplete paired-decode check records")
    headers=[r for r in rows if "fixture" in r]
    coverage=[r for r in rows if r.get("case")=="fixture-coverage"]
    if len(headers)!=1 or len(coverage)!=1:raise AssertionError("missing/duplicate fixture metadata")
    header=headers[0];capacities,lengths=GEOMETRY[fixture]
    counts=dict(expected_comparisons=comparisons,expected_paired_calls=pairs,expected_retired_checks=retired,expected_rejections=11,expected_reset_checks=resets)
    if (header.get("fixture")!=fixture or header.get("reference_mode")!=reference_mode or
        type(header.get("gpu_runner_owners")) is not int or header["gpu_runner_owners"]!=owners or
        header.get("standard_six_step_sequence") is not (fixture!="upper") or
        any(type(header.get(k)) is not int or header[k]!=v for k,v in counts.items())):
        raise AssertionError("incorrect fixture header")
    for key,values in (("capacities",capacities),("prompt_lengths",lengths)):
        if (not isinstance(header.get(key),list) or header[key]!=values or
            not all(type(value) is int for value in header[key])):raise AssertionError("incorrect fixture geometry")
    expected_coverage=dict(shifted_window=True,small_softmax=True,large_softmax=fixture in ("softmax","tiled","upper"),
                           B_exceeded_A_serial_scratch=fixture in ("scratch","upper"),tiled_prefill=fixture=="tiled")
    if any(coverage[0].get(k) is not v for k,v in expected_coverage.items()):raise AssertionError("missing fixture coverage")
    if any(r.get("exact_logits") is not True or r.get("exact_full_cache") is not True or
           r.get("logit_comparison")!="float_bits" or
           type(r.get("cache_bytes")) is not int or r["cache_bytes"]<=0 or
           any(type(r.get(k)) is not int for k in ("id","expected_id","cursor","expected_cursor","vocabulary")) or
           r["vocabulary"]<2 or not 0<=r["id"]<r["vocabulary"] or
           r.get("id")!=r.get("expected_id") or r.get("cursor")!=r.get("expected_cursor") for r in checked):
        raise AssertionError("paired logits/cache/position mismatch")
    if any(any(r.get(k) is not True for k in ("rejected","metadata_unchanged","full_caches_unchanged")) for r in rejected):
        raise AssertionError("rejection mutated paired state")
    if any(r.get("full_cache_unchanged_and_exact") is not True for r in inactive) or any(r.get("exact_full_cache") is not True for r in reset):
        raise AssertionError("retirement/reset cache mismatch")


@owned_main
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fixture",choices=list(FIXTURES),action="append")
    parser.add_argument("--reference-mode",choices=("normal","compact"),default="normal")
    parser.add_argument("--sanitizer",type=Path)
    parser.add_argument("--dump-kernel-launches",action="store_true")
    parser.add_argument("--timeout",type=float,default=600)
    args=parser.parse_args();fixtures=args.fixture or list(FIXTURES)
    if not 0<args.timeout<=7200 or len(set(fixtures))!=len(fixtures):raise ValueError("invalid fixture/deadline")
    if args.dump_kernel_launches and args.sanitizer is None:raise ValueError("launch dump requires sanitizer")
    args.output.mkdir(parents=True,exist_ok=False)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    binary_hash=digest(args.binary);tool_hash=digest(args.sanitizer) if args.sanitizer else None
    report=dict(status="running",binary_sha256=binary_hash,sanitizer_sha256=tool_hash,
                reference_mode=args.reference_mode,records=[],not_run=fixtures[:],worker_integration=False)
    def save():(args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    save();env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH","")
    try:
        for fixture in fixtures:
            command=[str(args.binary),"--experimental",str(args.checkpoint),str(args.output/fixture),fixture]
            if args.reference_mode!="normal":command.append(args.reference_mode)
            sanitizer_log=args.output/(fixture+".sanitizer.log")
            if args.sanitizer:
                options=[str(args.sanitizer),"--tool","memcheck","--error-exitcode","86","--log-file",str(sanitizer_log)]
                if args.dump_kernel_launches:options += ["--dump-kernel-launches","--print-session-details","--print-limit","0"]
                command=options+command
            record=dict(fixture=fixture,status="running",command=command)
            report["records"].append(record);report["not_run"].remove(fixture);save()
            stop=threading.Event();start=time.monotonic();observer_errors=[]
            with (args.output/(fixture+".application.log")).open("w") as application:
                owner=OwnedJob();observer=None
                try:
                    process=owner.start(command,env=env,stdout=application,stderr=subprocess.STDOUT)
                    record["owned_process_group"]=process.pid;save()
                    def observe():
                        try:
                            with (args.output/(fixture+".resources.jsonl")).open("w") as stream:
                                while True:
                                    stream.write(json.dumps(dict(seconds=time.monotonic()-start,**resource_snapshot(process.pid)))+"\n");stream.flush()
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
                if observer is None or observer.ident is None or observer.is_alive() or observer_errors:raise RuntimeError("resource observer incomplete: "+repr(observer_errors))
            record.update(returncode=code,timed_out=timed_out,seconds=time.monotonic()-start,group_cleanup_complete=True)
            if timed_out or code!=0:
                record["status"]="timeout" if timed_out else "failed";report["status"]="failed";save();return 86
            summary=json.loads((args.output/fixture/"summary.json").read_text())
            rows=[json.loads(line) for line in (args.output/fixture/"checks.jsonl").read_text().splitlines()]
            validate_case(fixture,summary,rows,args.reference_mode)
            if args.sanitizer and "ERROR SUMMARY: 0 errors" not in sanitizer_log.read_text():raise AssertionError("memcheck did not finish cleanly")
            record["status"]="passed";save()
        if digest(args.binary)!=binary_hash or (args.sanitizer and digest(args.sanitizer)!=tool_hash):raise AssertionError("probe/tool changed")
        report["status"]="passed"
    except BaseException as exc:
        report["status"],report["error"]="failed",str(exc);save();raise
    save();print(json.dumps(report,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
