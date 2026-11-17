"""Authenticated explicit-group HTTP gate and fixed-budget extended resource soak.

GPU memory observations are device-wide on Thor, not allocation attribution.
No duration or measured speed is a universal production SLA.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import threading
import time

from lfm3_server import LocalHTTPServer, Service, Settings
from lfm3_text_frontend import load_frontend
from lfm3_worker_client import WorkerClient
from check_lfm3_soak import process_sample
from lfm3_owned_process import owned_main


def resource_gate(baseline, sample, require_exact=False):
    for role in ("server", "worker"):
        if (sample[role]["rss_bytes"] > baseline[role]["rss_bytes"]+32*1024*1024 or
            sample[role]["descriptors"] > baseline[role]["descriptors"]+2):
            raise AssertionError("fixed warm RSS/descriptor budget exceeded: "+role)
    memory = sample["gpu"]
    if (type(memory.get("cuda_free_bytes")) is not int or type(memory.get("cuda_total_bytes")) is not int or
        not 0 < memory["cuda_free_bytes"] <= memory["cuda_total_bytes"] or
        memory["cuda_total_bytes"] != baseline["gpu"]["cuda_total_bytes"]):
        raise AssertionError("invalid device-wide CUDA memory reading")
    if require_exact and not exact_accounting(baseline["gpu"]):
        raise AssertionError("exact engine-buffer accounting required but the warm baseline lacks it")
    if exact_accounting(baseline["gpu"]):
        # G4 (validation/g4-gates-v2/PLAN.md): the worker's own engine-buffer bytes decide; on Thor's unified memory
        # the device-wide reading also moves with host activity and is recorded only.
        if (not exact_accounting(memory) or memory["app_live_bytes"] > baseline["gpu"]["app_live_bytes"]):
            raise AssertionError("warm engine-buffer bytes grew above the first warm sample")
    elif memory["cuda_free_bytes"] < baseline["gpu"]["cuda_free_bytes"]-128*1024*1024:
        raise AssertionError("fixed warm device-wide CUDA free-memory budget exceeded")


def post_close_recovered(empty, post_close, require_exact=False):
    """After closing every session: exact engine-buffer bytes back to the empty-session value when the worker reports
    them, else device-wide free memory within 16 MiB of it. Returns (recovered, how it was judged)."""
    if require_exact and not exact_accounting(empty):
        return False, "exact engine-buffer accounting required but missing or invalid"
    if exact_accounting(empty):
        return (exact_accounting(post_close) and post_close["app_live_bytes"] == empty["app_live_bytes"],
                "worker engine-buffer bytes (exact); device-wide readings recorded only")
    return (type(post_close.get("cuda_free_bytes")) is int and post_close["cuda_free_bytes"] >= empty["cuda_free_bytes"] - 16 * 1024 * 1024,
            "device-wide free memory (no engine-buffer accounting)")


def exact_accounting(stats):
    """True when the worker reports valid engine-buffer accounting (native build with LFM_ALLOCATION_METRICS=1)."""
    return type(stats.get("app_live_bytes")) is int and stats.get("app_metrics_valid") is True


def validate_response(response, reference, require_pairs=True):
    if not isinstance(reference,list) or len(reference)<2 or not all(type(token) is int for token in reference):
        raise AssertionError("need nonempty reference with actual decode work")
    count = response.get("paired_decode_calls")
    if (type(count) is not int or count != len(reference)-1 or count < (1 if require_pairs else 0) or
        response.get("gpu_batching") is not (count > 0) or
        not isinstance(response.get("results"), list) or len(response["results"]) != 2):
        raise AssertionError("HTTP did not acknowledge actual grouped decode")
    for lane in response["results"]:
        if (lane.get("generated_ids") != reference or not all(type(token) is int for token in lane["generated_ids"]) or
            lane.get("cancelled") is not False or lane.get("cache_valid") is not True or
            type(lane.get("paired_decode_calls")) is not int or lane.get("gpu_batching") is not (count > 0) or
            lane.get("paired_decode_calls") != count):
            raise AssertionError("HTTP batch lane output/participation mismatch")


@owned_main
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--control-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cycles",type=int,default=60)
    parser.add_argument("--minimum-seconds",type=float,default=600)
    parser.add_argument("--require-exact-accounting",action="store_true",
                        help="native-GEMM builds (G4): memory is judged only by the worker's exact engine-buffer bytes; "
                             "missing or invalid accounting fails instead of falling back to device-wide readings")
    args=parser.parse_args()
    if not 5<=args.cycles<=10000 or not 0<=args.minimum_seconds<=7200:
        raise ValueError("invalid bounded soak limits")
    args.output.mkdir(parents=True,exist_ok=False)
    reference=json.loads(args.control_result.read_text())["generated_ids"][:32]
    if len(reference)!=32:raise ValueError("need32 accepted control predictions")
    source_paths=[Path(__file__),Path(__file__).with_name("lfm3_server.py"),Path(__file__).with_name("lfm3_worker_client.py")]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    binary_hash=hashlib.sha256(args.worker.read_bytes()).hexdigest()
    report=dict(status="running",worker_sha256=binary_hash,source_sha256=hashes,
        limits=dict(warm_rss_growth_bytes=32*1024*1024,warm_fd_growth=2,warm_device_free_drop_bytes=128*1024*1024),
        requested_cycles=args.cycles,minimum_seconds=args.minimum_seconds,records=[],samples=[],latencies=[],
        scope="authenticated fixed-two-lane HTTP grouping, per-lane cancellation, exact IDs, bounded extended soak and device-wide memory observation; not automatic microbatching or a universal production SLA")
    def save():(args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    save();worker=service=server=thread=None;begin=time.monotonic()
    try:
        env=dict(os.environ);env.pop("LFM3_SERVER_TOKEN",None)
        env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH","")
        traces=args.output/"traces"
        worker=WorkerClient(args.worker,args.checkpoint,timeout=180,env=env,cancellation=True,batching=True,sampling_trace_root=traces)
        service=Service(worker,load_frontend(args.checkpoint),json.loads((args.checkpoint/"config.json").read_text()),
                        Settings(request_timeout=180,socket_timeout=3))
        bearer=secrets.token_urlsafe(32)
        server=LocalHTTPServer(("127.0.0.1",0),service,bearer)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(path,body,status=200,auth=True):
            connection=http.client.HTTPConnection("127.0.0.1",server.server_address[1],timeout=200)
            headers={"Content-Type":"application/json"}
            if auth:headers["Authorization"]="Bearer "+bearer
            started=time.monotonic()
            try:
                connection.request("POST",path,json.dumps(body).encode(),headers)
                response=connection.getresponse();raw=response.read(4*1024*1024+1)
                if len(raw)>4*1024*1024:raise AssertionError("oversized response")
                result=json.loads(raw)
                if response.status!=status:raise AssertionError((path,response.status,result))
                return result,time.monotonic()-started
            finally:connection.close()
        a,b=[call("/sessions",dict(capacity=512))[0]["session_id"] for _ in range(2)]
        messages=[dict(role="system",content="Respond with text."),dict(role="user",content="Hypothetically, might reflective recursion be a function of cognition?")]
        requests=[dict(session_id=sid,messages=messages,max_new_tokens=32,sampling=dict(do_sample=sampled,top_k=1,seed=42))
                  for sid,sampled in ((a,False),(b,True))]
        def batch():
            response,duration=call("/generate_batch",dict(requests=requests))
            validate_response(response,reference)
            return response,duration
        call("/generate_batch",dict(requests=requests),status=401,auth=False)
        call("/generate_batch",dict(requests=[requests[0],requests[0]]),status=400)
        for _ in range(3):batch()
        for sid in (a,b):call(f"/sessions/{sid}/close",{})
        empty_gpu=service._submit(lambda:worker.call(dict(op="stats")),lambda:None)
        if args.require_exact_accounting and not exact_accounting(empty_gpu):
            raise AssertionError("exact engine-buffer accounting required: start the worker with LFM_ALLOCATION_METRICS=1")
        if type(empty_gpu.get('cuda_free_bytes')) is not int:raise AssertionError('missing empty-session device baseline')
        a,b=[call("/sessions",dict(capacity=512))[0]["session_id"] for _ in range(2)]
        for request,sid in zip(requests,(a,b)):request['session_id']=sid
        for _ in range(3):batch()
        report['empty_session_gpu_baseline']=empty_gpu
        # Audit only this cancellation fixture; never change HTTP input schema.
        original=worker.call_batch_cancellable
        audit=threading.Event()
        def traced(lanes,events):
            return original([dict(lane,trace_sampling=True) for lane in lanes] if audit.is_set() else lanes,events)
        worker.call_batch_cancellable=traced
        target=worker.sequence+1;proof=traces/str(target)/"lane1/2.logits.bf16";audit.set()
        with ThreadPoolExecutor(1) as executor:
            future=executor.submit(call,"/generate_batch",dict(requests=requests))
            deadline=time.monotonic()+30
            while not proof.exists() and not future.done() and time.monotonic()<deadline:time.sleep(0.001)
            if not proof.exists() or future.done():raise AssertionError("HTTP group did not expose live paired decode proof")
            accepted=call(f"/sessions/{b}/cancel",{},status=202)[0]
            if accepted.get("cancel_requested") is not True:raise AssertionError("missing cancellation request acknowledgment")
            cancelled=future.result(timeout=180)[0]
        audit.clear();report["records"].append(dict(case="targeted-cancel-B",response=cancelled));save()
        x,y=cancelled["results"]
        if (x.get("generated_ids")!=reference or x.get("cancelled") is not False or y.get("cancelled") is not True or
            y.get("cache_valid") is not False or y.get("consumed_tokens")!=0 or
            not 2<=len(y.get("generated_ids",[]))<32 or y["generated_ids"]!=reference[:len(y["generated_ids"])]):
            raise AssertionError("HTTP grouped cancellation corrupted peer/partial results")
        (args.output/"cancelled-response.txt").write_text(y["text"]+"\n")
        call(f"/sessions/{b}/cancel",{},status=409)
        batch()
        def sample(cycle):
            gpu=service._submit(lambda:worker.call(dict(op="stats")),lambda:None)
            return dict(cycle=cycle,elapsed_seconds=time.monotonic()-begin,server=process_sample(os.getpid()),
                        worker=process_sample(worker.process.pid),gpu=gpu)
        baseline=sample(-1);report["baseline"]=baseline;save()
        soak_start=time.monotonic();cycle=0
        # Both count and duration requirements must be met; strict maximum below.
        while cycle<args.cycles or time.monotonic()-soak_start<args.minimum_seconds:
            if cycle>=10000 or time.monotonic()-soak_start>args.minimum_seconds+1800:
                raise AssertionError("bounded soak did not complete")
            if cycle<5:
                started=time.monotonic()
                for request in requests:
                    response,_=call("/generate",request)
                    if response.get("generated_ids")!=reference:raise AssertionError("serial baseline ID mismatch")
                report["latencies"].append(dict(mode="serial-pair",seconds=time.monotonic()-started))
            response,duration=batch()
            report["latencies"].append(dict(mode="group",seconds=duration))
            if cycle%10==9:
                call("/generate_batch",dict(requests=[requests[0],dict(requests[1],max_new_tokens=0)]),status=400)
                for sid in (a,b):call(f"/sessions/{sid}/reset",{})
                batch()
            observed=sample(cycle);report["samples"].append(observed);save();resource_gate(baseline,observed,args.require_exact_accounting)
            cycle+=1
        report["cycles_completed"]=cycle;report["soak_seconds"]=time.monotonic()-soak_start
        report["latency_summary"]={}
        for mode in ("serial-pair","group"):
            times=[row["seconds"] for row in report["latencies"] if row["mode"]==mode]
            report["latency_summary"][mode]=dict(samples=len(times),median_seconds=statistics.median(times),
                                               median_aggregate_tokens_per_second=64/statistics.median(times))
        (args.output/"control-response.txt").write_text(response["results"][0]["text"]+"\n")
        for sid in (a,b):call(f"/sessions/{sid}/close",{})
        post_close=service._submit(lambda:worker.call(dict(op='stats')),lambda:None)
        report['post_close_gpu']=post_close;save()
        recovered,report['memory_judged_by']=post_close_recovered(empty_gpu,post_close,args.require_exact_accounting)
        if (post_close.get('sessions')!=0 or post_close.get('reserved_tokens')!=0 or not recovered):
            raise AssertionError('post-close allocations not recovered (engine-buffer bytes exactly, else device-wide within 16 MiB)')
        if service.sessions or service.failed:raise AssertionError("leaked sessions/failed service")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest()!=binary_hash or any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[p.name] for p in source_paths):
            raise AssertionError("source or binary changed during soak")
        report["status"]="passed"
    except BaseException as exc:
        report["status"],report["error"]="failed",str(exc);raise
    finally:
        try:
            if server is not None:server.shutdown();server.server_close()
            if thread is not None:thread.join(timeout=10)
            if service is not None:
                service.close()
                if service.cleanup_failed:raise RuntimeError("batch service cleanup failed")
            elif worker is not None:worker.close()
        except BaseException as exc:
            report["status"],report["cleanup_error"]="failed",str(exc);raise
        finally:save()
    print(json.dumps({k:v for k,v in report.items() if k not in ("records","samples","latencies")},indent=2))


if __name__=="__main__":main()
