"""Resident-cache protocol gates. Explicitly not GPU batching/concurrency tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lfm3_oracle_live import oracle_live_count


class Worker:
    def __init__(self,binary,checkpoint,log,env,timeout=300,trace_root=None):
        self.log=log.open("wb")
        command=[str(binary),"--experimental",str(checkpoint)]
        if trace_root is not None:command += ["--sampling-trace-root",str(trace_root)]
        self.process=subprocess.Popen(command,env=env,
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,start_new_session=True)
        self.timeout=timeout;self.buffer=bytearray();self.selector=selectors.DefaultSelector()
        self.closed=False
        self.sequence=0
        os.set_blocking(self.process.stdin.fileno(),False)
        self.selector.register(self.process.stdout,selectors.EVENT_READ)
        try:
            ready=self.read()
            if ready.get("ready") is not True or type(ready.get("protocol")) is not int or ready["protocol"]!=2:
                raise RuntimeError("invalid handshake")
        except BaseException:
            self.close();raise
    def read(self,deadline=None):
        if deadline is None:deadline=time.monotonic()+self.timeout
        while b"\n" not in self.buffer:
            if len(self.buffer)>4*1024*1024:raise RuntimeError("worker frame exceeds limit")
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.selector.select(remaining):raise TimeoutError("worker response timeout")
            block=os.read(self.process.stdout.fileno(),65536)
            if not block:raise RuntimeError("worker exited before reply")
            self.buffer.extend(block)
        raw,_,rest=self.buffer.partition(b"\n");self.buffer=bytearray(rest)
        if len(raw)>4*1024*1024:raise RuntimeError("worker frame exceeds limit")
        result=json.loads(raw)
        if not isinstance(result,dict):raise RuntimeError("worker response must be object")
        return result
    def call(self,request,success=True):
        if "rpc_id" in request:raise ValueError("rpc_id is owned by transport")
        if len(json.dumps(request,allow_nan=False).encode())+32>4*1024*1024:raise ValueError("request exceeds frame limit")
        self.sequence+=1
        payload=(json.dumps(dict(request,rpc_id=self.sequence),allow_nan=False)+"\n").encode()
        deadline=time.monotonic()+self.timeout
        try:
            with selectors.DefaultSelector() as writable:
                writable.register(self.process.stdin,selectors.EVENT_WRITE)
                offset=0
                while offset<len(payload):
                    remaining=deadline-time.monotonic()
                    if remaining<=0 or not writable.select(remaining):raise TimeoutError("worker request write timeout")
                    try:offset+=os.write(self.process.stdin.fileno(),payload[offset:])
                    except BlockingIOError:continue
            response=self.read(deadline)
        except TimeoutError:
            self.close();raise
        if type(response.get("rpc_id")) is not int or response["rpc_id"]!=self.sequence or self.buffer:
            self.close();raise RuntimeError("uncorrelated/trailing worker response")
        del response["rpc_id"]
        if response.get("ok") is not success:raise AssertionError((request,response))
        return response
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait()
        self.selector.close();self.log.close()
        self.process.stdin.close();self.process.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--worker",type=Path,required=True)
    p.add_argument("--single",type=Path,required=True)
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--long-request",type=Path,help="Optional accepted32767-token prompt for resident32K budget gate")
    p.add_argument("--oracle",type=Path,help="Standalone text-sampling-oracle; required for non-preflight gates")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not args.preflight_only and args.oracle is None:raise ValueError("full resident gate requires sampling oracle")
    env=dict(os.environ)
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH","")
    records=[]
    worker_hash=hashlib.sha256(args.worker.read_bytes()).hexdigest()
    single_hash=hashlib.sha256(args.single.read_bytes()).hexdigest()
    oracle_hash=hashlib.sha256(args.oracle.read_bytes()).hexdigest() if args.oracle else None
    worker=Worker(args.worker,args.checkpoint,args.output/"worker.stderr.log",env,
                  trace_root=args.output/"sampling-traces" if args.oracle else None)
    def call(request,success=True):
        response=worker.call(request,success)
        # Large vocabulary logits are hashed separately, not repeated in reports.
        saved={k:v for k,v in response.items() if k!="last_logits"}
        records.append(dict(request=request,response=saved))
        return response
    try:
        call(dict(op="create",session_id="A",capacity=512))
        call(dict(op="create",session_id="B",capacity=512))
        call(dict(op="create",session_id="C",capacity=512),False)
        call(dict(op="create",session_id="A",capacity=512),False)
        call(dict(op="create",session_id="../bad",capacity=512),False)
        call(dict(op="generate",session_id="A",input_ids=[1],max_new_tokens=512),False)
        call(dict(op="generate",session_id="A",input_ids=[-1],max_new_tokens=1),False)
        call(dict(op="generate",session_id="A",input_ids=[125029],max_new_tokens=1),False)
        call(dict(op="generate",session_id="A",input_ids=[1],max_new_tokens=1,temperature=0),False)
        call(dict(op="generate",session_id="A",input_ids=[1],max_new_tokens=1,trace_directory="/tmp/forbidden"),False)
        stats=call(dict(op="stats"))
        if stats.get("weights_loaded") is not False or stats.get("reserved_tokens")!=1024:
            raise AssertionError("rejection allocated weights or changed quota")
        if not args.preflight_only:
            base=json.loads(args.request.read_text());prompt=base["input_ids"]
            model_config=json.loads((args.checkpoint/"config.json").read_text())
            text_config=model_config.get("text_config",model_config)
            vocabulary=text_config["vocab_size"]
            comparisons=[]
            def single_logits(name,input_ids,teacher,count=0,capacity=512):
                directory=args.output/name;directory.mkdir()
                request=dict(input_ids=input_ids,teacher_ids=teacher,max_new_tokens=count,capacity=capacity,trace_logits_only=True,
                             trace_directory=str(directory/"trace"))
                if capacity>4096:request["experimental_long_context"]=True
                (directory/"request.json").write_text(json.dumps(request))
                with (directory/"single.log").open("w") as stream:
                    subprocess.run([str(args.single),"--experimental",str(args.checkpoint),str(directory/"request.json"),
                                    str(directory/"result.json")],env=env,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
                output=json.loads((directory/"result.json").read_text())
                step=max(0,len(output["generated_ids"])-1) if count else len(teacher)
                generated=output["generated_ids"] if count else [output["teacher_argmax"][-1]]
                exact=directory/"trace"/f"{step}.logits.f32"
                if exact.exists():  # FP32-logit engine (native GEMM): the exact values it returns
                    raw=exact.read_bytes()
                    if len(raw)!=vocabulary*4:raise AssertionError("oracle logit vocabulary length mismatch")
                    return dict(logits=[x[0] for x in struct.iter_unpack("<f",raw)],generated=generated)
                raw=(directory/"trace"/f"{step}.logits.bf16").read_bytes()
                if len(raw)!=vocabulary*2:raise AssertionError("oracle logit vocabulary length mismatch")
                return dict(logits=[struct.unpack("<f",struct.pack("<I",x[0]<<16))[0] for x in struct.iter_unpack("<H",raw)],
                            generated=generated)
            expected0=single_logits("initial",prompt,[])
            expected1=single_logits("continued",prompt,[7])
            expected_b=single_logits("other",prompt+[8],[])
            expected_mismatch=single_logits("replacement",[1,2,3],[])
            def generate(name,id,inputs,expected,reused,count=1):
                result=call(dict(op="generate",session_id=id,input_ids=inputs,max_new_tokens=count,return_logits=True))
                diagnostic=dict(generated_ids=result.get("generated_ids"),expected_ids=expected["generated"],
                                reused_tokens=result.get("reused_tokens"),expected_reused_tokens=reused,
                                raw_float_logits_equal=result.get("last_logits")==expected["logits"])
                (args.output/(name+".comparison.json")).write_text(json.dumps(diagnostic,indent=2)+"\n")
                if (result.get("reused_tokens")!=reused or result.get("last_logits")!=expected["logits"] or
                    result.get("generated_ids")!=expected["generated"]):
                    raise AssertionError(f"{name}: prefix reuse/logits mismatch")
                consumed=len(inputs)+len(result["generated_ids"])-1
                snapshot=call(dict(op="inspect",session_id=id))
                if (result.get("consumed_tokens")!=consumed or snapshot.get("consumed_tokens")!=consumed or
                    snapshot.get("transcript_tokens")!=consumed+1 or snapshot.get("cache_valid") is not True or
                    snapshot.get("inflight") is not False):
                    raise AssertionError(f"{name}: committed transcript/final-unconsumed-token mismatch")
                raw=json.dumps(result["last_logits"],separators=(",",":")).encode()
                comparisons.append(dict(case=name,bitwise_float_values=True,reused_tokens=reused,generated_ids=result["generated_ids"],
                                        consumed_tokens=consumed,final_prediction_unconsumed=True,
                                        logits_sha256=hashlib.sha256(raw).hexdigest()))
                return result
            generate("A-initial","A",prompt,expected0,0)
            generate("B-isolated","B",prompt+[8],expected_b,0)
            before=call(dict(op="inspect",session_id="A"))
            call(dict(op="generate",session_id="A",input_ids=prompt,max_new_tokens=1,top_p=0),False)
            after=call(dict(op="inspect",session_id="A"))
            if before!=after:raise AssertionError("invalid metadata mutated A cache")
            generate("A-after-B-and-error","A",prompt+[7],expected1,len(prompt))
            generate("A-prefix-mismatch","A",[1,2,3],expected_mismatch,0)
            call(dict(op="reset",session_id="A"))
            inspected=call(dict(op="inspect",session_id="A"))
            if inspected["cache_valid"] or inspected["consumed_tokens"]:raise AssertionError("reset did not clear state")
            generate("A-reset","A",prompt,expected0,0)
            multiple=single_logits("multi-token",prompt,[],count=4)
            if len(multiple["generated"])!=4:raise AssertionError("fixture must exercise multiple decode forwards")
            generate("A-multi-token","A",prompt,multiple,0,count=4)
            continuation=prompt+multiple["generated"]+[7]
            continued=single_logits("after-generated",prompt,multiple["generated"]+[7])
            generate("A-reuse-generated-prefix","A",continuation,continued,len(prompt)+3)
            b_resumed=single_logits("B-resume",prompt+[8],[9])
            generate("B-after-A-advances-and-resets","B",prompt+[8,9],b_resumed,len(prompt)+1)
            from lfm3_text_frontend import load_frontend
            frontend=load_frontend(args.checkpoint)
            early_prompt=frontend.encode_messages([dict(role="system",content="Respond with text."),
                dict(role="user",content="What is 17 plus 25? Answer briefly.")],512,16)
            early=single_logits("early-stop",early_prompt,[],count=16)
            eos=text_config["eos_token_id"];stops=eos if isinstance(eos,list) else [eos]
            if "text_end_token_id" in model_config:stops=stops+[model_config["text_end_token_id"]]
            if not early["generated"] or len(early["generated"])>=16 or early["generated"][-1] not in stops:
                raise AssertionError("fixture must exercise natural early stopping")
            generate("A-early-stop","A",early_prompt,early,0,count=16)
            sampling_policy=dict(do_sample=True,temperature=0.7,top_k=42,top_p=0.963,min_p=0.01,repetition_penalty=1.1,seed=42)
            stochastic=dict(op="generate",session_id="A",input_ids=prompt,max_new_tokens=32,trace_sampling=True,**sampling_policy)
            sampled=call(stochastic)
            trace_id=sampled.get("sampling_trace")
            if type(trace_id) is not int or trace_id<1:raise AssertionError("missing resident sampling trace")
            oracle_request=dict(stochastic,suppress_tokens=base["suppress_tokens"])
            request_path=args.output/"resident-sampling.request.json"
            result_path=args.output/"resident-sampling.result.json"
            request_path.write_text(json.dumps(oracle_request,indent=2)+"\n")
            result_path.write_text(json.dumps(sampled,indent=2)+"\n")
            with (args.output/"resident-sampling-oracle.log").open("w") as stream:
                try:
                    # G6: the oracle replays the worker's bound, counting with the engine's own C++ function
                    subprocess.run([str(args.oracle),str(request_path),str(result_path),str(args.output/"sampling-traces"/str(trace_id)),
                                    str(vocabulary),str(args.checkpoint/"tokenizer.json")],
                                   env=env,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
                except subprocess.CalledProcessError as exc:raise AssertionError("resident sampling logit replay failed") from exc
            live=oracle_live_count(args.output/"resident-sampling-oracle.log")
            # Runtime evidence for G6: the worker reports the bound it used, and it must be the oracle's count
            # (replay alone cannot show it here: dead ids never rank inside top_k=42).
            if sampled.get("live_token_count")!=live:raise AssertionError("worker and oracle disagree on the live token count")
            if any(t>=live for t in sampled["generated_ids"]):raise AssertionError("worker emitted an id the tokenizer does not define")
            # Reset/interleaving must not advance the next request's RNG stream.
            call(dict(op="reset",session_id="A"))
            generate("B-between-sampled-replays","B",prompt+[8],expected_b,0)
            repeat=call(dict(stochastic,trace_sampling=False))
            if repeat.get("generated_ids")!=sampled.get("generated_ids") or not sampled.get("generated_ids"):
                raise AssertionError("resident sampled seed/reset/interleaving reproducibility failed")
            (args.output/"comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
        call(dict(op="close",session_id="A"));call(dict(op="close",session_id="B"))
        call(dict(op="create",session_id="large",capacity=32768))
        call(dict(op="create",session_id="overflow",capacity=1),False)
        if args.long_request and not args.preflight_only:
            long_prompt=json.loads(args.long_request.read_text())["input_ids"]
            if len(long_prompt)!=32767:raise ValueError("long request must contain32767 prompt tokens")
            expected_long=single_logits("resident32k-reference",long_prompt,[],capacity=32768)
            long_result=generate("resident32k-final-prediction","large",long_prompt,expected_long,0)
            before=call(dict(op="inspect",session_id="large"))
            call(dict(op="generate",session_id="large",input_ids=long_prompt+long_result["generated_ids"],max_new_tokens=1),False)
            after=call(dict(op="inspect",session_id="large"))
            if before!=after:raise AssertionError("32K overflow rejection mutated resident cache")
            (args.output/"comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
        call(dict(op="close",session_id="large"))
        call(dict(op="generate",session_id="A",input_ids=[1],max_new_tokens=1),False)
        call(dict(op="shutdown"))
        if worker.process.wait(timeout=10)!=0:raise AssertionError("unclean worker exit")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest()!=worker_hash or hashlib.sha256(args.single.read_bytes()).hexdigest()!=single_hash:
            raise AssertionError("worker/single binary changed during checks")
        if args.oracle and hashlib.sha256(args.oracle.read_bytes()).hexdigest()!=oracle_hash:raise AssertionError("sampling oracle changed")
        report=dict(status="passed",preflight_only=args.preflight_only,worker_sha256=worker_hash,
                    single_sha256=single_hash,records=records,
                    resident32k_checked=bool(args.long_request and not args.preflight_only),
                    resident_sampling_logit_replay=not args.preflight_only,oracle_sha256=oracle_hash,
                    scope="single-threaded resident cache isolation/prefix/lifecycle; not GPU batching or concurrent cancellation")
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))
    except BaseException as exc:
        (args.output/"summary.json").write_text(json.dumps(dict(status="failed",error=str(exc),records=records,
            worker_sha256=worker_hash,single_sha256=single_hash),indent=2)+"\n")
        raise
    finally:worker.close()


if __name__=="__main__":main()
