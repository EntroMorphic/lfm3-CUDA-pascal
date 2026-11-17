"""Counterexamples for resident acceptance assertions; no real worker/GPU."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types

import pytest

TOOL=Path(__file__).resolve().parents[1]/"tools/check_lfm3_worker.py"


@pytest.mark.parametrize("damage",[None,"empty-generation","wrong-generation","consumed","transcript","B-reuse","short-oracle","sampling-replay","live-mismatch","fp32-exact","fp32-short"])
def test_worker_gate_cannot_credit_broken_generation(tmp_path,monkeypatch,damage):
    spec=importlib.util.spec_from_file_location("worker_acceptance",TOOL)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    checkpoint=tmp_path/"checkpoint";checkpoint.mkdir()
    (checkpoint/"config.json").write_text(json.dumps(dict(vocab_size=8,eos_token_id=2)))
    for name in ["single","worker","oracle"]:(tmp_path/name).write_text("test fixture")
    request=tmp_path/"request.json";request.write_text(json.dumps(dict(input_ids=[1,2,3],suppress_tokens=[])))
    output=tmp_path/"evidence"
    monkeypatch.setattr(sys,"argv",[str(TOOL),"--checkpoint",str(checkpoint),"--worker",str(tmp_path/"worker"),
                                   "--single",str(tmp_path/"single"),"--request",str(request),"--output",str(output),"--oracle",str(tmp_path/"oracle")])
    # G4 "fp32-*": an FP32-logit engine returns values FP32 holds exactly but BF16 cannot (i + 2^-12) and
    # traces them exactly as STEP.logits.f32; the rounded .bf16 trace alone could not match them.
    logits=[i+2.0**-12 for i in range(8)] if damage and damage.startswith("fp32") else list(map(float,range(8)))
    class FakeWorker:
        def __init__(self,*args,**kwargs):self.sessions={};self.process=types.SimpleNamespace(wait=lambda timeout:0)
        def close(self):pass
        def call(self,req,success=True):
            if not success:return dict(ok=False,error="fixture rejection")
            op=req["op"];id=req.get("session_id")
            if op=="create":self.sessions[id]=dict(capacity=req["capacity"],consumed=[],transcript=[])
            elif op=="close":del self.sessions[id]
            elif op=="reset":self.sessions[id].update(consumed=[],transcript=[])
            elif op=="stats":return dict(ok=True,weights_loaded=False,reserved_tokens=1024)
            elif op=="inspect":
                s=self.sessions[id]
                return dict(ok=True,consumed_tokens=len(s["consumed"]),
                            transcript_tokens=len(s["transcript"])+(damage=="transcript" and bool(s["consumed"])),
                            cache_valid=bool(s["consumed"]),inflight=False)
            elif op=="generate":
                s=self.sessions[id];inputs=req["input_ids"];old=s["consumed"]
                reused=len(old) if old and len(old)<len(inputs) and inputs[:len(old)]==old else 0
                generated=[7,2] if inputs==[4,4] else [7]*req["max_new_tokens"]
                s["transcript"]=inputs+generated;s["consumed"]=(inputs+generated)[:-1]
                return dict(ok=True,generated_ids=[] if damage=="empty-generation" else [6] if damage=="wrong-generation" else generated,
                            sampling_trace=1,
                            # G6: the worker reports its token-choice bound; "live-mismatch" reports the output rows instead
                            live_token_count=128000 if damage=="live-mismatch" else 125033,
                            consumed_tokens=len(s["consumed"])+(damage=="consumed"),last_logits=logits,
                            reused_tokens=0 if damage=="B-reuse" and id=="B" else reused)
            return dict(ok=True)
    monkeypatch.setattr(gate,"Worker",FakeWorker)
    def single(command,**kwargs):
        if command[0]==str(tmp_path/"oracle"):
            if damage=="sampling-replay":raise gate.subprocess.CalledProcessError(1,command)
            # The real oracle reports the token count it bounded the replay with (G6); the gate requires it.
            kwargs["stdout"].write("PASS: replayed sampled tokens; non-greedy choice verified; live_token_count 125033\n")
            return
        req=json.loads(Path(command[-2]).read_text());dest=Path(command[-1])
        count=req["max_new_tokens"]
        generated=([7,2] if req["input_ids"]==[4,4] else [7]*count) if count else []
        result=dict(generated_ids=generated,teacher_argmax=[7]*(len(req["teacher_ids"])+1))
        dest.write_text(json.dumps(result))
        trace=Path(req["trace_directory"]);trace.mkdir()
        step=max(0,len(generated)-1) if count else len(req["teacher_ids"])
        raw=b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",x))[0]>>16) for x in logits)
        (trace/f"{step}.logits.bf16").write_bytes(raw[:-2] if damage=="short-oracle" else raw)
        if damage and damage.startswith("fp32"):
            exact=struct.pack("<8f",*logits)
            (trace/f"{step}.logits.f32").write_bytes(exact[:-4] if damage=="fp32-short" else exact)
    monkeypatch.setattr(gate.subprocess,"run",single)
    monkeypatch.setitem(sys.modules,"lfm3_text_frontend",types.SimpleNamespace(
        load_frontend=lambda path:types.SimpleNamespace(encode_messages=lambda *args:[4,4])))
    if damage and damage!="fp32-exact":
        with pytest.raises(AssertionError):gate.main()
        assert json.loads((output/"summary.json").read_text())["status"]=="failed"
    else:
        gate.main()
        assert json.loads((output/"summary.json").read_text())["status"]=="passed"
