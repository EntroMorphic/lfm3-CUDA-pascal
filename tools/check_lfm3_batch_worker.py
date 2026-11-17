"""Native grouped-decode acceptance against serial IDs and every logit bit.

Run exclusively on the GPU. This gate does not waive sanitizer or long-soak gates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

from lfm3_oracle_live import oracle_live_count
from lfm3_worker_client import WorkerClient
from lfm3_owned_process import owned_main


def require_live_agreement(results,live):
    """Runtime evidence for G6: every lane reports the token-choice bound it used, which must equal the
    oracle's count (both computed by lfm::tokenizer_token_count), and no lane emitted an id at or above it."""
    if not results:raise AssertionError('no lane results to check the live token count against')
    for lane_result in results:
        if lane_result.get('live_token_count')!=live:raise AssertionError('worker lane and oracle disagree on the live token count')
        if any(t>=live for t in lane_result.get('generated_ids',[])):raise AssertionError('worker lane emitted an id the tokenizer does not define')


def require_group(result, lanes, expected, require_pairs=True):
    if (not isinstance(lanes,list) or len(lanes)!=2 or not isinstance(expected,list) or len(expected)!=2 or
        any(not isinstance(row.get('generated_ids'),list) or not row['generated_ids'] for row in expected)):
        raise AssertionError("need two nonempty independent references")
    if (result.get("ok") is not True or type(result.get("paired_decode_calls")) is not int or
        result["paired_decode_calls"] < (1 if require_pairs else 0) or
        result.get("gpu_batching") is not (result["paired_decode_calls"] > 0)):
        raise AssertionError("group execution/real-pair metadata mismatch")
    results = result.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise AssertionError("missing lane results")
    if result['paired_decode_calls']!=min(len(row['generated_ids']) for row in expected)-1:
        raise AssertionError("paired-call count disagrees with completed decode steps")
    for lane, reference, actual in zip(lanes, expected, results):
        if (not isinstance(actual,dict) or actual.get("session_id") != lane["session_id"] or actual.get("ok") is not True or
            not isinstance(actual.get("generated_ids"),list) or not actual["generated_ids"] or
            not all(type(token) is int for token in actual["generated_ids"]) or
            actual.get("generated_ids") != reference["generated_ids"] or
            actual.get("do_sample") is not lane.get("do_sample", False) or
            type(actual.get("seed")) is not int or actual.get("seed") != lane.get("seed", 0) or
            actual.get("cancelled") is not False or actual.get("cache_valid") is not True or
            type(actual.get("paired_decode_calls")) is not int or
            actual["paired_decode_calls"] != result["paired_decode_calls"] or
            actual.get("gpu_batching") is not result["gpu_batching"]):
            raise AssertionError("per-lane output/sampler/participation metadata mismatch")
    return results


def compare_traces(root, serial, grouped, count):
    if type(count) is not int or count<1:
        raise AssertionError("trace proof needs positive coverage")
    def trace_path(value):
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise AssertionError("missing trace identity")
        path = root / str(value)
        if not path.resolve().is_relative_to(root.resolve()):
            raise AssertionError("unsafe trace identity")
        return path
    left, right = trace_path(serial), trace_path(grouped)
    if left.resolve()==right.resolve():raise AssertionError("reference and grouped trace identities alias")
    for index in range(count):
        name = f"{index}.logits.bf16"
        if os.path.samefile(left/name,right/name):raise AssertionError("reference and grouped trace files alias")
        a, b = (left/name).read_bytes(), (right/name).read_bytes()
        if len(a) != 256000 or a != b:
            raise AssertionError(f"serial/group full-logit bit mismatch at step {index}")
        exact = f"{index}.logits.f32"  # FP32-logit engine: compare the exact values too, and require both
        if (left/exact).exists() != (right/exact).exists():
            raise AssertionError(f"serial/group exact-trace presence mismatch at step {index}")
        if (left/exact).exists():
            a, b = (left/exact).read_bytes(), (right/exact).read_bytes()
            if len(a) != 512000 or a != b:
                raise AssertionError(f"serial/group exact FP32 logit mismatch at step {index}")
    return count


def validate_cancelled_history(before, after):
    if (type(before.get('transcript_tokens')) is not int or before['transcript_tokens']<1 or
        type(before.get('committed_consumed_tokens')) is not int or before['committed_consumed_tokens']<1):
        raise AssertionError('cancellation fixture needs a nonempty committed history')
    if (after.get('cache_valid') is not False or type(after.get('consumed_tokens')) is not int or after['consumed_tokens']!=0 or
        after.get('inflight') is not False or after.get('transcript_tokens')!=before['transcript_tokens'] or
        after.get('committed_consumed_tokens')!=before['committed_consumed_tokens']):
        raise AssertionError('cancelled group lost committed history or retained physical state')


@owned_main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--long-request", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trace_root = args.output/"traces"
    prompt = json.loads(args.request.read_text())["input_ids"]
    digest = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    oracle_hash=hashlib.sha256(args.oracle.read_bytes()).hexdigest()
    records = []
    report = dict(status="running", worker_sha256=digest, oracle_sha256=oracle_hash, records=records,
        scope="native grouped decoding, serial full-logit bits/IDs, independent policies, retirement, per-lane cancellation, stale tags and aggregate32K; sanitizer/HTTP/soak acceptance separate")
    def save():
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13/lib")+":"+env.get("LD_LIBRARY_PATH", "")
    worker = None
    try:
        worker = WorkerClient(args.worker, args.checkpoint, timeout=180, env=env, cancellation=True,
                              batching=True, sampling_trace_root=trace_root)
        def call(request):
            response = worker.call(request)
            records.append(dict(request=request, response=response));save()
            if response.get("ok") is not True:
                raise AssertionError("native operation rejected")
            return response
        def reset():
            for sid in ("A", "B"):
                call(dict(op="reset", session_id=sid))
        def pair(lanes, name):
            reset()
            reference = [call(dict(op="generate", **lane)) for lane in lanes]
            reset()
            grouped = worker.call_batch_cancellable(lanes, [threading.Event(), threading.Event()])
            records.append(dict(case=name, response=grouped));save()
            results = require_group(grouped, lanes, reference)
            for lane, expected, actual in zip(lanes, reference, results):
                if (actual.get("consumed_tokens") != len(lane["input_ids"])+len(actual["generated_ids"])-1 or
                    actual.get("reused_tokens") != 0):
                    raise AssertionError("fresh group cache accounting mismatch")
                report["bitwise_logit_steps"] = report.get("bitwise_logit_steps", 0)+compare_traces(
                    trace_root, expected.get("sampling_trace"), actual.get("sampling_trace"), len(actual["generated_ids"]))
            save()
            return reference, grouped
        for sid in ("A", "B"):
            call(dict(op="create", session_id=sid, capacity=512))
        lanes = [dict(session_id="A", input_ids=prompt, max_new_tokens=8, trace_sampling=True),
                 dict(session_id="B", input_ids=prompt+[8]*257, max_new_tokens=17, trace_sampling=True,
                      do_sample=True, temperature=0.7, top_k=42, top_p=0.963, min_p=0.01, repetition_penalty=1.1, seed=42)]
        _, stochastic_group=pair(lanes, "mixed-length-policy-retire-A")
        model=json.loads((args.checkpoint/'config.json').read_text())
        suppressed=[model[name] for name in ("audio_token_id","audio_out_token_id","image_token_id","video_token_id","mixed_start_token_id","audio_start_token_id")]
        request_path=args.output/'sampled-lane.request.json';result_path=args.output/'sampled-lane.result.json'
        request_path.write_text(json.dumps(dict(lanes[1],suppress_tokens=suppressed),indent=2)+'\n')
        sampled=stochastic_group['results'][1]
        result_path.write_text(json.dumps(sampled,indent=2)+'\n')
        with (args.output/'sampled-lane.oracle.log').open('w') as stream:
            # the worker chooses among the ids the tokenizer defines; the oracle replays the same rule, counting
            # with the engine's own C++ function from the checkpoint's tokenizer.json
            subprocess.run([str(args.oracle),str(request_path),str(result_path),str(trace_root/str(sampled['sampling_trace'])),
                            str(model['text_config']['vocab_size']),str(args.checkpoint/'tokenizer.json')],
                           check=True,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
        live=oracle_live_count(args.output/'sampled-lane.oracle.log');report['live_token_count']=live
        require_live_agreement(stochastic_group['results'],live)
        if any(t>=live for t in sampled['generated_ids']):raise AssertionError('worker emitted an id the tokenizer does not define')
        report['independent_non_greedy_logit_replay']=True;save()
        lanes = [dict(session_id="A", input_ids=prompt, max_new_tokens=32, trace_sampling=True),
                 dict(session_id="B", input_ids=prompt+[8], max_new_tokens=6, trace_sampling=True)]
        reference, grouped = pair(lanes, "retire-B")
        # Reuse both caches with independently committed histories.
        continued = [dict(lane, input_ids=lane["input_ids"]+expected["generated_ids"], max_new_tokens=3)
                     for lane, expected in zip(lanes, reference)]
        serial = [call(dict(op="generate", **lane)) for lane in continued]
        reset()
        worker.call_batch_cancellable(lanes, [threading.Event(), threading.Event()])
        resumed = worker.call_batch_cancellable(continued, [threading.Event(), threading.Event()])
        records.append(dict(case="resume-both", response=resumed));save()
        resumed_lanes = require_group(resumed, continued, serial)
        for lane, before, expected, actual in zip(lanes, reference, serial, resumed_lanes):
            if actual["reused_tokens"] != len(lane["input_ids"])+len(before["generated_ids"])-1:
                raise AssertionError("group cache reuse mismatch")
            report["bitwise_logit_steps"] += compare_traces(trace_root, expected["sampling_trace"], actual["sampling_trace"], len(actual["generated_ids"]))
        reset()
        eos_lanes=[dict(session_id='A',input_ids=prompt,max_new_tokens=128),
                   dict(session_id='B',input_ids=prompt,max_new_tokens=32)]
        eos_reference=[call(dict(op='generate',**lane)) for lane in eos_lanes]
        if not 32<len(eos_reference[0]['generated_ids'])<128:
            raise AssertionError('EOS fixture must stop after peer retirement and before budget')
        stop_ids=model.get('text_config',model).get('eos_token_id',[])
        if type(stop_ids) is int:stop_ids=[stop_ids]
        stop_ids=list(stop_ids)+[model['text_end_token_id']]
        if eos_reference[0]['generated_ids'][-1] not in stop_ids:
            raise AssertionError('early finish was not a configured stop token')
        reset()
        eos_group=worker.call_batch_cancellable(eos_lanes,[threading.Event(),threading.Event()])
        require_group(eos_group,eos_lanes,eos_reference)
        records.append(dict(case='EOS-after-peer-retirement',response=eos_group));save()
        before = [call(dict(op="inspect", session_id=sid)) for sid in ("A", "B")]
        invalid = worker.call(dict(op="generate_batch", lanes=[lanes[0], dict(lanes[1], max_new_tokens=0)]))
        if invalid.get("ok") is not False:
            raise AssertionError("invalid second lane was accepted")
        if before != [call(dict(op="inspect", session_id=sid)) for sid in ("A", "B")]:
            raise AssertionError("invalid second lane mutated state")
        # Cancel B only after both lanes have completed multiple decode steps.
        reset()
        cancel_lanes = [dict(lane, max_new_tokens=32) for lane in lanes]
        serial = [call(dict(op="generate", **lane)) for lane in cancel_lanes]
        # Keep nonempty committed histories: a cancelled fresh prefill must not
        # erase them, even though the prompt forces physical cache replacement.
        history = call(dict(op="inspect", session_id="B"))
        target = worker.sequence+1
        proof = trace_root/str(target)/"lane1/2.logits.bf16"
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(worker.call_batch_cancellable, cancel_lanes, [threading.Event(), threading.Event()])
            deadline = time.monotonic()+30
            while not proof.exists() and not future.done() and time.monotonic()<deadline:
                time.sleep(0.001)
            if not proof.exists() or future.done() or not worker.cancel("B"):
                raise AssertionError("no observed live paired work for targeted cancellation")
            cancelled = future.result(timeout=180)
        records.append(dict(case="cancel-B-while-A-continues", response=cancelled));save()
        a, b = cancelled["results"]
        if (a.get("generated_ids") != serial[0]["generated_ids"] or a.get("cancelled") is not False or
            b.get("cancelled") is not True or b.get("cache_valid") is not False or b.get("consumed_tokens") != 0 or
            not 2 <= len(b.get("generated_ids", [])) < 32 or b["generated_ids"] != serial[1]["generated_ids"][:len(b["generated_ids"])]):
            raise AssertionError("targeted cancellation damaged lane outputs")
        after = call(dict(op="inspect", session_id="B"))
        validate_cancelled_history(history,after)
        # A late tagged frame must never abort a later single RPC.
        os.write(worker.cancel_write_fd, struct.pack("<Q", (2<<32)|target))
        recovered = call(dict(op="generate", **cancel_lanes[1]))
        if recovered["generated_ids"] != serial[1]["generated_ids"]:
            raise AssertionError("stale group tag affected fresh single generation")
        event = threading.Event();event.set()
        pre = worker.call_batch_cancellable(cancel_lanes, [event, threading.Event()])
        records.append(dict(case="cancel-A-before-dispatch", response=pre));save()
        if (pre["results"][0].get("cancelled") is not True or pre["results"][0]["generated_ids"] != [] or
            pre["results"][1]["generated_ids"] != serial[1]["generated_ids"] or pre.get("gpu_batching") is not False):
            raise AssertionError("pre-dispatch lane cancellation/fallback metadata mismatch")
        for sid in ("A", "B"):
            call(dict(op="close", session_id=sid))
        if args.long_request:
            long_ids = json.loads(args.long_request.read_text())["input_ids"][:32763]
            if len(long_ids)!=32763:
                raise ValueError("need at least32763 long-context prompt IDs")
            call(dict(op="create", session_id="A", capacity=3))
            call(dict(op="create", session_id="B", capacity=32765))
            pair([dict(session_id="A", input_ids=[1], max_new_tokens=2, trace_sampling=True),
                  dict(session_id="B", input_ids=long_ids, max_new_tokens=2, trace_sampling=True)], "aggregate32K")
            for sid in ("A", "B"):
                call(dict(op="close", session_id=sid))
            report["aggregate32k_checked"] = True
        else:
            report["aggregate32k_checked"] = False
        call(dict(op="shutdown"))
        if worker.process.wait(timeout=10)!=0:
            raise AssertionError("worker did not exit cleanly after grouped gate")
        if hashlib.sha256(args.worker.read_bytes()).hexdigest()!=digest or hashlib.sha256(args.oracle.read_bytes()).hexdigest()!=oracle_hash:
            raise AssertionError("worker/oracle binary changed")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        except BaseException as exc:
            report["status"],report["cleanup_error"]="failed",str(exc)
            raise
        finally:
            save()
    print(json.dumps({key:value for key,value in report.items() if key!="records"}, indent=2))


if __name__=="__main__":
    main()
