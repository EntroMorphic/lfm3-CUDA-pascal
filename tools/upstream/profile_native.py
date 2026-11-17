"""Capture opt-in native NVTX/CUDA component timings and verify output IDs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_profile import analyze
from benchmark_common import assert_same_output, run_saved, sha256, validate_native, write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,required=True,help="Instrumented binary built with LIQUID_PROFILE_NVTX=1.")
    p.add_argument("--reference-runtime",type=Path,required=True,help="Uninstrumented, validated binary for exact ID comparison.")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--prompt-tokens",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--max-new-tokens",type=int,default=512)
    p.add_argument("--context-length",type=int,default=8192)
    p.add_argument("--temperature",type=float,default=.528)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--timeout",type=float,default=3600)
    p.add_argument("--reference-math",action="store_true")
    args=p.parse_args(argv)
    if (not 1 <= args.batch_size <= 65535 or args.max_new_tokens < 1 or args.context_length < 1 or
        not math.isfinite(args.temperature) or args.temperature < 0 or not 0 <= args.seed < 2**64 or
        not math.isfinite(args.timeout) or args.timeout <= 0): p.error("Invalid generation/profile settings")
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    summary=dict(complete=False, scope="Diagnostic eager component attribution; profiled rates are not benchmark throughput.",
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        binary_sha256=dict(instrumented=sha256(args.runtime),reference=sha256(args.reference_runtime)),
        prompt_sha256=sha256(args.prompt_tokens),config_sha256=sha256(args.model/"config.json"))
    try:
        vocabulary=json.loads((args.model/"config.json").read_text())["vocab_size"]
        def command(binary,directory):
            cmd=[str(binary.resolve()),"--model",str(args.model.resolve()),"--prompt-tokens",str(args.prompt_tokens.resolve()),
                "--batch-size",str(args.batch_size),"--temperature",str(args.temperature),"--seed",str(args.seed),
                "--context-length",str(args.context_length),"--max-new-tokens",str(args.max_new_tokens),
                "--top-k","0","--top-p","1","--min-p","0","--repetition-penalty","1",
                "--tokens-output",str(directory/"tokens"),"--no-cuda-graph"]
            if args.reference_math: cmd.append("--attention-math-dynamic")
            return cmd
        reference=output/"reference"
        run_saved(command(args.reference_runtime,reference),reference,args.timeout)
        expected=validate_native(reference,args.batch_size,vocabulary,args.max_new_tokens)
        warm=output/"instrumented-warmup"
        run_saved(command(args.runtime,warm),warm,args.timeout)
        assert_same_output(expected,validate_native(warm,args.batch_size,vocabulary,args.max_new_tokens))
        trace=output/"trace"
        prefix=output/"native"
        run_saved(["nsys","profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
                   "--cuda-graph-trace=node","--output="+str(prefix),*command(args.runtime,trace)],trace,args.timeout)
        traced=validate_native(trace,args.batch_size,vocabulary,args.max_new_tokens)
        assert_same_output(expected,traced)
        run_saved(["nsys","export","--type=sqlite","--output="+str(prefix.with_suffix('.sqlite')),
                   str(prefix.with_suffix('.nsys-rep'))],output/"export",args.timeout)
        report=analyze(prefix.with_suffix('.sqlite')); write_json(output/"components.json",report)
        if report["unmatched_decode_activities"]:
            raise ValueError("Unattributed GPU activity overlaps decode; inspect the retained trace")
        if report["unmatched_gpu_activities"]:
            # Setup/library activities may lack launch records. Keep them visible;
            # decode classification completeness is separately required.
            summary["unmatched_gpu_activities"]=report["unmatched_gpu_activities"]
        summary.update(complete=True,reference=expected,traced=traced)
    except BaseException as error:
        summary["error"]=repr(error); raise
    finally: write_json(output/"summary.json",summary)


if __name__ == "__main__": main()
