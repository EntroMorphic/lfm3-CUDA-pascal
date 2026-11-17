"""Compare real component timings with explicitly conditional resource models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from benchmark_common import write_json


def resource_reference(rows: list[dict], rooflines: list[str]) -> dict:
    streams = [r for r in rows if r["operation"] == "stream_read"]
    if not streams: raise ValueError("Missing streaming measurements")
    if any(not math.isfinite(r["us"]) or r["us"] <= 0 or r["logical_bytes"] <= 0 for r in streams):
        raise ValueError("Invalid streaming measurements")
    bandwidth = statistics.median(r["logical_bytes"] / (r["us"] * 1000) for r in streams)
    compute = []
    for text in rooflines:
        values = re.findall(r"^bf16_tensor_tflops: (.+)$", text, re.M)
        if len(values) != 1: raise ValueError("Missing/duplicate BF16 compute measurement")
        value = float(values[0])
        if not math.isfinite(value) or value <= 0: raise ValueError("Invalid BF16 compute measurement")
        compute.append(value)
    return dict(stream_read_gbps=bandwidth,
                square_bf16_tflops=statistics.median(compute) if compute else None)


def summarize(profile: dict, primitives: list[dict], resources: dict) -> dict:
    decode = [r for r in profile["phases"] if r["phase"] == "phase.decode"]
    if len(decode) != 1: raise ValueError("Expected one decode phase")
    projection_rows = [r for r in profile["projections"] if r["phase"] == "phase.decode"]
    if not projection_rows: raise ValueError("Missing decoded projection scopes")
    by_shape = defaultdict(list)
    for row in primitives:
        if row["operation"] == "canonical_gemm_poison":
            if not math.isfinite(row["us"]) or row["us"] <= 0: raise ValueError("Invalid GEMM measurement")
            by_shape[(row["input_width"], row["output_width"])].append(row["us"])
    weight_bytes = sum(r["scheduled_weight_bytes"] for r in projection_rows)
    flops = sum(r["scheduled_flops"] for r in projection_rows)
    logical_flops = sum(r["logical_flops"] for r in projection_rows)
    replay_us = 0.; missing = set()
    for row in projection_rows:
        values = dict(re.findall(r"(in|out|rows)=(\d+)", row["label"]))
        shape = (int(values["in"]), int(values["out"]))
        if shape not in by_shape: missing.add(shape)
        else: replay_us += statistics.median(by_shape[shape]) * ((int(values["rows"]) + 31) // 32)
    bandwidth_ms = weight_bytes / (resources["stream_read_gbps"] * 1e6)
    compute_ms = flops / (resources["square_bf16_tflops"] * 1e9) if resources["square_bf16_tflops"] else None
    kernel_components = [r for r in profile["components"] if r["phase"] == "phase.decode" and r["kind"] == "kernel"]
    kernel_total = sum(r["total_ns"] for r in kernel_components)
    if kernel_total <= 0: raise ValueError("Empty decode kernel accounting")
    return dict(scope="Conditional projection-only resource/replay model, not an attainable whole-inference floor or speedup prediction.",
        decode=decode[0], resource_reference=resources,
        scheduled_projection_weight_bytes=weight_bytes, scheduled_projection_flops=flops,
        logical_projection_flops=logical_flops,
        scheduled_to_logical_flops=flops/logical_flops,
        streaming_projection_time_model_ms=bandwidth_ms,
        square_compute_projection_time_model_ms=compute_ms,
        combined_projection_resource_model_ms=max(bandwidth_ms,compute_ms) if compute_ms is not None else None,
        cold_poisoned_gemm_replay_sum_ms=replay_us/1000 if not missing else None,
        missing_replay_shapes=[list(shape) for shape in sorted(missing)],
        kernels=[dict(component=r["component"], calls=r["calls"], total_ms=r["total_ns"]/1e6,
                      gpu_kernel_share_percent=100*r["total_ns"]/kernel_total) for r in kernel_components],
        assumptions=["Every scheduled projection tile reads its full matrix from the modeled streaming resource; cache residency can invalidate this assumption.",
                     "The square GEMM probe is an achieved resource reference, not narrow-GEMM throughput or a proved device ceiling.",
                     "Replay sum uses isolated constant-input, cold-weight N=32 GEMMs with output poisoning; layout, routing and other operators are outside that sum.",
                     "Kernel shares and instrumented uncovered intervals are attribution, not end-to-end optimization ceilings."])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True, help="components.json from profile_native.py")
    p.add_argument("--primitives", type=Path, required=True, help="Unprofiled component_floor JSONL stdout")
    p.add_argument("--roofline", type=Path, action="append", default=[], help="Repeat for each unprofiled roofline stdout")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists(): p.error("Output already exists")
    rows = [json.loads(line) for line in args.primitives.read_text().splitlines() if line.strip()]
    resources = resource_reference(rows, [path.read_text() for path in args.roofline])
    report = summarize(json.loads(args.profile.read_text()), rows, resources)
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("kernels","assumptions")}, indent=2))


if __name__ == "__main__": main()
