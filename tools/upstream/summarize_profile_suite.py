"""Summarize measured decode GPU durations across profile_native.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_common import write_json


def family(component: str) -> str:
    if component.startswith("expert/"): return "expert projections"
    if component == "expert.elementwise": return "expert elementwise"
    if component.startswith("mlp/"): return "dense MLP projections"
    if component == "mlp.elementwise": return "dense MLP elementwise"
    if component.startswith("conv/"): return "convolution projections"
    if component == "conv.elementwise": return "convolution elementwise"
    if component.startswith("attention/"): return "attention projections"
    if component in ("attention.core", "attention.preparation"): return "online attention core"
    if component.startswith("math."): return "math attention"
    if component.startswith("lm_head/") or component == "lm_head.selection": return "output head"
    if component.startswith("sampler."): return "sampler"
    if component.startswith("norm/"): return "normalization"
    if component.startswith("moe.routing"): return "MoE routing"
    if component == "embedding": return "embedding"
    if component == "control": return "control"
    return "other"


def summarize(root: Path) -> dict:
    root = root.resolve()
    results = []
    for directory in sorted(path.parent for path in root.glob("*/components.json")):
        profile = json.loads((directory / "components.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if not summary["complete"] or profile["unmatched_decode_activities"]:
            raise ValueError(f"Incomplete or unattributed profile: {directory.name}")
        for key in ("counts", "hashes", "token_sha256", "reasons"):
            if summary["reference"][key] != summary["traced"][key]:
                raise ValueError(f"Profile output mismatch ({key}): {directory.name}")
        decode = [row for row in profile["components"]
                  if row["phase"] == "phase.decode" and row["kind"] == "kernel"]
        phase = next(row for row in profile["phases"] if row["phase"] == "phase.decode")
        total = sum(row["total_ns"] for row in decode)
        if total != phase["kernel_ns"] or total <= 0:
            raise ValueError(f"Decode kernel accounting mismatch: {directory.name}")
        grouped = {}
        for row in decode:
            value = grouped.setdefault(family(row["component"]), dict(total_ns=0, calls=0))
            value["total_ns"] += row["total_ns"]
            value["calls"] += row["calls"]
        groups = [dict(family=name, **value, share_of_kernel_percent=100*value["total_ns"]/total)
                  for name,value in grouped.items()]
        groups.sort(key=lambda row: row["total_ns"], reverse=True)
        components = [dict(component=row["component"], total_ns=row["total_ns"], calls=row["calls"],
                           share_of_kernel_percent=100*row["total_ns"]/total) for row in decode]
        components.sort(key=lambda row: row["total_ns"], reverse=True)
        results.append(dict(model=directory.name, generated_tokens=summary["traced"]["total_tokens"],
                            traced_decode_tokens_per_second=summary["traced"]["rate"],
                            decode_kernel_ns=total, decode_kernel_ns_per_generated_token=total/summary["traced"]["total_tokens"],
                            decode_gpu_activity_union_ns=phase["gpu_activity_union_ns"],
                            decode_host_wall_ns=phase["host_wall_ns"], kernel_count=profile["kernel_count"],
                            groups=groups, components=components,
                            reference_output_sha256=summary["reference"]["token_sha256"]))
    if not results:
        raise ValueError("No component profiles found")
    return dict(complete=True,
                scope="Measured instrumented launch-attributed decode GPU kernel durations. Family totals are deterministic sums of native NVTX component labels; they are not FLOP, byte or hardware-counter estimates.",
                models=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error("Output already exists")
    report = summarize(args.profiles)
    write_json(args.output, report)
    for model in report["models"]:
        top = model["groups"][0]
        print(f"{model['model']}: {top['family']} {top['share_of_kernel_percent']:.2f}%")


if __name__ == "__main__": main()
