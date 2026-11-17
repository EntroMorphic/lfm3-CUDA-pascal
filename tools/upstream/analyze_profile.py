"""Attribute Nsight SQLite GPU activity to native NVTX launch scopes.

GPU time follows CUDA correlation IDs, not overlap with asynchronous host ranges.
Kernel durations and host/API durations are separate, non-additive measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import heapq
from pathlib import Path
import re
import sqlite3

from benchmark_common import write_json


def union_ns(intervals) -> int:
    total = 0; start = end = None
    for left, right in sorted(intervals):
        if right < left: raise ValueError("Negative activity duration")
        if start is None: start, end = left, right
        elif left <= end: end = max(end, right)
        else: total += end - start; start, end = left, right
    return total + (end - start if start is not None else 0)


def component(stack: list[str], kernel: str) -> str:
    # CUB/vendor libraries can push their own nested NVTX ranges. Keep their
    # kernel names in primitive reports, but attribute them to the enclosing
    # native operation rather than treating the vendor label as engine control.
    prefixes = ("phase.", "forward/", "linear/", "canonical_linear/", "norm/",
                "mlp/", "conv/", "attention/", "expert/", "sampler.", "math.")
    exact = {"attention.core", "lm_head", "moe.routing"}
    stack = [s for s in stack if s in exact or s.startswith(prefixes)]
    if not stack: return "unattributed"
    leaf = stack[-1]
    if leaf.startswith("math."): return leaf.split("/")[0]
    if leaf.startswith(("linear/", "canonical_linear/")):
        semantic = next((s.split("/")[0] for s in reversed(stack[:-1])
                         if s.startswith(("moe.routing", "expert/", "mlp/", "conv/", "attention/", "lm_head"))), "other")
        return semantic + "/" + leaf
    if leaf.startswith("norm/"): return leaf
    if "attention.core" in stack: return "attention.core"
    if "moe.routing" in stack: return "moe.routing"
    if leaf.startswith("sampler."): return leaf
    if any(s.startswith("expert/") for s in stack): return "expert.elementwise"
    if "lm_head" in stack: return "lm_head.selection"
    if any(s.startswith("mlp/") for s in stack): return "mlp.elementwise"
    if any(s.startswith("conv/") for s in stack): return "conv.elementwise"
    if any(s.startswith("attention/") for s in stack): return "attention.preparation"
    if "gather_embedding" in kernel: return "embedding"
    return "control"


def analyze(path: Path) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as con:
        return _analyze_connection(con,path)


def _analyze_connection(con, path: Path) -> dict:
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"}
    if not required <= tables: raise ValueError("Missing Nsight tables: " + str(required - tables))
    strings = {r["id"]: r["value"] for r in con.execute("SELECT id,value FROM StringIds")}
    ranges_by_thread = defaultdict(list); phases = []
    for r in con.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"):
        keys = r.keys()
        label = r["text"] if "text" in keys and r["text"] else strings.get(r["textId"], "") if "textId" in keys else ""
        if not label: continue
        item = (r["start"], r["end"], label)
        ranges_by_thread[r["globalTid"]].append(item)
        if label in ("phase.prefill", "phase.decode"): phases.append(item)
    if not any(p[2] == "phase.decode" for p in phases):
        raise ValueError("No native phase.decode range; build with LIQUID_PROFILE_NVTX=1 and trace nvtx")
    api_by_thread = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for r in con.execute(f"SELECT * FROM {table}"):
                api_by_thread[r["globalTid"]].append(dict(r))
    correlations = {}; api_rows = defaultdict(list)
    projection_ranges = []
    for tid, apis in api_by_thread.items():
        ranges = sorted(ranges_by_thread[tid], key=lambda r:(r[0],-r[1])); index = 0; active = {}; ends = []
        thread_phases = [r for r in ranges if r[2] in ("phase.prefill", "phase.decode")]
        for api in sorted(apis, key=lambda r:r["start"]):
            while index < len(ranges) and ranges[index][0] <= api["start"]:
                active[index] = ranges[index]; heapq.heappush(ends,(ranges[index][1],index)); index += 1
            while ends and ends[0][0] <= api["start"]:
                _, key = heapq.heappop(ends); active.pop(key, None)
            stack = [r[2] for r in sorted(active.values(), key=lambda r:(r[0],-r[1])) if r[1] >= api["end"]]
            phase = next((s for s in stack if s.startswith("phase.")), "setup")
            api["name"] = strings.get(api["nameId"], str(api["nameId"]))
            api_rows[(phase,api["name"])].append((api["start"],api["end"]))
            if api.get("correlationId"):
                key = api["correlationId"]
                # The runner traces exactly one native process. Duplicate IDs
                # across processes would make attribution ambiguous: fail closed.
                if key in correlations and correlations[key][0]["globalTid"] != tid:
                    raise ValueError("Ambiguous CUDA correlation across threads/processes")
                if key not in correlations or len(stack) > len(correlations[key][1]):
                    correlations[key] = (api, stack, phase)
        parents = []
        for start, end, label in ranges:
            while parents and parents[-1][1] <= start: parents.pop()
            match = re.fullmatch(r"(canonical_linear|linear)/in=(\d+)/out=(\d+)/rows=(\d+)", label)
            if not match:
                parents.append((start,end,label))
                continue
            phase = next((r[2] for r in thread_phases if r[0] <= start and r[1] >= end), "setup")
            kind, width_in, width_out, rows = match.groups(); width_in=int(width_in); width_out=int(width_out); rows=int(rows)
            logical_rows = rows
            if kind == "canonical_linear":
                expert = next((r[2] for r in reversed(parents) if r[2].startswith("expert/") and r[1] >= end), None)
                if expert is None: raise ValueError("Canonical expert projection lacks active-row range")
                logical_rows = int(expert.rsplit("active=",1)[1])
                if not 0 < logical_rows <= 32: raise ValueError("Invalid canonical expert active rows")
            tiles=(rows+31)//32
            projection_ranges.append(dict(phase=phase, label=label, logical_rows=logical_rows,
                scheduled_weight_bytes=2*width_in*width_out*tiles,
                scheduled_flops=2*width_in*width_out*32*tiles,
                logical_flops=2*width_in*width_out*logical_rows))
            parents.append((start,end,label))
    grouped = defaultdict(list); primitives = defaultdict(list); all_intervals=defaultdict(list)
    phase_intervals=defaultdict(list); kernel_count=0; unmatched=0
    for table, kind in [("CUPTI_ACTIVITY_KIND_KERNEL","kernel"),("CUPTI_ACTIVITY_KIND_MEMCPY","memcpy"),("CUPTI_ACTIVITY_KIND_MEMSET","memset")]:
        if table not in tables: continue
        for r in con.execute(f"SELECT * FROM {table}"):
            keys=r.keys(); correlation=r["correlationId"]
            if correlation in correlations:
                api, stack, phase = correlations[correlation]
            else:
                stack=[]; phase="unattributed"; unmatched+=1
            if kind == "kernel":
                name_id = r["demangledName"] if "demangledName" in keys else r["shortName"]
                name = strings.get(name_id, str(name_id)); kernel_count+=1
            elif kind == "memcpy": name=f"memcpy/kind={r['copyKind']}"
            else: name="memset"
            category = component(stack,name)
            interval=(r["start"],r["end"])
            if interval[1] < interval[0]: raise ValueError("Negative GPU duration")
            grouped[(phase,kind,category)].append(interval)
            primitives[(phase,kind,category,name)].append(interval)
            all_intervals[(phase,kind)].append(interval); phase_intervals[phase].append(interval)
    if not kernel_count: raise ValueError("No traced CUDA kernels")
    def stats(intervals):
        durations=sorted(b-a for a,b in intervals); size=len(durations)
        return dict(calls=size, total_ns=sum(durations), mean_ns=sum(durations)/size,
                    min_ns=durations[0], median_ns=(durations[(size-1)//2]+durations[size//2])/2,
                    p95_ns=durations[min(size-1, int(.95*size))])
    components=[dict(phase=p, kind=k, component=c, **stats(v)) for (p,k,c),v in grouped.items()]
    primitive_rows=[dict(phase=p,kind=k,component=c,name=n,**stats(v)) for (p,k,c,n),v in primitives.items()]
    phase_reports=[]
    unmatched_decode = 0
    for start,end,label in phases:
        if label == "phase.decode":
            unmatched_decode += sum(max(a,start) < min(b,end) for a,b in phase_intervals["unattributed"])
        intervals=phase_intervals[label]
        clipped=[(max(a,start),min(b,end)) for a,b in intervals if max(a,start)<min(b,end)]
        busy=union_ns(clipped)
        phase_reports.append(dict(phase=label, host_wall_ns=end-start, gpu_busy_within_host_range_ns=busy,
            uncovered_host_range_ns=end-start-busy,
            gpu_activity_union_ns=union_ns(intervals),
            kernel_ns=sum(b-a for a,b in all_intervals[(label,"kernel")]),
            memcpy_ns=sum(b-a for a,b in all_intervals[(label,"memcpy")]),
            memset_ns=sum(b-a for a,b in all_intervals[(label,"memset")])))
    return dict(scope="Instrumented launch-attributed GPU durations. Host/API times overlap GPU work; uncovered time is not proof of removable CPU overhead. Projection bytes model scheduled matrix reads, not measured DRAM traffic.",
                sqlite=str(path), phases=phase_reports, kernel_count=kernel_count, unmatched_gpu_activities=unmatched,
                unmatched_decode_activities=unmatched_decode,
                components=sorted(components,key=lambda r:r["total_ns"],reverse=True),
                primitives=sorted(primitive_rows,key=lambda r:r["total_ns"],reverse=True),
                apis=[dict(phase=p,name=n,**stats(v)) for (p,n),v in api_rows.items()],
                projections=projection_ranges)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite",type=Path); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists(): p.error("Output already exists")
    report=analyze(args.sqlite); write_json(args.output,report)
    for row in report["components"]:
        if row["phase"]=="phase.decode":
            print(f"{row['total_ns']/1e6:12.3f} ms  {row['calls']:9d}  {row['kind']:7}  {row['component']}")


if __name__ == "__main__": main()
