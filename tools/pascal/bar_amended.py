#!/usr/bin/env python3
"""The perplexity bar as amended by PLAN.md decision 6 (2026-11-18), on top of tools/ppl_compare.py, which is
unchanged and still run first.

Why (validation/prefill-arithmetic-v1, validation/round-once-v1): every window starts with the same token, so row 0
of every prefill is one fixed row shared by all windows, and the NLL at position 1 is one distribution, not a sample.
A difference in how that row is rounded is a single draw that the window bootstrap counts once per window. And the
bar's reference, HF SDPA in BF16, is itself an approximation: at 257 tokens the FP32 model is "worse" than SDPA by
the bar's own rule.

The amended rule, per length, with ppl_compare's statistic (paired mean difference of NLL, 95% bootstrap over
windows, EQUAL_EPS) and its verdicts:
  A = engine against HF SDPA            (the original bar)
  B = engine against the FP32 model     (the same statistic, the FP32 run as the reference)
  A length FAILS only if A is "worse" and B is "worse".
A length with A "worse" and B not "worse" is reported as "worse than SDPA, not worse than FP32": not a failure.
Without an FP32 run at a length, A alone decides, as before.
Reported and not judged: the FP32 model against SDPA; engine minus SDPA and minus FP32 at position 1 (no interval:
one distribution) and over positions 1-64.

Every file is loaded through ppl_compare's checks. Exit 0 PASS, 1 FAIL, 2 invalid input.
Usage: bar_amended.py --results DIR --anchor DIR [--engine pascal] [--anchor-engine hf-eager_blocked-fp32] [--output OUT.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppl_compare import InputError, bootstrap_ci, load_nll, load_windows, verdict  # noqa: E402


def judge(x, reference):
    d = (x - reference).mean(axis=1)
    ci = bootstrap_ci(d) if d.size > 1 else None
    return dict(delta=float(d.mean()), ci95=None if ci is None else [float(ci[0]), float(ci[1])], verdict=verdict(float(d.mean()), ci))


def run(a):
    summary = json.loads((a.results / "summary.json").read_text())
    rows = []
    lengths = sorted(int(k) for k in summary.get("windows", {}))
    if not lengths:
        raise InputError("summary.json lists no windows: nothing to compare")
    for length in lengths:
        windows, sha = load_windows(a.results, summary, length)
        count = windows.shape[0]
        engine, engine_sha = load_nll(a.results, a.engine, length, sha, count, True)
        sdpa, _ = load_nll(a.results, "hf-sdpa", length, sha, count, True)
        row = dict(length=length, windows=int(count), engine_nll_sha256=engine_sha, against_sdpa=judge(engine, sdpa))
        anchor = None
        if (a.anchor / f"nll-{a.anchor_engine}-{length}.f32").exists():
            anchor_summary = json.loads((a.anchor / "summary.json").read_text())
            if load_windows(a.anchor, anchor_summary, length)[1] != sha:
                raise InputError(f"the FP32 run's windows-{length}.i32 differ")
            anchor, _ = load_nll(a.anchor, a.anchor_engine, length, sha, count, True)
            meta = json.loads((a.anchor / f"nll-{a.anchor_engine}-{length}.meta.json").read_text())
            if meta.get("dtype") != "float32":
                raise InputError(f"the anchor at {length} is not recorded as float32")
            row["against_fp32"] = judge(engine, anchor)
            row["fp32_against_sdpa"] = judge(anchor, sdpa)
            row["position_1"] = dict(engine_minus_sdpa=float((engine - sdpa)[:, 0].mean()), engine_minus_fp32=float((engine - anchor)[:, 0].mean()))
            row["positions_1_64"] = dict(engine_minus_sdpa=float((engine - sdpa)[:, :64].mean()), engine_minus_fp32=float((engine - anchor)[:, :64].mean()))
        worse_a = row["against_sdpa"]["verdict"].startswith("worse")
        worse_b = anchor is None or row["against_fp32"]["verdict"].startswith("worse")
        invalid = "invalid" in (row["against_sdpa"]["verdict"], row.get("against_fp32", {}).get("verdict"))
        row["fails"] = bool(invalid or (worse_a and worse_b))
        row["reading"] = ("invalid" if invalid else "FAIL: worse than SDPA" + (" and worse than FP32" if anchor is not None else " (no FP32 run)") if row["fails"]
                          else "pass: worse than SDPA, not worse than FP32" if worse_a else "pass")
        rows.append(row)
    report = dict(status="complete", rule="PLAN.md decision 6", engine=a.engine, anchor_engine=a.anchor_engine, rows=rows,
                  passes=not any(r["fails"] for r in rows))
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    f = lambda j: "—" if j is None else f"{j['delta']:+.2e} " + ("(one window)" if j["ci95"] is None else f"[{j['ci95'][0]:+.2e}, {j['ci95'][1]:+.2e}]") + f" {j['verdict']}"
    for r in rows:
        print(f"{r['length']:6d}  {r['windows']:5d} windows\n    against SDPA: {f(r['against_sdpa'])}\n    against FP32: {f(r.get('against_fp32'))}"
              f"\n    FP32 against SDPA (information): {f(r.get('fp32_against_sdpa'))}")
        if "position_1" in r:
            print(f"    position 1: {r['position_1']['engine_minus_sdpa']:+.3e} from SDPA, {r['position_1']['engine_minus_fp32']:+.3e} from FP32;"
                  f" positions 1-64: {r['positions_1_64']['engine_minus_sdpa']:+.2e}, {r['positions_1_64']['engine_minus_fp32']:+.2e}")
        print(f"    {r['reading']}")
    print("AMENDED BAR:", "PASS" if report["passes"] else "FAIL")
    return 0 if report["passes"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, required=True); p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--engine", default="pascal"); p.add_argument("--anchor-engine", default="hf-eager_blocked-fp32")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.output: a.output.write_text(json.dumps(dict(status="running", engine=a.engine, passes=False)) + "\n")
    try:
        return run(a)
    except Exception as error:
        if a.output: a.output.write_text(json.dumps(dict(status="invalid input", engine=a.engine, passes=False, error=str(error))) + "\n")
        print(f"AMENDED BAR: NO VERDICT (invalid input): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
