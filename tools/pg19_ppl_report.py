#!/usr/bin/env python3
"""Report of a PG-19 long-context perplexity round (tools/thor/pg19_ppl_round.sh): ROUND/onepass and ROUND/chunked,
each a ppl_reference.py output (one prefix window per book: [BOS] + its first L - 1 tokens).

- Perplexity per engine and length, over every book long enough.
- One pass vs chunked at the lengths both ran: mean |dNLL| per token and the mean difference with a 95% bootstrap CI
  over books (chunking runs a different attention kernel once a chunk attends to a cached prefix).
- Context use: mean NLL by position bucket [2^k, 2^(k+1)) over the books of the longest length, per engine. A model
  that uses long context keeps the NLL flat or falling with position; one that breaks past a length shows a rise.
- Prefix consistency: a book's first L tokens are scored the same whatever the window length (causal), so the longest
  window's NLL over its first L - 1 positions is compared with the length-L run of the same book.
- With NATIVE_DIR (tools/thor/pg19_native_round.sh): native's NLL against HF's at each length, paired per book:
  mean(native - HF) with a 95% bootstrap CI over books; "worse" iff the CI lies above +1e-5 nats, "better" iff below
  -1e-5, else "equal" (the text engine's section 9 rule). HF is the one-pass run where it exists, else the chunked run.
  Native's meta must name the very windows file HF scored (by hash).
Usage: pg19_ppl_report.py ROUND_DIR [OUT.json [NATIVE_DIR]]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def load(run_dir):
    s = json.loads((run_dir / "summary.json").read_text())
    runs = {}
    for r in s["results"]:
        if "file" not in r:
            continue
        L = r["length"]
        w = s["windows"][str(L)]
        nll = np.fromfile(run_dir / r["file"], np.float32).reshape(w["count"], L - 1).astype(np.float64)
        runs[(r["engine"], L)] = dict(nll=nll, books=w["books"], seconds=r["seconds"], peak=r.get("peak_allocated_gib"))
    return runs


def ci(d, n=4000, seed=0):
    if len(d) < 2:
        return [float("nan"), float("nan")]
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


EPS = 1e-5


def main(round_dir, out_json=None, native_dir=None):
    round_dir = Path(round_dir)
    runs = {}
    for sub in ("onepass", "chunked"):
        if (round_dir / sub / "summary.json").exists():
            runs.update(load(round_dir / sub))
    report = dict(perplexity=[], onepass_vs_chunked=[], position_buckets={}, prefix_consistency=[])
    print("| engine | length | books | perplexity | mean NLL | seconds/book | peak GiB |\n|---|---|---|---|---|---|---|")
    for (e, L), r in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
        m = float(r["nll"].mean())
        report["perplexity"].append(dict(engine=e, length=L, books=len(r["books"]), mean_nll=m, perplexity=math.exp(m),
                                         seconds_per_book=r["seconds"] / len(r["books"]), peak_gib=r["peak"]))
        print(f"| {e} | {L} | {len(r['books'])} | {math.exp(m):.3f} | {m:.5f} | {r['seconds'] / len(r['books']):.1f} | "
              f"{r['peak'] or float('nan'):.1f} |")
    for (e, L), r in runs.items():
        if "-chunk" in e:
            base = (e.split("-chunk")[0], L)
            if base in runs:
                a, b = runs[base], r
                assert a["books"] == b["books"]
                d = (b["nll"] - a["nll"]).mean(1)
                x = dict(engines=[base[0], e], length=L, mean_abs_token=float(np.abs(b["nll"] - a["nll"]).mean()),
                         mean_diff=float(d.mean()), ci95=ci(d))
                report["onepass_vs_chunked"].append(x)
                print(f"\none pass vs chunked at {L}: mean |dNLL| per token {x['mean_abs_token']:.4f}, mean diff "
                      f"{x['mean_diff']:+.2e} CI [{x['ci95'][0]:+.2e}, {x['ci95'][1]:+.2e}]")
    longest = max(L for _, L in runs)
    for (e, L), r in runs.items():
        if L != longest:
            continue
        pos = np.arange(1, L)  # position of the predicted token = context tokens before it
        edges = [1] + [2 ** k for k in range(8, int(math.log2(L)) + 1)]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (pos >= lo) & (pos < hi)
            per_book = r["nll"][:, sel].mean(1)
            rows.append(dict(context=[lo, hi], mean_nll=float(per_book.mean()), per_book=per_book.tolist()))
        report["position_buckets"][e] = rows
        print(f"\n{e}, {len(r['books'])} books of {L} tokens: mean NLL by context position")
        print("| context | mean NLL | perplexity |\n|---|---|---|")
        for x in rows:
            print(f"| {x['context'][0]}-{x['context'][1] - 1} | {x['mean_nll']:.4f} | {math.exp(x['mean_nll']):.2f} |")
        for (e2, L2), r2 in runs.items():
            if L2 >= L:
                continue
            idx = [r2["books"].index(b) for b in r["books"] if b in r2["books"]]
            mine = [i for i, b in enumerate(r["books"]) if b in r2["books"]]
            d = np.abs(r["nll"][mine, :L2 - 1] - r2["nll"][idx])
            report["prefix_consistency"].append(dict(long=[e, L], short=[e2, L2], books=len(idx),
                                                    mean_abs_token=float(d.mean()), max_abs=float(d.max()),
                                                    identical=bool((d == 0).all())))
    if report["prefix_consistency"]:
        print("\nprefix consistency (longest window's first L-1 positions vs the length-L run):")
        for x in report["prefix_consistency"]:
            print(f"  {x['long'][0]} @{x['long'][1]} vs {x['short'][0]} @{x['short'][1]}: {x['books']} books, "
                  f"mean |dNLL| {x['mean_abs_token']:.4f}, max {x['max_abs']:.3f}{', identical' if x['identical'] else ''}")
    if native_dir:
        report["native_vs_hf"] = []
        print("\n| length | books | native ppl | HF ppl (engine) | mean(native - HF) | 95% CI | verdict |\n|---|---|---|---|---|---|---|")
        for L in sorted({L for _, L in runs}):
            nf = Path(native_dir) / f"nll-native-gemm-{L}.f32"
            if not nf.exists():
                continue
            hf = ("hf-sdpa", L) if ("hf-sdpa", L) in runs else next(k for k in runs if k[1] == L)
            meta = json.loads(nf.with_suffix(".meta.json").read_text())
            sub = "onepass" if hf[0] == "hf-sdpa" else "chunked"
            want = json.loads((round_dir / sub / "summary.json").read_text())["windows"][str(L)]["sha256"]
            assert meta["windows_sha256"] == want, f"native at {L} scored other windows"
            h = runs[hf]["nll"]
            n = np.fromfile(nf, np.float32).reshape(h.shape).astype(np.float64)
            d = (n - h).mean(1)
            c = ci(d)
            verdict = "worse" if c[0] > EPS else "better" if c[1] < -EPS else "equal"
            x = dict(length=L, books=h.shape[0], hf_engine=hf[0], native_perplexity=math.exp(n.mean()),
                     hf_perplexity=math.exp(h.mean()), mean_diff=float(d.mean()), ci95=c, verdict=verdict,
                     mean_abs_token=float(np.abs(n - h).mean()))
            report["native_vs_hf"].append(x)
            print(f"| {L} | {x['books']} | {x['native_perplexity']:.3f} | {x['hf_perplexity']:.3f} ({hf[0]}) | "
                  f"{x['mean_diff']:+.2e} | [{c[0]:+.2e}, {c[1]:+.2e}] | {verdict} |")
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
