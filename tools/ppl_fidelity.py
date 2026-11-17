#!/usr/bin/env python3
"""G4 tier 2b: is an engine's per-token NLL as close to an FP32 model's as HF SDPA's is? (validation/g4-gates-v2/PLAN.md)

Per token: A = |NLL_engine - NLL_anchor|, S = |NLL_sdpa - NLL_anchor|; the anchor is an FP32 run of ppl_reference
(default hf-eager_blocked-fp32: exact FP32 eager attention in blocks; see the PLAN amendment). Paired over windows:
D = mean(A) - mean(S), with ppl_compare's 95% paired bootstrap CI over windows and its epsilon. A length FAILS iff
the whole CI of D lies above +epsilon (the engine is statistically farther from FP32 than SDPA is). Also reported,
not judged: the same with hf-eager in place of hf-sdpa where it exists.

Every file is loaded through ppl_compare's checks: windows match summary.json, each NLL file's metadata binds it to
those windows and to its own hash. The anchor must be recorded as dtype float32.
The engine must not be an HF reference, must name its scorer, and must not be a byte copy of a reference file; with
--expect-sha256, every engine NLL file must have the hash listed there ("SHA  nll-ENGINE-L.f32" lines).
Exit codes: 0 PASS, 1 FAIL, 2 invalid input (no verdict; the report says so).
Usage: ppl_fidelity.py --anchor DIR --references DIR --engine-dir DIR --engine NAME [--expect-sha256 FILE] [--output OUT.json]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppl_compare import EQUAL_EPS, InputError, bootstrap_ci, load_nll, load_windows  # noqa: E402

LENGTHS = (256, 257, 4096, 32768)


def t_ci(window_diffs):
    """Student-t 95% CI over windows (information: the verdict uses ppl_compare's bootstrap, which is too narrow at
    9 windows). scipy if present, else a Cornish-Fisher quantile (within 0.3% for df >= 8)."""
    n = window_diffs.size
    if n < 2:
        return None  # no interval from one window
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, n - 1))
    except ImportError:
        z, df = 1.959963984540054, n - 1
        q = z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    half = q * float(window_diffs.std(ddof=1)) / n**0.5
    return [float(window_diffs.mean()) - half, float(window_diffs.mean()) + half]


def windows_sha(d, length):
    summary = json.loads((d / "summary.json").read_text())
    return load_windows(d, summary, length)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--anchor", type=Path, required=True, help="ppl_reference --dtype float32 output")
    p.add_argument("--anchor-engine", default="hf-eager_blocked-fp32", help="hf-ATTENTION-fp32")
    p.add_argument("--references", type=Path, required=True, help="directory with nll-hf-sdpa/-eager files")
    p.add_argument("--engine-dir", type=Path, required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--expect-sha256", type=Path, help="lines 'SHA  nll-ENGINE-L.f32' the engine files must match")
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS),
                   help="a subset of 256 257 4096 32768 (robustness checks only; the gate uses all four)")
    p.add_argument("--windows-dir", type=Path,
                   help="where windows-L.i32 and summary.json live, if not with the anchor (each NLL's metadata must bind "
                        "to those windows); the anchor's precision is then checked from its per-file metadata only")
    a = p.parse_args()
    if a.output:  # never leave an earlier verdict behind
        a.output.write_text(json.dumps(dict(status="running", engine=a.engine)) + "\n")
    try:
        return run(a)
    except Exception as error:  # any malformed input is "no verdict", never a FAIL
        if a.output:
            a.output.write_text(json.dumps(dict(status="invalid input", engine=a.engine, error=str(error))) + "\n")
        print(f"TIER 2b: NO VERDICT (invalid input): {error}", file=sys.stderr)
        return 2


def run(a):
    if a.engine.startswith("hf-"):
        raise InputError(f"{a.engine} is an HF reference, not an engine under test")
    if not set(a.lengths) <= set(LENGTHS):
        raise InputError(f"lengths must be among {LENGTHS}")
    expected = None
    if a.expect_sha256:
        expected = {}
        for line in a.expect_sha256.read_text().split("\n"):
            if line.strip():
                digest, name = line.split()
                expected[Path(name).name] = digest
        missing = [f"nll-{a.engine}-{L}.f32" for L in a.lengths if f"nll-{a.engine}-{L}.f32" not in expected]
        if missing:
            raise InputError(f"--expect-sha256 lists no hash for {missing}")
    windows_dir = a.windows_dir or a.anchor
    if a.windows_dir is None:  # the anchor's run-level precision record (ppl_reference summary.json)
        summary = json.loads((a.anchor / "summary.json").read_text())
        precision = summary.get("precision") or {}
        if summary.get("dtype") != "float32" or precision.get("matmul_allow_tf32") or precision.get("cudnn_allow_tf32"):
            raise InputError("the anchor's summary.json does not record FP32 with TF32 disabled")
    rows = []
    for length in a.lengths:
        windows, sha = windows_sha(windows_dir, length)
        for d in (a.references, a.engine_dir):
            if windows_sha(d, length)[1] != sha:
                raise InputError(f"{d}: windows-{length}.i32 differ from the anchor's")
        count = windows.shape[0]
        anchor, _ = load_nll(a.anchor, a.anchor_engine, length, sha, count, True)
        meta = json.loads((a.anchor / f"nll-{a.anchor_engine}-{length}.meta.json").read_text())
        attention = a.anchor_engine[len("hf-"):-len("-fp32")]
        if (not (a.anchor_engine.startswith("hf-") and a.anchor_engine.endswith("-fp32")) or meta.get("dtype") != "float32"
                or meta.get("attn_implementation") != attention):
            raise InputError(f"anchor at {length} is not recorded as float32 with {attention} attention")
        file_precision = meta.get("precision")  # per-file record (ppl_reference writes it since G4's red-team)
        if file_precision is not None and (file_precision.get("matmul_allow_tf32") or file_precision.get("cudnn_allow_tf32")):
            raise InputError(f"anchor at {length} records TF32 enabled")
        engine, engine_sha = load_nll(a.engine_dir, a.engine, length, sha, count, True)
        engine_meta = json.loads((a.engine_dir / f"nll-{a.engine}-{length}.meta.json").read_text())
        if not (isinstance(engine_meta.get("scorer_sha256"), str) and len(engine_meta["scorer_sha256"]) == 64):
            raise InputError(f"nll-{a.engine}-{length}.meta.json does not name the scorer that produced it")
        name = f"nll-{a.engine}-{length}.f32"
        if expected is not None and expected.get(name) != engine_sha:
            raise InputError(f"{name}: hash {engine_sha[:16]} is not the expected {str(expected.get(name))[:16]}")
        for ref_dir in (a.references, a.anchor):
            for ref_file in ref_dir.glob(f"nll-hf-*-{length}.f32"):
                if hashlib.sha256(ref_file.read_bytes()).hexdigest() == engine_sha:
                    raise InputError(f"{name} is a byte copy of {ref_file.name}")
        anchor = anchor.astype(np.float64)
        dev = lambda x: np.abs(x.astype(np.float64) - anchor).mean(axis=1)  # per-window mean |deviation|
        row = dict(length=length, windows=int(count), engine_nll_sha256=engine_sha, engine_mean_abs_dev=float(dev(engine).mean()))
        for ref in ("hf-sdpa", "hf-eager"):
            loaded = load_nll(a.references, ref, length, sha, count, ref == "hf-sdpa")
            if loaded is None:  # hf-eager does not exist above 4,096
                continue
            x = loaded[0]
            d = dev(engine) - dev(x)
            ci = bootstrap_ci(d)
            row[ref] = dict(ref_mean_abs_dev=float(dev(x).mean()), delta=float(d.mean()), ci95=list(ci),
                            t_ci95_information=t_ci(d), engine_closer_windows=int((d < 0).sum()))
        sd = row["hf-sdpa"]
        row["fails"] = sd["ci95"][0] > EQUAL_EPS
        row["verdict"] = ("farther than SDPA (FAIL)" if row["fails"] else
                          "closer than SDPA" if sd["ci95"][1] < -EQUAL_EPS else "as close as SDPA")
        rows.append(row)
    report = dict(status="complete", full_gate=list(a.lengths) == list(LENGTHS), definition=__doc__.split("\n\n")[0], epsilon=EQUAL_EPS, engine=a.engine,
                  anchor_engine=a.anchor_engine, lengths=a.lengths, rows=rows,
                  passes=not any(r["fails"] for r in rows))
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'length':>7}{'windows':>8}{'|engine-fp32|':>15}{'|sdpa-fp32|':>13}{'D':>11}   95% CI                     verdict")
    for r in rows:
        s = r["hf-sdpa"]
        print(f"{r['length']:>7}{r['windows']:>8}{r['engine_mean_abs_dev']:>15.3e}{s['ref_mean_abs_dev']:>13.3e}{s['delta']:>+11.2e}"
              f"   [{s['ci95'][0]:+.2e}, {s['ci95'][1]:+.2e}]   {r['verdict']}")
        if "hf-eager" in r:
            e = r["hf-eager"]
            print(f"{'':>7}{'':>8}{'':>15}{'eager':>13}{e['delta']:>+11.2e}   [{e['ci95'][0]:+.2e}, {e['ci95'][1]:+.2e}]   (information)")
    subset = "" if list(a.lengths) == list(LENGTHS) else f" (lengths {a.lengths} only: a diagnostic, not the gate)"
    print(f"TIER 2b: {'PASS' if report['passes'] else 'FAIL'}{subset}")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
