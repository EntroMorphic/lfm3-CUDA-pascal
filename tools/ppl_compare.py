"""Compare engine perplexity against the HF SDPA reference (the agreed quality bar).

Bar (decided 2026-11-06): the forward pass passes "so long as the perplexity is equal or better",
read statistically (adopted 2026-11-06): a length PASSES unless the engine is statistically worse,
i.e. unless the whole paired 95% CI of dNLL lies above +EQUAL_EPS. Verdicts: "worse" = CI entirely
above +EQUAL_EPS (FAIL); "better" = CI entirely below -EQUAL_EPS; "equal" = anything else (the CI
reaches into [-EQUAL_EPS, +EQUAL_EPS], so it includes or approaches zero). With one window (no CI)
the point estimate must be <= EQUAL_EPS.
  dNLL = mean NLL(engine) - mean NLL(hf-sdpa), in nats per token.
  EQUAL_EPS = 1e-5 nats/token. Measured scorer agreement (native vs HF eager) is <= 2e-6 per token
            and ~1e-9 on the mean, so EQUAL_EPS sits far above computation noise.
  resolution = CI half-width: the smallest uniform regression this corpus can detect at that length.
            A regression below the resolution can pass; it is reported so that is never hidden.
Informational (not the bar): equal_or_better (point dNLL <= EQUAL_EPS) and within_standard
(point dNLL <= |NLL(eager) - NLL(sdpa)| + EQUAL_EPS; 32K borrows the gap from the longest measured length).

Inputs are verified before any number is computed, and any failure is an error, never a PASS:
  - every length in summary.json is scored; an empty summary is refused;
  - windows-L.i32 matches the sha256 in summary.json and is [BOS] + content (BOS nowhere else);
  - every nll-ENGINE-L.f32 has nll-ENGINE-L.meta.json whose engine, length, windows_sha256 and
    nll_sha256 match, is finite, has W x (L-1) values, and the same shape as the reference;
  - HF files record the attention implementation their name claims (sdpa / eager);
  - the engine under test is not an HF configuration; its metadata names the scorer that produced it
    (scorer_sha256); and its file is not a byte copy of ANY HF reference file at that length.
compare-ENGINE.json is removed and replaced with a not-passing placeholder before any work, so an error
of any kind (missing file, malformed JSON, missing field) leaves a failing report, not a stale PASS. The
one case it cannot fix is a read-only directory (the old file can be neither removed nor rewritten); the
run then exits 2 and prints a WARNING naming the file as untrustworthy.
These bindings catch accidental mix-ups (wrong file, wrong windows, stale or copied results); they are
not a defense against deliberate forgery, which could alter a copy and recompute its hashes.
Paired 95% CI: bootstrap over windows (fixed seed), every engine scoring identical ids.

Usage: python ppl_compare.py RESULTS_DIR [--engine native]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EQUAL_EPS = 1e-5


class InputError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_windows(d, summary, length):
    info = summary["windows"][str(length)]
    path = d / info["file"]
    if sha256(path) != info["sha256"]:
        raise InputError(f"{path.name}: sha256 differs from summary.json")
    w = np.fromfile(path, dtype=np.int32)
    if w.size == 0 or w.size % length:
        raise InputError(f"{path.name}: {w.size} ids is not a whole number of length-{length} windows")
    w = w.reshape(-1, length)
    if w.shape[0] != info["count"]:
        raise InputError(f"{path.name}: {w.shape[0]} windows, summary says {info['count']}")
    bos = summary["bos_id"]
    if not (w[:, 0] == bos).all() or (w[:, 1:] == bos).any():
        raise InputError(f"{path.name}: windows are not [BOS] + content")
    return w, info["sha256"]


EXPECTED_ATTENTION = {"hf-sdpa": "sdpa", "hf-eager": "eager"}


def load_nll(d, engine, length, windows_sha, windows_count, required):
    p = d / f"nll-{engine}-{length}.f32"
    meta_path = d / f"nll-{engine}-{length}.meta.json"
    if not p.exists():
        if required:
            raise InputError(f"missing {p.name}")
        return None
    if not meta_path.exists():
        raise InputError(f"{p.name} has no {meta_path.name}: cannot prove which windows it scored")
    meta = json.loads(meta_path.read_text())
    if meta.get("engine") != engine:
        raise InputError(f"{meta_path.name}: records engine {meta.get('engine')!r}, not {engine!r}")
    if engine in EXPECTED_ATTENTION and meta.get("attn_implementation") != EXPECTED_ATTENTION[engine]:
        raise InputError(f"{meta_path.name}: ran attention {meta.get('attn_implementation')!r}, "
                         f"not {EXPECTED_ATTENTION[engine]!r}")
    if meta.get("length") != length:
        raise InputError(f"{meta_path.name}: length {meta.get('length')} != {length}")
    if meta.get("windows_sha256") != windows_sha:
        raise InputError(f"{meta_path.name}: scored different windows than windows-{length}.i32")
    if meta.get("nll_sha256") != sha256(p):
        raise InputError(f"{p.name}: contents differ from the sha256 recorded in its metadata")
    x = np.fromfile(p, dtype=np.float32).astype(np.float64)
    if x.size != windows_count * (length - 1):
        raise InputError(f"{p.name}: {x.size} values, expected {windows_count} x {length - 1}")
    if not np.isfinite(x).all():
        raise InputError(f"{p.name}: contains non-finite values")
    return x.reshape(windows_count, length - 1), meta["nll_sha256"]


def bootstrap_ci(window_diffs, seed=0, resamples=10000):
    rng = np.random.default_rng(seed)
    n = window_diffs.size
    idx = rng.integers(0, n, size=(resamples, n))
    means = window_diffs[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def passes_bar(delta, ci):
    """Pass unless statistically worse: the CI lower bound must not exceed EQUAL_EPS."""
    if not math.isfinite(delta) or (ci is not None and not all(map(math.isfinite, ci))):
        return False
    if ci is None:
        return delta <= EQUAL_EPS
    return ci[0] <= EQUAL_EPS


def verdict(delta, ci):
    if not math.isfinite(delta) or (ci is not None and not all(map(math.isfinite, ci))):
        return "invalid"
    if ci is None:
        return "equal-or-better (point)" if delta <= EQUAL_EPS else "worse (point)"
    if ci[1] < -EQUAL_EPS:
        return "better"
    if ci[0] > EQUAL_EPS:
        return "worse"
    return "equal"


def compare(d, engine):
    if engine.startswith("hf-"):
        raise InputError(f"engine {engine!r} is a reference configuration, not an engine under test")
    summary = json.loads((d / "summary.json").read_text())
    lengths = sorted(int(k) for k in summary.get("windows", {}))
    if not lengths:
        raise InputError("summary.json lists no windows: nothing to compare")
    rows, gaps = [], {}
    for length in lengths:
        windows, wsha = load_windows(d, summary, length)
        n = windows.shape[0]
        ref, ref_sha = load_nll(d, "hf-sdpa", length, wsha, n, required=True)
        eng, eng_sha = load_nll(d, engine, length, wsha, n, required=True)
        eager_loaded = load_nll(d, "hf-eager", length, wsha, n, required=False)
        eager = eager_loaded[0] if eager_loaded else None
        meta = json.loads((d / f"nll-{engine}-{length}.meta.json").read_text())
        if not (isinstance(meta.get("scorer_sha256"), str) and len(meta["scorer_sha256"]) == 64):
            raise InputError(f"nll-{engine}-{length}.meta.json does not name the scorer that produced it")
        for ref_file in sorted(d.glob(f"nll-hf-*-{length}.f32")):
            if sha256(ref_file) == eng_sha:
                raise InputError(f"nll-{engine}-{length}.f32 is byte-identical to {ref_file.name}: a copy, not a measurement")
        r = {"length": length, "windows": n, "tokens": ref.size,
             "ppl_sdpa": math.exp(ref.mean()), f"ppl_{engine}": math.exp(eng.mean())}
        diff = eng - ref
        r["delta_mean_nll"] = float(diff.mean())
        r["delta_ci95"] = bootstrap_ci(diff.mean(axis=1)) if n > 1 else None
        r["resolution"] = (r["delta_ci95"][1] - r["delta_ci95"][0]) / 2 if r["delta_ci95"] else None
        r["max_abs_token_delta"] = float(np.abs(diff).max())
        if eager is not None:
            r["ppl_eager"] = math.exp(eager.mean())
            gaps[length] = abs(float(eager.mean() - ref.mean()))
            r["tolerance"], r["tolerance_source"] = gaps[length], "measured at this length"
            r[f"{engine}_minus_eager_mean_nll"] = float((eng - eager).mean())
        elif gaps:
            src = max(gaps)
            r["tolerance"], r["tolerance_source"] = gaps[src], f"borrowed from length {src} (eager not measured here)"
        r["passes_bar"] = passes_bar(r["delta_mean_nll"], r["delta_ci95"])
        r["verdict"] = verdict(r["delta_mean_nll"], r["delta_ci95"])
        r["equal_or_better"] = r["delta_mean_nll"] <= EQUAL_EPS
        r["within_standard"] = r["equal_or_better"] or ("tolerance" in r and r["delta_mean_nll"] <= r["tolerance"] + EQUAL_EPS)
        rows.append(r)
    return {"engine": engine, "reference": "hf-sdpa", "corpus_sha256": summary["corpus_sha256"], "rows": rows,
            "equal_eps": EQUAL_EPS, "passes_bar": bool(rows) and all(r["passes_bar"] for r in rows),
            "all_equal_or_better": all(r["equal_or_better"] for r in rows),
            "all_within_standard": all(r["within_standard"] for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--engine", default="native")
    a = p.parse_args()
    out = a.results / f"compare-{a.engine}.json"
    try:
        # Remove any earlier report before doing anything (unlink works in a writable directory even when
        # the old file itself is read-only or root-owned), then write a not-passing placeholder, so only
        # a fully verified run can leave a PASS behind.
        out.unlink(missing_ok=True)
        out.write_text(json.dumps({"engine": a.engine, "passes_bar": False, "status": "incomplete"}, indent=2) + "\n")
        report = compare(a.results, a.engine)
    except Exception as e:  # any failure, of any type, is a FAIL with the original reason recorded
        reason = f"{type(e).__name__}: {e}"
        try:
            out.write_text(json.dumps({"engine": a.engine, "passes_bar": False, "input_error": reason}, indent=2) + "\n")
        except OSError as write_error:
            reason += f" (and the failure report could not be written: {type(write_error).__name__}: {write_error})"
        print(f"BAR: FAIL (input error: {reason})")
        try:
            stale = not (out.is_file() and json.loads(out.read_text()).get("passes_bar") is False)
        except Exception:  # unreadable or not a JSON object: certainly not this run's report
            stale = out.exists()
        if stale:
            print(f"WARNING: {out} still exists and was not replaced; do not trust it")
        return 2
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'length':>7} {'windows':>7} {'PPL sdpa':>10} {'PPL eager':>10} {'PPL ' + a.engine:>11} {'dNLL':>11} "
          f"{'95% CI':>25} {'resolution':>10}  {'verdict':>8} bar")
    for r in report["rows"]:
        ci = r["delta_ci95"]
        print(f"{r['length']:>7} {r['windows']:>7} {r['ppl_sdpa']:>10.4f} {r.get('ppl_eager', float('nan')):>10.4f} "
              f"{r['ppl_' + a.engine]:>11.4f} {r['delta_mean_nll']:>+11.2e} "
              f"{('[%+.2e, %+.2e]' % ci) if ci else '':>25} {(r['resolution'] or float('nan')):>10.2e}  "
              f"{r['verdict']:>8} {'PASS' if r['passes_bar'] else 'FAIL'}")
    print(f"BAR (perplexity equal or better, 95% CI reading): {'PASS' if report['passes_bar'] else 'FAIL'}")
    print(f"info: equal eps {EQUAL_EPS:g} nats/token; regressions smaller than a length's resolution cannot be detected; "
          f"point-estimate equal-or-better at all lengths: {report['all_equal_or_better']}; "
          f"within standard stack: {report['all_within_standard']}")
    return 0 if report["passes_bar"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
