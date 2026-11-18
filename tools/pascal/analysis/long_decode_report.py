#!/usr/bin/env python3
"""long-decode-v1, rule 4: the decode path against the prefill path at the same positions.

  long_decode_report.py --runs DIR --pascal DIR --thor DIR [--output FILE]

DIR/--runs holds CASE.default.json from tools/pascal/long_decode.sh. For each teacher-forced case, d is the NLL of
the next token from the decode step's logits minus the NLL the scorer (prefill) gave the same position of the same
window (nll-pascal-L.f32 in --pascal). The yardstick y is the Thor engine's scorer minus ours at the same positions
(nll-native-L.f32 in --thor): the difference between two arithmetics that both pass the perplexity bar.
A case passes if mean|d| <= 2 * mean|y|. Every value must be finite. Where a run records the windows and binary it
used (long_decode.sh does), the windows must be the ones the scorers read and the binary the same in every case.
Exit 0 if every case passes, 1 if one fails, 2 on bad input (the output file then says so).
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

CASES = ("control", "short", "middle", "end")
FACTOR = 2.0


def load_nll(directory, engine, length, windows):
    path = directory / f"nll-{engine}-{length}.f32"
    meta = json.loads((directory / f"nll-{engine}-{length}.meta.json").read_text())
    if meta["nll_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"{path}: contents differ from the sha256 in its metadata")
    x = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if x.size != windows * (length - 1):
        raise ValueError(f"{path}: {x.size} values, expected {windows} x {length - 1}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: holds a non-finite NLL")
    return x.reshape(windows, length - 1), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--pascal", type=pathlib.Path, required=True)
    ap.add_argument("--thor", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path)
    a = ap.parse_args()
    report, ok, binaries = {"factor": FACTOR, "cases": {}}, True, set()
    if a.output:
        a.output.write_text(json.dumps({"status": "running", "pass": False}) + "\n")
    try:
        for case in CASES:
            r = json.loads((a.runs / f"{case}.default.json").read_text())
            length = r["window_length"]
            windows = {4096: 72, 32768: 9}[length]
            ours, ours_meta = load_nll(a.pascal, "pascal", length, windows)
            thor, thor_meta = load_nll(a.thor, "native", length, windows)
            if ours_meta["windows_sha256"] != thor_meta["windows_sha256"]:
                raise ValueError(f"{case}: the two scorers read different windows")
            if "windows_sha256" in r and r["windows_sha256"] != ours_meta["windows_sha256"]:
                raise ValueError(f"{case}: decoded tokens from other windows than the scorers read")
            binaries.add(r.get("binary_sha256"))
            decode = np.array(r["nll"], dtype=np.float64)
            if not np.isfinite(decode).all():
                raise ValueError(f"{case}: a decode step's NLL is not finite")
            first = r["first_position"]  # the position step 0 predicts; NLL files start at position 1
            if decode.size != 1101 or first < 1 or first + decode.size > length:
                raise ValueError(f"{case}: {decode.size} NLL values from position {first}")
            cols = slice(first - 1, first - 1 + decode.size)
            prefill, other = ours[r["window"], cols], thor[r["window"], cols]
            d, y = decode - prefill, other - prefill
            passed = bool(np.abs(d).mean() <= FACTOR * np.abs(y).mean())
            ok &= passed
            report["cases"][case] = {
                "capacity": r["capacity"], "window": r["window"], "positions": [first, first + decode.size - 1],
                "mean_nll_decode": decode.mean(), "mean_nll_prefill": prefill.mean(), "mean_nll_thor_prefill": other.mean(),
                "mean_d": d.mean(), "mean_abs_d": np.abs(d).mean(), "max_abs_d": np.abs(d).max(),
                "mean_y": y.mean(), "mean_abs_y": np.abs(y).mean(), "max_abs_y": np.abs(y).max(),
                "ratio": np.abs(d).mean() / np.abs(y).mean(), "pass": passed}
            c = report["cases"][case]
            print(f"{case:8s} capacity {r['capacity']:6d} positions {first}-{first + decode.size - 1}: "
                  f"decode - prefill mean {c['mean_d']:+.3e}, mean|d| {c['mean_abs_d']:.3e}, max {c['max_abs_d']:.3e}; "
                  f"Thor - ours mean|y| {c['mean_abs_y']:.3e}, max {c['max_abs_y']:.3e}; "
                  f"ratio {c['ratio']:.2f} {'PASS' if passed else 'FAIL'}")
        if len(binaries) != 1:
            raise ValueError(f"the cases ran {len(binaries)} different binaries")
    except (OSError, KeyError, ValueError) as e:
        print(f"long_decode_report: NO VERDICT: {e}", file=sys.stderr)
        if a.output:
            a.output.write_text(json.dumps({"status": "invalid input", "pass": False, "error": str(e)}) + "\n")
        return 2
    report["pass"] = ok
    report["status"] = "complete"
    report["binary_sha256"] = binaries.pop()
    if a.output:
        a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"RULE 4 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
