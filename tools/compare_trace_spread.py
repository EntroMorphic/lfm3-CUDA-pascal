#!/usr/bin/env python3
"""Distance of BF16 implementations from an anchor forward pass, per traced activation and per logit step.

Each candidate is --candidate NAME ROOT/SUBDIR IDS_ROOT: its traces are ROOT/CASE/SUBDIR/*.bf16, and the prompt and tokens
it was fed are IDS_ROOT/CASE/reference.json "input_ids" and "generated_ids" (teacher-forced runs use the
reference that made their request). The anchor is ANCHOR_ROOT/CASE/reference with its own reference.json.
Every candidate must have the anchor's prompt. All candidates are compared on the same files: those the
anchor and every candidate traced, at steps k where tokens 0..k-1 agree for every candidate. Per file: relative RMS and max |error|. Per logit
step: KL(anchor || candidate) of the softmax over all logits, in nats, using each side's exact STEP.logits.f32
when present (its actual output), else its BF16 trace. Host numpy only.

Usage: compare_trace_spread.py ANCHOR_ROOT OUT.json --cases C1,C2 --candidate NAME ROOT/SUBDIR IDS_ROOT ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_trace_exact import exact_logit_steps  # noqa: E402


def load(path):
    raw = np.fromfile(path, np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64)


def distance(reference, other):
    error = other - reference
    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-8)
    return float(np.sqrt(np.mean(error**2)) / scale), float(np.max(np.abs(error)))


def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def kl(anchor, other):
    p, q = log_softmax(anchor), log_softmax(other)
    return float(np.sum(np.exp(p) * (p - q)))


def load_exact_logits(bf16_path):
    """A logits trace's exact FP32 sidecar when it exists (the run's actual output), else its BF16 file."""
    exact = bf16_path.with_name(bf16_path.name[: -len(".bf16")] + ".f32")
    return np.fromfile(exact, np.float32).astype(np.float64) if exact.exists() else load(bf16_path)


def reference(root, case):
    return json.loads((root / case / "reference.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("anchor", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--candidate", action="append", nargs=3, metavar=("NAME", "TRACES", "IDS_ROOT"), required=True)
    a = ap.parse_args()
    candidates = []
    for name, path, ids_root in a.candidate:
        root, subdir = Path(path).parent, Path(path).name
        candidates.append((name, root, subdir, Path(ids_root)))
    report = dict(definition=__doc__.split("\n\n")[0], anchor=str(a.anchor),
                  file_set="files traced by the anchor and by every candidate, at steps every candidate covers",
                  cases=[])
    for case in a.cases.split(","):
        anchor_ref = reference(a.anchor, case)
        anchor_ids = anchor_ref["generated_ids"]
        agree_by, skipped = {}, {}
        for name, root, subdir, ids_root in candidates:
            fed = reference(ids_root, case)
            # Every candidate must have been given the anchor's prompt; differing prompts are not comparable.
            if fed["input_ids"] != anchor_ref["input_ids"]:
                raise SystemExit(f"{case}: {name} was fed a different prompt than the anchor")
            agree = 0
            while agree < min(len(anchor_ids), len(fed["generated_ids"])) and anchor_ids[agree] == fed["generated_ids"][agree]:
                agree += 1
            agree_by[name] = agree
        last_step = min(agree_by.values())  # step k is fed tokens 0..k-1: comparable while they agree everywhere
        common, missing = [], {name: 0 for name, *_ in candidates}
        for ref in sorted((a.anchor / case / "reference").glob("*.bf16")):
            if int(ref.name.split(".")[0]) > last_step:
                continue
            present = True
            for name, root, subdir, _ in candidates:
                if not (root / case / subdir / ref.name).exists():
                    missing[name] += 1
                    present = False
            if present:
                common.append(ref)
        # Exact logits are all-or-none per run, and each .bf16 must be their rounding (lfm3_trace_exact).
        logit_steps = sorted(int(r.name.split(".")[0]) for r in common if ".logits." in r.name)
        exact = {"anchor": exact_logit_steps(a.anchor / case / "reference", logit_steps)}
        for name, root, subdir, _ in candidates:
            exact[name] = exact_logit_steps(root / case / subdir, logit_steps)
        row = dict(case=case, tokens_agree=agree_by, compared_through_step=last_step, common_files=len(common),
                   exact_fp32_logits=exact,
                   anchor_files_not_traced_by=missing, candidates={})
        for name, root, subdir, _ in candidates:
            rel, kls, logit_abs = [], [], []
            for ref in common:
                other = root / case / subdir / ref.name
                logit = ".logits." in ref.name
                r, o = (load_exact_logits(ref), load_exact_logits(other)) if logit else (load(ref), load(other))
                if r.shape != o.shape:
                    raise SystemExit(f"{case}: {name} {ref.name} has a different shape than the anchor")
                d, m = distance(r, o)
                if logit:
                    kls.append(kl(r, o))
                    logit_abs.append(m)
                else:
                    rel.append(d)
            row["candidates"][name] = dict(
                activation_files=len(rel), activation_rel_rms_median=float(np.median(rel)) if rel else None,
                activation_rel_rms_max=max(rel) if rel else None,
                logit_steps=len(kls), logit_kl_mean=float(np.mean(kls)) if kls else None,
                logit_kl_max=max(kls) if kls else None, logit_max_abs=max(logit_abs) if logit_abs else None)
        report["cases"].append(row)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
    for row in report["cases"]:
        print(f"{row['case']}: through step {row['compared_through_step']}, {row['common_files']} common files; "
              f"anchor files not traced by {row['anchor_files_not_traced_by']}")
        for name, c in row["candidates"].items():
            print(f"  {name:<7} act rel_rms median {f(c['activation_rel_rms_median'], '.4f')} max {f(c['activation_rel_rms_max'], '.4f')}"
                  f"  logits KL mean {f(c['logit_kl_mean'], '.2e')} max {f(c['logit_kl_max'], '.2e')} ({c['logit_steps']} steps)"
                  f"  max_abs {f(c['logit_max_abs'], '.3f')}")


if __name__ == "__main__":
    main()
