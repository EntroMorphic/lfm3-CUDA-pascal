"""Compare actual checkpoint activations; tolerances are explicit, never auto-relaxed."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--max-relative-rms", type=float, default=0.02)
    p.add_argument("--max-logit-error", type=float, default=0.5)
    p.add_argument("--require-exact", action="store_true", help="Also require bitwise-identical BF16 traces")
    args = p.parse_args()
    results = []
    for path in sorted((args.directory / "reference").glob("*.bf16")):
        native = args.directory / "native" / path.name
        if not native.exists():
            raise AssertionError(f"missing native trace {path.name}")
        ref_bytes, actual_bytes = path.read_bytes(), native.read_bytes()
        ref = torch.frombuffer(bytearray(ref_bytes), dtype=torch.bfloat16).float()
        actual = torch.frombuffer(bytearray(actual_bytes), dtype=torch.bfloat16).float()
        assert actual.shape == ref.shape, path.name
        error = actual - ref
        relative_rms = (error.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-8)).item()
        maximum = error.abs().max().item()
        passed = torch.isfinite(actual).all().item() and relative_rms <= args.max_relative_rms
        if path.name.endswith(".logits.bf16"):
            passed &= maximum <= args.max_logit_error
        if args.require_exact:
            passed &= ref_bytes == actual_bytes
        results.append(dict(name=path.name, relative_rms=relative_rms, max_abs=maximum,
                            bitwise_equal=ref_bytes == actual_bytes,
                            reference_sha256=hashlib.sha256(ref_bytes).hexdigest(),
                            native_sha256=hashlib.sha256(actual_bytes).hexdigest(),
                            exact_elements=int((actual == ref).sum()), elements=ref.numel(), passed=bool(passed)))
    assert results, "no reference traces"
    ref_meta = json.loads((args.directory / "reference.json").read_text())
    native_meta = json.loads((args.directory / "native.json").read_text())
    tokens_match = ref_meta["generated_ids"] == native_meta["teacher_argmax"]
    report = dict(max_relative_rms=args.max_relative_rms, max_logit_error=args.max_logit_error,
                  require_exact=args.require_exact, all_traces_bitwise_equal=all(x["bitwise_equal"] for x in results),
                  teacher_argmax_exact=tokens_match, checks=len(results),
                  failures=[r for r in results if not r["passed"]], results=results)
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if report["failures"] or not tokens_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
