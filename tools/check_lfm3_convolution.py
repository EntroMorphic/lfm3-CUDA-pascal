"""Verify distinct checkpoint prefill/decode contracts across lengths and magnitudes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from lfm3_bridge import Bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(9876)
    bridge = Bridge(args.library)
    spec = importlib.util.spec_from_file_location("checkpoint_conv", args.checkpoint / "lfm3_conv.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    checks = []

    def equal(name, actual, expected):
        assert torch.isfinite(actual).all(), name
        if not torch.equal(actual, expected):
            error = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f"{name}: not exact; max abs={error}")
        checks.append(dict(name=name, elements=actual.numel(), max_abs_error=0.0))

    with torch.inference_mode():
        for batch, width in ((1, 128), (3, 2048)):
            for length in (1, 2, 6, 7, 8, 17, 257):
                for dynamic in (False, True):
                    label = f"B={batch}/D={width}/T={length}/dynamic={dynamic}"
                    projected = torch.randn(batch, length, 3, width, device="cuda", dtype=torch.bfloat16)
                    if dynamic:
                        exponents = torch.randint(-20, 31, (batch, length, width), device="cuda")
                        projected[:, :, 2] = projected[:, :, 2] * torch.exp2(exponents.float()).bfloat16()
                    weights = torch.randn(width, 7, device="cuda", dtype=torch.bfloat16) * 0.2
                    inputs = (projected[:, :, 0] * projected[:, :, 2]).transpose(1, 2).contiguous()
                    expected = conv.lfm3_causal_conv1d(inputs, weights).transpose(1, 2) * projected[:, :, 1]
                    output = torch.full(expected.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
                    state = torch.full((batch, width, 7), float("nan"), device="cuda", dtype=torch.bfloat16)
                    bridge.call("conv7_prefill", projected, weights, state, output, batch, length, width)
                    equal("prefill/" + label, output, expected)
                    expected_state = torch.nn.functional.pad(inputs, (max(0, 7 - length), 0))[:, :, -7:].contiguous()
                    equal("prefill-cache/" + label, state, expected_state)
                    # Verify continuation from the prefill-produced cache, not a synthetic zero cache.
                    ref_state = expected_state.clone()
                    for step in range(4):
                        current = torch.randn(batch, 3, width, device="cuda", dtype=torch.bfloat16)
                        native = torch.empty(batch, width, device="cuda", dtype=torch.bfloat16)
                        ref = conv.lfm3_causal_conv1d_update(
                            (current[:, 0] * current[:, 2]).unsqueeze(-1), ref_state, weights).squeeze(-1) * current[:, 1]
                        bridge.call("conv7", current, weights, state, native, batch, width)
                        equal(f"decode/{step}/" + label, native, ref)
                        equal(f"decode-cache/{step}/" + label, state, ref_state)
                    # Repeat empty-cache prefill over dirty cache: no prior state may leak.
                    repeated = torch.empty_like(output)
                    bridge.call("conv7_prefill", projected, weights, state, repeated, batch, length, width)
                    equal("prefill-reset/" + label, repeated, expected)
                    equal("prefill-reset-cache/" + label, state, expected_state)
    report = dict(checks_passed=len(checks), checks=checks, seed=9876,
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  checkpoint_conv_sha256=hashlib.sha256((args.checkpoint / "lfm3_conv.py").read_bytes()).hexdigest(),
                  scope="exact BF16 empty-cache prefill and cached decode comparisons to the pinned checkpoint; no real model activations")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(checks)} exact convolution contract checks")


if __name__ == "__main__":
    main()
