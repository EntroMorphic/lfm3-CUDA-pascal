"""Audit-only probes: report boundary acceptance and numerical counterexamples."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_lfm3_ops import Bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    bridge = Bridge(args.library)
    torch.manual_seed(12345)
    findings = []
    observations = []

    def bf(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def should_reject(name, operation):
        try:
            operation()
        except (ValueError, TypeError) as exc:
            observations.append(dict(name=name, rejected=True, detail=str(exc)))
        else:
            findings.append(dict(name=name, problem="invalid input accepted without an error"))

    batch, heads, kv, dim, capacity = 1, 4, 2, 32, 16
    q, k, v = bf(batch, heads, dim), bf(batch, kv, capacity, dim), bf(batch, kv, capacity, dim)
    output = torch.empty_like(q)
    for position in (-1, capacity, capacity + 2):
        p = torch.tensor([position], device="cuda", dtype=torch.int32)
        should_reject(f"attention-position-{position}", lambda: bridge.call(
            "attention", q, k, v, p, output, batch, heads, kv, dim, capacity, 0))
    p = torch.zeros(batch, device="cuda", dtype=torch.int32)
    should_reject("attention-float32-query", lambda: bridge.call(
        "attention", q.float(), k, v, p, output, batch, heads, kv, dim, capacity, 0))
    strided = bf(batch, heads, dim * 2)[..., ::2]
    assert not strided.is_contiguous()
    should_reject("attention-strided-query", lambda: bridge.call(
        "attention", strided, k, v, p, output, batch, heads, kv, dim, capacity, 0))
    should_reject("attention-int64-position", lambda: bridge.call(
        "attention", q, k, v, p.long(), output, batch, heads, kv, dim, capacity, 0))

    # Exhaust broader rotary positions, beyond the six originally selected points.
    rows, heads, dim = 512, 4, 128
    positions = torch.randint(0, 32768, (rows,), device="cuda", dtype=torch.int32)
    for theta, pairs in ((1e6, 32), (1e4, 64)):
        x = bf(rows, heads, dim)
        inverse = 1.0 / theta ** (torch.arange(0, 2 * pairs, 2, device="cuda", dtype=torch.float32) / dim)
        inverse = torch.cat([inverse, torch.zeros(dim // 2 - pairs, device="cuda")])
        angles = positions.float()[:, None] * inverse[None, :]
        angles = torch.cat([angles, angles], -1)
        rotated = torch.cat([-x[..., dim // 2:], x[..., :dim // 2]], -1).float()
        expected = (x.float() * angles.cos().bfloat16().float()[:, None] +
                    rotated * angles.sin().bfloat16().float()[:, None]).bfloat16()
        actual = x.clone()
        bridge.call("rotary", actual, positions, rows, heads, dim, pairs, theta)
        difference = (actual.float() - expected.float()).abs()
        observations.append(dict(name=f"rope-broad/theta={theta}", max_abs=difference.max().item(),
                                 unequal=int((actual != expected).sum())))

    # Cancellation stress: compare to the actual checkpoint Triton decode routine.
    spec = importlib.util.spec_from_file_location("checkpoint_conv", args.checkpoint / "lfm3_conv.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    mismatches = []
    for exponent in (12, 20, 25, 30):
        width = 128
        projected = torch.ones(1, 3, width, device="cuda", dtype=torch.bfloat16)
        weights = torch.ones(width, 7, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(1, width, 7, device="cuda", dtype=torch.bfloat16)
        # Three large terms and tiny terms, with distinct signed channel permutations.
        values = torch.tensor([2.0 ** exponent, 1, -(2.0 ** exponent), 1, 0, 0], device="cuda", dtype=torch.bfloat16)
        for channel in range(width):
            state[0, channel, 1:] = values.roll(channel % 6)
        reference_state = state.clone()
        prefill_input = torch.cat([state[:, :, 1:], projected[:, 0:1].transpose(1, 2)], dim=-1)
        actual = torch.empty(1, width, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = conv.lfm3_causal_conv1d_update(projected[:, 0:1].transpose(1, 2), reference_state, weights).squeeze(-1)
            full_prefill = conv.lfm3_causal_conv1d(prefill_input, weights)
            prefill = full_prefill[:, :, -1]
        bridge.call("conv7", projected, weights, state, actual, 1, width)
        diff = (actual.float() - expected.float()).abs()
        native_prefill_input = torch.ones(1, 7, 3, width, device="cuda", dtype=torch.bfloat16)
        native_prefill_input[:, :, 2] = prefill_input.transpose(1, 2)
        native_prefill_output = torch.empty(1, 7, width, device="cuda", dtype=torch.bfloat16)
        native_prefill_state = torch.empty_like(state)
        bridge.call("conv7_prefill", native_prefill_input, weights, native_prefill_state, native_prefill_output, 1, 7, width)
        prefill_diff = (native_prefill_output.float() - full_prefill.transpose(1, 2).float()).abs()
        observations.append(dict(name=f"conv-cancellation/exponent={exponent}",
                                 decode_max_abs=diff.max().item(), prefill_max_abs=prefill_diff.max().item()))
        if prefill_diff.max() > 0 or not torch.equal(native_prefill_state, reference_state):
            findings.append(dict(name=f"conv-prefill-contract-{exponent}", problem="prefill output or cache disagrees with checkpoint"))
        if diff.max() > 0:
            mismatches.append(dict(exponent=exponent, max_abs=diff.max().item(),
                                   native=actual[0, :6].float().tolist(), reference=expected[0, :6].float().tolist(),
                                   checkpoint_prefill=prefill[0, :6].float().tolist()))
    if mismatches:
        findings.append(dict(name="conv-reduction-cancellation", problem="native sequential reduction differs from checkpoint Triton reduction", cases=mismatches))

    report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, findings=findings,
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  observations=observations, scope="adversarial probes against current candidate; original pre-remediation evidence retained separately")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_clean and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
