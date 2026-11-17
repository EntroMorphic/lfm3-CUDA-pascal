"""GPU numerical checks for isolated LFM3 kernels (not end-to-end inference).

Run inside the existing Transformers sandbox after compiling lfm3_ops_bridge.cu.
No ctypes function accepts arbitrary strided tensors: fixtures are contiguous CUDA BF16.
"""
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-reference", choices=("cuda", "cpu"), default="cuda",
                        help="CPU oracle avoids sanitizer instrumentation of third-party NVJET reference GEMMs")
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    bridge = Bridge(args.library)
    checks = []

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def check(name, actual, expected, *, atol=0.015625, rtol=0.01, exact=False):
        a, b = actual.float(), expected.float()
        assert torch.isfinite(a).all(), name
        diff = (a - b).abs()
        record = dict(name=name, elements=a.numel(), max_abs_error=diff.max().item(),
                      relative_l2=((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
                      unequal_elements=int((actual != expected).sum()),
                      atol=0.0 if exact else atol, rtol=0.0 if exact else rtol)
        torch.testing.assert_close(actual, expected, atol=record["atol"], rtol=record["rtol"], msg=name)
        checks.append(record)
        print(json.dumps(record), flush=True)

    # Full-head frequency progression with an identity tail for partial rotation.
    for heads in (4, 16):
        for theta, pairs in ((1e6, 32), (1e4, 64)):
            positions = torch.tensor([0, 1, 255, 256, 4096, 32767], device="cuda", dtype=torch.int32)
            x = rand(6, heads, 128)
            inverse = 1.0 / theta ** (torch.arange(0, 2 * pairs, 2, device="cuda", dtype=torch.float32) / 128)
            inverse = torch.cat([inverse, torch.zeros(64 - pairs, device="cuda")])
            angles = positions.float()[:, None] * inverse[None, :]
            angles = torch.cat([angles, angles], dim=-1)
            cosine, sine = angles.cos().bfloat16().float(), angles.sin().bfloat16().float()
            rotated = torch.cat([-x[..., 64:], x[..., :64]], dim=-1).float()
            expected = (x.float() * cosine[:, None, :] + rotated * sine[:, None, :]).bfloat16()
            actual = x.clone()
            bridge.call("rotary", actual, positions, 6, heads, 128, pairs, theta)
            check(f"rope/heads={heads}/theta={theta}/pairs={pairs}", actual, expected)
            if pairs < 64:
                check(f"rope-tail-a/heads={heads}", actual[..., pairs:64], x[..., pairs:64], exact=True)
                check(f"rope-tail-b/heads={heads}", actual[..., 64 + pairs:], x[..., 64 + pairs:], exact=True)

    for rows in (1, 3, 32):
        attended = rand(rows, 16, 128)
        logits = rand(rows, 16) * 4
        logits[0, :3] = torch.tensor([-80, 0, 80], device="cuda", dtype=torch.bfloat16)
        actual = torch.empty_like(attended)
        bridge.call("gate", attended, logits, actual, rows, 16, 128)
        check(f"head-gate/rows={rows}", actual, attended * logits.sigmoid()[..., None], exact=True)

    for rows, width in ((1, 128), (3, 1024), (3, 2048), (32, 2048)):
        update, residual, weight = rand(rows, width), rand(rows, width), rand(width)
        normalized = (update.float() * torch.rsqrt(update.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16()
        expected = normalized * weight + residual
        actual = torch.empty_like(update)
        bridge.call("sandwich", update, weight, residual, actual, rows, width, 1e-5)
        check(f"sandwich/rows={rows}/width={width}", actual, expected)

    # Use the pinned checkpoint's actual Triton convolution as an independent reference.
    spec = importlib.util.spec_from_file_location("checkpoint_conv", args.checkpoint / "lfm3_conv.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    for batch, width in ((1, 128), (3, 2048)):
        length = 17
        projected = rand(batch, length, 3, width)
        weights = rand(width, 7) * 0.2
        state = torch.zeros(batch, width, 7, device="cuda", dtype=torch.bfloat16)
        reference_state = state.clone()
        outputs = []
        with torch.inference_mode():
            inputs = (projected[:, :, 0] * projected[:, :, 2]).transpose(1, 2).contiguous()
            prefill = conv.lfm3_causal_conv1d(inputs, weights)
            for token in range(length):
                current = projected[:, token].contiguous()
                actual = torch.empty(batch, width, device="cuda", dtype=torch.bfloat16)
                bridge.call("conv7", current, weights, state, actual, batch, width)
                convolved = conv.lfm3_causal_conv1d_update(inputs[:, :, token:token + 1], reference_state, weights)
                expected = convolved.squeeze(-1) * current[:, 1]
                check(f"conv-decode/B={batch}/token={token}", actual, expected)
                check(f"conv-cache/B={batch}/token={token}", state, reference_state, exact=True)
                outputs.append(actual)
        actual = torch.stack(outputs, dim=1)
        expected = prefill.transpose(1, 2) * projected[:, :, 1]
        check(f"conv-prefill-equivalence/B={batch}", actual, expected)
        native_prefill = torch.empty(expected.shape, device="cuda", dtype=torch.bfloat16)
        native_prefill_state = torch.full_like(state, -7)
        bridge.call("conv7_prefill", projected, weights, native_prefill_state, native_prefill, batch, length, width)
        check(f"conv-native-prefill/B={batch}", native_prefill, expected, exact=True)
        check(f"conv-native-prefill-cache/B={batch}", native_prefill_state, reference_state, exact=True)
        # A reset must reproduce the first token exactly.
        state.zero_()
        reset = torch.empty_like(outputs[0])
        bridge.call("conv7", projected[:, 0].contiguous(), weights, state, reset, batch, width)
        check(f"conv-reset/B={batch}", reset, outputs[0], exact=True)

    # Per-row absolute positions; window includes the current token (256 total).
    for dim in (32, 128):
        batch, heads, kv_heads, capacity = 3, 16, 4, 520
        q, k, v = rand(batch, heads, dim), rand(batch, kv_heads, capacity, dim), rand(batch, kv_heads, capacity, dim)
        for pos in (0, 254, 255, 256, 257, 511):
            positions = torch.tensor([pos, max(0, pos - 3), min(capacity - 1, pos + 2)], device="cuda", dtype=torch.int32)
            for window in (0, 256):
                actual = torch.empty_like(q)
                bridge.call("attention", q, k, v, positions, actual, batch, heads, kv_heads, dim, capacity, window)
                expected_rows = []
                ref_q = q.to(args.attention_reference)
                ref_k = k.to(args.attention_reference)
                ref_v = v.to(args.attention_reference)
                for row, position in enumerate(positions.tolist()):
                    begin = max(0, position - window + 1) if window else 0
                    keys = ref_k[row, :, begin:position + 1].repeat_interleave(heads // kv_heads, dim=0)
                    values = ref_v[row, :, begin:position + 1].repeat_interleave(heads // kv_heads, dim=0)
                    scores = (ref_q[row, :, None, :] @ keys.transpose(-1, -2)) * (dim ** -0.5)
                    probability = scores.float().softmax(-1).bfloat16()
                    expected_rows.append((probability @ values).squeeze(1).to(q.device))
                check(f"attention/D={dim}/pos={pos}/window={window}", actual, torch.stack(expected_rows), atol=0.001, rtol=0.005)
                # Neither future keys nor future values may affect causal attention.
                old_k, old_v = k.clone(), v.clone()
                for row, position in enumerate(positions.tolist()):
                    k[row, :, position + 1:] = 100
                    v[row, :, position + 1:] = -100
                causal = torch.empty_like(q)
                bridge.call("attention", q, k, v, positions, causal, batch, heads, kv_heads, dim, capacity, window)
                check(f"attention-future-excluded/D={dim}/pos={pos}/window={window}", causal, actual, exact=True)
                k.copy_(old_k)
                v.copy_(old_v)
                if window and pos >= 256:
                    old = v.clone()
                    for row, position in enumerate(positions.tolist()):
                        v[row, :, :max(0, position - window + 1)] = 100
                    unaffected = torch.empty_like(q)
                    bridge.call("attention", q, k, v, positions, unaffected, batch, heads, kv_heads, dim, capacity, window)
                    check(f"attention-old-values-excluded/D={dim}/pos={pos}", unaffected, actual, exact=True)
                    v.copy_(old)

    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "seed": args.seed,
              "attention_reference": args.attention_reference,
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
              "checkpoint_conv_sha256": hashlib.sha256((args.checkpoint / "lfm3_conv.py").read_bytes()).hexdigest(),
              "checks_passed": len(checks), "checks": checks,
              "scope": "isolated CUDA operators, tolerance-based comparisons; not native model inference or exact end-to-end parity"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(checks)} isolated operator checks")


if __name__ == "__main__":
    main()
