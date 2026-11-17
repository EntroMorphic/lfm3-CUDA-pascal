"""Negative tests for the checked tensor wrapper and private raw-ABI metadata guard."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch

from lfm3_bridge import Bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge = Bridge(args.library)
    checks = []
    q = torch.ones(2, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(2, 2, 16, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    p = torch.tensor([0, 15], device="cuda", dtype=torch.int32)
    out = torch.full_like(q, -7)

    def attention(**changes):
        values = dict(q=q, k=k, v=v, p=p, out=out, batch=2, heads=4, kv=2, dim=32, capacity=16, window=0)
        values.update(changes)
        return bridge.call("attention", *values.values())

    def reject(name, operation):
        out.fill_(-7)
        try:
            operation()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"not rejected before kernel launch: {name}")
        torch.cuda.synchronize()
        assert torch.all(out == -7), f"rejected call wrote output: {name}"
        checks.append(name)

    for position in (-1, 16, 18, 2**31 - 1):
        reject(f"position-{position}", lambda position=position: attention(p=torch.tensor([0, position], device="cuda", dtype=torch.int32)))
    reject("float32-query", lambda: attention(q=q.float()))
    reject("int64-positions", lambda: attention(p=p.long()))
    reject("cpu-query", lambda: attention(q=q.cpu()))
    reject("strided-query", lambda: attention(q=torch.ones(2, 4, 64, device="cuda", dtype=torch.bfloat16)[..., ::2]))
    reject("wrong-shape", lambda: attention(q=q[:, :3].contiguous()))
    reject("writable-alias", lambda: attention(out=q))
    reject("requires-grad", lambda: attention(q=q.clone().requires_grad_()))
    for field, value in (("batch", 0), ("batch", -1), ("batch", True), ("batch", 2**32 + 2),
                         ("batch", 65536), ("heads", 3), ("kv", 0), ("dim", 257),
                         ("capacity", 4097), ("window", -1), ("window", 0.5)):
        reject(f"metadata/{field}={value}", lambda field=field, value=value: attention(**{field: value}))
    reject("unknown-op", lambda: bridge.call("bogus"))
    reject("wrong-argc", lambda: bridge.call("attention", q))

    projected = torch.ones(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(128, 7, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, 128, 7, device="cuda", dtype=torch.bfloat16)
    conv_out = torch.empty(2, 128, device="cuda", dtype=torch.bfloat16)
    reject("conv-state-shape", lambda: bridge.call("conv7", projected, weights, state[..., :6].contiguous(), conv_out, 2, 128))
    reject("conv-state-output-alias", lambda: bridge.call("conv7", projected, weights, state, state.flatten()[:256].view(2, 128), 2, 128))
    reject("gate-alias", lambda: bridge.call("gate", q, torch.ones(2, 4, device="cuda", dtype=torch.bfloat16), q, 2, 4, 32))
    reject("rope-odd-width", lambda: bridge.call("rotary", q, p, 2, 4, 31, 15, 1e6))
    reject("rope-too-many-pairs", lambda: bridge.call("rotary", q, p, 2, 4, 32, 17, 1e6))
    reject("rope-inf-theta", lambda: bridge.call("rotary", q, p, 2, 4, 32, 16, float("inf")))
    reject("rope-overflow-theta", lambda: bridge.call("rotary", q, p, 2, 4, 32, 16, 1e300))
    reject("rope-integer-overflow-theta", lambda: bridge.call("rotary", q, p, 2, 4, 32, 16, 10**400))
    reject("rope-underflow-theta", lambda: bridge.call("rotary", q, p, 2, 4, 32, 16, 1e-300))
    reject("sandwich-nan-epsilon", lambda: bridge.call("sandwich", q.view(2, 128), weights[:, 0].contiguous(), q.view(2, 128), torch.empty(2, 128, device="cuda", dtype=torch.bfloat16), 2, 128, float("nan")))

    # Bypass the Python checks deliberately: raw metadata/position validation
    # must fail without writing output or poisoning subsequent valid launches.
    raw = bridge._lib.lfm3_attention
    def raw_attention(positions, metadata=(2, 4, 2, 32, 16, 0), query=q):
        pointers = [ct.c_void_p(t.data_ptr()) for t in (query, k, v, positions, out)]
        return raw(*pointers, *metadata, ct.c_void_p(torch.cuda.current_stream().cuda_stream))

    for position in (-1, 16, 18, 2**31 - 1):
        out.fill_(-7)
        invalid = torch.tensor([0, position], device="cuda", dtype=torch.int32)
        assert raw_attention(invalid) != 0
        torch.cuda.synchronize()
        assert torch.all(out == -7)
        checks.append(f"raw-position-{position}")
    for metadata in ((65536, 4, 2, 32, 16, 0), (2, 4, 0, 32, 16, 0),
                     (2, 4, 2, 32, 4097, 0), (2, 4, 2, 32, 16, -1),
                     (65535, 65532, 4, 256, 4096, 0)):
        assert raw_attention(p, metadata) != 0
        checks.append(f"raw-metadata-{metadata}")
    assert raw_attention(p, query=q.cpu()) != 0
    checks.append("raw-host-pointer")
    attention()
    assert torch.all(out == 1)
    checks.append("valid-after-rejections")
    # Stream-local validation/launch and asynchronous input producers.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        v.fill_(2)
        attention()
        assert torch.all(out == 2)
    checks.append("nondefault-stream")
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2_000_000)
        v.fill_(3)
    # No explicit wait on producer: the checked harness must establish readiness.
    attention()
    assert torch.all(out == 3)
    checks.append("unjoined-producer-stream")
    graph = torch.cuda.CUDAGraph()
    marker = torch.zeros(1, device="cuda")
    rejected_capture = False
    with torch.cuda.graph(graph):
        marker.fill_(1)
        try:
            attention()
        except ValueError as exc:
            assert "graph capture" in str(exc)
            rejected_capture = True
    assert rejected_capture
    graph.replay()
    torch.cuda.synchronize()
    assert marker.item() == 1
    checks.append("graph-capture-rejected")
    report = {"checks_passed": len(checks), "checks": checks,
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "scope": "checked Python boundary, raw metadata/range rejection, producer readiness, stream recovery and graph rejection"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
