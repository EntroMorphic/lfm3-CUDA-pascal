"""Checked, synchronous, inference-only boundary for the isolated CUDA test ABI.

Raw C symbols are private test plumbing: pointers alone cannot encode PyTorch
shape/stride/alias contracts. Call this wrapper, not those symbols, from Python.
This is deliberately not a production/graph-capture/performance interface. A
device-wide producer barrier establishes readiness across input streams; the
caller must not mutate participating tensors concurrently with the call.
"""
import ctypes as ct
import math
from pathlib import Path
import struct

import torch


class Bridge:
    MAX_ELEMENTS = 2**31 - 1

    def __init__(self, path):
        self._lib = ct.CDLL(str(Path(path).resolve()))
        p, i, f = ct.c_void_p, ct.c_int, ct.c_float
        self._signatures = {
            "rotary": [p, p, i, i, i, i, f, p],
            "gate": [p, p, p, i, i, i, p],
            "sandwich": [p, p, p, p, i, i, f, p],
            "conv7": [p, p, p, p, i, i, p],
            "conv7_prefill": [p, p, p, p, i, i, i, p],
            "attention": [p, p, p, p, p, i, i, i, i, i, i, p],
        }
        for name, signature in self._signatures.items():
            fn = getattr(self._lib, "lfm3_" + name)
            fn.argtypes, fn.restype = signature, i

    @staticmethod
    def _int(value, name, minimum=1, maximum=2**31 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name}: invalid integer metadata")
        return value

    @staticmethod
    def _float32(value, name):
        if type(value) not in (int, float) or value <= 0:
            raise ValueError(f"{name}: expected positive finite scalar")
        try:
            if not math.isfinite(value):
                raise ValueError(f"{name}: expected positive finite scalar")
            converted = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"{name}: float32 overflow") from exc
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError(f"{name}: invalid float32 value")

    def _contract(self, name, args):
        if name not in self._signatures or len(args) != len(self._signatures[name]) - 1:
            raise ValueError("unknown operator or incorrect argument count")
        i = self._int
        if name == "rotary":
            x, positions, rows, heads, dim, pairs, theta = args
            i(rows, "rows", maximum=65535); i(heads, "heads", maximum=65535)
            i(dim, "head_dim", maximum=1024); i(pairs, "pairs", minimum=0, maximum=dim // 2)
            if dim % 2:
                raise ValueError("RoPE requires even head_dim")
            self._float32(theta, "theta")
            return [x, positions], [(rows, heads, dim), (rows,)], {1}, {0}
        if name == "gate":
            x, logits, output, rows, heads, dim = args
            i(rows, "rows", maximum=65535); i(heads, "heads", maximum=65535); i(dim, "head_dim", maximum=1024)
            return [x, logits, output], [(rows, heads, dim), (rows, heads), (rows, heads, dim)], set(), {2}
        if name == "sandwich":
            x, weight, residual, output, rows, width, epsilon = args
            i(rows, "rows", maximum=65535); i(width, "width", maximum=1048576)
            self._float32(epsilon, "epsilon")
            return [x, weight, residual, output], [(rows, width), (width,), (rows, width), (rows, width)], set(), {3}
        if name == "conv7":
            projected, weight, state, output, batch, width = args
            i(batch, "batch", maximum=65535); i(width, "width", maximum=1048576)
            return [projected, weight, state, output], [(batch, 3, width), (width, 7), (batch, width, 7), (batch, width)], set(), {2, 3}
        if name == "conv7_prefill":
            projected, weight, state, output, batch, length, width = args
            i(batch, "batch", maximum=65535); i(length, "length", maximum=32768); i(width, "width", maximum=1048576)
            return [projected, weight, state, output], [(batch, length, 3, width), (width, 7),
                                                       (batch, width, 7), (batch, length, width)], set(), {2, 3}
        q, k, v, positions, output, batch, heads, kv, dim, capacity, window = args
        i(batch, "batch", maximum=65535); i(heads, "heads", maximum=65535); i(kv, "kv_heads", maximum=65535)
        i(dim, "head_dim", maximum=256); i(capacity, "capacity", maximum=4096); i(window, "window", minimum=0)
        if heads % kv:
            raise ValueError("query heads must be divisible by KV heads")
        return [q, k, v, positions, output], [(batch, heads, dim), (batch, kv, capacity, dim),
                                             (batch, kv, capacity, dim), (batch,), (batch, heads, dim)], {3}, {4}

    def call(self, name, *args):
        tensors, shapes, integer_tensors, writes = self._contract(name, args)
        device = None
        spans = []
        for index, (tensor, shape) in enumerate(zip(tensors, shapes)):
            if math.prod(shape) > self.MAX_ELEMENTS:
                raise ValueError("tensor/launch size exceeds supported range")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("expected tensor argument")
            dtype = torch.int32 if index in integer_tensors else torch.bfloat16
            if tensor.layout != torch.strided or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"tensor {index}: wrong layout, dtype or shape")
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.is_conj() or tensor.is_neg() or tensor.requires_grad:
                raise ValueError(f"tensor {index}: expected contiguous CUDA inference tensor")
            if device is None:
                device = tensor.device
            if tensor.device != device:
                raise ValueError("all tensors must reside on the same CUDA device")
            spans.append((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()))
        for left in range(len(tensors)):
            for right in range(left + 1, len(tensors)):
                if (left in writes or right in writes) and max(spans[left][0], spans[right][0]) < min(spans[left][1], spans[right][1]):
                    raise ValueError("writable tensor overlaps another argument")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("test bridge does not support CUDA graph capture")
            # This correctness harness is intentionally synchronous, not timed.
            # Tensor objects do not expose their producer stream for validation.
            torch.cuda.synchronize(device)
            if name in {"attention", "rotary"}:
                positions = tensors[3 if name == "attention" else 1]
                maximum = args[-2] if name == "attention" else 2**31 - 1
                if positions.min().item() < 0 or positions.max().item() >= maximum:
                    raise ValueError("position outside valid cache/rotary range")
            pointers = [ct.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else x for x in args]
            stream = torch.cuda.current_stream(device)
            status = getattr(self._lib, "lfm3_" + name)(*pointers, ct.c_void_p(stream.cuda_stream))
            if status:
                raise RuntimeError(f"{name}: CUDA validation/launch failed: {status}")
            stream.synchronize()
