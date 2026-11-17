"""Exact FP32 logit traces (G4): STEP.logits.f32 written next to STEP.logits.bf16 by an FP32-logit engine.

An engine that returns and samples from FP32 logits (the native-GEMM build) writes the exact values as
STEP.logits.f32 and their round-to-nearest-even BF16 rounding as STEP.logits.bf16, the established format. A
gate that reads the .bf16 file must know it is the rounding of what the engine used, and must not accept a
partial set.
"""
import array
import sys
from pathlib import Path


def bf16_rounding_of(exact: bytes) -> bytes:
    """Round-to-nearest-even BF16 bytes of little-endian FP32 bytes. Rejects nonfinite values and bad sizes."""
    if len(exact) % 4:
        raise ValueError("FP32 trace size is not a multiple of 4 bytes")
    words = array.array("I")
    words.frombytes(exact)
    if sys.byteorder != "little":
        words.byteswap()
    for word in words:
        if (word >> 23) & 0xFF == 0xFF:  # inf or NaN: never a valid logit
            raise ValueError("nonfinite value in FP32 trace")
    out = array.array("H", (((w + 0x7FFF + ((w >> 16) & 1)) >> 16) & 0xFFFF for w in words))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def exact_logit_steps(directory: Path, steps) -> bool:
    """Checks STEP.logits.f32 for the given steps: present for all or none; each .bf16 is its rounding.

    Returns True when the exact traces are present (and consistent), False when none are. Raises otherwise.
    """
    steps = list(steps)
    present = [(directory / f"{step}.logits.f32").exists() for step in steps]
    if any(present) and not all(present):
        missing = [step for step, ok in zip(steps, present) if not ok]
        raise ValueError(f"FP32 logit traces missing for steps {missing[:8]} (all or none)")
    if not any(present):
        return False
    for step in steps:
        rounded = (directory / f"{step}.logits.bf16").read_bytes()
        if bf16_rounding_of((directory / f"{step}.logits.f32").read_bytes()) != rounded:
            raise ValueError(f"step {step}: BF16 logit trace is not the rounding of the FP32 trace")
    return True

