#!/usr/bin/env python3
"""validation/round-once-v1, rule 0. Runs in the container: is every output of the round-once prefill kernel the
correctly rounded value of its exact dot product? Builds native/tests/round_once_check.cu, feeds it BF16 operands
(model-like magnitudes; exponents spread over 100 binades with cancellation; subnormal and zero operands; sums
built to sit exactly on, just above and just below a rounding boundary of BF16 and of FP32; a 32,768-term sum), and compares every output with Python integer arithmetic
rounded once, to nearest even. Also counts how many outputs the plain kernel gets differently (information only).
    python3 tools/pascal/round_once_check.py [BUILD_DIR=/workspace/build/round-once-check]"""
import array, os, random, struct, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/build/round-once-check"); B.mkdir(parents=True, exist_ok=True)
SCALE = 266  # a product of two BF16 values is an integer multiple of 2^-266

def bf16_bits(v):
    """The BF16 bit pattern of a float that is exactly representable in BF16."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    assert struct.unpack("<f", struct.pack("<f", v))[0] == v and bits & 0xFFFF == 0, v
    return bits >> 16

def scaled(bits):
    """A finite BF16 bit pattern as an integer in units of 2^-133 (zeros and subnormals included)."""
    sign, exponent, fraction = bits >> 15, (bits >> 7) & 0xFF, bits & 0x7F
    assert exponent < 255
    value = ((128 if exponent else 0) + fraction) << (max(exponent, 1) - 1)
    return -value if sign else value

def round_once(total, precision):
    """total * 2^-SCALE rounded to nearest even at `precision` bits, as a Python float (exact)."""
    if total == 0: return 0.0
    magnitude, shift = abs(total), max(0, abs(total).bit_length() - precision)
    quotient, remainder = magnitude >> shift, magnitude & ((1 << shift) - 1)
    if shift and (remainder > (1 << (shift - 1)) or (remainder == (1 << (shift - 1)) and quotient & 1)): quotient += 1
    value = float(quotient) * 2.0 ** (shift - SCALE)
    return -value if total < 0 else value

def random_bf16(rng, low, high):
    return bf16_bits((1 if rng.random() < 0.5 else -1) * (128 + rng.randrange(128)) / 128 * 2.0 ** rng.randint(low, high))

def boundary_cases():
    """Lists of (a, b) terms whose sum sits on a rounding boundary, or a hair above or below it. Each starts with
    +2^40 and ends with -2^40, so the running sum is large while the small terms arrive."""
    ties = {"BF16 tie, even below": [(1.0, 1.0), (1.0, 2.0 ** -8)],                      # 1 + 2^-8
            "BF16 tie, odd below": [(1.0, 1.0), (1.0, 2.0 ** -7), (1.0, 2.0 ** -8)],    # 1 + 2^-7 + 2^-8
            "FP32 tie, even below": [(1.0, 1.0), (2.0 ** -12, 2.0 ** -12)],             # 1 + 2^-24
            "FP32 tie, odd below": [(1.0, 1.0), (2.0 ** -12, 2.0 ** -11), (2.0 ** -12, 2.0 ** -12)]}
    cases = []
    for terms in ties.values():
        for sign in (1.0, -1.0):          # the whole sum negated too
            for tiny in (None, 15, -15, 35, -35, 50, -50, 120, -120):  # +-2^-30, 2^-70, 2^-100, 2^-240
                case = [(2.0 ** 20, 2.0 ** 20)] + [(sign * a, b) for a, b in terms]
                if tiny: case.append(((1.0 if tiny > 0 else -1.0) * 2.0 ** -abs(tiny), 2.0 ** -abs(tiny)))
                cases.append(case + [(-(2.0 ** 20), 2.0 ** 20)])
    return cases

def problem(rng, rows, n, k, kind):
    def fill():
        if kind == "model": return None
        if kind == "spread": return random_bf16(rng, -60, 40)
        if kind == "tiny": return rng.choice([0, 0x8000, rng.randrange(1, 128), 0x8000 | rng.randrange(1, 128), random_bf16(rng, -126, -100)])
        return rng.choice([0, 0x8000])
    x = [[fill() if kind != "model" else random_bf16(rng, -3, 3) for _ in range(k)] for _ in range(rows)]
    w = [[fill() if kind != "model" else random_bf16(rng, -8, -3) for _ in range(k)] for _ in range(n)]
    if kind == "tiny":   # subnormal and zero operands, with a few normal terms so that results are in the normal range
        for r in range(rows):
            for i in range(0, k, 16): x[r][i] = random_bf16(rng, -20, 0)
        for c in range(n):
            for i in range(0, k, 16): w[c][i] = random_bf16(rng, -20, 0)
    if kind == "boundary":  # pair (row, column) = case; each case has its own k positions, zero for every other pair
        for index, terms in enumerate(boundary_cases()):
            r, c = index % rows, index // rows
            assert c < n and index * 8 + len(terms) <= k
            for t, (a, b) in enumerate(terms):
                x[r][index * 8 + t] = bf16_bits(a); w[c][index * 8 + t] = bf16_bits(b)
    return x, w

def run(exe, x, w, rows, n, k, layout, output, env):
    array.array("H", [v for row in x for v in row]).tofile(open(B / "x.bf16", "wb"))
    flat = [v for col in w for v in col] if layout == "kc" else [w[c][i] for i in range(k) for c in range(n)]
    array.array("H", flat).tofile(open(B / "w.bf16", "wb"))
    subprocess.run([str(exe), str(B / "x.bf16"), str(rows), str(k), str(B / "w.bf16"), str(n), layout, output, str(B / "out.bin")],
                   check=True, env={**os.environ, **env})
    out = array.array("H" if output == "bf16" else "I"); out.fromfile(open(B / "out.bin", "rb"), rows * n)
    return out

def main():
    flags = ["-O3", "--fmad=false", "-std=c++20", "-arch=sm_61", "-Wno-deprecated-gpu-targets", "-DLFM_GEMM_ROUND_ONCE",
             "-I", str(ROOT / "native/include")]
    exe = B / "round-once-check"
    subprocess.run(["nvcc", *flags, "-o", str(exe), str(ROOT / "native/tests/round_once_check.cu")], check=True)
    rng = random.Random(20261118); ok = True
    for kind, rows, n, k in (("model", 5, 40, 2048), ("model", 3, 24, 7680), ("spread", 5, 40, 2048), ("spread", 9, 16, 512), ("tiny", 5, 16, 512),
                              ("boundary", 9, 8, 576), ("spread", 3, 8, 32768)):
        x, w = problem(rng, rows, n, k, kind)
        xs = [[scaled(v) for v in row] for row in x]; ws = [[scaled(v) for v in col] for col in w]
        exact = [[sum(a * b for a, b in zip(xs[r], ws[c])) for c in range(n)] for r in range(rows)]
        for layout in ("kc", "kn"):
            for output, precision in (("bf16", 8), ("f32", 24)):
                want = [round_once(exact[r][c], precision) for r in range(rows) for c in range(n)]
                if output == "bf16": want_bits = [bf16_bits(v) for v in want]
                else: want_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in want]
                once = run(exe, x, w, rows, n, k, layout, output, {"LFM_ONCE_ROWS": "1000000"})
                plain = run(exe, x, w, rows, n, k, layout, output, {"LFM_ONCE_ROWS": "0"})
                wrong = sum(1 for a, b in zip(once, want_bits) if a != b)
                plain_wrong = sum(1 for a, b in zip(plain, want_bits) if a != b)
                ok &= wrong == 0
                print(f"{kind:8s} {rows} x {n} x {k:4d} {layout} {output:4s}: round-once {len(want_bits) - wrong}/{len(want_bits)} correctly rounded"
                      f" ({'PASS' if wrong == 0 else 'FAIL'}); plain kernel differs from the correct rounding in {plain_wrong}", flush=True)
    print("round_once_check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
