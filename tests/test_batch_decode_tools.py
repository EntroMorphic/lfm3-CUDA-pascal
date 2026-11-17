"""Host tests of the batched-decode gate tools: tools/batch_decode_fidelity.py and tools/check_gpu_select.py, on
synthetic inputs whose verdicts are known."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import batch_decode_fidelity as fid  # noqa: E402
import check_gpu_select as sel  # noqa: E402

L, P, R = 64, 40, 12


def write_case(d, engine_noise, serial_noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    fp32 = rng.gamma(2.0, 1.0, (R, L - 1)).astype(np.float32)
    sdpa = (fp32 + rng.normal(0, 0.02, fp32.shape)).astype(np.float32)
    eager = (fp32 + rng.normal(0, 0.03, fp32.shape)).astype(np.float32)
    dec = fp32[:, P - 1:L - 1]
    (dec + rng.normal(0, engine_noise, dec.shape)).astype(np.float32).tofile(d / "x.batch.f32")
    (dec + rng.normal(0, serial_noise, dec.shape)).astype(np.float32).tofile(d / "x.serial.f32")
    for name, a in (("fp32", fp32), ("sdpa", sdpa), ("eager", eager)):
        a.tofile(d / f"{name}.f32")
    w = np.arange(R * L, dtype=np.int32)
    w.tofile(d / "w.i32")
    return hashlib.sha256((d / "w.i32").read_bytes()).hexdigest()


def run(d, sha):
    return fid.main(str(d / "x"), L, P, str(d / "fp32.f32"), str(d / "sdpa.f32"), str(d / "eager.f32"),
                    str(d / "w.i32"), sha)


def test_fidelity_passes_an_engine_closer_to_fp32_than_sdpa(tmp_path):
    assert run(tmp_path, write_case(tmp_path, engine_noise=0.0))


def test_fidelity_fails_an_engine_farther_from_fp32_than_sdpa(tmp_path):
    assert not run(tmp_path, write_case(tmp_path, engine_noise=0.2))


def test_fidelity_refuses_other_windows(tmp_path):
    write_case(tmp_path, 0.0)
    with pytest.raises(SystemExit):
        run(tmp_path, "0" * 64)


def write_draws(d, k, p, counts):
    np.asarray(counts, np.int64).tofile(d / f"case-{k}.counts.i64")
    np.asarray(p, np.float64).tofile(d / f"case-{k}.probs.f64")


def test_select_check_passes_exact_draws_and_fails_biased_ones(tmp_path):
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(200) * 0.3)
    p[17] = 0.0
    p /= p.sum()
    write_draws(tmp_path, 0, p, np.bincount(rng.choice(200, 200000, p=p), minlength=200))
    assert sel.main(str(tmp_path), 1)
    q = p.copy(); q[np.argmax(p)] *= 1.2; q /= q.sum()  # a 20% bias on the likeliest id
    write_draws(tmp_path, 0, p, np.bincount(rng.choice(200, 200000, p=q), minlength=200))
    assert not sel.main(str(tmp_path), 1)


def test_select_check_fails_a_zero_probability_draw(tmp_path):
    rng = np.random.default_rng(4)
    p = np.full(50, 1 / 49); p[5] = 0.0
    c = np.bincount(rng.choice(50, 50000, p=p), minlength=50); c[5] = 1
    write_draws(tmp_path, 0, p, c)
    assert not sel.main(str(tmp_path), 1)
