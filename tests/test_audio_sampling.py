"""B2 (journal/lfm3-tts-quality-bar.md): the native speech-frame sampler (native/include/lfm/audio_sampling.hpp)
against an independent analytic oracle: softmax(penalize(logits) / 0.4), penalty 1.2 on the codebook's codes in
the last 64 accepted frames, as HF's RepetitionPenaltyLogitsProcessor then TemperatureLogitsWarper. The logits are
real depth-decoder logits (FP64 anchor, tts-museum-male; tests/data_audio_sampling_fixture.npz) at six positions
with 0 to 64 history codes (the trace's raw windows, sorted; none holds a repeated code: the penalty makes repeats
rare, 9.7% of this trace's 64-frame windows hold one, over every codebook at a stride of 8 frames (validation/audio-control-v1/repeat-rate.txt),
so the repeat case is built in test_repeated_history_codes_are_one_penalty).
Seeds are fixed, so the chi-squared outcomes are reproducible.
"""
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# tools/lfm3_audio_control_report.py points this at a copy of the headers with a planted bug
INCLUDE = os.environ.get("LFM_AUDIO_INCLUDE", str(ROOT / "native/include"))
FIXTURE = np.load(Path(__file__).with_name("data_audio_sampling_fixture.npz"))
POSITIONS = range(len(FIXTURE["codebook"]))
P_MIN = 1e-3  # every chi-squared test must clear this


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    exe = tmp_path_factory.mktemp("audio_sampling") / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", INCLUDE,
                    str(ROOT / "native/tests/audio_sampling_oracle.cpp"), "-o", str(exe)], check=True)
    return exe


def run(oracle, args, stdin=""):
    return subprocess.run([str(oracle), *map(str, args)], input=stdin, capture_output=True, text=True,
                          check=True).stdout.split()


def case_input(i):
    history = FIXTURE[f"history{i}"]
    return " ".join(repr(float(v)) for v in FIXTURE["logits"][i]) + f"\n{len(history)} " + " ".join(map(str, history))


def analytic(i):
    logits = FIXTURE["logits"][i].astype(np.float64)
    seen = np.zeros(2017, bool)
    seen[FIXTURE[f"history{i}"]] = True
    s = np.where(seen, np.where(logits < 0, logits * 1.2, logits / 1.2), logits) / 0.4
    p = np.exp(s - s.max())
    return p / p.sum()


def chi2(counts, expected_p):
    n = counts.sum()
    expected = expected_p * n
    keep = expected >= 5
    obs = np.append(counts[keep], counts[~keep].sum())
    exp = np.append(expected[keep], expected[~keep].sum())
    if exp[-1] < 5:  # fold a small tail into the smallest kept bin
        j = np.argmin(exp[:-1]); obs[j] += obs[-1]; exp[j] += exp[-1]; obs, exp = obs[:-1], exp[:-1]
    return stats.chisquare(obs, exp * obs.sum() / exp.sum()).pvalue, int(len(obs))


@pytest.mark.parametrize("i", POSITIONS)
def test_exact_distribution_matches_analytic(oracle, i):
    out = run(oracle, ["dist", int(FIXTURE["codebook"][i])], case_input(i))
    p = np.zeros(2017)
    for code, prob in zip(out[::2], out[1::2]):
        p[int(code)] = float(prob)
    # FP32 logits widened to long double vs float64 here: agreement to ~1e-12 absolute
    assert np.abs(p - analytic(i)).max() < 1e-12


@pytest.mark.parametrize("i", POSITIONS)
def test_draws_follow_the_distribution(oracle, i):
    counts = np.array(run(oracle, ["codes", 60000, 1000 + i, int(FIXTURE["codebook"][i])], case_input(i)), np.int64)
    p, bins = chi2(counts, analytic(i))
    assert counts.sum() == 60000 and bins >= 2
    assert p > P_MIN, (p, bins)


def test_history_changes_the_distribution():
    # a fixture position whose history holds high-probability codes: the penalty must matter there
    i = next(i for i in POSITIONS if len(FIXTURE[f"history{i}"]))
    logits = FIXTURE["logits"][i].astype(np.float64)
    plain = np.exp((logits - logits.max()) / 0.4); plain /= plain.sum()
    assert np.abs(plain - analytic(i)).max() > 1e-3


def test_reveal_order_is_uniform(oracle):
    n = 120000
    out = np.array(run(oracle, ["order", n, 7]), np.int64)
    position, together = out[:144].reshape(12, 12), out[144:]
    assert (position.sum(0) == n).all() and (position.sum(1) == n).all()
    assert stats.chisquare(position.ravel(), np.full(144, n / 12)).pvalue > P_MIN
    assert len(together) == len(list(combinations(range(12), 2)))
    # Each permutation puts exactly 18 pairs in a shared pass, so only the per-pair counts are informative:
    # every pair shares a pass with probability 3/11 (Bonferroni over the 66 pairs).
    assert together.sum() == 18 * n
    assert min(stats.binomtest(int(k), n, 3 / 11).pvalue for k in together) > P_MIN / 66


def frame_case():
    """Codebook c scores its own logits (fixture position c % 6, rolled by 157 c over the 2016 speech codes) and has
    its own 64-frame history (that position's history codes rolled likewise, padded with its least likely code)."""
    logits, history = np.zeros((12, 2017)), np.zeros((64, 12), np.int64)
    for c in range(12):
        i, shift = c % 6, 157 * c
        x = FIXTURE["logits"][i].astype(np.float64)
        logits[c, :2016] = np.roll(x[:2016], shift)
        logits[c, 2016] = x[2016]
        h = (FIXTURE[f"history{i}"] + shift) % 2016
        history[:, c] = np.concatenate([h, np.full(64 - len(h), np.argmin(logits[c, :2016]))])
    return logits, history


def test_frame_protocol_and_marginals(oracle):
    # Each codebook has its own logits and history, so a code written to the wrong codebook, scored from another
    # codebook's logits or penalized by another's history changes that codebook's marginal. The oracle also checks
    # that every pass is given exactly the codes the earlier passes sampled.
    n = 20000
    logits, history = frame_case()
    stdin = f"64 {' '.join(map(str, history.ravel()))}\n" + " ".join(repr(float(v)) for v in logits.ravel())
    counts = np.array(run(oracle, ["frame", n, 11], stdin), np.int64).reshape(12, 2017)
    for c in range(12):
        p = analytic_of(logits[c], history[:, c])
        assert stats.chisquare(*map(np.asarray, _binned(counts[c], p))).pvalue > P_MIN / 12, c


def analytic_of(logits, history, temperature=0.4, top_k=0):
    seen = np.zeros(2017, bool)
    seen[history] = True
    s = np.where(seen, np.where(logits < 0, logits * 1.2, logits / 1.2), logits)
    if top_k:
        s = np.where(s >= np.sort(s)[-top_k], s, -np.inf)
    s = s / temperature
    p = np.exp(s - s.max())
    return p / p.sum()


def dist_of(oracle, args, stdin):
    out = run(oracle, ["dist", *args], stdin)
    p = np.zeros(2017)
    for code, prob in zip(out[::2], out[1::2]):
        p[int(code)] = float(prob)
    return p


def test_repeated_history_codes_are_one_penalty(oracle):
    # the fixture's real windows hold no repeat, so repeat them here: HF penalizes a code once (gather/scatter).
    # Flat synthetic logits, so a per-occurrence penalty on the most likely code would move the distribution.
    i = 2
    logits = np.random.default_rng(4).normal(0, 0.5, 2017).astype(np.float32)
    h = FIXTURE[f"history{i}"]
    top = int(np.argmax(logits[:2016]))
    repeated = np.concatenate([h[:30], h[:30], [top] * 8])  # 68 codes, the window keeps the last 64
    stdin = " ".join(repr(float(v)) for v in logits) + f"\n{len(repeated)} " + " ".join(map(str, repeated))
    p = dist_of(oracle, [int(FIXTURE["codebook"][i])], stdin)
    window = repeated[-64:]
    assert np.abs(p - analytic_of(logits.astype(np.float64), window)).max() < 1e-12
    # a per-occurrence penalty would differ here
    x = logits.astype(np.float64).copy()
    for code in window:
        x[code] = x[code] * 1.2 if x[code] < 0 else x[code] / 1.2
    multi = np.exp((x - x.max()) / 0.4); multi /= multi.sum()
    assert np.abs(p - multi).max() > 1e-3


@pytest.mark.parametrize("i", [1, 3])
@pytest.mark.parametrize("top_k", [1, 5, 50])
def test_top_k(oracle, i, top_k):
    stdin = case_input(i)
    want = analytic_of(FIXTURE["logits"][i].astype(np.float64), FIXTURE[f"history{i}"], 0.4, top_k)
    assert np.abs(dist_of(oracle, [int(FIXTURE["codebook"][i]), 0.4, top_k], stdin) - want).max() < 1e-12
    counts = np.array(run(oracle, ["codes", 20000, 30 + top_k, int(FIXTURE["codebook"][i]), 0.4, top_k], stdin), np.int64)
    assert counts[want == 0].sum() == 0
    if top_k > 1:
        assert chi2(counts, want)[0] > P_MIN


@pytest.mark.parametrize("i", POSITIONS)
def test_temperature_zero_is_greedy(oracle, i):
    # HF: temperature 0 builds no warpers, so the code is the argmax of the penalized logits (top_k ignored)
    best = int(np.argmax(analytic_of(FIXTURE["logits"][i].astype(np.float64), FIXTURE[f"history{i}"], 1.0)))
    for top_k in (0, 5):
        counts = np.array(run(oracle, ["codes", 50, 3, int(FIXTURE["codebook"][i]), 0, top_k], case_input(i)), np.int64)
        assert counts[best] == 50
        point = np.zeros(2017); point[best] = 1
        assert (dist_of(oracle, [int(FIXTURE["codebook"][i]), 0, top_k], case_input(i)) == point).all()


def test_top_k_above_the_codes_keeps_them_all(oracle):
    # HF's TopKLogitsWarper clamps top_k to the vocabulary
    p = dist_of(oracle, [int(FIXTURE["codebook"][3]), 0.4, 5000], case_input(3))
    assert np.abs(p - analytic(3)).max() < 1e-12


@pytest.mark.parametrize("top_k", [0, 50])  # 0: the double path; 50: the TextSampler path
def test_code_stream_is_not_a_same_seed_text_stream(oracle, top_k):
    # integration may give text and speech one seed: their draws must not share uniforms
    i = 3
    args = ["pair", 4000, 9, int(FIXTURE["codebook"][i])] + ([0.4, top_k] if top_k else [])
    out = np.array(run(oracle, args, case_input(i)), np.int64).reshape(-1, 2)
    p = analytic(i)
    if top_k:
        keep = np.argsort(-p, kind="stable")[:top_k]
        q = np.zeros_like(p); q[keep] = p[keep]; p = q / q.sum()
    agree = (out[:, 0] == out[:, 1]).mean()
    # independent streams agree with probability sum p^2; a shared stream would agree on every draw
    assert abs(agree - (p ** 2).sum()) < 5 * np.sqrt((p ** 2).sum() / 4000) + 0.01, agree


def _binned(counts, p):
    exp = p * counts.sum()
    keep = exp >= 5
    obs = np.append(counts[keep], counts[~keep].sum())
    e = np.append(exp[keep], exp[~keep].sum())
    return obs, e * obs.sum() / e.sum()


def test_window_and_eos_refusal(oracle):
    out = run(oracle, ["window"])
    assert list(map(int, out[:-1])) == list(range(1, 65)) and out[-1] == "refused-eos"


def test_seeds_reproduce_and_differ(oracle):
    a = run(oracle, ["codes", 2000, 5, int(FIXTURE["codebook"][1])], case_input(1))
    b = run(oracle, ["codes", 2000, 5, int(FIXTURE["codebook"][1])], case_input(1))
    c = run(oracle, ["codes", 2000, 6, int(FIXTURE["codebook"][1])], case_input(1))
    assert a == b and a != c
