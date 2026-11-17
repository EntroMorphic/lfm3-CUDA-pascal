#!/usr/bin/env python3
"""validation/audio-control-v1/report.json: the B2 and A4 record (#2 step 3), reproducible on any machine with g++.

- clean: tests/test_audio_sampling.py and tests/test_audio_modes.py, per test outcome.
- numbers: the B2 figures the README quotes and the tests only bound (exact-distribution differences, chi-squared p).
- planted: each bug below is planted in a copy of native/include (LFM_AUDIO_INCLUDE) and the relevant test file is
  rerun; a bug counts as caught iff at least one test fails. Every plant must be caught.
Usage: lfm3_audio_control_report.py OUT_JSON
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLING, MODES = "audio_sampling.hpp", "audio_modes.hpp"
TS, TM = "tests/test_audio_sampling.py", "tests/test_audio_modes.py"
PLANTS = [
    ("codes written to the reveal position, scored from the first pick's logits", SAMPLING, TS,
     "codes[o[i]] = sample_code(o[i], logits.data() + size_t(o[i]) * kCodes);",
     "codes[i] = sample_code(i, logits.data() + size_t(o[next]) * kCodes);"),
    ("passes scored without the codes revealed so far", SAMPLING, TS,
     "score(codes.data(), masked, picked, logits.data());",
     "{ Frame zero{}; score(zero.data(), masked, picked, logits.data()); }"),
    ("Sattolo's shuffle (cyclic permutations only) for the reveal order", SAMPLING, TS,
     "o[below(std::uint64_t(i) + 1)]", "o[below(std::uint64_t(i))]"),
    ("the code engine shares the seed of a TextSampler", SAMPLING, TS, "t.seed = mix(o.seed);", "t.seed = o.seed;"),
    ("the double-path code engine shares a same-seed TextSampler's stream", SAMPLING, TS, "code_rng_(mix(options.seed)),",
     "code_rng_(options.seed),"),
    ("the double path penalizes from codebook 0's history", SAMPLING, TS, "seen[f[size_t(codebook)]] = true;",
     "seen[f[0]] = true;"),
    ("the double path ignores the repetition penalty", SAMPLING, TS, "if (seen[i]) v = v < 0", "if (false) v = v < 0"),
    ("the double path ignores the temperature", SAMPLING, TS, "std::exp((score[i] - maximum) / options_.temperature)",
     "std::exp(score[i] - maximum)"),
    ("temperature 0 is not greedy", SAMPLING, TS, "t.do_sample = o.temperature > 0;", "t.do_sample = true;"),
    ("top_k ignored", SAMPLING, TS, "t.top_k = std::min(o.top_k, kCodes);", "t.top_k = 0;"),
    ("greedy distribution() reports a temperature-1 softmax", SAMPLING, TS,
     "if (options_.temperature == 0) d = ", "if (false) d = "),
    ("top_k above 2017 refused instead of clamped", SAMPLING, TS, "t.top_k = std::min(o.top_k, kCodes);",
     "t.top_k = o.top_k;"),
    ("the window keeps 63 frames", SAMPLING, TS, "while (history_.size() > size_t(options_.repetition_window))",
     "while (history_.size() >= size_t(options_.repetition_window))"),
    ("EOS only when all 12 codebooks are EOS", MODES, TM,
     "const bool eos = std::find(raw.begin(), raw.end(), kEosCode) != raw.end();",
     "const bool eos = std::count(raw.begin(), raw.end(), kEosCode) == kCodebooks;"),
    ("cadence still counted after <|text_end|>", MODES, TM, "if (interleaving && !text_done_ && --slots_left_ == 0)",
     "if (interleaving && --slots_left_ == 0)"),
    ("text-only requests may sample <|audio_start|>", MODES, TM, "if (o_.suppress_audio_out || text_only_)",
     "if (o_.suppress_audio_out)"),
    ("a replayed EOS frame does not resume text", MODES, TM, "mode_ = last_frame_eos  ", "mode_ = false  "),
    ("a replayed EOS frame resumes text even when the prompt ends elsewhere", MODES, TM,
     "last_replayed_frame && last_prompt_token == t_.audio_out &&", "last_replayed_frame &&"),
    ("<|audio_out|> accepted without its replayed frame", MODES, TM,
     "if (last_prompt_token == t_.audio_out && !last_replayed_frame)", "if (false)"),
    ("the caller's cache is ignored", MODES, TM, "r.decode = o_.caller_cache || ", "r.decode = "),
    ("EOS frames reported as accepted speech", MODES, TM, "r.eos = eos;", "r.eos = false;"),
]


def pytest(test_file, include=None):
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "r.xml"
        env = dict(os.environ, **({"LFM_AUDIO_INCLUDE": include} if include else {}))
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
                        str(ROOT / test_file)], cwd=ROOT, env=env, capture_output=True, text=True)
        cases = ET.parse(xml).getroot().iter("testcase")
        return {c.get("name"): ("failed" if c.find("failure") is not None or c.find("error") is not None
                                else "skipped" if c.find("skipped") is not None else "passed") for c in cases}


def numbers():
    sys.path.insert(0, str(ROOT / "tests"))
    import numpy as np
    import test_audio_sampling as t
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "oracle"
        subprocess.run(["g++", "-std=c++17", "-O2", "-I", t.INCLUDE, str(ROOT / "native/tests/audio_sampling_oracle.cpp"),
                        "-o", str(exe)], check=True)
        out = dict(exact_max_abs=[], draws_chi2=[])
        for i in t.POSITIONS:
            cb = int(t.FIXTURE["codebook"][i])
            p = t.dist_of(exe, [cb], t.case_input(i))
            out["exact_max_abs"].append(float(np.abs(p - t.analytic(i)).max()))
            counts = np.array(t.run(exe, ["codes", 60000, 1000 + i, cb], t.case_input(i)), np.int64)
            pv, bins = t.chi2(counts, t.analytic(i))
            out["draws_chi2"].append(dict(p=float(pv), bins=bins))
        order = np.array(t.run(exe, ["order", 120000, 7]), np.int64)
        out["order_position_chi2_p"] = float(t.stats.chisquare(order[:144]).pvalue)
        out["order_pairs_min_bonferroni_p"] = float(min(1.0, 66 * min(t.stats.binomtest(int(k), 120000, 3 / 11).pvalue
                                                              for k in order[144:])))
        logits, history = t.frame_case()
        stdin = f"64 {' '.join(map(str, history.ravel()))}\n" + " ".join(repr(float(v)) for v in logits.ravel())
        counts = np.array(t.run(exe, ["frame", 20000, 11], stdin), np.int64).reshape(12, 2017)
        out["frame_marginal_min_bonferroni_p"] = float(min(1.0, 12 * min(
            t.stats.chisquare(*map(np.asarray, t._binned(counts[c], t.analytic_of(logits[c], history[:, c])))).pvalue
            for c in range(12))))
        return out


def main(out_json):
    clean = {f: pytest(f) for f in (TS, TM)}
    planted = []
    for name, header, test_file, old, new in PLANTS:
        with tempfile.TemporaryDirectory() as d:
            inc = Path(d) / "include"
            shutil.copytree(ROOT / "native/include", inc)
            h = inc / "lfm" / header
            s = h.read_text()
            assert s.count(old) == 1, (name, old)
            h.write_text(s.replace(old, new))
            r = pytest(test_file, str(inc))
        failed = sorted(k for k, v in r.items() if v == "failed")
        planted.append(dict(bug=name, header=header, tests=len(r), failed=len(failed), failing=failed, caught=bool(failed)))
        print(f"{'caught' if failed else 'MISSED'} {len(failed):3d}/{len(r)}  {name}")
    report = dict(clean={f: dict(passed=sum(v == "passed" for v in r.values()), total=len(r), outcomes=r)
                         for f, r in clean.items()},
                  numbers=numbers(), planted=planted)
    Path(out_json).write_text(json.dumps(report, indent=1) + "\n")
    ok = all(v["passed"] == v["total"] for v in report["clean"].values()) and all(p["caught"] for p in planted)
    print({f: f"{v['passed']}/{v['total']}" for f, v in report["clean"].items()}, "ALL CAUGHT" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1]) else 1)
