"""Host-only fast-path parity, independent argmax, and unchanged RNG mapping."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "lfm/text_sampling.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <string>

using lfm::TextSampler;
using lfm::TextSamplingOptions;

// Independently select the adjusted maximum, without sorting/filtering or flags.
int oracle(const std::vector<float>& x, const TextSamplingOptions& o,
           const std::vector<int>& history, const std::vector<int>& blocked) {
  int best = -1;
  long double maximum = -std::numeric_limits<long double>::infinity();
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    if (std::find(blocked.begin(), blocked.end(), i) != blocked.end()) continue;
    long double score = x[i];
    if (std::find(history.begin(), history.end(), i) != history.end()) {
      if (score < 0) score *= static_cast<long double>(o.repetition_penalty);
      else score /= static_cast<long double>(o.repetition_penalty);
    }
    if (best == -1 || score > maximum) { best = i; maximum = score; }
  }
  return best;
}

std::string outcome(const std::function<int()>& call) {
  try { return "id:" + std::to_string(call()); }
  catch (const std::invalid_argument& e) { return std::string("invalid:") + e.what(); }
  catch (const std::overflow_error& e) { return std::string("overflow:") + e.what(); }
}

void parity(const std::vector<float>& x, TextSamplingOptions o,
            const std::vector<int>& history = {}, const std::vector<int>& blocked = {}) {
  TextSampler s(o);
  const auto expected = outcome([&] { return s.distribution(x.data(), x.size(), history, blocked).front().token_id; });
  const auto actual = outcome([&] { return s.sample(x.data(), x.size(), history, blocked); });
  assert(actual == expected);
  if (actual.rfind("id:", 0) == 0)
    assert(actual == "id:" + std::to_string(oracle(x, o, history, blocked)));
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  if (mode == "exhaustive") {
    // All 5^3 score tuples, every history subset and nonempty allowed subset.
    const float values[] = {-2, -0.0f, 0.0f, 1, 2};
    for (float a : values) for (float b : values) for (float c : values)
      for (int h = 0; h < 8; ++h) for (int mask = 0; mask < 7; ++mask) {
        std::vector<int> history, blocked;
        for (int i = 0; i < 3; ++i) {
          if (h & (1 << i)) { history.push_back(i); history.push_back(i); }
          if (mask & (1 << i)) { blocked.push_back(i); blocked.push_back(i); }
        }
        for (double penalty : {0.5, 1.0, 2.0}) for (int policy = 0; policy < 4; ++policy) {
          TextSamplingOptions o;
          o.do_sample = policy == 3;
          o.top_k = policy == 3 ? 1 : policy;
          o.repetition_penalty = penalty;
          o.temperature = mask % 2 ? 0.01 : 100.0;
          o.top_p = h % 2 ? 0.1 : 1.0;
          o.min_p = h % 3 ? 1.0 : 0.0;
          parity({a, b, c}, o, history, blocked);
        }
      }
  } else if (mode == "extremes") {
    const float f = std::numeric_limits<float>::max();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const double d = std::numeric_limits<double>::denorm_min();
    const double huge = std::numeric_limits<double>::max();
    for (double temperature : {d, 1.0, huge})
      for (double penalty : {d, 0.5, 1.0, 2.0, huge})
        for (double p : {d, 0.5, 1.0}) for (double m : {0.0, d, 0.5, 1.0})
          for (int policy = 0; policy < 3; ++policy) {
            TextSamplingOptions o;
            o.do_sample = policy == 2; o.top_k = policy == 0 ? 0 : 1;
            o.temperature = temperature; o.repetition_penalty = penalty;
            o.top_p = p; o.min_p = m;
            for (const auto& x : std::vector<std::vector<float>>{
                     {f, -f, tiny, -tiny}, {-f, -f, -f, -f},
                     {f, f, f, f}, {-0.0f, 0.0f, -tiny, tiny}}) {
              parity(x, o, {0, 0, 1, 3, 3});
              parity(x, o, {0, 0, 1, 3, 3}, {0, 0, 2});
            }
          }
    // Production-scale vocabulary: late winner, low-ID tie and full validation.
    std::vector<float> x(128000, -1);
    x.back() = 3; x[123] = 3;
    for (bool sample : {false, true}) {
      TextSamplingOptions o; o.do_sample = sample; o.top_k = sample ? 1 : 0;
      parity(x, o); parity(x, o, {}, {123, 123});
      x.back() = std::numeric_limits<float>::quiet_NaN();
      parity(x, o, {}, {127999});
      x.back() = 3;
    }
  } else if (mode == "invalid") {
    const float x[] = {1, 2, 3};
    for (bool sample : {false, true}) for (std::int64_t k : {0, 1, 4}) {
      if (sample && k != 1) continue;
      TextSamplingOptions o; o.do_sample = sample; o.top_k = k;
      TextSampler s(o);
      const auto reject = [&](const float* ptr, std::size_t n,
                              const std::vector<int>& h, const std::vector<int>& b) {
        auto expected = outcome([&] { return s.distribution(ptr, n, h, b).front().token_id; });
        assert(expected.rfind("invalid:", 0) == 0);
        assert(expected == outcome([&] { return s.sample(ptr, n, h, b); }));
      };
      reject(nullptr, 3, {}, {}); reject(x, 0, {}, {});
      reject(x, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1, {}, {});
      for (int bad : {-1, 3, std::numeric_limits<int>::max()}) {
        reject(x, 3, {0, 0, bad}, {}); reject(x, 3, {}, {0, 0, bad});
      }
      reject(x, 3, {}, {0, 1, 2, 2});
      for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        const float broken[] = {3, 2, bad};
        reject(broken, 3, {}, {}); reject(broken, 3, {2, 2}, {2, 2});
      }
      if (k > 3) reject(x, 3, {}, {});
    }
  } else if (mode == "rng") {
    // General policies still match the documented one-draw inverse CDF,
    // including singleton calls followed by multi-token support and rejection.
    for (int k : {0, 2, 4}) {
      TextSamplingOptions o; o.do_sample = true; o.top_k = k; o.seed = 42;
      TextSampler s(o); std::mt19937_64 engine(o.seed);
      const float x[] = {0, 1, 2, 3};
      for (int step = 0; step < 100; ++step) {
        const std::vector<int> blocked = step % 2 ? std::vector<int>{} : std::vector<int>{0, 1, 2};
        const auto candidates = s.distribution(x, 4, {}, blocked);
        const long double u = static_cast<long double>(engine() >> 11) / 9007199254740992.0L;
        int expected = candidates.back().token_id; long double sum = 0;
        for (const auto& c : candidates) { sum += c.probability; if (u < sum) { expected = c.token_id; break; } }
        assert(s.sample(x, 4, {}, blocked) == expected);
        assert(outcome([&] { return s.sample(x, 4, {}, {0, 1, 2, 3}); }).rfind("invalid:", 0) == 0);
      }
    }
  } else { assert(false); }
  std::cout << "ok\n";
}
'''


@pytest.fixture(scope="module", params=["optimized", "ubsan"])
def fastpath_probe(tmp_path_factory, request):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required for CPU sampler tests")
    directory = tmp_path_factory.mktemp("text-sampling-fastpath-" + request.param)
    source = directory / "probe.cpp"
    source.write_text(SOURCE)
    binary = directory / "probe"
    flags = ["-O2"] if request.param == "optimized" else [
        "-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all",
    ]
    subprocess.run(
        [compiler, "-std=c++17", *flags, "-Wall", "-Wextra", "-Werror",
         "-pedantic", "-I", str(ROOT / "native/include"), str(source),
         "-o", str(binary)], check=True,
    )
    return binary


@pytest.mark.parametrize("scenario", ["exhaustive", "extremes", "invalid", "rng"])
def test_fastpath(fastpath_probe, scenario):
    result = subprocess.run([str(fastpath_probe), scenario], text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_fastpath_rng_draw_contract():
    # Fixed options make top-k1 draws unobservable through public token IDs.
    # Guard the explicit draw without exposing/mutating the private RNG/policy.
    text = (ROOT / "native/include/lfm/text_sampling.hpp").read_text()
    branch = text.split("if (!options_.do_sample || options_.top_k == 1) {", 1)[1]
    branch = branch.split("const auto candidates = distribution(", 1)[0]
    assert branch.count("rng_()") == 1
    assert "if (options_.do_sample) (void)rng_();\n      return best;" in branch
    assert branch.index("if (best < 0) throw") < branch.index("rng_()")
