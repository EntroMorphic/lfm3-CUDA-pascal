"""Compile the production sampler with a host compiler; no CUDA/GPU required."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "lfm/text_sampling.hpp"
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <string>

using lfm::TextSampler;
using lfm::TextSamplingOptions;

void invalid(const std::function<void()>& call) {
  bool caught = false;
  try { call(); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
}
void near(long double a, long double b, long double tolerance = 1e-7L) {
  assert(std::abs(a - b) < tolerance);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string test = argv[1];
  const float logits[] = {0, 1, 2, 3};
  if (test == "greedy") {
    TextSampler sampler;
    assert(sampler.sample(logits, 4) == 3);
    const float ties[] = {2, 2, 2, -1};
    assert(sampler.sample(ties, 4) == 0);
    assert(sampler.sample(ties, 4, {}, {0, 0}) == 1);
    assert(sampler.sample(logits, 4, {}, {3, 2, 1}) == 0);
    auto d = sampler.distribution(ties, 4);
    assert(d[0].token_id == 0 && d[1].token_id == 1 && d[2].token_id == 2);
  } else if (test == "invalid_options") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double v : {-1.0, 0.0, nan, inf, -inf}) {
      TextSamplingOptions o;
      o.temperature = v; invalid([&] { TextSampler s(o); });
      o = {}; o.repetition_penalty = v; invalid([&] { TextSampler s(o); });
      o = {}; o.top_p = v; invalid([&] { TextSampler s(o); });
    }
    for (double v : {-1.0, 1.001, nan, inf, -inf}) {
      TextSamplingOptions o;
      o.min_p = v; invalid([&] { TextSampler s(o); });
    }
    TextSamplingOptions o;
    o.top_p = 1.001; invalid([&] { TextSampler s(o); });
    o = {}; o.top_k = -1; invalid([&] { TextSampler s(o); });
    o.top_k = 5; TextSampler s(o); invalid([&] { s.sample(logits, 4); });
    o.top_k = std::numeric_limits<std::int64_t>::max();
    TextSampler huge(o); invalid([&] { huge.sample(logits, 4); });
  } else if (test == "invalid_inputs") {
    TextSampler s;
    invalid([&] { s.sample(nullptr, 4); });
    invalid([&] { s.sample(logits, 0); });
    invalid([&] { s.sample(logits, static_cast<std::size_t>(INT32_MAX) + 1); });
    for (int id : {-1, 4, INT32_MAX}) {
      invalid([&] { s.sample(logits, 4, {id}); });
      invalid([&] { s.sample(logits, 4, {}, {id}); });
    }
    invalid([&] { s.sample(logits, 4, {}, {0, 1, 2, 3}); });
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
      const float bad[] = {0, value};
      invalid([&] { s.sample(bad, 2); });
      invalid([&] { s.sample(bad, 2, {}, {1}); });
    }
  } else if (test == "repetition") {
    TextSamplingOptions o; o.repetition_penalty = 2;
    TextSampler s(o);
    const float positive[] = {4, 3, 1};
    assert(s.sample(positive, 3, {0, 0, 0}) == 1);
    auto a = s.distribution(positive, 3, {0});
    auto b = s.distribution(positive, 3, {0, 0, 0});
    for (std::size_t i = 0; i < a.size(); ++i) near(a[i].probability, b[i].probability);
    const float negative[] = {-1, -1.5f, -4};
    assert(s.sample(negative, 3, {0}) == 1);
    o.repetition_penalty = .5; TextSampler encourage(o);
    assert(encourage.sample(positive, 3, {1}) == 1);
    assert(encourage.sample(negative, 3, {1}) == 1);
    const float zeros[] = {-0.0f, 0.0f};
    assert(s.sample(zeros, 2, {0}) == 0);
  } else if (test == "filters") {
    const float equal[] = {0, 0, 0, 0};
    TextSamplingOptions o; o.top_k = 2;
    auto d = TextSampler(o).distribution(equal, 4);
    assert(d.size() == 2 && d[0].token_id == 0 && d[1].token_id == 1);
    near(d[0].probability, .5L);
    o = {}; o.top_p = .5;
    d = TextSampler(o).distribution(equal, 4); assert(d.size() == 2);
    o.top_p = .50001;
    d = TextSampler(o).distribution(equal, 4); assert(d.size() == 3);
    o.top_p = std::numeric_limits<double>::denorm_min();
    d = TextSampler(o).distribution(equal, 4); assert(d.size() == 1);
    o = {}; o.min_p = 1;
    d = TextSampler(o).distribution(equal, 4); assert(d.size() == 4);
    d = TextSampler(o).distribution(logits, 4); assert(d.size() == 1);
    o.min_p = .3;
    d = TextSampler(o).distribution(logits, 4); assert(d.size() == 2);
    o.top_k = 3; o.top_p = .7;
    d = TextSampler(o).distribution(logits, 4); assert(d.size() == 1);
    o = {}; o.temperature = 2;
    const float pair[] = {0, 2};
    d = TextSampler(o).distribution(pair, 2);
    near(d[0].probability, 1.L / (1.L + std::exp(-1.L)));
  } else if (test == "extremes") {
    const float extreme[] = {std::numeric_limits<float>::max(),
                             -std::numeric_limits<float>::max(), 0};
    for (double temp : {std::numeric_limits<double>::denorm_min(),
                        std::numeric_limits<double>::max(), 1.0}) {
      for (double penalty : {std::numeric_limits<double>::denorm_min(),
                             std::numeric_limits<double>::max(), 1.0}) {
        TextSamplingOptions o; o.temperature = temp; o.repetition_penalty = penalty;
        TextSampler s(o);
        auto d = s.distribution(extreme, 3, {0, 1});
        long double total = 0;
        for (auto c : d) {
          assert(std::isfinite(c.probability) && c.probability > 0);
          total += c.probability;
        }
        near(total, 1.L, 1e-15L);
        assert(s.sample(extreme, 3, {0, 1}) == 0);
      }
    }
    TextSamplingOptions o; o.temperature = std::numeric_limits<double>::denorm_min();
    const float equal[] = {std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    auto d = TextSampler(o).distribution(equal, 2);
    assert(d.size() == 2); near(d[0].probability, .5L);
  } else if (test == "rng") {
    TextSamplingOptions o; o.do_sample = true; o.seed = 42;
    TextSampler a(o), b(o), untouched(o);
    const float equal[] = {0, 0, 0, 0};
    // First mt19937_64(42) output is 13930160852258120406, u ~= .755.
    assert(a.sample(equal, 4) == 3);
    assert(b.sample(equal, 4) == 3);
    for (int i = 0; i < 1000; ++i) {
      a.distribution(equal, 4);
      invalid([&] { a.sample(equal, 4, {}, {0, 1, 2, 3}); });
      assert(a.sample(equal, 4) == b.sample(equal, 4));
    }
    TextSampler copy = a;
    for (int i = 0; i < 100; ++i) assert(copy.sample(equal, 4) == a.sample(equal, 4));
    TextSampler singleton(o);
    singleton.sample(equal, 4, {}, {1, 2, 3});
    untouched.sample(equal, 4);
    assert(singleton.sample(equal, 4) == untouched.sample(equal, 4));
    o.seed = 43; TextSampler other(o);
    bool differs = false;
    for (int i = 0; i < 100; ++i) differs |= other.sample(equal, 4) != b.sample(equal, 4);
    assert(differs);
  } else if (test == "properties") {
    std::mt19937_64 random(7331);
    for (int iteration = 0; iteration < 1000; ++iteration) {
      std::vector<float> values(17);
      for (auto& value : values) value = static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 13;
      TextSamplingOptions o; o.do_sample = true; o.seed = random();
      o.temperature = std::pow(10., static_cast<int>(random() % 13) - 6);
      o.repetition_penalty = std::pow(10., static_cast<int>(random() % 9) - 4);
      o.top_k = random() % 18;
      o.top_p = (1 + random() % 100) / 100.;
      o.min_p = (random() % 101) / 100.;
      TextSampler s(o);
      const auto d = s.distribution(values.data(), values.size(), {1, 1, 3}, {2, 4});
      assert(!d.empty());
      long double total = 0;
      for (std::size_t i = 0; i < d.size(); ++i) {
        assert(d[i].token_id != 2 && d[i].token_id != 4);
        assert(d[i].probability > 0 && std::isfinite(d[i].probability));
        if (i) assert(d[i - 1].probability >= d[i].probability);
        total += d[i].probability;
      }
      near(total, 1.L, 1e-15L);
      for (int draw = 0; draw < 10; ++draw) {
        const int id = s.sample(values.data(), values.size(), {1, 1, 3}, {2, 4});
        assert(std::any_of(d.begin(), d.end(), [id](const lfm::TextCandidate& c) { return c.token_id == id; }));
      }
    }
  } else if (test == "statistics") {
    TextSamplingOptions o; o.do_sample = true; o.seed = 123; o.top_k = 3;
    const float values[] = {0, std::log(2.f), std::log(3.f), -100};
    TextSampler s(o);
    std::array<int, 4> counts{};
    for (int i = 0; i < 40000; ++i) ++counts[s.sample(values, 4)];
    assert(counts[3] == 0);
    near(counts[0] / 40000.L, 1.L / 6, .012L);
    near(counts[1] / 40000.L, 2.L / 6, .012L);
    near(counts[2] / 40000.L, 3.L / 6, .012L);
    o.min_p = .8; TextSampler restricted(o);
    for (int i = 0; i < 1000; ++i) assert(restricted.sample(values, 4) == 2);
  } else { assert(false); }
  std::cout << "ok\n";
}
'''


@pytest.fixture(scope="module")
def sampling_probe(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required for CPU sampler tests")
    directory = tmp_path_factory.mktemp("text-sampling")
    source = directory / "probe.cpp"
    source.write_text(SOURCE)
    binary = directory / "probe"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-pedantic", "-I", str(ROOT / "native/include"), str(source),
         "-o", str(binary)], check=True,
    )
    return binary


@pytest.mark.parametrize("scenario", [
    "greedy", "invalid_options", "invalid_inputs", "repetition", "filters",
    "extremes", "rng", "properties", "statistics",
])
def test_text_sampling(sampling_probe, scenario):
    result = subprocess.run([str(sampling_probe), scenario], text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
