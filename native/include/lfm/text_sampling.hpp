#pragma once

// Host-only text sampling. See audit/SAMPLING.md for the serving contract.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace lfm {

struct TextSamplingOptions {
  bool do_sample = false;
  double temperature = 1.0;
  std::int64_t top_k = 0;  // Zero disables; otherwise must not exceed vocab size.
  double top_p = 1.0;
  double min_p = 0.0;
  double repetition_penalty = 1.0;
  std::uint64_t seed = 0;
};

struct TextCandidate {
  int token_id;
  long double probability;
};

class TextSampler {
 public:
  explicit TextSampler(TextSamplingOptions options = {})
      : options_(options), rng_(options.seed) {
    if (!std::isfinite(options.temperature) || options.temperature <= 0 ||
        options.top_k < 0 || !std::isfinite(options.top_p) ||
        options.top_p <= 0 || options.top_p > 1 ||
        !std::isfinite(options.min_p) || options.min_p < 0 || options.min_p > 1 ||
        !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0)
      throw std::invalid_argument("invalid text sampling options");
  }

  // History is a set semantically: repeated occurrences receive ONE penalty.
  // Suppression is explicit, not encoded as +/-infinity in the logits.
  // Returns positive probabilities in score-descending, token-ID-ascending order.
  // Does not advance RNG, even when do_sample=true.
  std::vector<TextCandidate> distribution(
      const float* logits, std::size_t vocab_size,
      const std::vector<int>& history = {},
      const std::vector<int>& suppressed = {}) const {
    if (!logits || vocab_size == 0 ||
        vocab_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid text logits/vocabulary");
    if (static_cast<std::uint64_t>(options_.top_k) > vocab_size)
      throw std::invalid_argument("top_k exceeds vocabulary size");
    std::vector<unsigned char> seen(vocab_size, 0), blocked(vocab_size, 0);
    const auto mark = [vocab_size](const std::vector<int>& ids,
                                 std::vector<unsigned char>& flags) {
      for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= vocab_size)
          throw std::invalid_argument("text token ID out of vocabulary bounds");
        flags[static_cast<std::size_t>(id)] = 1;
      }
    };
    mark(history, seen);
    mark(suppressed, blocked);
    struct Score { int id; long double value; };
    std::vector<Score> scores;
    scores.reserve(vocab_size);
    for (std::size_t i = 0; i < vocab_size; ++i) {
      // Validate even suppressed entries: NaNs must not hide in a request.
      if (!std::isfinite(logits[i]))
        throw std::invalid_argument("text logits must all be finite");
      if (blocked[i]) continue;
      long double value = logits[i];
      if (seen[i]) {
        const long double penalty = options_.repetition_penalty;
        value = value < 0 ? value * penalty : value / penalty;
      }
      if (!std::isfinite(value))
        throw std::overflow_error("repetition penalty exceeds host arithmetic range");
      scores.push_back({static_cast<int>(i), value});
    }
    if (scores.empty()) throw std::invalid_argument("all text tokens suppressed");
    std::sort(scores.begin(), scores.end(), [](const Score& a, const Score& b) {
      return a.value > b.value || (a.value == b.value && a.id < b.id);
    });
    if (options_.top_k && static_cast<std::size_t>(options_.top_k) < scores.size())
      scores.resize(static_cast<std::size_t>(options_.top_k));

    // Subtract BEFORE dividing by temperature: huge positive logits and tiny
    // temperatures cannot overflow to +inf or produce inf-inf/NaN softmax.
    // Negative overflow/exp underflow represents zero mass and is intentional.
    std::vector<TextCandidate> result;
    result.reserve(scores.size());
    const long double maximum = scores.front().value;
    long double total = 0;
    for (const auto& score : scores) {
      const long double weight = std::exp(
          (score.value - maximum) / static_cast<long double>(options_.temperature));
      // Maximum has weight exactly 1; relative min_p needs no normalization.
      if (weight > 0 && weight >= static_cast<long double>(options_.min_p)) {
        result.push_back({score.id, weight});
        total += weight;
      }
    }
    if (options_.top_p < 1.0) {
      const long double threshold = static_cast<long double>(options_.top_p) * total;
      long double cumulative = 0;
      std::size_t keep = 0;
      do { cumulative += result[keep++].probability; }
      while (keep < result.size() && cumulative < threshold);
      result.resize(keep);  // Include the token crossing the nucleus threshold.
      total = cumulative;
    }
    for (auto& candidate : result) candidate.probability /= total;
    return result;
  }

  int sample(const float* logits, std::size_t vocab_size,
             const std::vector<int>& history = {},
             const std::vector<int>& suppressed = {}) {
    if (!options_.do_sample || options_.top_k == 1) {
      // The maximum survives temperature, min_p and top_p filtering. Avoid
      // sorting/softmax, but retain distribution()'s validation and arithmetic.
      if (!logits || vocab_size == 0 ||
          vocab_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid text logits/vocabulary");
      if (static_cast<std::uint64_t>(options_.top_k) > vocab_size)
        throw std::invalid_argument("top_k exceeds vocabulary size");
      std::vector<unsigned char> seen(vocab_size, 0), blocked(vocab_size, 0);
      const auto mark = [vocab_size](const std::vector<int>& ids,
                                   std::vector<unsigned char>& flags) {
        for (int id : ids) {
          if (id < 0 || static_cast<std::size_t>(id) >= vocab_size)
            throw std::invalid_argument("text token ID out of vocabulary bounds");
          flags[static_cast<std::size_t>(id)] = 1;
        }
      };
      mark(history, seen);
      mark(suppressed, blocked);
      int best = -1;
      long double maximum = 0;
      for (std::size_t i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(logits[i]))
          throw std::invalid_argument("text logits must all be finite");
        if (blocked[i]) continue;
        long double value = logits[i];
        if (seen[i]) {
          const long double penalty = options_.repetition_penalty;
          value = value < 0 ? value * penalty : value / penalty;
        }
        if (!std::isfinite(value))
          throw std::overflow_error("repetition penalty exceeds host arithmetic range");
        // Ascending iteration and strict comparison retain the lowest-ID tie.
        if (best < 0 || value > maximum) {
          best = static_cast<int>(i);
          maximum = value;
        }
      }
      if (best < 0) throw std::invalid_argument("all text tokens suppressed");
      // Successful stochastic singleton calls still consume exactly one draw;
      // greedy and rejected calls consume none, as on the general path.
      if (options_.do_sample) (void)rng_();
      return best;
    }
    const auto candidates = distribution(logits, vocab_size, history, suppressed);
    if (!options_.do_sample) return candidates.front().token_id;
    // Exactly one engine draw for every successful stochastic call (including
    // singleton support). Fixed 53-bit mapping avoids std::distribution variance.
    const long double uniform = static_cast<long double>(rng_() >> 11) *
                                (1.0L / 9007199254740992.0L);
    long double cumulative = 0;
    for (const auto& candidate : candidates) {
      cumulative += candidate.probability;
      if (uniform < cumulative) return candidate.token_id;
    }
    return candidates.back().token_id;  // Final normalization roundoff only.
  }

 private:
  TextSamplingOptions options_;
  std::mt19937_64 rng_;
};

}  // namespace lfm
