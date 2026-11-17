#pragma once
// Host-side sampling of one speech frame (HF `sample_audio_frame`, #2 step 3, bar B2 in
// journal/lfm3-tts-quality-bar.md):
//   - the 12 codebooks are revealed in a uniformly random order, 4 per pass over 3 passes (`maskgit_schedule`);
//   - each pass scores the codebooks it reveals, from the codes revealed so far;
//   - each code is drawn from softmax(penalize(logits) / temperature): the repetition penalty (divide positive
//     logits, multiply negative ones) applies once to every code this codebook emitted in the last 64 accepted
//     frames. These are HF's defaults for speech (0.4, 1.2, 64; no top-k).
//   - as HF's `_build_warpers`: temperature 0 is greedy (argmax of the penalized logits), and top_k > 0 keeps the k
//     best penalized codes. HF's unset temperature (unit temperature with top_k, else greedy) is the caller's to map.
//     Top-k ties at the k-th score keep the lowest codes (TextSampler's policy); HF keeps every tie. A top_k above
//     the 2017 codes keeps them all, as HF's TopKLogitsWarper clamps it.
// Draws: the usual case (temperature > 0, no top-k) has its own path in double precision: the penalty, the softmax
// exp((s - max) / T) and the inverse CDF in code order, one 64-bit Mersenne Twister draw (53 bits) per code. Its
// probabilities equal lfm::TextSampler's long-double ones to double rounding (B2 checks both against the analytic
// distribution). The long-double path costs ~4 ms per code on Thor's ARM cores, where long double is software
// binary128; this one costs microseconds. Greedy and top-k use lfm::TextSampler (the text engine's sampler).
// The code engine and the order engine are seeded from two splitmix64 mixes of the seed, so neither shares a stream
// with a TextSampler built from the same seed.
// PyTorch's Philox stream cannot be reproduced, so equality with HF is distributional, not bitwise.
//
// History lifetime, as HF's `_generate`: one FrameSampler per request; `accept` every frame the mode machine reports
// as not EOS (Outcome::eos false), across interleaved blocks and audio -> text -> audio switches; never seeded from
// the prompt's frames; never cleared within a request.
#include <lfm/text_sampling.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <stdexcept>
#include <vector>

namespace lfm::audio {

constexpr int kCodebooks = 12, kCodes = 2017, kEosCode = 2016;
constexpr std::array<int, 3> kSchedule = {4, 4, 4};

struct FrameSamplingOptions {
  double temperature = 0.4;
  double repetition_penalty = 1.2;
  int repetition_window = 64;  // accepted frames
  std::uint64_t seed = 0;
  int top_k = 0;  // 0: no top-k
};

using Frame = std::array<int, kCodebooks>;

class FrameSampler {
 public:
  explicit FrameSampler(FrameSamplingOptions options = {})
      : options_(checked(options)), codes_(text_options(options_)), code_rng_(mix(options.seed)),
        order_rng_(mix(mix(options.seed))) {}

  // A uniformly random reveal order (Fisher-Yates with unbiased bounded draws).
  std::array<int, kCodebooks> order() {
    std::array<int, kCodebooks> o;
    for (int i = 0; i < kCodebooks; ++i) o[i] = i;
    for (int i = kCodebooks - 1; i > 0; --i) std::swap(o[i], o[below(std::uint64_t(i) + 1)]);
    return o;
  }

  // One code of `codebook` from its 2017 logits, penalized by the codebook's recent history.
  int sample_code(int codebook, const float* logits) {
    if (codebook < 0 || codebook >= kCodebooks) throw std::invalid_argument("audio codebook out of range");
    if (options_.temperature == 0 || options_.top_k) return codes_.sample(logits, kCodes, history_codes(codebook));
    bool seen[kCodes] = {};
    for (const Frame& f : history_) seen[f[size_t(codebook)]] = true;
    double score[kCodes], maximum = -INFINITY;
    for (int i = 0; i < kCodes; ++i) {
      if (!std::isfinite(logits[i])) throw std::invalid_argument("audio logits must all be finite");
      double v = logits[i];
      if (seen[i]) v = v < 0 ? v * options_.repetition_penalty : v / options_.repetition_penalty;
      score[i] = v;
      maximum = std::max(maximum, v);
    }
    double total = 0;
    for (int i = 0; i < kCodes; ++i) total += (score[i] = std::exp((score[i] - maximum) / options_.temperature));
    const double target = double(code_rng_() >> 11) * (1.0 / 9007199254740992.0) * total;
    double cumulative = 0;
    int last = 0;
    for (int i = 0; i < kCodes; ++i) {
      if (score[i] <= 0) continue;
      last = i;
      cumulative += score[i];
      if (target < cumulative) return i;
    }
    return last;  // rounding at the very end of the CDF
  }

  // One frame drawn pass by pass (a batch interleaves many): begin() draws the reveal order; while !done(), the caller
  // scores the codebooks `picked` from `codes` where !`masked`, then take() draws them from those logits. sample_frame
  // is the same steps in one call.
  struct Draw {
    std::array<int, kCodebooks> order{};
    Frame codes{};
    bool masked[kCodebooks], picked[kCodebooks];
    int pass = 0, next = 0;
    bool done() const { return pass == int(kSchedule.size()); }
  };
  Draw begin() {
    Draw d;
    d.order = order();
    for (int c = 0; c < kCodebooks; ++c) d.masked[c] = true;
    pick(d);
    return d;
  }
  // `logits`: at least the picked codebooks' 2017 logits, codebook c at logits + c * 2017 (host).
  void take(Draw& d, const float* logits) {
    if (d.done()) throw std::logic_error("audio frame draw already complete");
    const int count = kSchedule[size_t(d.pass)];
    for (int i = d.next; i < d.next + count; ++i) d.codes[d.order[i]] = sample_code(d.order[i], logits + size_t(d.order[i]) * kCodes);
    for (int i = d.next; i < d.next + count; ++i) d.masked[d.order[i]] = false;
    d.next += count;
    ++d.pass;
    if (!d.done()) pick(d);
  }

  // One frame. `score(codes, masked, picked, logits)` must write the logits of every picked codebook c to
  // logits[c * 2017 ...] (host), from `codes` where !masked. Returns the raw codes (any may be the EOS code 2016).
  template <class Score>
  Frame sample_frame(Score&& score) {
    Draw d = begin();
    std::vector<float> logits(size_t(kCodebooks) * kCodes);
    while (!d.done()) {
      score(d.codes.data(), d.masked, d.picked, logits.data());
      take(d, logits.data());
    }
    return d.codes;
  }

  // An accepted (non-EOS) frame joins the penalty history; HF keeps the last `repetition_window`.
  void accept(const Frame& frame) {
    for (int code : frame)
      if (code < 0 || code >= kEosCode) throw std::invalid_argument("accepted audio frame holds EOS or invalid code");
    if (!options_.repetition_window) return;
    history_.push_back(frame);
    while (history_.size() > size_t(options_.repetition_window)) history_.pop_front();
  }
  void clear_history() { history_.clear(); }
  // The codes the penalty applies to for `codebook` (with repeats; the sampler treats them as a set).
  std::vector<int> history_codes(int codebook) const {
    std::vector<int> h;
    for (const Frame& f : history_) h.push_back(f.at(size_t(codebook)));
    return h;
  }
  // The exact sampling distribution of one code (no draw), for checking. Greedy (temperature 0): the argmax alone.
  std::vector<TextCandidate> distribution(int codebook, const float* logits) const {
    auto d = codes_.distribution(logits, kCodes, history_codes(codebook));
    if (options_.temperature == 0) d = {{d.front().token_id, 1.0L}};  // the highest penalized score, lowest code
    return d;
  }

 private:
  static void pick(Draw& d) {
    for (int c = 0; c < kCodebooks; ++c) d.picked[c] = false;
    for (int i = d.next; i < d.next + kSchedule[size_t(d.pass)]; ++i) d.picked[d.order[i]] = true;
  }
  static FrameSamplingOptions checked(FrameSamplingOptions o) {
    if (o.repetition_window < 0) throw std::invalid_argument("invalid audio repetition window");
    if (!(o.temperature >= 0) || !std::isfinite(o.temperature)) throw std::invalid_argument("invalid audio temperature");
    if (o.top_k < 0) throw std::invalid_argument("invalid audio top_k");
    return o;
  }
  static TextSamplingOptions text_options(const FrameSamplingOptions& o) {
    TextSamplingOptions t;
    t.do_sample = o.temperature > 0;  // HF: temperature 0 builds no warpers, i.e. argmax
    t.temperature = o.temperature > 0 ? o.temperature : 1.0;
    t.top_k = std::min(o.top_k, kCodes);
    t.repetition_penalty = o.repetition_penalty;
    t.seed = mix(o.seed);
    return t;
  }
  static std::uint64_t mix(std::uint64_t x) {  // splitmix64: decorrelates the order engine from the code engine
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }
  std::uint64_t below(std::uint64_t n) {  // uniform in [0, n), by rejection
    const std::uint64_t threshold = (0 - n) % n;
    for (;;) {
      const std::uint64_t r = order_rng_();
      if (r >= threshold) return r % n;
    }
  }

  FrameSamplingOptions options_;
  TextSampler codes_;
  std::mt19937_64 code_rng_, order_rng_;
  std::deque<Frame> history_;
};

}  // namespace lfm::audio
