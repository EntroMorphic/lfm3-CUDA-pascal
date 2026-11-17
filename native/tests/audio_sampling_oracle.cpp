// Oracle for lfm/audio_sampling.hpp (B2). Reads stdin, writes stdout; one mode per call:
//   dist CODEBOOK [T K]     stdin: 2017 logits, H, then H history codes -> "code probability" lines (exact)
//   codes N SEED CODEBOOK [T K]   same stdin -> 2017 counts of N draws
//   (T K: temperature, 0 for greedy, and top_k; default 0.4 0)
//   pair N SEED CODEBOOK [T K]   same stdin -> N lines: a FrameSampler draw, then a same-seed TextSampler draw
//   order N SEED        -> 144 counts [position][codebook], then 66 counts of codebook pairs sharing a pass
//   frame N SEED        stdin: H, H frames of 12 history codes, then 12 x 2017 logits (codebook c's own) -> checks
//                       the pass protocol and that each pass sees the codes sampled before it; prints 12 x 2017
//                       counts of the frame's codes
//   window              accepts 65 frames, prints codebook 0's history (the window keeps the last 64)
#include <lfm/audio_sampling.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace lfm::audio;

static void read_case(std::vector<float>& logits, FrameSampler& sampler, int codebook) {
  logits.resize(kCodes);
  for (auto& v : logits) std::cin >> v;
  int h;
  std::cin >> h;
  for (int i = 0; i < h; ++i) {
    Frame f{};
    std::cin >> f[size_t(codebook)];
    sampler.accept(f);
  }
}

static FrameSamplingOptions options(int argc, char** argv, int at, std::uint64_t seed) {
  FrameSamplingOptions o{0.4, 1.2, 64, seed};
  if (argc == at + 2) o.temperature = std::atof(argv[at]), o.top_k = std::atoi(argv[at + 1]);
  return o;
}

int main(int argc, char** argv) try {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "dist" && (argc == 3 || argc == 5)) {
    FrameSampler sampler(options(argc, argv, 3, 0));
    std::vector<float> logits;
    const int c = std::atoi(argv[2]);
    read_case(logits, sampler, c);
    std::cout << std::setprecision(18);
    for (const auto& d : sampler.distribution(c, logits.data())) std::cout << d.token_id << " " << d.probability << "\n";
  } else if (mode == "codes" && (argc == 5 || argc == 7)) {
    FrameSampler sampler(options(argc, argv, 5, std::strtoull(argv[3], nullptr, 10)));
    std::vector<float> logits;
    const int c = std::atoi(argv[4]);
    read_case(logits, sampler, c);
    std::vector<long> counts(kCodes);
    for (long n = std::atol(argv[2]); n--;) ++counts[size_t(sampler.sample_code(c, logits.data()))];
    for (long v : counts) std::cout << v << "\n";
  } else if (mode == "pair" && (argc == 5 || argc == 7)) {
    // the same seed: FrameSampler's code draws next to a TextSampler's with the same options
    const std::uint64_t seed = std::strtoull(argv[3], nullptr, 10);
    const FrameSamplingOptions o = options(argc, argv, 5, seed);
    FrameSampler sampler(o);
    std::vector<float> logits;
    const int c = std::atoi(argv[4]);
    read_case(logits, sampler, c);
    lfm::TextSamplingOptions t;
    t.do_sample = true, t.temperature = o.temperature, t.top_k = o.top_k, t.repetition_penalty = 1.2, t.seed = seed;
    lfm::TextSampler text(t);
    const auto history = sampler.history_codes(c);
    for (long n = std::atol(argv[2]); n--;)
      std::cout << sampler.sample_code(c, logits.data()) << " " << text.sample(logits.data(), kCodes, history) << "\n";
  } else if (mode == "order" && argc == 4) {
    FrameSampler sampler({0.4, 1.2, 64, std::strtoull(argv[3], nullptr, 10)});
    std::vector<long> position(144), together(66);
    for (long n = std::atol(argv[2]); n--;) {
      const auto o = sampler.order();
      int pass[kCodebooks];
      for (int i = 0; i < kCodebooks; ++i) ++position[size_t(i * 12 + o[i])], pass[o[i]] = i / 4;
      for (int a = 0, k = 0; a < kCodebooks; ++a)
        for (int b = a + 1; b < kCodebooks; ++b, ++k) together[size_t(k)] += pass[a] == pass[b];
    }
    for (long v : position) std::cout << v << "\n";
    for (long v : together) std::cout << v << "\n";
  } else if (mode == "frame" && argc == 4) {
    FrameSampler sampler({0.4, 1.2, 64, std::strtoull(argv[3], nullptr, 10)});
    int h;
    std::cin >> h;
    for (int i = 0; i < h; ++i) {
      Frame f;
      for (int& c : f) std::cin >> c;
      sampler.accept(f);
    }
    std::vector<float> own(size_t(kCodebooks) * kCodes);
    for (auto& v : own) std::cin >> v;
    if (!std::cin) throw std::runtime_error("bad input");
    std::vector<long> counts(size_t(kCodebooks) * kCodes);
    for (long n = std::atol(argv[2]); n--;) {
      int call = 0, revealed = 0;
      bool was_masked[kCodebooks];
      for (bool& m : was_masked) m = true;
      // seen[p][c]: the code of revealed codebook c that pass p was given (-1 while masked)
      int seen[3][kCodebooks];
      const Frame f = sampler.sample_frame([&](const int* codes, const bool* masked, const bool* picked, float* out) {
        if (call >= 3) throw std::runtime_error("more than 3 passes");
        int picks = 0, still = 0;
        for (int c = 0; c < kCodebooks; ++c) {
          if (picked[c] && !masked[c]) throw std::runtime_error("picked a revealed codebook");
          if (masked[c] != was_masked[c]) throw std::runtime_error("mask changed outside a reveal");
          picks += picked[c];
          still += masked[c];
          seen[call][c] = masked[c] ? -1 : codes[c];
          if (picked[c]) std::copy(own.begin() + long(c) * kCodes, own.begin() + long(c + 1) * kCodes, out + size_t(c) * kCodes);
        }
        if (picks != kSchedule[size_t(call)] || still != kCodebooks - revealed)
          throw std::runtime_error("pass protocol");
        for (int c = 0; c < kCodebooks; ++c) if (picked[c]) was_masked[c] = false;
        revealed += picks;
        ++call;
      });
      if (call != 3) throw std::runtime_error("expected 3 passes");
      // every later pass must see exactly the codes the frame returns for the codebooks revealed before it
      for (int p = 0; p < 3; ++p)
        for (int c = 0; c < kCodebooks; ++c)
          if (seen[p][c] != -1 && seen[p][c] != f[size_t(c)]) throw std::runtime_error("a pass saw a code the frame lost");
      for (int c = 0; c < kCodebooks; ++c) ++counts[size_t(c) * kCodes + size_t(f[size_t(c)])];
    }
    for (long v : counts) std::cout << v << "\n";
  } else if (mode == "window" && argc == 2) {
    FrameSampler sampler;
    for (int i = 0; i < 65; ++i) {
      Frame f{};
      f[0] = i;
      sampler.accept(f);
    }
    for (int v : sampler.history_codes(0)) std::cout << v << "\n";
    try {
      Frame eos{};
      eos[3] = kEosCode;
      sampler.accept(eos);
      std::cout << "accepted-eos\n";
    } catch (const std::invalid_argument&) {
      std::cout << "refused-eos\n";
    }
  } else {
    std::cerr << "bad arguments\n";
    return 2;
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio_sampling_oracle: " << e.what() << "\n";
  return 1;
}
