#pragma once
// Audio input (#3 audio in): the LFM3-Omni log-mel front end (feature_extraction_lfm3_omni.py, Lfm3OmniFeatureExtractor
// with the checkpoint's processor_config.json), on the host.
//
// 16 kHz mono samples, right-padded with zeros to n_fft + k * hop; frames of 400 samples every 160 (center=False),
// a periodic Hann window, the power spectrum of each frame's 201 bins, 128 Slaney mel filters over 0-8000 Hz
// (transformers.audio_utils.mel_filter_bank, norm="slaney", mel_scale="slaney"), log10 clamped below at -12, then
// (x + 4) / 4. The spectrum and the mel products are computed in double here (HF: float32 FFT and matmul), so the
// features agree with HF's to float rounding; the encoder reads them rounded to BF16.
// Layout as HF's input_features for one clip: [128 mel][frames]. Host-only.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace lfm::audio_features {

inline constexpr int kRate = 16000, kFft = 400, kHop = 160, kBins = kFft / 2 + 1, kMels = 128, kStride = 8;
// Host copies of one clip's stages (the parity check against HF): [128][T] features, the downsampler's [N][768], the
// encoder's [N][768] and the projected [N][2048], BF16 values as float.
struct Dump {
  std::vector<float> mel, down, encoded, features;
};

inline constexpr double kLogFloor = -12.0, kOffset = 4.0, kDivisor = 4.0;

// The padded length and frame count of a clip of n samples (Lfm3OmniFeatureExtractor.pad_length / num_frames).
inline size_t padded_length(size_t n) { return n < size_t(kFft) ? size_t(kFft) : n + (kHop - (n - kFft) % kHop) % kHop; }
inline size_t frames(size_t n) { return 1 + (padded_length(n) - kFft) / kHop; }
// Encoder tokens (placeholders) for a clip of n samples: one per 8 frames, rounded up.
inline size_t tokens(size_t n) { return (frames(n) + kStride - 1) / kStride; }

namespace detail {
inline double hz_to_mel(double f) {
  const double logstep = 27.0 / std::log(6.4);
  return f >= 1000.0 ? 15.0 + std::log(f / 1000.0) * logstep : 3.0 * f / 200.0;
}
inline double mel_to_hz(double m) {
  const double logstep = std::log(6.4) / 27.0;
  return m >= 15.0 ? 1000.0 * std::exp(logstep * (m - 15.0)) : 200.0 * m / 3.0;
}
// numpy.linspace(start, stop, num): start + i * step, the last element set to stop
inline std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> y(num);
  const double step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) y[i] = start + i * step;
  y[num - 1] = stop;
  return y;
}
}  // namespace detail

// [bin][mel], as mel_filter_bank returns it (float64), then as HF uses it (cast to float32)
inline const std::vector<float>& mel_filters() {
  static const std::vector<float> filters = [] {
    const auto mel = detail::linspace(detail::hz_to_mel(0.0), detail::hz_to_mel(kRate / 2), kMels + 2);
    std::vector<double> hz(mel.size());
    for (size_t i = 0; i < mel.size(); ++i) hz[i] = detail::mel_to_hz(mel[i]);
    const auto fft = detail::linspace(0.0, kRate / 2, kBins);
    std::vector<float> f(size_t(kBins) * kMels);
    for (int b = 0; b < kBins; ++b)
      for (int m = 0; m < kMels; ++m) {
        const double down = -(hz[m] - fft[b]) / (hz[m + 1] - hz[m]);
        const double up = (hz[m + 2] - fft[b]) / (hz[m + 2] - hz[m + 1]);
        const double tri = std::max(0.0, std::min(down, up));
        f[size_t(b) * kMels + m] = float(tri * (2.0 / (hz[m + 2] - hz[m])));
      }
    return f;
  }();
  return filters;
}

// The clip's features, [128][frames(n)] (row-major: mel-major, as HF's input_features[0]).
inline std::vector<float> log_mel(const std::vector<float>& samples) {
  if (samples.empty()) throw std::runtime_error("audio features: empty clip");
  const size_t T = frames(samples.size());
  std::vector<float> x(padded_length(samples.size()), 0.0f);
  std::copy(samples.begin(), samples.end(), x.begin());
  // torch.hann_window(400) (periodic), in float32
  static const std::vector<float> window = [] {
    std::vector<float> w(kFft);
    for (int n = 0; n < kFft; ++n) w[n] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * n / kFft));
    return w;
  }();
  static const std::vector<double> cosine = [] {
    std::vector<double> c(kFft);
    for (int i = 0; i < kFft; ++i) c[i] = std::cos(2.0 * M_PI * i / kFft);
    return c;
  }();
  static const std::vector<double> sine = [] {
    std::vector<double> s(kFft);
    for (int i = 0; i < kFft; ++i) s[i] = std::sin(2.0 * M_PI * i / kFft);
    return s;
  }();
  const auto& filters = mel_filters();
  std::vector<float> out(size_t(kMels) * T);
  std::vector<double> frame(kFft), power(kBins), mel(kMels);
  for (size_t t = 0; t < T; ++t) {
    for (int n = 0; n < kFft; ++n) frame[n] = double(x[t * kHop + n] * window[n]);  // the windowed float32 sample
    for (int k = 0; k < kBins; ++k) {
      double re = 0, im = 0;
      for (int n = 0, i = 0; n < kFft; ++n, i += k) {
        if (i >= kFft) i -= kFft;
        re += frame[n] * cosine[i];
        im -= frame[n] * sine[i];
      }
      power[k] = re * re + im * im;
    }
    std::fill(mel.begin(), mel.end(), 0.0);
    for (int k = 0; k < kBins; ++k)
      for (int m = 0; m < kMels; ++m) mel[m] += double(filters[size_t(k) * kMels + m]) * power[k];
    for (int m = 0; m < kMels; ++m)
      out[size_t(m) * T + t] = float((std::max(std::log10(mel[m]), kLogFloor) + kOffset) / kDivisor);
  }
  return out;
}

}  // namespace lfm::audio_features
