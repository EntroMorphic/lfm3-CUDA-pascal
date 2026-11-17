// The vision front end after decoding: see lfm/image_preprocess.hpp for what it reproduces and why.
#include <lfm/image_preprocess.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace lfm::image {

Size smart_resize(int height, int width, int factor, std::uint64_t min_pixels, std::uint64_t max_pixels) {
  // Python: max(h, w) / min(h, w) > 200 (a correctly rounded double, as here: both sides are exact in a double).
  const double aspect = double(std::max(height, width)) / double(std::min(height, width));
  if (aspect > 200) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", aspect);
    throw DecodeError("aspect_ratio", std::string("absolute aspect ratio must be smaller than 200, got ") + buf);
  }
  const double f = factor;
  // round() ties to even: nearbyint in the default rounding mode
  long long h_bar = (long long)std::nearbyint(double(height) / f) * factor;
  long long w_bar = (long long)std::nearbyint(double(width) / f) * factor;
  const double hw = double(std::uint64_t(height) * std::uint64_t(width));  // exact: under 2^53 after decoding
  if (std::uint64_t(h_bar * w_bar) > max_pixels) {
    const double beta = std::sqrt(hw / double(max_pixels));
    h_bar = std::max<long long>(factor, (long long)std::floor(double(height) / beta / f) * factor);
    w_bar = std::max<long long>(factor, (long long)std::floor(double(width) / beta / f) * factor);
  } else if (std::uint64_t(h_bar * w_bar) < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / hw);
    h_bar = (long long)std::ceil(double(height) * beta / f) * factor;
    w_bar = (long long)std::ceil(double(width) * beta / f) * factor;
  }
  return Size{int(h_bar), int(w_bar)};
}

namespace {

// Which multiply-adds torch's compiled weight code fuses is not visible from its source, so it is explicit here,
// independent of this file's compiler flags, and was settled by measurement (validation/images-v1):
// LFM_RESIZE_FMA_MODE 0 no fusion; 1 the cubic polynomial fused; 2 also every use of center = scale * (i + 0.5).
// 0 and 1 give the same effective int16 kernel on every size pair tried (381,000 exhaustive up to 1,500, 200,000
// random up to 20,000); 2 differs on 8 of the random pairs, and on those torch matches 0 and 1, not 2 (x86 and
// aarch64). Kept switchable for re-measuring when torch changes.
#ifndef LFM_RESIZE_FMA_MODE
#define LFM_RESIZE_FMA_MODE 0
#endif

// HelperInterpCubic::aa_filter<double, true>: Keys cubic convolution, a = -0.5 (the constants 5a, 8a, 4a folded at
// compile time, as in torch).
double cubic_filter(double x) {
  constexpr double a = -0.5;
  x = std::abs(x);
  if (x < 1.0) {
    if constexpr (LFM_RESIZE_FMA_MODE == 0) return ((a + 2) * x - (a + 3)) * x * x + 1;
    return std::fma(std::fma(a + 2, x, -(a + 3)) * x, x, 1.0);
  }
  if (x < 2.0) {
    if constexpr (LFM_RESIZE_FMA_MODE == 0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return std::fma(std::fma(std::fma(a, x, -(5 * a)), x, 8 * a), x, -(4 * a));
  }
  return 0.0;
}

struct Taps {
  std::vector<int> xmin, xsize;
  std::vector<std::int16_t> weights;  // out_size rows of max_interp_size
  int max_interp_size = 0;
  unsigned precision = 0;
};

// HelperInterpBase::_compute_index_ranges_weights (antialias) and _compute_index_ranges_int16_weights.
Taps taps(int input_size, int output_size) {
  Taps t;
  const double scale = double(input_size) / output_size;  // area_pixel_compute_scale, align_corners=False
  const double support = (scale >= 1.0) ? (4 * 0.5) * scale : 4 * 0.5;
  t.max_interp_size = int(std::ceil(support)) * 2 + 1;
  const int m = t.max_interp_size;
  std::vector<double> w(std::size_t(output_size) * m, 0.0);
  t.xmin.resize(output_size);
  t.xsize.resize(output_size);
  double wt_max = 0.0;
  for (int i = 0; i < output_size; ++i) {
    const double ip5 = i + 0.5;
    const double center = scale * ip5;
    double total = 0.0;
    const double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    const double lo = LFM_RESIZE_FMA_MODE == 2 ? std::fma(scale, ip5, -support) : center - support;
    const double hi = LFM_RESIZE_FMA_MODE == 2 ? std::fma(scale, ip5, support) : center + support;
    const std::int64_t xmin = std::max(static_cast<std::int64_t>(lo + 0.5), std::int64_t(0));
    std::int64_t xsize = std::min(static_cast<std::int64_t>(hi + 0.5), std::int64_t(input_size)) - xmin;
    xsize = std::clamp(xsize, std::int64_t(0), std::int64_t(m));
    double* wi = &w[std::size_t(i) * m];
    for (std::int64_t j = 0; j < xsize; ++j) {
      const double d = LFM_RESIZE_FMA_MODE == 2 ? std::fma(-scale, ip5, double(j + xmin)) : double(j + xmin) - center;
      const double v = cubic_filter((d + 0.5) * invscale);
      wi[j] = v;
      total += v;
    }
    double wt_max_i = 0.0;
    if (total != 0.0) {
      for (std::int64_t j = 0; j < xsize; ++j) {
        wi[j] /= total;
        wt_max_i = std::max(wt_max_i, wi[j]);
      }
    }
    wt_max = std::max(wt_max, wt_max_i);
    t.xmin[i] = int(xmin);
    t.xsize[i] = int(xsize);
  }
  unsigned p = 0;
  for (p = 0; p < 22; ++p) {
    const int next = (int)(0.5 + wt_max * (1 << (p + 1)));
    if (next >= (1 << 15)) break;
  }
  t.precision = p;
  t.weights.resize(w.size());
  for (std::size_t k = 0; k < w.size(); ++k) {
    const double v = w[k] * (1 << p);
    t.weights[k] = std::int16_t((v < 0) ? (int)(-0.5 + v) : (int)(0.5 + v));
  }
  return t;
}

// One pass along x (step = 3, the next pixel) or y (step = 3 * width, the next row): basic_loop_aa_horizontal /
// vertical<uint8_t>.
void pass(const std::uint8_t* src, int src_w, int src_h, std::uint8_t* dst, int dst_w, int dst_h, bool horizontal,
          const Taps& t) {
  const int m = t.max_interp_size;
  const int round = 1 << (t.precision - 1);
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const int o = horizontal ? x : y;
      const std::int16_t* wt = &t.weights[std::size_t(o) * m];
      const int n = t.xsize[o];
      for (int c = 0; c < 3; ++c) {
        int acc = round;
        if (horizontal) {
          const std::uint8_t* s = src + (std::size_t(y) * src_w + t.xmin[o]) * 3 + c;
          for (int j = 0; j < n; ++j) acc += int(s[std::size_t(j) * 3]) * wt[j];
        } else {
          const std::uint8_t* s = src + (std::size_t(t.xmin[o]) * src_w + x) * 3 + c;
          for (int j = 0; j < n; ++j) acc += int(s[std::size_t(j) * src_w * 3]) * wt[j];
        }
        dst[(std::size_t(y) * dst_w + x) * 3 + c] = std::uint8_t(std::clamp(acc >> t.precision, 0, 255));
      }
    }
  }
  (void)src_h;
}

}  // namespace

Rgb resize_bicubic_aa(const Rgb& in, int out_width, int out_height) {
  if (out_width == in.width && out_height == in.height) return in;
  Rgb cur = in;
  if (out_width != in.width) {
    const Taps t = taps(in.width, out_width);
    Rgb next;
    next.width = out_width;
    next.height = cur.height;
    next.pixels.resize(std::size_t(next.width) * next.height * 3);
    pass(cur.pixels.data(), cur.width, cur.height, next.pixels.data(), next.width, next.height, true, t);
    cur = std::move(next);
  }
  if (out_height != in.height) {
    const Taps t = taps(in.height, out_height);
    Rgb next;
    next.width = cur.width;
    next.height = out_height;
    next.pixels.resize(std::size_t(next.width) * next.height * 3);
    pass(cur.pixels.data(), cur.width, cur.height, next.pixels.data(), next.width, next.height, false, t);
    cur = std::move(next);
  }
  return cur;
}

int tokens(const Rgb& image, const PreprocessConfig& cfg) {
  const int factor = cfg.patch * cfg.merge;
  const Size s = smart_resize(image.height, image.width, factor, cfg.min_pixels, cfg.max_pixels);
  return (s.height / cfg.patch) * (s.width / cfg.patch) / (cfg.merge * cfg.merge);  // grid_t = 1, as preprocess
}

PixelValues preprocess(const Rgb& image, const PreprocessConfig& cfg) {
  const int factor = cfg.patch * cfg.merge;
  const Size s = smart_resize(image.height, image.width, factor, cfg.min_pixels, cfg.max_pixels);
  const Rgb r = resize_bicubic_aa(image, s.width, s.height);
  PixelValues pv;
  pv.grid_h = s.height / cfg.patch;
  pv.grid_w = s.width / cfg.patch;
  pv.row_size = 3 * cfg.temporal * cfg.patch * cfg.patch;
  pv.tokens = pv.grid_t * pv.grid_h * pv.grid_w / (cfg.merge * cfg.merge);
  pv.values.resize(std::size_t(pv.grid_h) * pv.grid_w * pv.row_size);
  // HF's fused rescale and normalize: mean = std = 0.5 * (1 / (1/255)) = 127.5 in float32; (x - 127.5f) / 127.5f.
  float lut[256];
  for (int v = 0; v < 256; ++v) lut[v] = (float(v) - 127.5f) / 127.5f;
  const int P = cfg.patch, M = cfg.merge, T = cfg.temporal;
  float* out = pv.values.data();
  for (int bh = 0; bh < pv.grid_h / M; ++bh)
    for (int bw = 0; bw < pv.grid_w / M; ++bw)
      for (int mh = 0; mh < M; ++mh)
        for (int mw = 0; mw < M; ++mw) {
          const int y0 = (bh * M + mh) * P, x0 = (bw * M + mw) * P;
          for (int c = 0; c < 3; ++c)
            for (int t = 0; t < T; ++t)
              for (int py = 0; py < P; ++py)
                for (int px = 0; px < P; ++px)
                  *out++ = lut[r.pixels[(std::size_t(y0 + py) * r.width + (x0 + px)) * 3 + c]];
        }
  return pv;
}

}  // namespace lfm::image

namespace lfm::image::detail {

// A test hook (lfm3-image-decode-check --taps): an FNV-1a fingerprint of the resize taps for one dimension, so the
// LFM_RESIZE_FMA_MODE variants can be compared over many sizes without images.
std::uint64_t taps_fingerprint(int input_size, int output_size) {
  const Taps t = taps(input_size, output_size);
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](std::int64_t v) {
    for (int k = 0; k < 8; ++k) {
      h ^= std::uint64_t(v >> (8 * k)) & 0xFF;
      h *= 1099511628211ull;
    }
  };
  // the effective kernel only: (input index, weight) for every nonzero weight a pass reads (taps past xsize are never
  // read; a zero-weight tap at the edge changes nothing)
  mix(t.precision);
  for (int i = 0; i < output_size; ++i) {
    mix(-1);
    for (int j = 0; j < t.xsize[i]; ++j) {
      const std::int16_t w = t.weights[std::size_t(i) * t.max_interp_size + j];
      if (w != 0) {
        mix(t.xmin[i] + j);
        mix(w);
      }
    }
  }
  return h;
}

}  // namespace lfm::image::detail
