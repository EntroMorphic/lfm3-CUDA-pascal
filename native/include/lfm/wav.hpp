#pragma once
// Audio input (#3 audio in): a WAV file's samples as float, as libsndfile (soundfile.read(dtype="float32")) gives them:
// 16-bit PCM as s / 32768, 32-bit IEEE float as is. Mono only; the rate is returned and checked by the caller (the
// model's feature extractor takes 16 kHz, and resampling is not this engine's job). Host-only.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm::wav {

struct Audio {
  std::vector<float> samples;
  int rate = 0;
};

// Parses a whole RIFF/WAVE file held in memory. Refuses anything but mono 16-bit PCM or 32-bit float (format 1 or 3,
// or WAVE_FORMAT_EXTENSIBLE carrying one of them), a block align or byte rate inconsistent with the format, a
// truncated chunk, and non-finite float samples. A data chunk whose size is 0xFFFFFFFF (written by a stream that could
// not seek back) runs to the end of the file, whole samples only, as libsndfile reads it.
inline Audio parse(const std::string& bytes) {
  auto fail = [](const std::string& why) -> Audio { throw std::runtime_error("wav: " + why); };
  auto u16 = [&](size_t at) { uint16_t v; std::memcpy(&v, bytes.data() + at, 2); return v; };
  auto u32 = [&](size_t at) { uint32_t v; std::memcpy(&v, bytes.data() + at, 4); return v; };
  if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") || bytes.compare(8, 4, "WAVE")) return fail("not a RIFF/WAVE file");
  int format = 0, channels = 0, bits = 0, rate = 0, align = 0;
  uint32_t byte_rate = 0;
  bool have_fmt = false;
  for (size_t at = 12; at + 8 <= bytes.size();) {
    std::string id = bytes.substr(at, 4);  // for messages: printable ASCII only (the bytes are the caller's input)
    for (char& c : id)
      if (c < 32 || c > 126) c = '?';
    size_t size = u32(at + 4);
    const size_t body = at + 8;
    if (id == "data" && size == 0xFFFFFFFFu && have_fmt && align > 0) size = (bytes.size() - body) / align * align;
    if (size > bytes.size() - body) return fail("truncated " + id + " chunk");
    if (id == "fmt ") {
      if (size < 16) return fail("short fmt chunk");
      format = u16(body), channels = u16(body + 2), rate = int(u32(body + 4)), byte_rate = u32(body + 8),
      align = u16(body + 12), bits = u16(body + 14);
      if (format == 0xFFFE) {  // WAVE_FORMAT_EXTENSIBLE: the subformat GUID's first two bytes are the format
        if (size < 40) return fail("short extensible fmt chunk");
        format = u16(body + 24);
      }
      have_fmt = true;
    } else if (id == "data") {
      if (!have_fmt) return fail("data before fmt");
      if (channels != 1) return fail("expected mono audio, got " + std::to_string(channels) + " channels");
      if (rate <= 0) return fail("invalid sample rate");
      if (align != channels * bits / 8 || byte_rate != uint32_t(rate) * uint32_t(align))
        return fail("block align or byte rate inconsistent with the format");
      Audio a;
      a.rate = rate;
      if (format == 1 && bits == 16) {
        if (size % 2) return fail("odd 16-bit data size");
        a.samples.resize(size / 2);
        for (size_t i = 0; i < a.samples.size(); ++i) {
          int16_t s;
          std::memcpy(&s, bytes.data() + body + 2 * i, 2);
          a.samples[i] = float(s) / 32768.0f;
        }
      } else if (format == 3 && bits == 32) {
        if (size % 4) return fail("float data size not a multiple of 4");
        a.samples.resize(size / 4);
        std::memcpy(a.samples.data(), bytes.data() + body, size);
        for (float v : a.samples)
          if (!std::isfinite(v)) return fail("non-finite sample");
      } else {
        return fail("expected 16-bit PCM or 32-bit float samples (format " + std::to_string(format) + ", " +
                    std::to_string(bits) + " bits)");
      }
      return a;
    }
    at = body + size + (size & 1);  // chunks are padded to even sizes
  }
  return fail("no data chunk");
}

inline Audio read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("wav: cannot open " + path);
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return parse(bytes);
}

}  // namespace lfm::wav
