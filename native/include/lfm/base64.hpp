#pragma once
// Strict standard base64 (RFC 4648, "+/" alphabet, "=" padding, no whitespace) for audio input (#3): a WAV clip
// travels base64 in the /speech request and in the engine frame; streamed audio out travels the same way. Host-only.
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfm::base64 {

// Refuses a length that is not a multiple of 4, a character outside the alphabet, padding anywhere but the last one or
// two characters, and nonzero bits under the padding (so every byte string has exactly one accepted encoding).
inline std::string decode(std::string_view s) {
  auto fail = []() -> std::string { throw std::invalid_argument("invalid base64"); };
  if (s.size() % 4) return fail();
  auto value = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  std::string out;
  out.reserve(s.size() / 4 * 3);
  for (size_t i = 0; i < s.size(); i += 4) {
    const bool last = i + 4 == s.size();
    const int pad = last ? (s[i + 3] == '=') + (s[i + 2] == '=' && s[i + 3] == '=') : 0;
    int v[4];
    for (int k = 0; k < 4; ++k) {
      if (k >= 4 - pad) { v[k] = 0; continue; }
      v[k] = value(s[i + k]);
      if (v[k] < 0) return fail();
    }
    const unsigned n = unsigned(v[0]) << 18 | unsigned(v[1]) << 12 | unsigned(v[2]) << 6 | unsigned(v[3]);
    out += char(n >> 16);
    if (pad < 2) out += char((n >> 8) & 255);
    else if (n & 0xFFFF) return fail();
    if (pad < 1) out += char(n & 255);
    else if (n & 0xFF) return fail();
  }
  return out;
}

inline std::string encode(std::string_view bytes) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const size_t left = bytes.size() - i;
    unsigned n = unsigned(static_cast<unsigned char>(bytes[i])) << 16;
    if (left > 1) n |= unsigned(static_cast<unsigned char>(bytes[i + 1])) << 8;
    if (left > 2) n |= unsigned(static_cast<unsigned char>(bytes[i + 2]));
    out += t[n >> 18];
    out += t[(n >> 12) & 63];
    out += left > 1 ? t[(n >> 6) & 63] : '=';
    out += left > 2 ? t[n & 63] : '=';
  }
  return out;
}

}  // namespace lfm::base64
