// lfm3-image-decode-check: the native front end's decoded pixels (and, with --preprocess, its pixel_values), per
// image, to compare with HF's (tools/hash_lfm3_image_front.py; validation/images-v1 rule 1). Host only.
// Usage: lfm3-image-decode-check [--preprocess] SOURCE...
//          LABEL=DIR                 every *.jpg, *.jpeg, *.png in DIR, sorted, as the Python tool does
//          LABEL=noise:SEED:SIZES    SplitMix64 noise images (as the Python tool's noise source), one per "W H" line
//                                    of the file SIZES, named LABEL/iii-WxH
//        lfm3-image-decode-check --resize CASES.txt   (tools/hash_torch_resize.py's cases: the resize alone)
//        lfm3-image-decode-check --selftest
// Prints one JSON line per image: {"name", "rgb_size": [w, h], "rgb_sha256"} or {"name", "load_error": "code: msg"};
// with --preprocess also "grid_thw", "tokens", "pixel_values_sha256" (float32 bytes, little-endian) or
// "process_error". --resize prints {"case", "in_sha256", "out_sha256"} per case.
#include <lfm/image_decode.hpp>
#include <lfm/image_preprocess.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace lfm::image::detail {
std::uint64_t taps_fingerprint(int input_size, int output_size);
}

namespace {

// SHA-256 (FIPS 180-4), checked by --selftest against the standard vectors.
struct Sha256 {
  std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> block{};
  std::size_t used = 0;
  std::uint64_t bits = 0;
  static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void compress() {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = std::uint32_t(block[4 * i]) << 24 | std::uint32_t(block[4 * i + 1]) << 16 |
             std::uint32_t(block[4 * i + 2]) << 8 | block[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const std::uint8_t* p, std::size_t n) {
    bits += std::uint64_t(n) * 8;
    while (n) {
      const std::size_t take = std::min(n, 64 - used);
      std::copy(p, p + take, block.begin() + used);
      used += take; p += take; n -= take;
      if (used == 64) { compress(); used = 0; }
    }
  }
  std::string hex() {
    const std::uint64_t total = bits;
    const std::uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (used != 56) update(&zero, 1);
    for (int i = 7; i >= 0; --i) { const std::uint8_t b = std::uint8_t(total >> (8 * i)); update(&b, 1); }
    std::string out;
    char buf[9];
    for (auto v : h) { std::snprintf(buf, sizeof buf, "%08x", v); out += buf; }
    return out;
  }
};

std::string sha256(const std::vector<std::uint8_t>& v) {
  Sha256 s;
  s.update(v.data(), v.size());
  return s.hex();
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (std::uint8_t(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o;
}

// SplitMix64 as a counter (the Python tools' noise_rgb): word i is the output for state key + (i + 1) * golden.
std::vector<std::uint8_t> noise_bytes(std::uint64_t key, std::size_t n) {
  std::vector<std::uint8_t> out(n);
  for (std::size_t i = 0; i < n; i += 8) {
    std::uint64_t z = key + std::uint64_t(i / 8 + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    for (std::size_t k = 0; k < 8 && i + k < n; ++k) out[i + k] = std::uint8_t(z >> (8 * k));
  }
  return out;
}

std::string sha256_floats(const std::vector<float>& v) {
  static_assert(sizeof(float) == 4);
  Sha256 s;
  s.update(reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * 4);  // little-endian hosts (x86, aarch64)
  return s.hex();
}

void emit(const std::string& name, const lfm::image::Rgb& img, bool preprocess) {
  std::printf("{\"name\": \"%s\", \"rgb_size\": [%d, %d], \"rgb_sha256\": \"%s\"", json_escape(name).c_str(),
              img.width, img.height, sha256(img.pixels).c_str());
  if (preprocess) {
    try {
      const auto pv = lfm::image::preprocess(img);
      std::printf(", \"grid_thw\": [%d, %d, %d], \"tokens\": %d, \"pixel_values_sha256\": \"%s\"", pv.grid_t,
                  pv.grid_h, pv.grid_w, pv.tokens, sha256_floats(pv.values).c_str());
    } catch (const lfm::image::DecodeError& e) {
      std::printf(", \"process_error\": \"%s: %s\"", e.code.c_str(), json_escape(e.what()).c_str());
    }
  }
  std::printf("}\n");
}

int resize_cases(const char* path) {
  std::ifstream in(path);
  long long iw, ih, ow, oh;
  unsigned long long key;
  int i = 0;
  while (in >> iw >> ih >> ow >> oh >> key) {
    lfm::image::Rgb img;
    img.width = int(iw);
    img.height = int(ih);
    img.pixels = noise_bytes(key, std::size_t(iw * ih * 3));
    const auto out = lfm::image::resize_bicubic_aa(img, int(ow), int(oh));
    std::printf("{\"case\": %d, \"in_sha256\": \"%s\", \"out_sha256\": \"%s\"}\n", i++, sha256(img.pixels).c_str(),
                sha256(out.pixels).c_str());
  }
  return 0;
}

// --taps MAX STEP: the resize taps' fingerprint for every input size 1..MAX against every output size 1..MAX in STEP
// strides (and every multiple of 32 up to MAX), one line "in out fingerprint".
int taps_sweep(int max, int step) {
  for (int in = 1; in <= max; ++in)
    for (int out = 1; out <= max; ++out)
      if (out % step == 0 || out % 32 == 0)
        std::printf("%d %d %016llx\n", in, out, (unsigned long long)lfm::image::detail::taps_fingerprint(in, out));
  return 0;
}

// --taps-random N SEED: N random pairs, input 1..20000, output a multiple of 32 up to 20000 (what smart_resize gives).
int taps_random(int n, unsigned long long seed) {
  for (int k = 0; k < n; ++k) {
    const auto r = noise_bytes(seed * 1000003ull + std::uint64_t(k), 8);
    std::uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v |= std::uint64_t(r[b]) << (8 * b);
    const int in = int(v % 20000) + 1, out = 32 * (int((v >> 32) % 625) + 1);
    std::printf("%d %d %016llx\n", in, out, (unsigned long long)lfm::image::detail::taps_fingerprint(in, out));
  }
  return 0;
}

int selftest() {
  const std::string a = sha256({}), b = sha256({'a', 'b', 'c'});
  std::vector<std::uint8_t> million(1000000, 'a');
  const std::string c = sha256(million);
  const bool ok = a == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" &&
                  b == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
                  c == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
  std::printf("sha256 selftest %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--selftest") return selftest();
  if (argc == 3 && std::string(argv[1]) == "--resize") return resize_cases(argv[2]);
  if (argc == 4 && std::string(argv[1]) == "--taps") return taps_sweep(std::atoi(argv[2]), std::atoi(argv[3]));
  if (argc == 4 && std::string(argv[1]) == "--taps-random") return taps_random(std::atoi(argv[2]), std::stoull(argv[3]));
  if (argc < 2) {
    std::fprintf(stderr, "usage: lfm3-image-decode-check [--preprocess] SOURCE... | --resize CASES | --selftest\n");
    return 2;
  }
  bool preprocess = false;
  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "--preprocess") {
      preprocess = true;
      continue;
    }
    const auto eq = arg.find('=');
    const std::string label = arg.substr(0, eq), dir = arg.substr(eq + 1);
    if (dir.rfind("noise:", 0) == 0) {  // noise:SEED:SIZES
      const auto c2 = dir.find(':', 6);
      const unsigned long long seed = std::stoull(dir.substr(6, c2 - 6));
      std::ifstream sizes(dir.substr(c2 + 1));
      long long w, h;
      int i = 0;
      while (sizes >> w >> h) {
        lfm::image::Rgb img;
        img.width = int(w);
        img.height = int(h);
        img.pixels = noise_bytes((seed << 32) | (std::uint64_t(w) << 16) | std::uint64_t(h), std::size_t(w * h * 3));
        char name[64];
        std::snprintf(name, sizeof name, "%03d-%lldx%lld", i++, w, h);
        emit(label + "/" + name, img, preprocess);
      }
      continue;
    }
    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      std::string ext = e.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
      std::ifstream in(f, std::ios::binary);
      const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      const std::string name = label + "/" + f.filename().string();
      try {
        const auto img = lfm::image::load_rgb(bytes.data(), bytes.size());
        emit(name, img, preprocess);
      } catch (const lfm::image::DecodeError& e) {
        std::printf("{\"name\": \"%s\", \"load_error\": \"%s: %s\"}\n", json_escape(name).c_str(), e.code.c_str(),
                    json_escape(e.what()).c_str());
      }
    }
  }
  return 0;
}
