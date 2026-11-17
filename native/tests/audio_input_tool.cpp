// Host tool for the audio-input parsers and front end (tests/test_audio_input_host.py): one JSON line per input.
//   audio-input-tool wav FILE...      -> {"ok":true,"rate":R,"samples":N,"sum":S} or {"ok":false,"error":"..."}
//   audio-input-tool b64 FILE...      -> {"ok":true,"hex":"..."} or {"ok":false,"error":"..."} (FILE holds the text)
//   audio-input-tool mel WAV OUT.f32  -> writes lfm::audio_features::log_mel of the clip ([128][frames] float32) and
//                                        prints {"frames":T,"tokens":N}
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/wav.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
std::string slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot read ") + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) o += (c == '"' || c == '\\') ? std::string("\\") + c : (unsigned char)c < 32 ? std::string(" ") : std::string(1, c);
  return o;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) throw std::runtime_error("usage: audio-input-tool wav|b64 FILE... | mel WAV OUT.f32");
  const std::string mode = argv[1];
  if (mode == "mel") {
    const auto a = lfm::wav::read(argv[2]);
    const auto mel = lfm::audio_features::log_mel(a.samples);
    std::ofstream f(argv[3], std::ios::binary);
    f.write(reinterpret_cast<const char*>(mel.data()), std::streamsize(mel.size() * sizeof(float)));
    std::cout << "{\"frames\":" << lfm::audio_features::frames(a.samples.size())
              << ",\"tokens\":" << lfm::audio_features::tokens(a.samples.size()) << "}\n";
    return f ? 0 : 1;
  }
  for (int i = 2; i < argc; ++i) {
    try {
      if (mode == "wav") {
        const auto a = lfm::wav::parse(slurp(argv[i]));
        double sum = 0;
        for (float v : a.samples) sum += v;
        std::printf("{\"ok\":true,\"rate\":%d,\"samples\":%zu,\"sum\":%.9g}\n", a.rate, a.samples.size(), sum);
      } else if (mode == "b64") {
        const auto bytes = lfm::base64::decode(slurp(argv[i]));
        std::string hex;
        for (unsigned char c : bytes) { char h[3]; std::snprintf(h, 3, "%02x", c); hex += h; }
        std::cout << "{\"ok\":true,\"hex\":\"" << hex << "\"}\n";
      } else {
        throw std::runtime_error("unknown mode");
      }
    } catch (const std::exception& e) {
      std::cout << "{\"ok\":false,\"error\":\"" << escape(e.what()) << "\"}\n";
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "audio-input-tool: " << e.what() << "\n";
  return 1;
}
