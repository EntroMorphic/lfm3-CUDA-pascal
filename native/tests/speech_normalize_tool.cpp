// Host tool for tests/test_speech_normalize.py: each input line is a JSON string; each output line is the JSON string
// of lfm::speech_normalize::normalize of it.
//   speech-normalize-tool IN.jsonl OUT.jsonl
#include <lfm/lfm3_text_request.hpp>
#include <lfm/model_config.hpp>
#include <lfm/speech_normalize.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: speech-normalize-tool IN.jsonl OUT.jsonl");
  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  std::string line;
  size_t n = 0;
  while (std::getline(in, line)) {
    const auto j = lfm::Json::parse(line);
    out << lfm::lfm3_text::json_string(lfm::speech_normalize::normalize(j.string())) << "\n";
    ++n;
  }
  if (!out) throw std::runtime_error("write failed");
  std::cerr << n << " lines\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "speech-normalize-tool: " << e.what() << "\n";
  return 1;
}
