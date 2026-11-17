// Audio input (#3): the native renderer against HF's processor on whole sets of requests (validation/audio-in-v1).
//   audio-prompt-check TOKENIZER.json CASES.jsonl
// Each line: {"case": name, "messages": [...] (user audio parts {"type": "audio", "audio": WAV path}), "prompt_ids":
// [...] (HF's apply_chat_template for the same conversation)}. Renders the messages natively (lfm3_text::parse_messages
// with audio, encode_messages, expand_audio with each clip's lfm::audio_features::tokens from the WAV file), checks the
// spans (lfm3_speech::check_audio_spans) and compares with prompt_ids. Prints each mismatch and a summary; exit 1 on
// any mismatch or error. Host-only.
#include <lfm/lfm3_text.hpp>
#include <lfm/speech_run.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: audio-prompt-check TOKENIZER.json CASES.jsonl");
  const auto tok = lfm::lfm3_text::Tokenizer::load(argv[1]);
  std::ifstream in(argv[2]);
  if (!in) throw std::runtime_error(std::string("cannot read ") + argv[2]);
  std::string line;
  size_t cases = 0, equal = 0, failed = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    ++cases;
    const auto j = lfm::Json::parse(line);
    const std::string name = j.at("case").string();
    try {
      const auto messages = lfm::lfm3_text::parse_messages(j.at("messages"), /*allow_audio=*/true);
      std::vector<size_t> clip_tokens;
      for (const auto& m : messages)
        for (const auto& p : m.parts)
          if (p.audio) {
            const auto a = lfm::wav::read(p.data);
            if (a.rate != lfm::audio_features::kRate || a.samples.empty()) throw std::runtime_error("not a 16 kHz clip");
            clip_tokens.push_back(lfm::audio_features::tokens(a.samples.size()));
          }
      auto ids = lfm::lfm3_text::encode_messages(tok, messages, 32768, 1);
      if (!clip_tokens.empty()) ids = lfm::lfm3_text::expand_audio(ids, clip_tokens, 32768, 1);
      lfm3_speech::check_audio_spans(ids, clip_tokens);
      std::vector<int> want;
      for (const auto& x : j.at("prompt_ids").array()) want.push_back(x.integer());
      if (ids == want) {
        ++equal;
      } else {
        ++failed;
        size_t k = 0;
        while (k < ids.size() && k < want.size() && ids[k] == want[k]) ++k;
        std::cout << "DIFFER " << name << ": native " << ids.size() << " ids, HF " << want.size() << ", first difference at "
                  << k << "\n";
      }
    } catch (const std::exception& e) {
      ++failed;
      std::cout << "ERROR " << name << ": " << e.what() << "\n";
    }
  }
  std::cout << "{\"cases\":" << cases << ",\"equal\":" << equal << ",\"failed\":" << failed << "}\n";
  return failed || !cases ? 1 : 0;
} catch (const std::exception& e) {
  std::cerr << "audio-prompt-check: " << e.what() << "\n";
  return 1;
}
