// Image input (#4): the native renderer against HF's processor (validation/images-v1).
//   image-prompt-check TOKENIZER.json CASES.jsonl
// Each line: {"case": name, "messages": [...] (user image parts {"type": "image", "image": path}), "prompt_ids": [...]
// (HF's apply_chat_template for the same conversation)}. Renders the messages natively (lfm3_text::parse_messages with
// images, encode_messages, expand_media with each image's token count from the native front end: lfm::image::load_rgb
// then lfm::image::preprocess) and compares with prompt_ids. Prints each mismatch and a summary; exit 1 on any mismatch
// or error. Host-only.
#include <lfm/image_decode.hpp>
#include <lfm/image_preprocess.hpp>
#include <lfm/lfm3_text.hpp>

#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: image-prompt-check TOKENIZER.json CASES.jsonl");
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
      const auto messages = lfm::lfm3_text::parse_messages(j.at("messages"), /*allow_audio=*/false, /*allow_image=*/true);
      std::vector<lfm::lfm3_text::MediaTokens> media;
      for (const auto& m : messages)
        for (const auto& p : m.parts)
          if (p.image) {
            std::ifstream f(p.data, std::ios::binary);
            if (!f) throw std::runtime_error("cannot read " + p.data);
            const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const auto pv = lfm::image::preprocess(lfm::image::load_rgb(bytes.data(), bytes.size()));
            media.push_back({lfm::lfm3_text::kImageToken, size_t(pv.tokens)});
          }
      auto ids = lfm::lfm3_text::encode_messages(tok, messages, 32768, 1);
      if (!media.empty()) ids = lfm::lfm3_text::expand_media(ids, media, 32768, 1);
      std::vector<int> want;
      for (const auto& x : j.at("prompt_ids").array()) want.push_back(int(x.integer()));
      if (ids == want) {
        ++equal;
      } else {
        ++failed;
        size_t k = 0;
        while (k < ids.size() && k < want.size() && ids[k] == want[k]) ++k;
        std::cout << "MISMATCH " << name << ": native " << ids.size() << " tokens, HF " << want.size()
                  << ", first difference at " << k << "\n";
      }
    } catch (const std::exception& e) {
      ++failed;
      std::cout << "ERROR " << name << ": " << e.what() << "\n";
    }
  }
  std::cout << cases << " cases: " << equal << " equal, " << failed << " differ or failed\n";
  return failed ? 1 : 0;
} catch (const std::exception& e) {
  std::cerr << "image-prompt-check: " << e.what() << "\n";
  return 2;
}
