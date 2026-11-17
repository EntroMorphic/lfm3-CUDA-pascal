// Host tool for the native LFM3 text frontend (lfm/lfm3_text.hpp), used by the differential tests.
//
//   lfm3-text-tool TOKENIZER.json encode IN.jsonl OUT.jsonl   {"text": s}              -> {"ids": [..]}
//   lfm3-text-tool TOKENIZER.json decode IN.jsonl OUT.jsonl   {"ids": [..], "skip": b} -> {"text": s, "stream": s, "held": n, "tail": s}
//   lfm3-text-tool TOKENIZER.json chat   IN.jsonl OUT.jsonl   {"messages": [..], "capacity": n, "max_new_tokens": n}
//                                                             -> {"prompt": s, "ids": [..]}
//   lfm3-text-tool TOKENIZER.json split  IN.jsonl OUT.jsonl   {"text": s}              -> {"sentences": [..]}
//   lfm3-text-tool TOKENIZER.json spaces IN.jsonl OUT.jsonl   {"from": a, "to": b}     -> {"spaces": [code points]}
//   lfm3-text-tool TOKENIZER.json speech IN.jsonl OUT.jsonl   {"messages": [..], "capacity": n, "max_new_tokens": n}
//                                                             -> {"texts": [..], "ids": [[..]..]}  (lfm/speech_text.hpp)
// A case the frontend rejects yields {"error": message}. "stream" is the concatenated output of the
// incremental decoder, which must equal "text"; "held" is the most bytes it ever held back (at most 3)
// and "tail" what finish() released (only replacement characters: an unfinished character at the end).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "lfm/lfm3_text.hpp"
#include "lfm/speech_text.hpp"

namespace {

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (const unsigned char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else out += static_cast<char>(c);
  }
  return out + "\"";
}

std::string ids_json(const std::vector<int>& ids) {
  std::string out = "[";
  for (size_t i = 0; i < ids.size(); ++i) out += (i ? "," : "") + std::to_string(ids[i]);
  return out + "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: lfm3-text-tool TOKENIZER.json encode|decode|chat|split|spaces|speech IN.jsonl OUT.jsonl\n";
    return 2;
  }
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const auto tok = lfm::lfm3_text::Tokenizer::load(argv[1]);
    const auto t1 = std::chrono::steady_clock::now();
    const std::string mode = argv[2];
    if (mode != "encode" && mode != "decode" && mode != "chat" && mode != "split" && mode != "spaces" && mode != "speech") throw std::runtime_error("unknown mode " + mode);
    std::ifstream in(argv[3], std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + argv[3]);
    std::ofstream out(argv[4], std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(std::string("cannot create ") + argv[4]);
    std::string line;
    size_t cases = 0, errors = 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::string result;
      try {
        const lfm::Json j = lfm::Json::parse(line);
        if (mode == "encode") {
          result = "{\"ids\":" + ids_json(tok.encode(j.at("text").string())) + "}";
        } else if (mode == "decode") {
          std::vector<int> ids;
          for (const auto& v : j.at("ids").array()) ids.push_back(v.integer());
          const bool skip = j.at("skip").boolean();
          lfm::lfm3_text::Tokenizer::StreamDecoder stream(tok, skip);
          std::string streamed;
          size_t held = 0;
          for (const int id : ids) {
            streamed += stream.push(id);
            held = std::max(held, stream.pending_bytes());
          }
          const std::string tail = stream.finish();
          streamed += tail;
          result = "{\"text\":" + quote(tok.decode(ids, skip)) + ",\"stream\":" + quote(streamed) +
                   ",\"held\":" + std::to_string(held) + ",\"tail\":" + quote(tail) + "}";
        } else if (mode == "split") {
          result = "{\"sentences\":[";
          const auto sentences = lfm::speech_text::split_sentences(j.at("text").string());
          for (size_t i = 0; i < sentences.size(); ++i) result += (i ? "," : "") + quote(sentences[i]);
          result += "]}";
        } else if (mode == "spaces") {
          std::vector<int> spaces;
          for (int c = j.at("from").integer(); c < j.at("to").integer(); ++c)
            if (lfm::speech_text::is_space(static_cast<std::uint32_t>(c))) spaces.push_back(c);
          result = "{\"spaces\":" + ids_json(spaces) + "}";
        } else if (mode == "speech") {
          lfm::lfm3_text::check_capacity(j.at("capacity").integer(), j.at("max_new_tokens").integer());
          const auto chats = lfm::speech_text::sentence_requests(lfm::lfm3_text::parse_messages(j.at("messages")));
          std::string texts = "[", ids = "[";
          for (size_t k = 0; k < chats.size(); ++k) {
            texts += (k ? "," : "") + quote(chats[k].back().content);
            ids += (k ? "," : "") + ids_json(lfm::lfm3_text::encode_messages(tok, chats[k], j.at("capacity").integer(),
                                                                             j.at("max_new_tokens").integer()));
          }
          result = "{\"texts\":" + texts + "],\"ids\":" + ids + "]}";
        } else {
          lfm::lfm3_text::check_capacity(j.at("capacity").integer(), j.at("max_new_tokens").integer());
          const auto messages = lfm::lfm3_text::parse_messages(j.at("messages"));
          const std::string prompt = lfm::lfm3_text::render_chat(messages);
          const auto ids = lfm::lfm3_text::encode_messages(tok, messages, j.at("capacity").integer(),
                                                           j.at("max_new_tokens").integer());
          result = "{\"prompt\":" + quote(prompt) + ",\"ids\":" + ids_json(ids) + "}";
        }
      } catch (const std::exception& e) {
        ++errors;
        result = "{\"error\":" + quote(e.what()) + "}";
      }
      out << result << '\n';
      ++cases;
    }
    out.close();
    if (!out) throw std::runtime_error("write failed");
    const auto t2 = std::chrono::steady_clock::now();
    std::cerr << "lfm3-text-tool " << mode << ": " << cases << " cases, " << errors << " errors; load "
              << std::chrono::duration<double>(t1 - t0).count() << " s, run "
              << std::chrono::duration<double>(t2 - t1).count() << " s; tokens " << tok.token_count() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "lfm3-text-tool: " << e.what() << "\n";
    return 1;
  }
}
