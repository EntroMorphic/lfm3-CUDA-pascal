#pragma once
// Request/response glue between the engine binaries and the native text frontend (lfm3_text.hpp).
//
// Contract, shared by liquid-lfm3 and liquid-lfm3-worker:
//   "messages": [...]     instead of "input_ids": the binary renders the chat template and tokenizes
//                         natively, with tools/lfm3_text_frontend.py encode_messages rules (capacity,
//                         max_new_tokens reservation) and make_request's modality-placeholder rejection.
//                         Implies "return_text".
//   "return_text": true   the response carries "text": the generated ids minus a final stop token,
//                         decoded with special tokens skipped, exactly as tools/lfm3_chat.py and
//                         tools/lfm3_server.py decode them.
// Requests with neither field are untouched: same parsing, same output bytes.
//
// Known, deliberate differences at the JSON boundary (outputs are identical for every input both accept):
// lfm::Json rejects duplicate keys (as tools/lfm3_chat.read_json does on the serving path), keeps numbers
// as doubles so a whole-valued float such as 16.0 is accepted where Python requires an int (the existing
// contract of every native request field), and rejects lone surrogates, NaN and nesting beyond 64 at
// parse time with its own messages. Decode skips ids the tokenizer does not define (as tokenizers
// does); the engines never generate them (G6).
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lfm/lfm3_text.hpp"

namespace lfm::lfm3_text {

// JSON string literal for valid UTF-8 text (decode output always is): escapes quote, backslash and
// control characters, keeps every other character as UTF-8.
inline std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (const unsigned char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c < 0x20 || c == 0x7F) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else out += static_cast<char>(c);
  }
  return out + "\"";
}

inline bool wants_text(const Json& request) {
  return request.has("messages") || (request.has("return_text") && request.at("return_text").boolean());
}

// Loads the native tokenizer and requires it to agree with the engine's verified token count (G6).
inline std::unique_ptr<Tokenizer> load_checked(const std::string& path, int live) {
  auto t = std::make_unique<Tokenizer>(Tokenizer::load(path));
  if (t->token_count() != live) throw std::runtime_error("native tokenizer disagrees with the verified token count");
  return t;
}

// Prompt ids for a "messages" request; `suppressed` are the modality placeholders text input may not use.
inline std::vector<int> message_input_ids(const Tokenizer& t, const Json& request, int capacity, int max_new_tokens,
                                          const std::vector<int>& suppressed) {
  if (request.has("input_ids")) throw std::runtime_error("messages cannot be combined with input_ids");
  // tools/lfm3_chat.make_request order: integer ranges, then the messages, then encode_messages
  // (which adds the prompt-plus-output fit).
  if (capacity < 1 || capacity > kMaxContextTokens) throw std::runtime_error("lfm3 chat: capacity must be an integer in 1..262144");
  if (max_new_tokens < 1 || max_new_tokens > kMaxContextTokens)
    throw std::runtime_error("lfm3 chat: max_new_tokens must be an integer in 1..262144");
  const auto messages = parse_messages(request.at("messages"));
  auto ids = encode_messages(t, messages, capacity, max_new_tokens);
  for (const int id : ids)
    if (std::find(suppressed.begin(), suppressed.end(), id) != suppressed.end())
      throw std::runtime_error("multimodal placeholder input is unsupported in text-only mode");
  return ids;
}

inline std::string generated_text(const Tokenizer& t, std::vector<int> generated, const std::vector<int>& stops) {
  if (!generated.empty() && std::find(stops.begin(), stops.end(), generated.back()) != stops.end()) generated.pop_back();
  return t.decode(generated, true);
}

}  // namespace lfm::lfm3_text
