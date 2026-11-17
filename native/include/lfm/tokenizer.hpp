#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace lfm {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json, size_t vocabulary_limit = 16777216);
  std::vector<int> encode(const std::string& text) const;
  std::vector<int> encode_chat(const std::string& user_prompt) const;
  std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const;
  size_t vocab_size() const { return id_to_token_.size(); }

 private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> merge_rank_;
  bool native_chat_supported_ = true;
  bool generation_thinking_ = false;
};

}  // namespace lfm
