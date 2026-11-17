#include <lfm/tokenizer.hpp>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <charconv>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfm {
namespace {

void skip_space(std::string_view text, size_t& position) {
  while (position < text.size() &&
         (text[position] == ' ' || text[position] == '\n' || text[position] == '\r' || text[position] == '\t')) {
    ++position;
  }
}

uint32_t hex4(std::string_view text, size_t position) {
  if (position + 4 > text.size()) throw std::runtime_error("invalid tokenizer Unicode escape");
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const char character = text[position + i];
    value <<= 4;
    if (character >= '0' && character <= '9') value |= character - '0';
    else if (character >= 'a' && character <= 'f') value |= character - 'a' + 10;
    else if (character >= 'A' && character <= 'F') value |= character - 'A' + 10;
    else throw std::runtime_error("invalid tokenizer Unicode escape");
  }
  return value;
}

void append_utf8(std::string& output, uint32_t codepoint) {
  if (codepoint <= 0x7f) output.push_back(static_cast<char>(codepoint));
  else if (codepoint <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

// ByteLevel tokens can end mid-codepoint, or represent arbitrary invalid bytes.
// Match the official decoder's UTF-8 lossy conversion (maximal subparts), not
// raw byte emission. Valid UTF-8 is copied unchanged, including embedded NULs.
std::string utf8_lossy(std::string_view bytes) {
  std::string output;
  size_t i = 0;
  while (i < bytes.size()) {
    const auto first = static_cast<unsigned char>(bytes[i]);
    if (first < 0x80) { output.push_back(bytes[i++]); continue; }
    const int length = first >= 0xc2 && first <= 0xdf ? 2 :
                       first >= 0xe0 && first <= 0xef ? 3 :
                       first >= 0xf0 && first <= 0xf4 ? 4 : 0;
    if (!length) { append_utf8(output, 0xfffd); ++i; continue; }
    size_t valid = 1;
    for (; valid < static_cast<size_t>(length) && i + valid < bytes.size(); ++valid) {
      const auto byte = static_cast<unsigned char>(bytes[i + valid]);
      if (byte < 0x80 || byte > 0xbf || (valid == 1 &&
          ((first == 0xe0 && byte < 0xa0) || (first == 0xed && byte > 0x9f) ||
           (first == 0xf0 && byte < 0x90) || (first == 0xf4 && byte > 0x8f)))) break;
    }
    if (valid == static_cast<size_t>(length)) output.append(bytes.substr(i, valid));
    else append_utf8(output, 0xfffd);
    i += valid;
  }
  return output;
}

std::string parse_json_string(std::string_view text, size_t& position) {
  skip_space(text, position);
  if (position >= text.size() || text[position++] != '"') throw std::runtime_error("invalid tokenizer JSON string");
  std::string output;
  while (position < text.size()) {
    const char character = text[position++];
    if (character == '"') return output;
    if (character != '\\') {
      output.push_back(character);
      continue;
    }
    if (position >= text.size()) throw std::runtime_error("invalid tokenizer JSON escape");
    const char escape = text[position++];
    if (escape == 'u') {
      uint32_t codepoint = hex4(text, position);
      position += 4;
      if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
        if (position + 6 > text.size() || text[position] != '\\' || text[position + 1] != 'u') {
          throw std::runtime_error("invalid tokenizer surrogate pair");
        }
        position += 2;
        const uint32_t low = hex4(text, position);
        position += 4;
        if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("invalid tokenizer surrogate pair");
        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
      }
      append_utf8(output, codepoint);
    } else {
      switch (escape) {
        case '"': output.push_back('"'); break;
        case '\\': output.push_back('\\'); break;
        case '/': output.push_back('/'); break;
        case 'b': output.push_back('\b'); break;
        case 'f': output.push_back('\f'); break;
        case 'n': output.push_back('\n'); break;
        case 'r': output.push_back('\r'); break;
        case 't': output.push_back('\t'); break;
        default: throw std::runtime_error("unsupported tokenizer JSON escape");
      }
    }
  }
  throw std::runtime_error("unterminated tokenizer JSON string");
}

std::vector<uint32_t> utf8_codepoints(std::string_view text) {
  std::vector<uint32_t> output;
  for (size_t i = 0; i < text.size();) {
    const uint8_t first = static_cast<uint8_t>(text[i++]);
    if (first < 0x80) output.push_back(first);
    else {
      int continuation = first < 0xe0 ? 1 : first < 0xf0 ? 2 : 3;
      uint32_t codepoint = first & (0x7f >> continuation);
      if (i + continuation > text.size()) throw std::runtime_error("invalid UTF-8 in tokenizer vocabulary");
      for (int j = 0; j < continuation; ++j) {
        const uint8_t byte = static_cast<uint8_t>(text[i++]);
        if ((byte & 0xc0) != 0x80) throw std::runtime_error("invalid UTF-8 in tokenizer vocabulary");
        codepoint = (codepoint << 6) | (byte & 0x3f);
      }
      output.push_back(codepoint);
    }
  }
  return output;
}

std::unordered_map<uint32_t, uint8_t> byte_decoder() {
  std::vector<int> bytes;
  for (int value = 33; value <= 126; ++value) bytes.push_back(value);
  for (int value = 161; value <= 172; ++value) bytes.push_back(value);
  for (int value = 174; value <= 255; ++value) bytes.push_back(value);
  std::vector<uint32_t> codepoints(bytes.begin(), bytes.end());
  int extra = 0;
  for (int value = 0; value < 256; ++value) {
    bool present = false;
    for (const int byte : bytes) present |= byte == value;
    if (!present) {
      bytes.push_back(value);
      codepoints.push_back(256 + extra++);
    }
  }
  std::unordered_map<uint32_t, uint8_t> decoder;
  for (size_t i = 0; i < bytes.size(); ++i) decoder.emplace(codepoints[i], static_cast<uint8_t>(bytes[i]));
  return decoder;
}

std::vector<std::string> byte_encoder() {
  const auto decoder = byte_decoder();
  std::vector<std::string> encoder(256);
  for (const auto& [codepoint, byte] : decoder) append_utf8(encoder[byte], codepoint);
  return encoder;
}

std::string pair_key(const std::string& first, const std::string& second) {
  std::string key = first;
  key.push_back('\0');
  key += second;
  return key;
}

bool is_special(std::string_view token) {
  return token.size() >= 4 && token.starts_with("<|") && token.ends_with("|>");
}

}  // namespace

Tokenizer::Tokenizer(const std::string& tokenizer_json, size_t vocabulary_limit) {
  // Match the runtime's maximum vocabulary, and use the checkpoint's tighter
  // bound when provided. Reject sparse malicious IDs before growing storage.
  if (vocabulary_limit == 0 || vocabulary_limit > 16777216)
    throw std::runtime_error("invalid tokenizer vocabulary limit");
  auto validate_id = [&](int id) {
    if (id < 0 || static_cast<size_t>(id) >= vocabulary_limit)
      throw std::runtime_error("tokenizer ID exceeds vocabulary limit");
  };
  const auto template_path = std::filesystem::path(tokenizer_json).parent_path() / "chat_template.jinja";
  if (std::filesystem::exists(template_path)) {
    native_chat_supported_ = false;
    if (std::filesystem::file_size(template_path) <= 4 * 1024 * 1024) {
      std::ifstream source(template_path);
      const std::string chat((std::istreambuf_iterator<char>(source)), {});
      const std::string marker = "{%- if add_generation_prompt -%}";
      const auto start = chat.rfind(marker);
      if (start != std::string::npos) {
        size_t p = start + marker.size(); skip_space(chat, p);
        if (chat.compare(p, 3, "{{-") == 0) {
          p += 3;
          try {
            const auto literal = parse_json_string(chat, p);
            skip_space(chat, p);
            if (chat.compare(p, 3, "-}}") == 0) {
              p += 3; skip_space(chat, p);
              const std::string end = "{%- endif -%}";
              if (chat.compare(p, end.size(), end) == 0) {
                p += end.size(); skip_space(chat, p);
                native_chat_supported_ = p == chat.size() &&
                    (literal == "<|im_start|>assistant\n" || literal == "<|im_start|>assistant\n<think>");
                generation_thinking_ = literal == "<|im_start|>assistant\n<think>";
              }
            }
          } catch (const std::runtime_error&) { /* Unknown templates require explicit prompt tokens. */ }
        }
      }
    }
  }
  std::ifstream stream(tokenizer_json, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open tokenizer: " + tokenizer_json);
  const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const size_t model = text.find("\"model\"");
  size_t position = text.find("\"vocab\"", model);
  if (model == std::string::npos || position == std::string::npos) throw std::runtime_error("tokenizer BPE vocabulary not found");
  position = text.find('{', position);
  if (position == std::string::npos) throw std::runtime_error("invalid tokenizer vocabulary");
  ++position;
  while (true) {
    skip_space(text, position);
    if (position >= text.size()) throw std::runtime_error("unterminated tokenizer vocabulary");
    if (text[position] == '}') break;
    const std::string token = parse_json_string(text, position);
    skip_space(text, position);
    if (position >= text.size() || text[position++] != ':') throw std::runtime_error("invalid tokenizer vocabulary member");
    skip_space(text, position);
    int id = 0;
    const auto parsed = std::from_chars(text.data() + position, text.data() + text.size(), id);
    if (parsed.ec != std::errc() || id < 0) throw std::runtime_error("invalid tokenizer vocabulary ID");
    position = static_cast<size_t>(parsed.ptr - text.data());
    validate_id(id);
    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(static_cast<size_t>(id) + 1);
    if (!id_to_token_[id].empty()) throw std::runtime_error("duplicate tokenizer vocabulary ID");
    id_to_token_[id] = token;
    token_to_id_[token] = id;
    skip_space(text, position);
    if (text[position] == ',') ++position;
    else if (text[position] != '}') throw std::runtime_error("invalid tokenizer vocabulary separator");
  }
  size_t added = text.find("\"added_tokens\"");
  if (added == std::string::npos) throw std::runtime_error("tokenizer added-token table not found");
  added = text.find('[', added);
  if (added == std::string::npos) throw std::runtime_error("invalid tokenizer added-token table");
  ++added;
  while (true) {
    skip_space(text, added);
    if (added >= text.size()) throw std::runtime_error("unterminated tokenizer added-token table");
    if (text[added] == ']') break;
    if (text[added] != '{') throw std::runtime_error("invalid tokenizer added-token record");
    const size_t object_begin = added++;
    int depth = 1;
    bool in_string = false;
    bool escaped = false;
    while (added < text.size() && depth > 0) {
      const char character = text[added++];
      if (in_string) {
        if (escaped) escaped = false;
        else if (character == '\\') escaped = true;
        else if (character == '"') in_string = false;
      } else if (character == '"') in_string = true;
      else if (character == '{') ++depth;
      else if (character == '}') --depth;
    }
    if (depth != 0) throw std::runtime_error("unterminated tokenizer added-token record");
    const std::string_view object(text.data() + object_begin, added - object_begin);
    size_t id_position = object.find("\"id\"");
    id_position = object.find(':', id_position);
    if (id_position == std::string_view::npos) throw std::runtime_error("added token has no ID");
    ++id_position;
    skip_space(object, id_position);
    int id = 0;
    const auto parsed = std::from_chars(object.data() + id_position, object.data() + object.size(), id);
    if (parsed.ec != std::errc() || id < 0) throw std::runtime_error("invalid added-token ID");
    size_t content_position = object.find("\"content\"");
    content_position = object.find(':', content_position);
    if (content_position == std::string_view::npos) throw std::runtime_error("added token has no content");
    ++content_position;
    const std::string content = parse_json_string(object, content_position);
    validate_id(id);
    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(static_cast<size_t>(id) + 1);
    if (!id_to_token_[id].empty() && id_to_token_[id] != content) {
      throw std::runtime_error("added token conflicts with BPE vocabulary");
    }
    id_to_token_[id] = content;
    token_to_id_[content] = id;
    skip_space(text, added);
    if (text[added] == ',') ++added;
    else if (text[added] != ']') throw std::runtime_error("invalid tokenizer added-token separator");
  }
  if (id_to_token_.empty()) throw std::runtime_error("empty tokenizer vocabulary");

  size_t merges = text.find("\"merges\"", position);
  if (merges == std::string::npos) throw std::runtime_error("tokenizer merge table not found");
  merges = text.find('[', merges);
  if (merges == std::string::npos) throw std::runtime_error("invalid tokenizer merge table");
  ++merges;
  int rank = 0;
  while (true) {
    skip_space(text, merges);
    if (merges >= text.size()) throw std::runtime_error("unterminated tokenizer merge table");
    if (text[merges] == ']') break;
    if (text[merges++] != '[') throw std::runtime_error("invalid tokenizer merge record");
    const std::string first = parse_json_string(text, merges);
    skip_space(text, merges);
    if (text[merges++] != ',') throw std::runtime_error("invalid tokenizer merge pair");
    const std::string second = parse_json_string(text, merges);
    skip_space(text, merges);
    if (text[merges++] != ']') throw std::runtime_error("invalid tokenizer merge record");
    merge_rank_.emplace(pair_key(first, second), rank++);
    skip_space(text, merges);
    if (text[merges] == ',') ++merges;
    else if (text[merges] != ']') throw std::runtime_error("invalid tokenizer merge separator");
  }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  static const char* pattern =
      R"REGEX((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)REGEX";
  int error = 0;
  PCRE2_SIZE error_offset = 0;
  pcre2_code* regex = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern),
      PCRE2_ZERO_TERMINATED,
      PCRE2_UTF | PCRE2_UCP,
      &error,
      &error_offset,
      nullptr);
  if (!regex) throw std::runtime_error("cannot compile tokenizer regex at " + std::to_string(error_offset));
  pcre2_match_data* match = pcre2_match_data_create_from_pattern(regex, nullptr);
  if (!match) {
    pcre2_code_free(regex);
    throw std::runtime_error("cannot allocate tokenizer regex match data");
  }
  const auto encoder = byte_encoder();
  std::vector<int> ids;
  size_t offset = 0;
  while (offset < text.size()) {
    const int result = pcre2_match(
        regex,
        reinterpret_cast<PCRE2_SPTR>(text.data()),
        text.size(),
        offset,
        0,
        match,
        nullptr);
    if (result < 0) {
      pcre2_match_data_free(match);
      pcre2_code_free(regex);
      throw std::runtime_error("tokenizer regex did not cover input at byte " + std::to_string(offset));
    }
    PCRE2_SIZE* matches = pcre2_get_ovector_pointer(match);
    if (matches[0] != offset || matches[1] <= matches[0]) {
      pcre2_match_data_free(match);
      pcre2_code_free(regex);
      throw std::runtime_error("tokenizer regex produced a gap or empty match");
    }
    std::vector<std::string> symbols;
    symbols.reserve(matches[1] - matches[0]);
    for (size_t index = matches[0]; index < matches[1]; ++index) {
      symbols.push_back(encoder[static_cast<uint8_t>(text[index])]);
    }
    while (symbols.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      std::string best_pair;
      for (size_t index = 0; index + 1 < symbols.size(); ++index) {
        const std::string key = pair_key(symbols[index], symbols[index + 1]);
        const auto iterator = merge_rank_.find(key);
        if (iterator != merge_rank_.end() && iterator->second < best_rank) {
          best_rank = iterator->second;
          best_pair = key;
        }
      }
      if (best_pair.empty()) break;
      std::vector<std::string> merged;
      for (size_t index = 0; index < symbols.size();) {
        if (index + 1 < symbols.size() && pair_key(symbols[index], symbols[index + 1]) == best_pair) {
          merged.push_back(symbols[index] + symbols[index + 1]);
          index += 2;
        } else {
          merged.push_back(symbols[index++]);
        }
      }
      symbols = std::move(merged);
    }
    for (const std::string& symbol : symbols) {
      const auto iterator = token_to_id_.find(symbol);
      if (iterator == token_to_id_.end()) throw std::runtime_error("BPE symbol missing from vocabulary");
      ids.push_back(iterator->second);
    }
    offset = matches[1];
  }
  pcre2_match_data_free(match);
  pcre2_code_free(regex);
  return ids;
}

std::vector<int> Tokenizer::encode_chat(const std::string& user_prompt) const {
  if (!native_chat_supported_) throw std::runtime_error("unsupported native generation prompt; use --prompt-tokens");
  auto special = [&](const char* text) {
    const auto found = token_to_id_.find(text);
    if (found == token_to_id_.end()) throw std::runtime_error("unsupported native chat template: missing " + std::string(text) + "; use --prompt-tokens");
    return found->second;
  };
  std::vector<int> ids = {special("<|startoftext|>"), special("<|im_start|>")};
  auto append = [&](const std::string& text) {
    const auto encoded = encode(text);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
  };
  append("user\n" + user_prompt);
  ids.push_back(special("<|im_end|>"));
  append("\n");
  ids.push_back(special("<|im_start|>"));
  append("assistant\n");
  if (generation_thinking_) ids.push_back(special("<think>"));
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) const {
  const auto decoder = byte_decoder();
  std::string output;
  for (const int id : token_ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) throw std::out_of_range("token ID out of range");
    const std::string& token = id_to_token_[id];
    if (skip_special_tokens && is_special(token)) continue;
    for (const uint32_t codepoint : utf8_codepoints(token)) {
      const auto iterator = decoder.find(codepoint);
      if (iterator == decoder.end()) throw std::runtime_error("token contains a character outside ByteLevel alphabet");
      output.push_back(static_cast<char>(iterator->second));
    }
  }
  return utf8_lossy(output);
}

}  // namespace lfm
