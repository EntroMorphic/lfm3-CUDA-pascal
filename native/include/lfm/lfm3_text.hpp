#pragma once
// Native LFM3 text frontend: chat template (text subset), byte-level BPE tokenizer and detokenizer.
//
// Reference behaviour (bit-exact target): Rust `tokenizers` 0.23.3 reading the checkpoint's
// tokenizer.json, and Transformers' Jinja render of chat_template.jinja for the text-chat subset that
// tools/lfm3_text_frontend.py accepts. Nothing here is a general implementation: the loader verifies
// that tokenizer.json has exactly the pipeline described below and refuses anything else.
//
//   encode:  added tokens split out first (exact, leftmost-longest; 140 tokens, none normalized)
//            -> no normalizer
//            -> Split(regex, Isolated): the regex is evaluated by a hand scanner (scan_piece) over
//               character classes measured from the reference's Oniguruma (lfm3_text_classes.hpp)
//            -> ByteLevel (bytes as BPE symbols; the char<->byte map only matters for vocab keys)
//            -> BPE, ignore_merges (a whole piece in the vocabulary is one token), ranked merges
//               applied lowest rank first, leftmost on ties (tokenizers' word.rs merge_all)
//   decode:  special-flagged added tokens skipped; runs of base tokens -> bytes -> lossy UTF-8
//            (Rust from_utf8_lossy); a kept added token is inserted verbatim and ends the run. A
//            skipped special token does not end the run (measured). Whether a kept one ends it is
//            not observable here: every added token is ASCII, and an ASCII byte ends any partial
//            UTF-8 sequence either way.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lfm/lfm3_text_classes.hpp"
#include "lfm/model_config.hpp"

namespace lfm::lfm3_text {

// ---------------------------------------------------------------- UTF-8

// Strict decoder (RFC 3629): returns false on any ill-formed sequence.
inline bool next_code_point(std::string_view s, size_t& i, std::uint32_t& cp) {
  const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  if (c < 0x80) { cp = c; ++i; return true; }
  int need = 0;
  std::uint32_t lo = 0x80, hi = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) { need = 1; cp = c & 0x1F; }
  else if (c >= 0xE0 && c <= 0xEF) {
    need = 2; cp = c & 0x0F;
    if (c == 0xE0) lo = 0xA0;
    if (c == 0xED) hi = 0x9F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    need = 3; cp = c & 0x07;
    if (c == 0xF0) lo = 0x90;
    if (c == 0xF4) hi = 0x8F;
  } else return false;
  for (int k = 1; k <= need; ++k) {
    if (i + k >= s.size()) return false;
    const unsigned char t = b(i + k);
    if (t < (k == 1 ? lo : 0x80) || t > (k == 1 ? hi : 0xBF)) return false;
    cp = (cp << 6) | (t & 0x3F);
  }
  i += need + 1;
  return true;
}

inline bool valid_utf8(std::string_view s) {
  size_t i = 0;
  std::uint32_t cp = 0;
  while (i < s.size())
    if (!next_code_point(s, i, cp)) return false;
  return true;
}

// Rust String::from_utf8_lossy: each maximal ill-formed subpart becomes one U+FFFD.
// Returns the number of bytes of the next unit at i and whether it is valid; `complete` is false when
// the unit is an ill-formed tail that more bytes could still complete (used by the stream decoder).
struct Utf8Unit { size_t length; bool valid; bool could_continue; };
inline Utf8Unit utf8_unit(std::string_view s, size_t i) {
  const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  if (c < 0x80) return {1, true, false};
  int width = 0;
  if (c >= 0xC2 && c <= 0xDF) width = 2;
  else if (c >= 0xE0 && c <= 0xEF) width = 3;
  else if (c >= 0xF0 && c <= 0xF4) width = 4;
  else return {1, false, false};
  size_t k = 1;
  for (; k < static_cast<size_t>(width); ++k) {
    if (i + k >= s.size()) return {k, false, true};  // truncated: more input could complete it
    const unsigned char t = b(i + k);
    unsigned lo = 0x80, hi = 0xBF;
    if (k == 1) {
      if (c == 0xE0) lo = 0xA0;
      if (c == 0xED) hi = 0x9F;
      if (c == 0xF0) lo = 0x90;
      if (c == 0xF4) hi = 0x8F;
    }
    if (t < lo || t > hi) return {k, false, false};
  }
  return {k, true, false};
}

inline void append_lossy(std::string& out, std::string_view bytes) {
  static constexpr char kReplacement[] = "\xEF\xBF\xBD";
  for (size_t i = 0; i < bytes.size();) {
    const Utf8Unit u = utf8_unit(bytes, i);
    if (u.valid) out.append(bytes.substr(i, u.length));
    else out.append(kReplacement, 3);
    i += u.length;
  }
}

// ---------------------------------------------------------------- character classes

enum class CharClass : std::uint8_t { Letter, Number, Space, Other };

template <size_t N>
inline bool in_ranges(const CodeRange (&r)[N], std::uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < r[mid].first) hi = mid;
    else if (cp > r[mid].last) lo = mid + 1;
    else return true;
  }
  return false;
}

inline CharClass classify(std::uint32_t cp) {
  if (in_ranges(kLetter, cp)) return CharClass::Letter;
  if (in_ranges(kNumber, cp)) return CharClass::Number;
  if (in_ranges(kSpace, cp)) return CharClass::Space;
  return CharClass::Other;
}

struct Chars {
  std::vector<std::uint32_t> cp;
  std::vector<CharClass> cls;
  std::vector<size_t> offset;  // byte offset of each char, plus the end
};

inline Chars decode_chars(std::string_view s) {
  static const auto ascii = [] {
    std::vector<CharClass> t(128);
    for (std::uint32_t c = 0; c < 128; ++c) t[c] = classify(c);
    return t;
  }();
  Chars out;
  out.cp.reserve(s.size()); out.cls.reserve(s.size()); out.offset.reserve(s.size() + 1);
  size_t i = 0;
  while (i < s.size()) {
    out.offset.push_back(i);
    std::uint32_t cp = 0;
    if (!next_code_point(s, i, cp)) throw std::runtime_error("lfm3 text: invalid UTF-8");
    out.cp.push_back(cp);
    out.cls.push_back(cp < 128 ? ascii[cp] : classify(cp));
  }
  out.offset.push_back(s.size());
  return out;
}

// Length (in chars) of the regex match starting at p. The regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+|\s+(?!\S)|\s+
// matches at every position (every char is a letter, number, whitespace or other), so a
// leftmost-first search from p always starts at p and the first alternative that matches wins.
inline size_t scan_piece(const Chars& t, size_t p) {
  const size_t n = t.cp.size();
  const auto cls = [&](size_t i) { return t.cls[i]; };
  const auto is = [&](size_t i, CharClass c) { return i < n && t.cls[i] == c; };
  const auto crlf = [&](size_t i) { return i < n && (t.cp[i] == '\r' || t.cp[i] == '\n'); };
  // 1. contractions; case-insensitive in Oniguruma's sense, measured: ASCII case pairs, plus U+017F
  //    (long s) for 's. The suffixes are disjoint on their first letter, so their order is moot.
  if (t.cp[p] == '\'' && p + 1 < n) {
    const auto low = [](std::uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    const std::uint32_t a = low(t.cp[p + 1]);
    if (a == 's' || t.cp[p + 1] == kLongS || a == 't' || a == 'm' || a == 'd') return 2;
    if (p + 2 < n) {
      const std::uint32_t b = low(t.cp[p + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
  }
  // 2. [^\r\n\p{L}\p{N}]?\p{L}+ (the greedy optional prefix is taken only if a letter follows)
  {
    size_t q = p;
    if (cls(p) != CharClass::Letter && cls(p) != CharClass::Number && !crlf(p) && is(p + 1, CharClass::Letter)) q = p + 1;
    if (is(q, CharClass::Letter)) {
      while (is(q, CharClass::Letter)) ++q;
      return q - p;
    }
  }
  // 3. \p{N}{1,3}
  if (cls(p) == CharClass::Number) {
    size_t q = p;
    while (q < p + 3 && is(q, CharClass::Number)) ++q;
    return q - p;
  }
  // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
  {
    size_t q = (t.cp[p] == ' ' && is(p + 1, CharClass::Other)) ? p + 1 : p;
    if (is(q, CharClass::Other)) {
      while (is(q, CharClass::Other)) ++q;
      while (crlf(q)) ++q;
      return q - p;
    }
  }
  // Only whitespace is left at p.
  size_t q = p;
  while (is(q, CharClass::Space)) ++q;
  // 5. \s*[\r\n]+ : through the last CR/LF of the whitespace run
  for (size_t i = q; i > p; --i)
    if (crlf(i - 1)) return i - p;
  // 6. \s+(?!\S) : the run, or the run minus its last char when a non-space follows
  if (q == n) return q - p;
  if (q - p >= 2) return q - p - 1;
  // 7. \s+
  return q - p;
}

// Pre-tokenized pieces of a segment (no added tokens inside), as byte ranges.
inline std::vector<std::string_view> pretokenize(std::string_view s) {
  std::vector<std::string_view> pieces;
  if (s.empty()) return pieces;
  const Chars t = decode_chars(s);
  for (size_t p = 0; p < t.cp.size();) {
    const size_t len = scan_piece(t, p);
    if (len == 0) throw std::logic_error("lfm3 text: empty regex match");
    pieces.push_back(s.substr(t.offset[p], t.offset[p + len] - t.offset[p]));
    p += len;
  }
  return pieces;
}

// ---------------------------------------------------------------- byte-level alphabet

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100.. in byte order.
inline const std::vector<std::uint32_t>& byte_to_char() {
  static const std::vector<std::uint32_t> table = [] {
    std::vector<std::uint32_t> t(256);
    std::uint32_t next = 256;
    for (std::uint32_t b = 0; b < 256; ++b) {
      const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      t[b] = printable ? b : next++;
    }
    return t;
  }();
  return table;
}

// The raw bytes a byte-level vocabulary string stands for; false if a char is outside the alphabet.
inline bool byte_level_to_bytes(std::string_view token, std::string& bytes) {
  static const std::unordered_map<std::uint32_t, unsigned char> inverse = [] {
    std::unordered_map<std::uint32_t, unsigned char> m;
    const auto& t = byte_to_char();
    for (int b = 0; b < 256; ++b) m.emplace(t[b], static_cast<unsigned char>(b));
    return m;
  }();
  bytes.clear();
  size_t i = 0;
  while (i < token.size()) {
    std::uint32_t cp = 0;
    if (!next_code_point(token, i, cp)) return false;
    const auto it = inverse.find(cp);
    if (it == inverse.end()) return false;
    bytes.push_back(static_cast<char>(it->second));
  }
  return true;
}

// ---------------------------------------------------------------- tokenizer

inline constexpr const char* kPretokenizerRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

class Tokenizer {
 public:
  struct Added { std::string content; int id; bool special; };

  static Tokenizer load(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open tokenizer: " + path);
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size <= 0 || size > 256 * 1024 * 1024) throw std::runtime_error("invalid tokenizer size");
    stream.seekg(0);
    const std::string text((std::istreambuf_iterator<char>(stream)), {});
    return from_json(Json::parse(text));
  }

  static Tokenizer from_json(const Json& j) {
    const auto fail = [](const std::string& what) { throw std::runtime_error("lfm3 tokenizer: unsupported " + what); };
    const auto is_null = [](const Json& v) { return std::holds_alternative<std::nullptr_t>(v.value); };
    const auto obj = [&](const Json& v, const char* what) -> const Json::Object& {
      const auto* o = std::get_if<Json::Object>(&v.value);
      if (!o) fail(what);
      return *o;
    };
    const auto keys = [&](const Json& v, std::vector<std::string> expected, const char* what) {
      const auto& o = obj(v, what);
      std::sort(expected.begin(), expected.end());
      std::vector<std::string> got;
      for (const auto& kv : o) got.push_back(kv.first);
      if (got != expected) fail(std::string(what) + " fields");
    };
    keys(j, {"version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "model",
             "post_processor", "decoder"}, "tokenizer.json");
    if (!is_null(j.at("truncation")) || !is_null(j.at("padding"))) fail("truncation/padding");
    if (!is_null(j.at("normalizer"))) fail("normalizer");
    // pre_tokenizer: Sequence[Split(this regex, Isolated, invert=false), ByteLevel(no prefix, no regex)]
    const Json& pre = j.at("pre_tokenizer");
    keys(pre, {"type", "pretokenizers"}, "pre_tokenizer");
    if (pre.at("type").string() != "Sequence" || pre.at("pretokenizers").array().size() != 2) fail("pre_tokenizer");
    const Json& split = pre.at("pretokenizers").array()[0];
    keys(split, {"type", "pattern", "behavior", "invert"}, "Split");
    keys(split.at("pattern"), {"Regex"}, "Split pattern");
    if (split.at("type").string() != "Split" || split.at("behavior").string() != "Isolated" ||
        split.at("invert").boolean() || split.at("pattern").at("Regex").string() != kPretokenizerRegex)
      fail("Split pre-tokenizer");
    const Json& bl = pre.at("pretokenizers").array()[1];
    keys(bl, {"type", "add_prefix_space", "trim_offsets", "use_regex"}, "ByteLevel pre-tokenizer");
    if (bl.at("type").string() != "ByteLevel" || bl.at("add_prefix_space").boolean() || bl.at("use_regex").boolean())
      fail("ByteLevel pre-tokenizer");
    // post_processor ByteLevel only adjusts offsets; decoder ByteLevel is the byte map.
    const Json& post = j.at("post_processor");
    keys(post, {"type", "add_prefix_space", "trim_offsets", "use_regex"}, "post_processor");
    if (post.at("type").string() != "ByteLevel") fail("post_processor");
    const Json& dec = j.at("decoder");
    keys(dec, {"type", "add_prefix_space", "trim_offsets", "use_regex"}, "decoder");
    if (dec.at("type").string() != "ByteLevel") fail("decoder");
    const Json& model = j.at("model");
    keys(model, {"type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk",
                 "byte_fallback", "ignore_merges", "vocab", "merges"}, "model");
    if (model.at("type").string() != "BPE" || !is_null(model.at("dropout")) || !is_null(model.at("unk_token")) ||
        !is_null(model.at("continuing_subword_prefix")) || !is_null(model.at("end_of_word_suffix")) ||
        model.at("fuse_unk").boolean() || model.at("byte_fallback").boolean() || !model.at("ignore_merges").boolean())
      fail("BPE model options");

    Tokenizer t;
    const auto& vocab = obj(model.at("vocab"), "vocab");
    int base_count = 0;
    for (const auto& kv : vocab) base_count = std::max(base_count, kv.second.integer() + 1);
    if (base_count != static_cast<int>(vocab.size())) throw std::runtime_error("lfm3 tokenizer: base ids not contiguous");
    t.base_bytes_.assign(base_count, std::string());
    std::vector<bool> seen(base_count, false);
    std::unordered_map<std::string, int> by_string;
    std::string bytes;
    for (const auto& kv : vocab) {
      const int id = kv.second.integer();
      if (seen[id]) throw std::runtime_error("lfm3 tokenizer: base vocabulary repeats an id");
      seen[id] = true;
      if (!byte_level_to_bytes(kv.first, bytes) || bytes.empty())
        throw std::runtime_error("lfm3 tokenizer: vocabulary token outside the byte-level alphabet");
      t.base_bytes_[id] = bytes;
      if (!t.vocab_.emplace(bytes, id).second) throw std::runtime_error("lfm3 tokenizer: two tokens share bytes");
      by_string.emplace(kv.first, id);
    }
    for (int b = 0; b < 256; ++b) {
      const auto it = t.vocab_.find(std::string(1, static_cast<char>(b)));
      if (it == t.vocab_.end()) throw std::runtime_error("lfm3 tokenizer: byte missing from vocabulary");
      t.byte_id_[b] = it->second;
    }
    const auto& merges = model.at("merges").array();
    t.merges_.reserve(merges.size() * 2);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
      const auto& m = merges[rank].array();
      if (m.size() != 2) fail("merge format");
      const std::string a = m[0].string(), b = m[1].string();
      const auto ia = by_string.find(a), ib = by_string.find(b), ij = by_string.find(a + b);
      if (ia == by_string.end() || ib == by_string.end() || ij == by_string.end())
        throw std::runtime_error("lfm3 tokenizer: merge refers to a missing token");
      if (!t.merges_.emplace(pair_key(ia->second, ib->second), Merge{static_cast<std::uint32_t>(rank), ij->second}).second)
        throw std::runtime_error("lfm3 tokenizer: duplicate merge");
    }
    int count = base_count;
    for (const auto& a : j.at("added_tokens").array()) {
      keys(a, {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}, "added token");
      if (a.at("single_word").boolean() || a.at("lstrip").boolean() || a.at("rstrip").boolean() ||
          a.at("normalized").boolean())
        fail("added-token options");
      Added added{a.at("content").string(), a.at("id").integer(), a.at("special").boolean()};
      if (added.content.empty() || added.content[0] != '<' || !valid_utf8(added.content)) fail("added-token content");
      if (added.id != count) throw std::runtime_error("lfm3 tokenizer: added ids must follow the base vocabulary in order");
      if (by_string.count(added.content)) fail("added token that is also a base token");
      ++count;
      t.added_.push_back(std::move(added));
    }
    for (const auto& a : t.added_)
      for (const auto& b : t.added_)
        if (&a != &b && a.content == b.content) fail("duplicate added token");
    t.base_count_ = base_count;
    return t;
  }

  int token_count() const { return base_count_ + static_cast<int>(added_.size()); }
  int base_count() const { return base_count_; }
  const std::vector<Added>& added_tokens() const { return added_; }
  bool is_special(int id) const { return id >= base_count_ && id < token_count() && added_[id - base_count_].special; }

  // Text (valid UTF-8) -> ids, as Tokenizer.encode(text, add_special_tokens=False).
  std::vector<int> encode(std::string_view text) const {
    if (!valid_utf8(text)) throw std::runtime_error("lfm3 tokenizer: input is not valid UTF-8");
    std::vector<int> ids;
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
      int best = -1;
      size_t best_len = 0;
      if (text[i] == '<')
        for (size_t k = 0; k < added_.size(); ++k) {
          const auto& c = added_[k].content;
          if (c.size() > best_len && text.compare(i, c.size(), c) == 0) { best = static_cast<int>(k); best_len = c.size(); }
        }
      // Leftmost-longest: the first position with any match wins, then its longest match. Every
      // added token starts with '<' (checked at load).
      if (best < 0) { ++i; continue; }
      encode_segment(text.substr(start, i - start), ids);
      ids.push_back(added_[best].id);
      i += best_len;
      start = i;
    }
    encode_segment(text.substr(start), ids);
    return ids;
  }

  // ids -> text, as Tokenizer.decode(ids, skip_special_tokens). Unknown ids are skipped, as there.
  std::string decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string out, run;
    for (const int id : ids) {
      if (id >= 0 && id < base_count_) run += base_bytes_[id];
      else if (id >= base_count_ && id < token_count()) {
        const Added& a = added_[id - base_count_];
        if (skip_special_tokens && a.special) continue;
        append_lossy(out, run);
        run.clear();
        out += a.content;
      }
    }
    append_lossy(out, run);
    return out;
  }

  // Incremental decode: the concatenation of every push() and the final finish() equals decode(ids).
  // Bytes that could still complete a UTF-8 character are held back until they can be resolved.
  class StreamDecoder {
   public:
    StreamDecoder(const Tokenizer& t, bool skip_special_tokens) : t_(t), skip_(skip_special_tokens) {}
    std::string push(int id) {
      std::string out;
      if (id >= 0 && id < t_.base_count_) {
        pending_ += t_.base_bytes_[id];
        size_t i = 0;
        while (i < pending_.size()) {
          const Utf8Unit u = utf8_unit(pending_, i);
          if (u.could_continue) break;
          if (u.valid) out.append(pending_, i, u.length);
          else out.append("\xEF\xBF\xBD", 3);
          i += u.length;
        }
        pending_.erase(0, i);
      } else if (id >= t_.base_count_ && id < t_.token_count()) {
        const Added& a = t_.added_[id - t_.base_count_];
        if (skip_ && a.special) return out;
        append_lossy(out, pending_);
        pending_.clear();
        out += a.content;
      }
      return out;
    }
    size_t pending_bytes() const { return pending_.size(); }  // at most 3: an unfinished character
    std::string finish() {
      std::string out;
      append_lossy(out, pending_);
      pending_.clear();
      return out;
    }
   private:
    const Tokenizer& t_;
    bool skip_;
    std::string pending_;
  };

 private:
  struct Merge { std::uint32_t rank; int id; };
  static std::uint64_t pair_key(int a, int b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
  }

  void encode_segment(std::string_view segment, std::vector<int>& ids) const {
    for (const std::string_view piece : pretokenize(segment)) bpe(piece, ids);
  }

  // tokenizers models/bpe: ignore_merges, then Word::merge_all with a min-heap on (rank, position).
  void bpe(std::string_view piece, std::vector<int>& ids) const {
    if (const auto it = vocab_.find(std::string(piece)); it != vocab_.end()) { ids.push_back(it->second); return; }
    struct Symbol { int c; long prev, next; size_t len; };
    std::vector<Symbol> sym(piece.size());
    for (size_t i = 0; i < piece.size(); ++i)
      sym[i] = {byte_id_[static_cast<unsigned char>(piece[i])], static_cast<long>(i) - 1,
                i + 1 < piece.size() ? static_cast<long>(i + 1) : -1, 1};
    struct Item { std::uint32_t rank; size_t pos; int id; };
    const auto later = [](const Item& x, const Item& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
    std::priority_queue<Item, std::vector<Item>, decltype(later)> queue(later);
    const auto lookup = [&](int a, int b) -> const Merge* {
      const auto it = merges_.find(pair_key(a, b));
      return it == merges_.end() ? nullptr : &it->second;
    };
    for (size_t i = 0; i + 1 < sym.size(); ++i)
      if (const Merge* m = lookup(sym[i].c, sym[i + 1].c)) queue.push({m->rank, i, m->id});
    while (!queue.empty()) {
      const Item top = queue.top();
      queue.pop();
      Symbol& cur = sym[top.pos];
      if (cur.len == 0 || cur.next == -1) continue;
      const size_t next_pos = static_cast<size_t>(cur.next);
      const Symbol right = sym[next_pos];
      const Merge* still = lookup(cur.c, right.c);
      if (!still || still->id != top.id) continue;  // expired entry
      cur.c = top.id;
      cur.len += right.len;
      cur.next = right.next;
      sym[next_pos].len = 0;
      if (right.next > -1) sym[static_cast<size_t>(right.next)].prev = static_cast<long>(top.pos);
      if (cur.prev >= 0)
        if (const Merge* m = lookup(sym[static_cast<size_t>(cur.prev)].c, cur.c))
          queue.push({m->rank, static_cast<size_t>(cur.prev), m->id});
      if (cur.next > -1)
        if (const Merge* m = lookup(cur.c, sym[static_cast<size_t>(cur.next)].c)) queue.push({m->rank, top.pos, m->id});
    }
    for (const Symbol& s : sym)
      if (s.len) ids.push_back(s.c);
  }

  int base_count_ = 0;
  int byte_id_[256] = {};
  std::vector<std::string> base_bytes_;
  std::unordered_map<std::string, int> vocab_;  // raw bytes -> id
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::vector<Added> added_;
};

// ---------------------------------------------------------------- chat template (text subset)

// Python str.strip() whitespace (Jinja's |trim): Unicode White_Space plus U+001C..U+001F.
inline bool python_space(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

inline std::string python_strip(std::string_view s) {
  const Chars t = decode_chars(s);
  size_t a = 0, b = t.cp.size();
  while (a < b && python_space(t.cp[a])) ++a;
  while (b > a && python_space(t.cp[b - 1])) --b;
  return std::string(s.substr(t.offset[a], t.offset[b] - t.offset[a]));
}

// A text part, or (audio input, parse_messages with allow_audio) an audio clip: `data` holds the part's "audio"
// string (the caller's reference to the clip: a path, or base64 WAV bytes) and the part renders as the template's
// <|vision_start|><|audio|><|vision_end|>; or (image input, allow_image) an image: `data` holds the part's "image"
// string (a path, or base64 JPEG/PNG bytes) and the part renders as <|vision_start|><|image|><|vision_end|>.
struct ContentPart { std::string text; bool audio = false; std::string data; bool image = false; };
struct Message {
  std::string role;
  bool content_is_list = false;
  std::string content;              // when a string
  std::vector<ContentPart> parts;   // when a list of text parts
  bool has_thinking = false;
  std::string thinking;             // thinking / reasoning / reasoning_content
  bool has_tool_call_id = false;
  std::string tool_call_id;
};

inline constexpr int kMaxContextTokens = 262144;  // the engine ceiling; ModelConfig holds each checkpoint's
inline constexpr size_t kMaxTextBytes = 4 * 1024 * 1024;
inline constexpr size_t kMaxMessages = 4096;

// tools/lfm3_text_frontend.py RESERVED: <\|[^<>]*\|>|CONTINUE_FINAL_MESSAGE_TAG
inline bool has_reserved(std::string_view s) {
  if (s.find("CONTINUE_FINAL_MESSAGE_TAG ") != std::string_view::npos) return true;
  for (size_t i = s.find("<|"); i != std::string_view::npos; i = s.find("<|", i + 1)) {
    const size_t k = s.find_first_of("<>", i + 2);
    if (k != std::string_view::npos && s[k] == '>' && k >= i + 3 && s[k - 1] == '|') return true;
  }
  return false;
}

// Validation of tools/lfm3_text_frontend.py validate_messages, on the JSON form of the messages.
// allow_audio (speech input, #3): a user message's content list may also hold {type: "audio", audio: string} parts.
// allow_image (image input, #4): likewise {type: "image", image: string} parts.
inline std::vector<Message> parse_messages(const Json& j, bool allow_audio = false, bool allow_image = false) {
  const auto bad = [](const std::string& why) { throw std::runtime_error("lfm3 chat: " + why); };
  const auto* list = std::get_if<Json::Array>(&j.value);
  if (!list || list->empty() || list->size() > kMaxMessages) bad("messages must contain 1..4096 messages");
  size_t total = 0;
  const auto text = [&](const Json& v, const char* name) -> std::string {
    const auto* s = std::get_if<std::string>(&v.value);
    if (!s) bad(std::string(name) + " must be a string");
    if (has_reserved(*s)) bad(std::string(name) + ": reserved chat/multimodal placeholder is not text input");
    if (!valid_utf8(*s)) bad(std::string(name) + ": invalid Unicode");
    if (s->size() > kMaxTextBytes) bad(std::string(name) + ": text byte limit exceeded");
    return *s;
  };
  std::vector<Message> out;
  for (size_t index = 0; index < list->size(); ++index) {
    const auto* o = std::get_if<Json::Object>(&(*list)[index].value);
    if (!o) bad("each message must be a mapping");
    Message m;
    const auto role = o->find("role");
    const std::string* r = role == o->end() ? nullptr : std::get_if<std::string>(&role->second.value);
    if (!r || (*r != "system" && *r != "user" && *r != "assistant" && *r != "tool"))
      bad("role must be system, user, assistant, or tool");
    m.role = *r;
    if (m.role == "system" && index != 0) bad("system is supported only as the first message");
    for (const auto& kv : *o) {
      const std::string& k = kv.first;
      const bool ok = k == "role" || k == "content" ||
                      (m.role == "assistant" && (k == "thinking" || k == "reasoning" || k == "reasoning_content")) ||
                      (m.role == "tool" && k == "tool_call_id");
      if (!ok) bad("unsupported message fields (tool calls/definitions are not supported)");
    }
    const auto content = o->find("content");
    if (content == o->end()) bad("every message requires text content");
    if (std::holds_alternative<std::string>(content->second.value)) {
      m.content = text(content->second, "content");
      total += m.content.size();
    } else if (const auto* parts = std::get_if<Json::Array>(&content->second.value)) {
      m.content_is_list = true;
      std::string joined;
      for (const Json& part : *parts) {
        const auto* po = std::get_if<Json::Object>(&part.value);
        const auto type = po ? po->find("type") : Json::Object::const_iterator();
        if (allow_audio && po && po->size() == 2 && type != po->end() && po->count("audio") &&
            std::holds_alternative<std::string>(type->second.value) && type->second.string() == "audio") {
          if (m.role != "user") bad("audio parts are supported in user messages only");
          const auto* data = std::get_if<std::string>(&po->at("audio").value);
          if (!data || data->empty()) bad("an audio part's audio must be a nonempty string");
          m.parts.push_back({std::string(), true, *data});
          // the text on either side of an audio part is not adjacent in the prompt (the part renders as its span
          // tokens between them), so each run of text parts is checked as joined text on its own
          text(Json{joined}, "joined content");
          joined.clear();
          continue;
        }
        if (allow_image && po && po->size() == 2 && type != po->end() && po->count("image") &&
            std::holds_alternative<std::string>(type->second.value) && type->second.string() == "image") {
          if (m.role != "user") bad("image parts are supported in user messages only");
          const auto* data = std::get_if<std::string>(&po->at("image").value);
          if (!data || data->empty()) bad("an image part's image must be a nonempty string");
          m.parts.push_back({std::string(), false, *data, true});
          text(Json{joined}, "joined content");  // as for audio: the span separates the text on either side
          joined.clear();
          continue;
        }
        if (!po || po->size() != 2 || type == po->end() || !po->count("text") ||
            !std::holds_alternative<std::string>(type->second.value) || type->second.string() != "text")
          bad("content parts must be text-only {type: text, text: string}");
        m.parts.push_back({text(po->at("text"), "content part"), false, std::string()});
        total += m.parts.back().text.size();
        joined += m.parts.back().text;
      }
      text(Json{joined}, "joined content");
    } else {
      bad("content must be a string or a list of text parts; multimodal input is unsupported");
    }
    // As the frontend: count the reasoning fields first, then validate the one that is present.
    int reasoning = 0;
    for (const char* key : {"thinking", "reasoning", "reasoning_content"}) reasoning += o->count(key) ? 1 : 0;
    if (reasoning > 1) bad("use only one assistant reasoning field");
    for (const char* key : {"thinking", "reasoning", "reasoning_content"})
      if (o->count(key)) {
        m.has_thinking = true;
        m.thinking = text(o->at(key), key);
        total += m.thinking.size();
      }
    if (o->count("tool_call_id")) {
      m.has_tool_call_id = true;
      m.tool_call_id = text(o->at("tool_call_id"), "tool_call_id");
      total += m.tool_call_id.size();
      if (m.tool_call_id.empty()) bad("tool_call_id must not be empty");
    }
    if (total > kMaxTextBytes) bad("conversation text byte limit exceeded");
    out.push_back(std::move(m));
  }
  return out;
}

// chat_template.jinja with add_generation_prompt=True, enable_thinking=False, preserve_thinking=False,
// no tools, for messages that passed parse_messages.
inline std::string render_chat(const std::vector<Message>& all) {
  const auto content_truthy = [](const Message& m) { return m.content_is_list ? !m.parts.empty() : !m.content.empty(); };
  const auto parse_content = [](const Message& m) {
    if (!m.content_is_list) return m.content;
    std::string s;
    for (const auto& p : m.parts)
      s += p.audio ? std::string("<|vision_start|><|audio|><|vision_end|>")
                   : p.image ? std::string("<|vision_start|><|image|><|vision_end|>") : p.text;
    return s;
  };
  std::string out = "<|startoftext|>";
  size_t first = 0;
  std::string system_prompt;
  if (!all.empty() && all[0].role == "system") {
    if (content_truthy(all[0])) system_prompt = parse_content(all[0]);
    first = 1;
  }
  if (!system_prompt.empty()) out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  const size_t n = all.size() - first;
  const auto msg = [&](size_t i) -> const Message& { return all[first + i]; };
  long last_user = -1;
  for (size_t i = 0; i < n; ++i)
    if (msg(i).role == "user") last_user = static_cast<long>(i);
  std::vector<std::string> carry;  // text items of merged tool messages
  const auto tool_items = [&](const Message& m) {
    std::vector<std::string> items;
    if (!m.content_is_list) { if (!m.content.empty()) items.push_back(m.content); }
    else for (const auto& p : m.parts) items.push_back(p.text);
    return items;
  };
  const auto tool_response_list = [](const std::vector<std::string>& items) {
    std::string s = "<tool_response>\n";
    for (size_t k = 0; k < items.size(); ++k) s += (k ? "\n" : "") + items[k];
    return s + "\n</tool_response>";
  };
  for (size_t i = 0; i < n; ++i) {
    const Message& m = msg(i);
    const bool merge = m.role == "tool" && i + 1 < n && msg(i + 1).role == "tool" && m.has_tool_call_id &&
                       !m.tool_call_id.empty() && msg(i + 1).has_tool_call_id && msg(i + 1).tool_call_id == m.tool_call_id;
    const std::vector<std::string> items = m.role == "tool" ? tool_items(m) : std::vector<std::string>{};
    if (merge) { carry.insert(carry.end(), items.begin(), items.end()); continue; }
    out += "<|im_start|>" + m.role + "\n";
    if (m.role == "assistant") {
      const bool keep = static_cast<long>(i) > last_user;
      const std::string check = m.content_is_list ? std::string() : m.content;
      if (m.has_thinking && keep) out += "<think>" + m.thinking + "</think>";
      else if (keep && check.find("</think>") == std::string::npos) out += "<think></think>";
      std::string content = content_truthy(m) ? parse_content(m) : std::string();
      if (!keep) {
        const size_t at = content.rfind("</think>");
        if (at != std::string::npos) content = python_strip(std::string_view(content).substr(at + 8));
        content = "<think></think>" + content;
      }
      // CONTINUE_FINAL_MESSAGE_TAG handling is an identity here: the tag is removed and re-appended
      // with no tool calls in between, and validated text cannot contain it anyway.
      out += content + "<|im_end|>\n";
    } else if (m.role == "tool") {
      if (!carry.empty()) {
        std::vector<std::string> merged = carry;
        merged.insert(merged.end(), items.begin(), items.end());
        out += tool_response_list(merged);
        carry.clear();
      } else if (m.content_is_list && !m.parts.empty()) {
        out += tool_response_list(tool_items(m));
      } else {
        const std::string c = m.content_is_list ? std::string() : m.content;
        out += c.find("<tool_response>") != std::string::npos ? c : "<tool_response>\n" + c + "\n</tool_response>";
      }
      out += "<|im_end|>\n";
    } else {
      if (content_truthy(m)) out += parse_content(m);
      out += "<|im_end|>\n";
    }
  }
  out += "<|im_start|>assistant\n<think></think>";
  return out;
}

// TextFrontend.encode_messages checks these before it validates the messages; callers that parse
// messages themselves call this first to keep the same order of rejections.
inline void check_capacity(int capacity, int max_new_tokens) {
  if (capacity < 1 || capacity > kMaxContextTokens) throw std::runtime_error("lfm3 chat: capacity must be an integer in 1..262144");
  if (max_new_tokens < 1 || max_new_tokens > kMaxContextTokens)
    throw std::runtime_error("lfm3 chat: max_new_tokens must be an integer in 1..262144");
  if (max_new_tokens >= capacity) throw std::runtime_error("lfm3 chat: capacity must fit a nonempty prompt plus max_new_tokens");
}

// TextFrontend.encode_messages: rendered prompt ids, reserving every requested output token.
inline std::vector<int> encode_messages(const Tokenizer& tokenizer, const std::vector<Message>& messages,
                                        int capacity, int max_new_tokens) {
  check_capacity(capacity, max_new_tokens);
  std::vector<int> ids = tokenizer.encode(render_chat(messages));
  if (ids.empty()) throw std::runtime_error("lfm3 chat: empty prompt");
  if (ids.size() + static_cast<size_t>(max_new_tokens) > static_cast<size_t>(capacity))
    throw std::runtime_error("lfm3 chat: prompt (" + std::to_string(ids.size()) + ") + output (" +
                             std::to_string(max_new_tokens) + ") exceeds capacity (" + std::to_string(capacity) + ")");
  return ids;
}

// Audio input: the processor's expansion of each clip's single <|audio|> (as the template renders it) into its
// encoder token count, clip by clip in order (lfm::audio_features::tokens); then the capacity check of encode_messages.
inline constexpr int kAudioToken = 125031;  // <|audio|>
inline std::vector<int> expand_audio(const std::vector<int>& ids, const std::vector<size_t>& clip_tokens, int capacity,
                                     int max_new_tokens, int audio_token = kAudioToken) {
  std::vector<int> out;
  size_t clip = 0;
  for (int id : ids) {
    if (id != audio_token) { out.push_back(id); continue; }
    if (clip >= clip_tokens.size()) throw std::runtime_error("lfm3 chat: more <|audio|> placeholders than clips");
    if (clip_tokens[clip] == 0) throw std::runtime_error("lfm3 chat: an audio clip without tokens");
    if (clip_tokens[clip] > size_t(capacity)) throw std::runtime_error("lfm3 chat: audio clip exceeds capacity");
    out.insert(out.end(), clip_tokens[clip++], audio_token);
    if (out.size() > size_t(capacity)) break;  // refused below
  }
  if (out.size() <= size_t(capacity) && clip != clip_tokens.size())
    throw std::runtime_error("lfm3 chat: fewer <|audio|> placeholders than clips");
  if (out.size() + static_cast<size_t>(max_new_tokens) > static_cast<size_t>(capacity))
    throw std::runtime_error("lfm3 chat: prompt (" + std::to_string(out.size()) + "+) + output (" +
                             std::to_string(max_new_tokens) + ") exceeds capacity (" + std::to_string(capacity) + ")");
  return out;
}

// Audio and image input together: each placeholder (<|audio|> or <|image|>, as the template renders them, one per part)
// expanded to its media's token count, the media in prompt order (`media`: the placeholder each expects and its count;
// an image's count is its grid_h * grid_w / 4, lfm::image::PixelValues::tokens); then the capacity check of
// encode_messages. A prompt with audio only is expand_audio's, which this does not replace.
inline constexpr int kImageToken = 125030;  // <|image|>
struct MediaTokens { int placeholder = 0; size_t tokens = 0; };
inline std::vector<int> expand_media(const std::vector<int>& ids, const std::vector<MediaTokens>& media, int capacity,
                                     int max_new_tokens) {
  std::vector<int> out;
  size_t at = 0;
  for (int id : ids) {
    if (id != kAudioToken && id != kImageToken) { out.push_back(id); continue; }
    if (at >= media.size()) throw std::runtime_error("lfm3 chat: more media placeholders than media");
    if (media[at].placeholder != id) throw std::runtime_error("lfm3 chat: media out of order with their placeholders");
    if (media[at].tokens == 0) throw std::runtime_error("lfm3 chat: media without tokens");
    if (media[at].tokens > size_t(capacity)) throw std::runtime_error("lfm3 chat: media exceeds capacity");
    out.insert(out.end(), media[at++].tokens, id);
    if (out.size() > size_t(capacity)) break;  // refused below
  }
  if (out.size() <= size_t(capacity) && at != media.size())
    throw std::runtime_error("lfm3 chat: fewer media placeholders than media");
  if (out.size() + static_cast<size_t>(max_new_tokens) > static_cast<size_t>(capacity))
    throw std::runtime_error("lfm3 chat: prompt (" + std::to_string(out.size()) + "+) + output (" +
                             std::to_string(max_new_tokens) + ") exceeds capacity (" + std::to_string(capacity) + ")");
  return out;
}

}  // namespace lfm::lfm3_text
