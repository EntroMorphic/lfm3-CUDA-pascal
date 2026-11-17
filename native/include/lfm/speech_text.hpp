#pragma once
// Long-form TTS by sentences (#2 step 4, validation/speech-long-form-v1): the native text splitter and the
// per-sentence requests.
//
// Why: the backbone's sliding-attention layers (window 256) lose sight of the text being read about 180 frames
// into an utterance, and long utterances then skip and loop. One fresh request per sentence never gets there.
//
// split_sentences is tools/lfm3_speech_chunked_hf.py sentences(), exactly (tests/test_speech_text.py); the rule was
// chosen by test (validation/speech-split-v1: WER, then Tripp's listening):
//   1. cut the stripped text at each whitespace run that follows . ! or ?, except after a title (Mr. Dr. vs. ...: never)
//      or after an abbreviation (a.m. U.S. Jan. no. ...) or an ellipsis when the next word does not start with an
//      ASCII capital (after any opening quote or bracket);
//   2. a part under kMinWords words (str.split()) joins the next one; a short last part joins the one before;
//   3. every chunk becomes its words joined by single spaces, and one over kCapWords words is cut at its last clause
//      mark (a word ending in , ; : -- or an em or en dash, or followed by a lone - -- em or en dash) that leaves both
//      sides kMinWords long and the first at most kCapWords; with none, into the fewest equal pieces; repeated.
// Whitespace is Python's str.isspace (the same set as re's Unicode \s), pinned code point by code point.
// sentence_requests turns a [system?, user] chat into one chat per sentence of the user text, each rendered
// later exactly as a single request is (lfm3_text::encode_messages).
#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "lfm/lfm3_text.hpp"
#include "lfm/speech_normalize.hpp"

namespace lfm::speech_text {

inline constexpr size_t kMinWords = 6;
inline constexpr size_t kCapWords = 49;  // the longest single chunk that read cleanly on every seed (speech-split-v1)

inline bool in_list(std::string_view word, std::initializer_list<std::string_view> list) {
  std::string low(word);
  for (char& c : low) if (c >= 'A' && c <= 'Z') c = char(c + 32);
  for (auto w : list) if (low == w) return true;
  return false;
}
// Titles never end a sentence; an abbreviation, or a word ending in "...", ends one only before a word that starts with
// an ASCII capital after any opening quote or bracket (" ' U+201C U+2018 ( [). Red-team 2026-11-09: no. jr. sr. are
// abbreviations (they often end sentences), vs. is a title.
inline bool is_title(std::string_view w) { return in_list(w, {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "vs."}); }
inline bool is_abbreviation(std::string_view w) {
  return in_list(w, {"a.m.", "p.m.", "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "no.", "jr.", "sr.", "jan.", "feb.", "mar.",
                     "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."});
}
inline bool starts_capital(std::string_view word) {  // after any opening quotes or brackets
  for (;;) {
    if (!word.empty() && (word[0] == '"' || word[0] == '\'' || word[0] == '(' || word[0] == '[')) word.remove_prefix(1);
    else if (word.starts_with("\u201c") || word.starts_with("\u2018")) word.remove_prefix(3);
    else break;
  }
  return !word.empty() && word[0] >= 'A' && word[0] <= 'Z';
}

// Python str.isspace(): bidirectional class WS, B or S, or general category Zs.
inline bool is_space(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

inline size_t count_words(std::string_view s) {
  size_t i = 0, words = 0;
  bool in_word = false;
  std::uint32_t c = 0;
  while (i < s.size()) {
    if (!lfm3_text::next_code_point(s, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    const bool sp = is_space(c);
    words += !sp && !in_word;
    in_word = !sp;
  }
  return words;
}

// The words of s (maximal runs of non-whitespace), as views into s.
inline std::vector<std::string_view> split_words(std::string_view s) {
  std::vector<std::string_view> words;
  size_t i = 0, word = std::string_view::npos;
  while (i < s.size()) {
    const size_t at = i;
    std::uint32_t c = 0;
    if (!lfm3_text::next_code_point(s, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    if (is_space(c)) {
      if (word != std::string_view::npos) words.push_back(s.substr(word, at - word)), word = std::string_view::npos;
    } else if (word == std::string_view::npos) {
      word = at;
    }
  }
  if (word != std::string_view::npos) words.push_back(s.substr(word));
  return words;
}

inline std::string join_words(const std::vector<std::string_view>& w, size_t from, size_t to) {
  std::string out;
  for (size_t k = from; k < to; ++k) (out += (k > from ? " " : "")) += w[k];
  return out;
}

inline std::uint32_t last_code_point(std::string_view w) {
  size_t i = w.size() - 1;
  while (i > 0 && (static_cast<unsigned char>(w[i]) & 0xC0) == 0x80) --i;
  std::uint32_t c = 0;
  lfm3_text::next_code_point(w, i, c);
  return c;
}

inline bool is_mark(std::string_view word, std::string_view next) {
  const std::uint32_t last = last_code_point(word);
  return last == ',' || last == ';' || last == ':' || last == 0x2014 || last == 0x2013 || word.ends_with("--") ||
         next == "-" || next == "--" || next == "\u2014" || next == "\u2013";
}

// The chunk's words joined by single spaces; over cap words, cut at the last clause mark at or before word cap that
// leaves both sides kMinWords long, or with none into the fewest equal pieces; repeated on the rest (linear).
inline void cap_words(const std::string& chunk, size_t cap, std::vector<std::string>& out) {
  const auto w = split_words(chunk);
  size_t start = 0;
  while (w.size() - start > cap) {
    const size_t n = w.size() - start;
    size_t cut = 0;
    for (size_t i = start; i < std::min(start + cap, w.size() - 1); ++i) {
      const size_t k = i + 1 - start;
      if (k >= kMinWords && n - k >= kMinWords && is_mark(w[i], w[i + 1])) cut = k;
    }
    if (cut == 0) {  // the fewest equal pieces of at most cap words
      const size_t pieces = (n + cap - 1) / cap;
      cut = (n + pieces - 1) / pieces;
    }
    out.push_back(join_words(w, start, start + cut));
    start += cut;
  }
  out.push_back(join_words(w, start, w.size()));
}

inline std::vector<std::string> split_sentences(std::string_view text, size_t min_words = kMinWords,
                                                size_t cap = kCapWords) {
  struct Cp { std::uint32_t c; size_t at; };
  std::vector<Cp> cps;
  for (size_t i = 0; i < text.size();) {
    const size_t at = i;
    std::uint32_t c = 0;
    if (!lfm3_text::next_code_point(text, i, c)) throw std::runtime_error("speech text: invalid UTF-8");
    cps.push_back({c, at});
  }
  const auto byte = [&](size_t k) { return k < cps.size() ? cps[k].at : text.size(); };
  size_t b = 0, e = cps.size();
  while (b < e && is_space(cps[b].c)) ++b;
  while (e > b && is_space(cps[e - 1].c)) --e;
  std::vector<std::string> parts;
  if (b == e) return parts;
  size_t start = b, prev_end = b;
  for (size_t i = b + 1; i < e;) {  // cps[b] is not a space, so a run always has a word before it
    if (!is_space(cps[i].c)) { ++i; continue; }
    size_t j = i;
    while (is_space(cps[j].c)) ++j;  // cps[e - 1] is not a space: the run ends inside the text
    const std::string_view before = text.substr(byte(prev_end), byte(i) - byte(prev_end));
    const std::uint32_t prev = cps[i - 1].c;
    size_t k = j;
    while (k < e && !is_space(cps[k].c)) ++k;
    const std::string_view next = text.substr(byte(j), byte(k) - byte(j));
    prev_end = j;
    if ((prev == '.' || prev == '!' || prev == '?') && !is_title(before) &&
        !((is_abbreviation(before) || before.ends_with("...")) && !starts_capital(next))) {
      parts.emplace_back(text.substr(byte(start), byte(i) - byte(start)));
      start = j;
    }
    i = j;
  }
  parts.emplace_back(text.substr(byte(start), byte(e) - byte(start)));
  std::vector<std::string> merged;
  for (auto& p : parts) {
    if (!merged.empty() && count_words(merged.back()) < min_words) merged.back() += " " + p;
    else merged.push_back(std::move(p));
  }
  if (merged.size() > 1 && count_words(merged.back()) < min_words) {
    merged[merged.size() - 2] += " " + merged.back();
    merged.pop_back();
  }
  std::vector<std::string> out;
  for (const auto& m : merged) cap_words(m, cap, out);
  return out;
}

// One chat per sentence of the user text: [system?, user] in, the same system and one sentence each out.
inline std::vector<std::vector<lfm3_text::Message>> sentence_requests(const std::vector<lfm3_text::Message>& messages) {
  const bool sys = messages.size() == 2 && messages[0].role == "system";
  if (!(messages.size() == 1 || sys) || messages.back().role != "user")
    throw std::runtime_error("speech split: messages must be [system, user] or [user]");
  const auto& user = messages.back();
  if (user.content_is_list) throw std::runtime_error("speech split: the user content must be a string");
  const auto sentences = split_sentences(user.content);
  if (sentences.empty()) throw std::runtime_error("speech split: the user text is empty");
  // each chunk copies the messages without the user text (copying it whole for every sentence made the split's memory
  // quadratic in the text's length: audio-assurance-v1 red-team)
  auto base = messages;
  base.back().content = std::string();
  std::vector<std::vector<lfm3_text::Message>> out;
  out.reserve(sentences.size());
  for (const auto& s : sentences) {
    out.push_back(base);
    out.back().back().content = s;
  }
  return out;
}

// Numbers, symbols and abbreviations in words (lfm/speech_normalize.hpp, chosen by validation/speech-normalize-v1) in
// every user message's text, before the split: for speech output read from text (TTS). Each normalized text is held to
// the text byte limit (expansion is bounded: about six bytes per input byte at most).
inline void normalize_user_text(std::vector<lfm3_text::Message>& messages) {
  auto fix = [](std::string& s) {
    s = speech_normalize::normalize(s);
    if (s.size() > lfm3_text::kMaxTextBytes) throw std::runtime_error("speech normalize: text byte limit exceeded");
  };
  for (auto& m : messages) {
    if (m.role != "user") continue;
    if (!m.content_is_list) fix(m.content);
    else for (auto& p : m.parts) if (!p.audio) fix(p.text);
  }
}

}  // namespace lfm::speech_text
