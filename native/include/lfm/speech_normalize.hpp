#pragma once
// Text normalization for speech (validation/speech-normalize-v1, which chose these rules by test): numbers, symbols and
// abbreviations written out as words before the sentence split, American English. A native port of
// tools/speech_normalize.py, the reference: the same passes in the same order, each a hand-written scanner that finds
// the same matches as the reference's regular expression (leftmost, non-overlapping, with the same backtracking
// outcome), so the output is byte for byte the reference's (tests/test_speech_normalize.py). No std::regex: every pass
// is linear in the text apart from bounded look-arounds. Matching is ASCII (the reference's (?a)): digits, word
// characters, spaces and word boundaries are ASCII; the non-ASCII symbols (− £ € ° × – “) are matched as UTF-8
// sequences; the few places where the reference counts code points (context windows, Python's str.rstrip) count code
// points here too. Host-only.
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfm::speech_normalize {

namespace detail {

inline bool digit(char c) { return c >= '0' && c <= '9'; }
inline bool upper(char c) { return c >= 'A' && c <= 'Z'; }
inline bool lower(char c) { return c >= 'a' && c <= 'z'; }
inline bool alpha(char c) { return upper(c) || lower(c); }
inline bool word(char c) { return alpha(c) || digit(c) || c == '_'; }
inline bool space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// The text one pass reads, with the reference's character tests at byte positions (out of range: no character).
struct Text {
  std::string_view s;
  size_t n() const { return s.size(); }
  char at(size_t i) const { return i < s.size() ? s[i] : '\0'; }
  bool is_digit(size_t i) const { return i < s.size() && digit(s[i]); }
  bool is_word(size_t i) const { return i < s.size() && word(s[i]); }
  bool is_space(size_t i) const { return i < s.size() && space(s[i]); }
  bool prev_word(size_t i) const { return i > 0 && word(s[i - 1]); }
  bool boundary(size_t i) const { return prev_word(i) != is_word(i); }  // \b
  bool starts(size_t i, std::string_view x) const { return i <= s.size() && s.substr(i).substr(0, x.size()) == x; }
  size_t digits_end(size_t i) const { while (is_digit(i)) ++i; return i; }
  size_t spaces_end(size_t i) const { while (is_space(i)) ++i; return i; }
  // [\d,]* scans (amount_end) start at every digit after a comma; any start inside a run ends where the run ends, so
  // the last run's span is kept and a pass stays linear ("1,1,1,..." was quadratic: audio-assurance-v1 red-team)
  mutable size_t run_lo = 0, run_hi = 0;
  // \d{1,3}(?:,\d{3})+ chains (the decimals pass): a start inside a chain ends where the chain ends ("1,000,000,..."
  // was quadratic); [\w-] runs before a 4-digit number (the year test): later numbers in the same run reuse its start
  mutable size_t chain_lo = 0, chain_hi = 0, hy_lo = 0, hy_hi = 0;
  size_t chain_end(size_t de) const {  // from the end of a 1-3 digit run: past every ",ddd" group
    if (chain_lo < de && de <= chain_hi) return chain_hi;
    size_t e = de;
    while (at(e) == ',' && digits_end(e + 1) == e + 4) e += 4;
    chain_lo = de, chain_hi = e;
    return e;
  }
  // whitespace-delimited tokens that are links or addresses ("@", "://", "www.", or a domain), computed once a pass
  mutable std::vector<std::pair<size_t, size_t>> link_spans{};
  mutable bool links_done = false;
  bool in_link(size_t i) const;
  size_t hyphen_run_start(size_t p) const {  // the start of the [\w-] run ending at p
    if (hy_lo < p && p <= hy_hi) return hy_lo;
    size_t w = p;
    while (w > 0 && (word(s[w - 1]) || s[w - 1] == '-')) {
      if (w == hy_hi && hy_lo < hy_hi) { w = hy_lo; break; }
      --w;
    }
    hy_lo = w, hy_hi = p;
    return w;
  }
};

// (?a)@|://|www\.|[A-Za-z0-9]\.(?:com|org|net|edu|gov|io|dev|ai|app|co|uk|de)(?![A-Za-z0-9]) anywhere in a token
inline bool link_token(std::string_view tok) {
  if (tok.find('@') != std::string_view::npos || tok.find("://") != std::string_view::npos ||
      tok.find("www.") != std::string_view::npos)
    return true;
  static const char* const tlds[] = {"com", "org", "net", "edu", "gov", "io", "dev", "ai", "app", "co", "uk", "de"};
  auto alnum = [](char c) { return alpha(c) || digit(c); };
  for (size_t k = 1; k < tok.size(); ++k) {
    if (tok[k] != '.' || !alnum(tok[k - 1])) continue;
    for (const char* tld : tlds) {
      const std::string_view d(tld);
      if (tok.substr(k + 1, d.size()) == d && (k + 1 + d.size() == tok.size() || !alnum(tok[k + 1 + d.size()])))
        return true;
    }
  }
  return false;
}
inline bool Text::in_link(size_t i) const {
  if (!links_done) {
    links_done = true;
    for (size_t a = 0; a < s.size();) {
      if (space(s[a])) { ++a; continue; }
      size_t b = a;
      while (b < s.size() && !space(s[b])) ++b;
      if (link_token(s.substr(a, b - a))) link_spans.push_back({a, b});
      a = b;
    }
  }
  size_t lo = 0, hi = link_spans.size();  // the last span starting at or before i
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (link_spans[mid].first <= i) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && i < link_spans[lo - 1].second;
}

// A pass: at each position, the matcher returns the match's end and fills the replacement, or 0 for no match there.
template <class Matcher>
std::string pass(const std::string& in, Matcher&& match) {
  const Text t{in};
  std::string out;
  out.reserve(in.size() + in.size() / 4);
  std::string rep;
  for (size_t i = 0; i < in.size();) {
    rep.clear();
    const size_t end = match(t, i, rep);
    if (end > i) { out += rep; i = end; }
    else out += in[i++];
  }
  return out;
}

inline const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                              "nineteen"};
inline const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
inline const char* kMonths[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
                                "October", "November", "December"};

// Numbers as decimal digit strings, of any length (the reference uses Python's integers).
inline std::string strip_zeros(std::string_view d) {
  size_t k = 0;
  while (k + 1 < d.size() && d[k] == '0') ++k;
  return std::string(d.substr(k));
}
inline int small(std::string_view d) {  // the value of a digit string known to be short (< 10 digits)
  int v = 0;
  for (char c : d) v = v * 10 + (c - '0');
  return v;
}
inline bool is_zero(std::string_view d) { return strip_zeros(d) == "0"; }
inline bool equals(std::string_view d, long v) { const auto s = strip_zeros(d); return s.size() < 10 && small(s) == v; }
inline bool in_range(std::string_view d, long lo, long hi) {
  const auto s = strip_zeros(d);
  if (s.size() >= 10) return false;
  const long v = small(s);
  return lo <= v && v <= hi;
}

inline std::string cardinal_small(int n) {  // n < 1000
  if (n < 20) return kOnes[n];
  if (n < 100) return std::string(kTens[n / 10]) + (n % 10 ? std::string("-") + kOnes[n % 10] : "");
  return std::string(kOnes[n / 100]) + " hundred" + (n % 100 ? " " + cardinal_small(n % 100) : "");
}
// cardinal(int(d)): scales trillion, billion, million, thousand, largest first, recursively (as the reference)
inline std::string cardinal_upto12(const std::string& d) {  // d stripped, 1..12 digits
  if (d.size() <= 3) return cardinal_small(small(d));
  static const std::pair<size_t, const char*> scales[] = {{9, "billion"}, {6, "million"}, {3, "thousand"}};
  for (const auto& [k, name] : scales)
    if (d.size() > k) {
      const std::string hi = d.substr(0, d.size() - k), lo = strip_zeros(d.substr(d.size() - k));
      return cardinal_upto12(hi) + " " + name + (lo != "0" ? " " + cardinal_upto12(lo) : "");
    }
  return {};  // unreachable
}
// Above 12 digits the reference recurses on all but the last 12 ("... trillion" + the last 12): the same words built
// in one pass, most significant group first, so a long digit run costs linear time and no stack (a 400,000-digit
// ordinal overflowed the handler thread's stack: audio-assurance-v1 red-team).
inline std::string cardinal(std::string_view digits_in) {
  const std::string d = strip_zeros(digits_in);
  if (d.size() <= 12) return cardinal_upto12(d);
  const size_t groups = (d.size() - 1) / 12, head = d.size() - 12 * groups;
  std::string out = cardinal_upto12(d.substr(0, head));
  for (size_t g = 0; g < groups; ++g) {
    const std::string lo = strip_zeros(std::string_view(d).substr(head + 12 * g, 12));
    out += " trillion";
    if (lo != "0") out += " " + cardinal_upto12(lo);
  }
  return out;
}
inline std::string cardinal(long n) { return cardinal(std::to_string(n)); }

inline std::string ordinal(std::string_view d) {
  const std::string w = cardinal(d);
  const size_t last_space = w.rfind(' ');
  const std::string last_word = last_space == std::string::npos ? w : w.substr(last_space + 1);
  const char sep = last_word.find('-') != std::string::npos ? '-' : ' ';
  const size_t at = w.rfind(sep);
  const std::string head = at == std::string::npos ? "" : w.substr(0, at + 1);
  std::string last = at == std::string::npos ? w : w.substr(at + 1);
  static const std::pair<const char*, const char*> irregular[] = {
      {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"},
      {"nine", "ninth"}, {"twelve", "twelfth"}};
  bool done = false;
  for (const auto& [a, b] : irregular)
    if (last == a) { last = b; done = true; break; }
  if (!done) last = !last.empty() && last.back() == 'y' ? last.substr(0, last.size() - 1) + "ieth" : last + "th";
  return head + last;
}
inline std::string ordinal(long n) { return ordinal(std::to_string(n)); }

// pairs(n): years, decades and house numbers (n < 100000)
inline std::string pairs(long n) {
  if (n < 100 || (n % 1000 == 0 && n < 10000) || (2000 <= n && n < 2010)) return cardinal(n);
  if (n < 1000)
    return n % 100 == 0 ? cardinal(n)
                        : std::string(kOnes[n / 100]) + " " +
                              (n % 100 >= 10 ? cardinal(n % 100) : std::string("oh ") + kOnes[n % 100]);
  const long hi = n / 100, lo = n % 100;
  if (lo == 0) return cardinal(hi) + " hundred";
  return cardinal(hi) + " " + (lo >= 10 ? cardinal(lo) : std::string("oh ") + kOnes[lo]);
}
inline std::string plural_pairs(const std::string& w) {
  const size_t at = w.rfind(' ');
  const std::string head = at == std::string::npos ? "" : w.substr(0, at + 1);
  const std::string last = at == std::string::npos ? w : w.substr(at + 1);
  return head + (!last.empty() && last.back() == 'y' ? last.substr(0, last.size() - 1) + "ies" : last + "s");
}
inline std::string spell_digits(std::string_view d, const char* zero = "zero") {
  std::string out;
  for (char c : d) {
    if (!out.empty()) out += ' ';
    out += c == '0' ? zero : kOnes[c - '0'];
  }
  return out;
}
inline std::string without_commas(std::string_view s) {
  std::string out;
  for (char c : s) if (c != ',') out += c;
  return out;
}
// number(s): commas dropped; "whole.frac" as "<whole> point <digits>"
inline std::string number(std::string_view raw) {
  const std::string s = without_commas(raw);
  const size_t dot = s.find('.');
  if (dot == std::string::npos) return cardinal(s);
  const std::string whole = s.substr(0, dot);
  return cardinal(whole.empty() ? std::string("0") : whole) + " point " + spell_digits(std::string_view(s).substr(dot + 1));
}

// "\s*$|\s+[A-Z...]" at i: only whitespace remains, or whitespace then a capital (or, with quotes, a quote or bracket)
inline bool sentence_end(const Text& t, size_t i, bool quotes) {
  const size_t j = t.spaces_end(i);
  if (j == t.n()) return true;
  if (j == i) return false;
  if (upper(t.at(j))) return true;
  return quotes && (t.at(j) == '"' || t.at(j) == '\'' || t.at(j) == '(' || t.starts(j, "\xE2\x80\x9C"));
}
inline bool cont(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }
// the byte offset k code points before `end` (0 if fewer)
inline size_t back_code_points(std::string_view s, size_t end, size_t k) {
  size_t i = end;
  while (k > 0 && i > 0) {
    --i;
    while (i > 0 && cont(s[i])) --i;
    --k;
  }
  return i;
}
// Python's str.isspace for the code point ending at `end`; returns its start, or npos when not a space
inline size_t python_space_before(std::string_view s, size_t end) {
  if (end == 0) return std::string::npos;
  size_t i = end - 1;
  while (i > 0 && cont(s[i])) --i;
  const auto* p = reinterpret_cast<const unsigned char*>(s.data() + i);
  const size_t len = end - i;
  uint32_t cp;
  if (len == 1 && p[0] < 0x80) cp = p[0];
  else if (len == 2 && (p[0] & 0xE0) == 0xC0) cp = (p[0] & 0x1Fu) << 6 | (p[1] & 0x3Fu);
  else if (len == 3 && (p[0] & 0xF0) == 0xE0) cp = (p[0] & 0x0Fu) << 12 | (p[1] & 0x3Fu) << 6 | (p[2] & 0x3Fu);
  else return std::string::npos;  // 4-byte code points are never spaces; invalid UTF-8 is not either
  const bool sp = (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
                  cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
                  cp == 0x205F || cp == 0x3000;
  return sp ? i : std::string::npos;
}
inline size_t python_rstrip(std::string_view s, size_t end) {
  for (size_t at; (at = python_space_before(s, end)) != std::string::npos;) end = at;
  return end;
}
inline std::vector<std::string> letter_runs(std::string_view s) {  // re.findall("[A-Za-z]+")
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    if (!alpha(s[i])) { ++i; continue; }
    size_t j = i;
    while (j < s.size() && alpha(s[j])) ++j;
    out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}
inline std::string ascii_lower(std::string s) {
  for (char& c : s) if (upper(c)) c = char(c - 'A' + 'a');
  return s;
}
// an optional minus sign ("-" or U+2212) at i: its length
inline size_t minus_sign(const Text& t, size_t i) {
  if (t.at(i) == '-') return 1;
  return t.starts(i, "\xE2\x88\x92") ? 3 : 0;
}
// \d[\d,]*(?:\.\d+)? at i (the first character a digit): its end
inline size_t amount_end(const Text& t, size_t i) {
  if (!t.is_digit(i)) return i;
  size_t j;
  if (t.run_lo <= i && i < t.run_hi) {
    j = t.run_hi;
  } else {
    j = i + 1;
    while (t.is_digit(j) || t.at(j) == ',') ++j;
    t.run_lo = i, t.run_hi = j;
  }
  if (t.at(j) == '.' && t.is_digit(j + 1)) j = t.digits_end(j + 1);
  return j;
}
// \d+(?:\.\d+)? at i: its end
inline size_t decimal_end(const Text& t, size_t i) {
  if (!t.is_digit(i)) return i;
  size_t j = t.digits_end(i);
  if (t.at(j) == '.' && t.is_digit(j + 1)) j = t.digits_end(j + 1);
  return j;
}
inline bool not_after(const Text& t, size_t i, std::string_view chars, bool or_word) {  // (?<![\w<chars>])
  if (i == 0) return true;
  const char c = t.s[i - 1];
  return !(or_word && word(c)) && chars.find(c) == std::string_view::npos;
}

// value comparison of two digit strings (any length)
inline int compare_numbers(std::string_view a, std::string_view b) {
  const std::string x = strip_zeros(a), y = strip_zeros(b);
  if (x.size() != y.size()) return x.size() < y.size() ? -1 : 1;
  return x < y ? -1 : (x == y ? 0 : 1);
}
// "to the sixth", "squared", "cubed", "to the minus fifth" (the reference's power(n), n = -digits when neg)
inline std::string power(std::string_view d, bool neg) {
  if (neg && !is_zero(d)) return "to the minus " + ordinal(d);
  if (equals(d, 2)) return "squared";
  if (equals(d, 3)) return "cubed";
  return "to the " + ordinal(d);
}
// an exponent as written: whole through power(), decimal as a number ("to the zero point five")
inline std::string exponent(std::string_view d, bool neg) {
  if (d.find('.') != std::string_view::npos) return std::string("to the ") + (neg ? "minus " : "") + number(d);
  return power(d, neg);
}
inline std::string fraction_words(std::string_view n, std::string_view d) {
  const bool one = equals(n, 1);
  if (equals(d, 2)) return cardinal(n) + (one ? " half" : " halves");
  if (equals(d, 4)) return cardinal(n) + (one ? " quarter" : " quarters");
  return cardinal(n) + " " + ordinal(d) + (one ? "" : "s");
}
// \s*=\s*[-−]?\d at i: an equation's right-hand side follows
inline bool equation_after(const Text& t, size_t i) {
  const size_t j = t.spaces_end(i);
  if (t.at(j) != '=') return false;
  const size_t k = t.spaces_end(j + 1);
  return t.is_digit(k + minus_sign(t, k));
}
// a space when an ASCII letter follows a number directly ("22B", "6am"); not before a plural "s" or "ers" ("747s",
// "49ers": the letters stay on the number, as before the audit)
inline const char* glue(const Text& t, size_t end, bool whole = true) {  // after a decimal: "2.5s" -> "two point five s"
  if (!alpha(t.at(end))) return "";
  if (whole && t.at(end) == 's' && !alpha(t.at(end + 1))) return "";
  if (whole && t.starts(end, "ers") && !alpha(t.at(end + 3))) return "";
  return " ";
}
// a < b for two decimal strings, exactly
inline bool decimal_less(std::string_view a, std::string_view b) {
  const size_t da = a.find('.'), db = b.find('.');
  const std::string ai = strip_zeros(a.substr(0, da)), bi = strip_zeros(b.substr(0, db));
  if (ai.size() != bi.size()) return ai.size() < bi.size();
  if (ai != bi) return ai < bi;
  std::string af(da == std::string_view::npos ? "" : a.substr(da + 1)), bf(db == std::string_view::npos ? "" : b.substr(db + 1));
  const size_t w = std::max(af.size(), bf.size());
  af.resize(w, '0'), bf.resize(w, '0');
  return af < bf;
}
// [aApP]\.?[mM]\b at j
inline bool meridiem_at(const Text& x, size_t j) {
  const char a = x.at(j);
  if (a != 'a' && a != 'A' && a != 'p' && a != 'P') return false;
  size_t k = j + 1;
  if (x.at(k) == '.' && (x.at(k + 1) == 'm' || x.at(k + 1) == 'M')) k += 2;
  else if (x.at(k) == 'm' || x.at(k) == 'M') k += 1;
  else return false;
  return x.boundary(k);
}
// "six" -> "sixes", "twenty" -> "twenties", "forty-seven" -> "forty-sevens"
inline std::string plural_words(const std::string& w) {
  if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ies";
  return w + (!w.empty() && w.back() == 'x' ? "es" : "s");
}
// the match [i, end) as written (a rule that declines after matching, as the reference's re.sub returning group 0)
inline size_t as_written(const Text& t, size_t i, size_t end, std::string& r) {
  r = std::string(t.s.substr(i, end - i));
  return end;
}
// ([A-Za-z]+):? +\Z in the 40 code points before i: that word, lowercased, or ""
inline std::string word_before(const Text& t, size_t i) {
  const size_t lo = back_code_points(t.s, i, 40);
  size_t q = i;
  while (q > lo && t.s[q - 1] == ' ') --q;
  if (q == i) return "";
  if (q > lo && t.s[q - 1] == ':') --q;
  size_t w = q;
  while (w > lo && alpha(t.s[w - 1])) --w;
  return w == q ? "" : ascii_lower(std::string(t.s.substr(w, q - w)));
}
// the last three words in the 30 code points before start, then (with_after) the word after end ( +([A-Za-z]+))
inline std::vector<std::string> words_near(const Text& t, size_t start, size_t end, bool with_after) {
  const size_t lo = back_code_points(t.s, start, 30);
  auto runs = letter_runs(t.s.substr(lo, start - lo));
  std::vector<std::string> out;
  for (size_t k = runs.size() > 3 ? runs.size() - 3 : 0; k < runs.size(); ++k) out.push_back(ascii_lower(runs[k]));
  if (with_after && t.at(end) == ' ') {
    size_t j = end;
    while (t.at(j) == ' ') ++j;
    size_t k = j;
    while (alpha(t.at(k))) ++k;
    if (k > j) out.push_back(ascii_lower(std::string(t.s.substr(j, k - j))));
  }
  return out;
}
inline bool any_of_words(const std::vector<std::string>& words, std::initializer_list<const char*> cues) {
  for (const auto& w : words)
    for (const char* c : cues)
      if (w == c) return true;
  return false;
}
// a run of lo..hi digits at i (the next character not a digit): its end, or 0
inline size_t run_of(const Text& t, size_t i, size_t lo, size_t hi) {
  const size_t e = t.digits_end(i);
  return e - i >= lo && e - i <= hi ? e : 0;
}

}  // namespace detail

inline std::string normalize(const std::string& text) {
  using namespace detail;
  std::string t = text;

  // ISO dates: \b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b(?!-)   (validation/audio-assurance-v1
  // normalizer-audit.md, as every pass marked "audit" below)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i) || x.digits_end(i) < i + 4 || x.at(i + 4) != '-') return 0;
    const char m0 = x.at(i + 5), m1 = x.at(i + 6), d0 = x.at(i + 8), d1 = x.at(i + 9);
    const bool month = (m0 == '0' && m1 >= '1' && m1 <= '9') || (m0 == '1' && m1 >= '0' && m1 <= '2');
    const bool day = (d0 == '0' && d1 >= '1' && d1 <= '9') || ((d0 == '1' || d0 == '2') && digit(d1)) ||
                     (d0 == '3' && (d1 == '0' || d1 == '1'));
    if (!month || x.at(i + 7) != '-' || !day || !x.boundary(i + 10) || x.at(i + 10) == '-') return 0;
    if (x.in_link(i)) return as_written(x, i, i + 10, r);
    r = std::string(kMonths[small(x.s.substr(i + 5, 2)) - 1]) + " " + ordinal(x.s.substr(i + 8, 2)) + ", " +
        pairs(small(x.s.substr(i, 4)));
    return i + 10;
  });
  // audit: international numbers: (?<![\w.+])\+(\d{1,3})((?:[ .-]\d{1,4}(?!\d)){2,5})\b(?![.-]?\d), 8 digits or more
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (x.at(i) != '+' || !not_after(x, i, ".+", true)) return 0;
    const size_t c = run_of(x, i + 1, 1, 3);
    if (!c) return 0;
    std::vector<std::pair<size_t, size_t>> groups{{i + 1, c}};
    for (size_t j = c; groups.size() < 6;) {
      if (x.at(j) != ' ' && x.at(j) != '.' && x.at(j) != '-') break;
      const size_t e = run_of(x, j + 1, 1, 4);
      if (!e) break;
      groups.push_back({j + 1, e});
      j = e;
    }
    for (size_t k = groups.size(); k >= 3; --k) {  // the most groups whose end passes \b(?![.-]?\d)
      const size_t end = groups[k - 1].second;
      if (x.is_word(end) || ((x.at(end) == '.' || x.at(end) == '-') && x.is_digit(end + 1))) continue;
      size_t total = 0;
      for (size_t g = 0; g < k; ++g) total += groups[g].second - groups[g].first;
      if (total < 8 || x.in_link(i)) return as_written(x, i, end, r);
      r = "plus ";
      for (size_t g = 0; g < k; ++g)
        r += (g ? ", " : "") + spell_digits(x.s.substr(groups[g].first, groups[g].second - groups[g].first), "oh");
      return end;
    }
    return 0;
  });

  // phone numbers: \(?\b(\d{3})\)?[-. ](\d{3})[-.](\d{4})\b   (audit: "." between the last groups too)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    size_t j = i + (x.at(i) == '(' ? 1 : 0);
    if (!x.boundary(j) || x.digits_end(j) < j + 3) return 0;
    const size_t a = j;
    j += 3;
    if (x.at(j) == ')') ++j;
    if (x.at(j) != '-' && x.at(j) != '.' && x.at(j) != ' ') return 0;
    const size_t b = ++j;
    if (x.digits_end(j) < j + 3 || (x.at(j + 3) != '-' && x.at(j + 3) != '.')) return 0;
    const size_t c = j + 4;
    if (x.digits_end(c) < c + 4 || !x.boundary(c + 4)) return 0;
    r = spell_digits(x.s.substr(a, 3), "oh") + ", " + spell_digits(x.s.substr(b, 3), "oh") + ", " +
        spell_digits(x.s.substr(c, 4), "oh");
    return c + 4;
  });
  // (?<![\d.,-])\b(\d{3})-(\d{4})\b(?![-\d])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, "0123456789.,-", false) || !x.boundary(i)) return 0;
    if (x.digits_end(i) < i + 3 || x.at(i + 3) != '-' || x.digits_end(i + 4) < i + 8) return 0;
    const size_t e = i + 8;
    if (!x.boundary(e) || x.at(e) == '-' || x.is_digit(e)) return 0;
    r = spell_digits(x.s.substr(i, 3), "oh") + ", " + spell_digits(x.s.substr(i + 4, 4), "oh");
    return e;
  });
  // audit: a plus sign on a number: (?<![\w.+])\+(?=[$£€]?\d) ; between digits: (?<=\d)\+(?=\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (x.at(i) != '+' || !not_after(x, i, ".+", true)) return 0;
    size_t j = i + 1;
    if (x.at(j) == '$') j += 1;
    else if (x.starts(j, "\xC2\xA3")) j += 2;
    else if (x.starts(j, "\xE2\x82\xAC")) j += 3;
    if (!x.is_digit(j)) return 0;
    if (x.in_link(i)) return as_written(x, i, i + 1, r);
    r = "plus ";
    return i + 1;
  });
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (i == 0 || !digit(x.s[i - 1]) || x.at(i) != '+' || !x.is_digit(i + 1)) return 0;
    if (x.in_link(i)) return as_written(x, i, i + 1, r);
    r = " plus ";
    return i + 1;
  });
  // audit: feet and inches: (?<![\w.,'])(\d+)' ?(\d{1,2})(?:"|”|'')
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,'", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i || x.at(a) != '\'') return 0;
    const size_t b0 = a + 1 + (x.at(a + 1) == ' ' ? 1 : 0), b = run_of(x, b0, 1, 2);
    if (!b) return 0;
    size_t e;
    if (x.at(b) == '"') e = b + 1;
    else if (x.starts(b, "\xE2\x80\x9D")) e = b + 3;
    else if (x.starts(b, "''")) e = b + 2;
    else return 0;
    const std::string_view f = x.s.substr(i, a - i), n = x.s.substr(b0, b - b0);
    // under 12 inches, and not an angle: (?:°|\bdeg\.?) *$ before the number
    size_t q = i;
    while (q > 0 && x.s[q - 1] == ' ') --q;
    const std::string_view before = x.s.substr(0, q);
    auto deg_at = [&](size_t k) { return k >= 3 && before.substr(k - 3, 3) == "deg" && (k == 3 || !word(before[k - 4])); };
    const bool angle = (before.size() >= 2 && before.substr(before.size() - 2) == "\xC2\xB0") ||
                       deg_at(before.size()) || (!before.empty() && before.back() == '.' && deg_at(before.size() - 1));
    if (small(n) > 11 || angle || x.in_link(i)) return as_written(x, i, e, r);
    r = cardinal(f) + (f == "1" ? " foot " : " feet ") + cardinal(n) + (n == "1" ? " inch" : " inches");
    return e;
  });
  // audit: dotted numbers: (?<![\w.,/])\d+(?:\.\d+){2,}\b(?!\.\d) ; a date (15.03.2024 day first; 3.15.2024 month first
  // when the day is above 12; 2024.03.15); four valid octets: an IPv4 address digit by digit with "dot" (a three-digit
  // group, or a network word before it), else a version; grouped thousands (1.234.567, no version word before it);
  // otherwise a version with "point"
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,/", true)) return 0;
    std::vector<std::string_view> g;
    size_t e = x.digits_end(i);
    if (e == i) return 0;
    g.push_back(x.s.substr(i, e - i));
    while (x.at(e) == '.' && x.is_digit(e + 1)) {
      const size_t k = x.digits_end(e + 1);
      g.push_back(x.s.substr(e + 1, k - e - 1));
      e = k;
    }
    if (g.size() < 3 || x.is_word(e)) return 0;  // fewer groups would leave ".\d" next
    if (x.in_link(i)) return as_written(x, i, e, r);
    const auto near = words_near(x, i, i, false);
    if (g.size() == 3 && g[2].size() == 4 && in_range(g[2], 1000, 2099) && g[0].size() <= 2 && g[1].size() <= 2) {
      if (in_range(g[0], 1, 31) && in_range(g[1], 1, 12)) {
        r = std::string(kMonths[small(g[1]) - 1]) + " " + ordinal(g[0]) + ", " + pairs(small(g[2]));
        return e;
      }
      if (in_range(g[0], 1, 12) && in_range(g[1], 13, 31)) {
        r = std::string(kMonths[small(g[0]) - 1]) + " " + ordinal(g[1]) + ", " + pairs(small(g[2]));
        return e;
      }
    }
    if (g.size() == 3 && g[0].size() == 4 && in_range(g[0], 1000, 2099) && g[1].size() == 2 && g[2].size() == 2 &&
        in_range(g[1], 1, 12) && in_range(g[2], 1, 31)) {
      r = std::string(kMonths[small(g[1]) - 1]) + " " + ordinal(g[2]) + ", " + pairs(small(g[0]));
      return e;
    }
    bool octets = g.size() == 4, three = false;
    for (const auto& v : g) octets = octets && v.size() <= 3 && small(v) <= 255, three = three || v.size() == 3;
    const bool ip = octets && (three || any_of_words(near, {"ip", "address", "ping", "connect", "server", "host", "mask",
                                                            "netmask", "gateway", "dns", "subnet", "router"}));
    bool thousands = !octets && g[0].size() <= 3 && g[0][0] != '0';
    for (size_t k = 1; k < g.size(); ++k) thousands = thousands && g[k].size() == 3;
    if (!ip && thousands &&
        !any_of_words(near, {"version", "versions", "release", "releases", "v", "build", "update", "upgrade", "patch",
                             "ip", "address", "ping", "connect", "server", "host", "mask", "netmask", "gateway", "dns",
                             "subnet", "router"})) {
      std::string all;
      for (const auto& v : g) all += v;
      r = cardinal(all);
      return e;
    }
    r.clear();
    for (const auto& v : g) {
      if (!r.empty()) r += ip ? " dot " : " point ";
      r += ip || (v.size() > 1 && v[0] == '0') ? spell_digits(v) : cardinal(v);  // a zero-led version part: "22.04.3"
    }
    return e;
  });
  // audit: scientific notation: (?<![\w.,/#-])(\d+(?:\.\d+)?)([eE])([-+−]?)(\d+(?:\.\d+)?)\b(?![-.]?\w) (a capital E
  // without a decimal point or a sign is left as written: "Room 2E1");
  // powers: (?<![\w.,/-])(\d+(?:\.\d+)?)\^([-−]?)(\d+(?:\.\d+)?)\b(?!\.\d)
  // a minus sign before them: (?<![\w.])[-−](?=\d+(?:\.\d+)?(?:\^|[eE][-+−]?\d)) -> "minus "  (-10^9, -1e-5)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t m = minus_sign(x, i);
    if (!m) return 0;
    const size_t a = decimal_end(x, i + m);
    if (a == i + m) return 0;
    bool ok = x.at(a) == '^';
    if (!ok && (x.at(a) == 'e' || x.at(a) == 'E')) {
      size_t k = a + 1;
      if (const size_t mm = minus_sign(x, k)) k += mm;
      else if (x.at(k) == '+') k += 1;
      ok = x.is_digit(k);
    }
    if (!ok) return 0;
    if (x.in_link(i)) return as_written(x, i, i + m, r);
    r = "minus ";
    return i + m;
  });
  for (const bool sci : {true, false}) {
    t = pass(t, [sci](const Text& x, size_t i, std::string& r) -> size_t {
      if (!not_after(x, i, sci ? ".,/#-" : ".,/-", true)) return 0;
      const size_t a = decimal_end(x, i);
      if (a == i) return 0;
      size_t j = a;
      if (sci ? (x.at(j) != 'e' && x.at(j) != 'E') : x.at(j) != '^') return 0;
      const bool capital = x.at(j) == 'E';
      ++j;
      bool neg = false, sign = false;
      if (const size_t m = minus_sign(x, j)) neg = sign = true, j += m;
      else if (sci && x.at(j) == '+') sign = true, j += 1;
      const size_t e = decimal_end(x, j);  // the exponent, a decimal part read (audit red-team: "2^0.5")
      if (e == j || !x.boundary(e)) return 0;
      if (sci ? ((x.at(e) == '-' || x.at(e) == '.') && x.is_word(e + 1)) : (x.at(e) == '.' && x.is_digit(e + 1)))
        return 0;
      if (x.in_link(i) || (sci && capital && !sign && a == x.digits_end(i))) return as_written(x, i, e, r);
      r = number(x.s.substr(i, a - i)) + (sci ? " times ten " : " ") + exponent(x.s.substr(j, e - j), neg);
      if (!sci) {  // (?:( ?)[-−–]\4(?=\d))? : 2^31-1, "two to the thirty-first minus one"
        const bool sp = x.at(e) == ' ';
        size_t k = e + (sp ? 1 : 0);
        size_t m = minus_sign(x, k);
        if (!m && x.starts(k, "\xE2\x80\x93")) m = 3;
        if (m && (!sp || x.at(k + m) == ' ') && x.is_digit(k + m + (sp ? 1 : 0))) {
          r += " minus ";
          return k + m + (sp ? 1 : 0);
        }
      }
      return e;
    });
  }
  // audit: a minus sign on money: (?<![\w.])[-−](?=[$£€]\d) ; money ranges, no space or a space on both sides, not a
  // subtraction: (?<![\w.])([$£€]\d[\d,]*(?:\.\d+)?)( ?)[-–]\2(?=[$£€]?\d)(?![$£€]?\d[\d,]*(?:\.\d+)?\s*=) -> "\1 to "
  auto currency = [](const Text& x, size_t i) -> size_t {  // a currency sign at i: its length
    if (x.at(i) == '$') return 1;
    if (x.starts(i, "\xC2\xA3")) return 2;
    return x.starts(i, "\xE2\x82\xAC") ? 3 : 0;
  };
  t = pass(t, [&](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t m = minus_sign(x, i);
    if (!m) return 0;
    const size_t c = currency(x, i + m);
    if (!c || !x.is_digit(i + m + c)) return 0;
    if (x.in_link(i)) return as_written(x, i, i + m, r);
    r = "minus ";
    return i + m;
  });
  t = pass(t, [&](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t c0 = currency(x, i);
    if (!c0) return 0;
    const size_t a = amount_end(x, i + c0);
    if (a == i + c0) return 0;
    const bool sp = x.at(a) == ' ';
    size_t j = a + (sp ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else return 0;
    if (sp) {
      if (x.at(j) != ' ') return 0;
      ++j;
    }
    const size_t k = j + currency(x, j);
    if (!x.is_digit(k)) return 0;
    if (x.in_link(i)) return as_written(x, i, j, r);
    // red-team round 3: ascending, not chained ("$60-$13-$6"), no "=" later in the clause ("$450 - $75 - $120 = $255")
    const size_t b = amount_end(x, k);
    size_t q = i;
    while (q > 0 && space(x.s[q - 1])) --q;
    bool chained_before = false;  // an amount, then a dash, before it (not a bullet "- $5 - $10")
    if (q > 0 && (x.s[q - 1] == '-' || (q >= 3 && x.s.substr(q - 3, 3) == "\xE2\x80\x93"))) {
      size_t k = x.s[q - 1] == '-' ? q - 1 : q - 3;
      while (k > 0 && space(x.s[k - 1])) --k;
      chained_before = k > 0 && digit(x.s[k - 1]);
    }
    size_t c = x.spaces_end(b);
    bool chained_after = false;
    if (x.at(c) == '-' || x.starts(c, "\xE2\x80\x93")) {
      c = x.spaces_end(c + (x.at(c) == '-' ? 1 : 3));
      chained_after = x.is_digit(c + currency(x, c));
    }
    bool equation = false;  // "=" before [;!?\n] or "." + whitespace / the window's end, in 120 code points after b
    for (size_t p = b, cp = 0; p < x.n() && cp < 120; ++cp) {
      const char ch = x.s[p];
      size_t len = 1;
      while (p + len < x.n() && cont(x.s[p + len])) ++len;
      if (ch == ';' || ch == '!' || ch == '?' || ch == '\n') break;
      if (ch == '.' && (cp + 1 == 120 || p + 1 == x.n() || space(x.s[p + 1]))) break;
      if (ch == '=') { equation = true; break; }
      p += len;
    }
    if (chained_before || chained_after || equation ||
        !decimal_less(without_commas(x.s.substr(i + c0, a - i - c0)), without_commas(x.s.substr(k, b - k))))
      return as_written(x, i, j, r);
    r = std::string(x.s.substr(i, a - i)) + " to ";
    return j;
  });

  // money: ([$£€])(\d[\d,]*(?:\.\d+)?)(\s+(million|billion|trillion|thousand)\b)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const char* const units[][4] = {{"dollar", "dollars", "cent", "cents"}, {"pound", "pounds", "penny", "pence"},
                                           {"euro", "euros", "cent", "cents"}};
    int cur;
    size_t j;
    if (x.at(i) == '$') cur = 0, j = i + 1;
    else if (x.starts(i, "\xC2\xA3")) cur = 1, j = i + 2;
    else if (x.starts(i, "\xE2\x82\xAC")) cur = 2, j = i + 3;
    else return 0;
    const size_t e = amount_end(x, j);
    if (e == j) return 0;
    const std::string value = without_commas(x.s.substr(j, e - j));
    size_t end = e;
    const size_t k = x.spaces_end(e);
    if (k > e)
      for (const char* scale : {"million", "billion", "trillion", "thousand"})
        if (x.starts(k, scale) && x.boundary(k + std::string_view(scale).size())) {
          end = k + std::string_view(scale).size();
          r = number(value) + " " + scale + " " + units[cur][1];
          return end;
        }
    // audit: (bn|[kKMB])\b right after the amount: $40k, $5M, $3.2B, $2bn
    const char* letter = nullptr;
    size_t letter_end = 0;
    if (x.starts(e, "bn") && x.boundary(e + 2)) letter = "billion", letter_end = e + 2;
    else if ((x.at(e) == 'k' || x.at(e) == 'K') && x.boundary(e + 1)) letter = "thousand", letter_end = e + 1;
    else if (x.at(e) == 'M' && x.boundary(e + 1)) letter = "million", letter_end = e + 1;
    else if (x.at(e) == 'B' && x.boundary(e + 1)) letter = "billion", letter_end = e + 1;
    if (letter) { r = number(value) + " " + letter + " " + units[cur][1]; return letter_end; }
    const size_t dot = value.find('.');
    const std::string whole = value.substr(0, dot);
    const std::string frac = dot == std::string::npos ? "" : value.substr(dot + 1);
    if (frac.size() > 2) { r = number(value) + " " + units[cur][1]; return end; }  // audit: $3.999
    const int cents = frac.empty() ? 0 : small((frac + "00").substr(0, 2));
    if (!is_zero(whole) || !cents) r = cardinal(whole) + " " + (equals(whole, 1) ? units[cur][0] : units[cur][1]);
    if (cents) r += (r.empty() ? "" : " and ") + cardinal(long(cents)) + " " + (cents == 1 ? units[cur][2] : units[cur][3]);
    return end;
  });
  // numeric dates: \b(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])/(\d{4})\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    auto alternatives = [&](size_t at, bool day) {  // the lengths the alternation can take at `at`, in its order
      std::vector<size_t> out;
      if (!day) {
        if (x.at(at) == '1' && x.at(at + 1) >= '0' && x.at(at + 1) <= '2') out.push_back(2);
      } else {
        if (x.at(at) == '3' && (x.at(at + 1) == '0' || x.at(at + 1) == '1')) out.push_back(2);
        if ((x.at(at) == '1' || x.at(at) == '2') && x.is_digit(at + 1)) out.push_back(2);
      }
      if (x.at(at) == '0' && x.at(at + 1) >= '1' && x.at(at + 1) <= '9') out.push_back(2);
      if (x.at(at) >= '1' && x.at(at) <= '9') out.push_back(1);
      return out;
    };
    for (size_t m : alternatives(i, false)) {
      if (x.at(i + m) != '/') continue;
      const size_t d0 = i + m + 1;
      for (size_t d : alternatives(d0, true)) {
        const size_t y0 = d0 + d;
        if (x.at(y0) != '/' || x.digits_end(y0 + 1) < y0 + 5 || !x.boundary(y0 + 5)) continue;
        r = std::string(kMonths[small(x.s.substr(i, m)) - 1]) + " " + ordinal(x.s.substr(d0, d)) + ", " +
            pairs(small(x.s.substr(y0 + 1, 4)));
        return y0 + 5;
      }
    }
    return 0;
  });
  // audit: day first: \b(1[3-9]|2\d|3[01])/(1[0-2]|0?[1-9])/(\d{4})\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    const char d0 = x.at(i), d1 = x.at(i + 1);
    const bool day = (d0 == '1' && d1 >= '3' && d1 <= '9') || (d0 == '2' && digit(d1)) || (d0 == '3' && (d1 == '0' || d1 == '1'));
    if (!day || x.at(i + 2) != '/') return 0;
    const size_t m0 = i + 3;
    std::vector<size_t> months;
    if (x.at(m0) == '1' && x.at(m0 + 1) >= '0' && x.at(m0 + 1) <= '2') months.push_back(2);
    if (x.at(m0) == '0' && x.at(m0 + 1) >= '1' && x.at(m0 + 1) <= '9') months.push_back(2);
    if (x.at(m0) >= '1' && x.at(m0) <= '9') months.push_back(1);
    for (size_t m : months) {
      const size_t y0 = m0 + m;
      if (x.at(y0) != '/' || x.digits_end(y0 + 1) < y0 + 5 || !x.boundary(y0 + 5)) continue;
      if (x.in_link(i)) return as_written(x, i, y0 + 5, r);
      r = std::string(kMonths[small(x.s.substr(m0, m)) - 1]) + " " + ordinal(x.s.substr(i, 2)) + ", " +
          pairs(small(x.s.substr(y0 + 1, 4)));
      return y0 + 5;
    }
    return 0;
  });
  // audit: h:mm:ss(.s): (?<!:)\b(\d{1,2}):([0-5]\d):([0-5]\d(?:\.\d+)?)\b(?![:.]\d|:) ; a duration ("in", "ran", "runs",
  // "duration", ..., or more than 23 hours; a duration word beats a leading zero) in hours, minutes and seconds; a time
  // of day (a.m./p.m. after it, a leading zero, "at"/"by"/..., or 13 hours or more) as a clock; otherwise as written
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i) || (i > 0 && x.s[i - 1] == ':')) return 0;
    const size_t h = run_of(x, i, 1, 2);
    if (!h || x.at(h) != ':') return 0;
    auto two = [&](size_t at) { return x.at(at) >= '0' && x.at(at) <= '5' && x.is_digit(at + 1); };
    if (!two(h + 1) || x.at(h + 3) != ':' || !two(h + 4)) return 0;
    const size_t end = (x.at(h + 6) == '.' && x.is_digit(h + 7)) ? x.digits_end(h + 7) : h + 6;
    if (!x.boundary(end) || x.at(end) == ':' || (x.at(end) == '.' && x.is_digit(end + 1))) return 0;
    if (x.in_link(i)) return as_written(x, i, end, r);
    const std::string_view hs = x.s.substr(i, h - i), sec = x.s.substr(h + 4, end - h - 4);
    const int hh = small(hs), mm = small(x.s.substr(h + 1, 2));
    bool sec_zero = true;
    for (char c : sec) sec_zero = sec_zero && (c == '0' || c == '.');
    const std::string sec_words = number(sec) + (sec == "01" ? " second" : " seconds");
    const std::string w = word_before(x, i);
    auto one_of = [&](std::initializer_list<const char*> cues) {
      for (const char* c : cues) if (w == c) return true;
      return false;
    };
    const bool meridiem = meridiem_at(x, x.spaces_end(end));
    if (hh > 23 || (!meridiem && one_of({"in", "ran", "run", "time", "times", "lasted", "lasts", "took", "takes",
                                         "finished", "clocked", "record", "split", "length", "runtime", "duration",
                                         "runs"}))) {
      const std::pair<int, const char*> parts[] = {{hh, "hour"}, {mm, "minute"}};
      r.clear();
      for (const auto& [v, unit] : parts)
        if (v) r += (r.empty() ? "" : " ") + cardinal(long(v)) + " " + unit + (v == 1 ? "" : "s");
      if (!sec_zero) r += (r.empty() ? "" : " ") + sec_words;
      if (r.empty()) r = "zero seconds";
      return end;
    }
    if (meridiem || (hs.size() == 2 && hs[0] == '0') || one_of({"at", "by", "until", "till", "from", "since"}) || hh >= 13) {
      r = cardinal(long(hh)) + (mm == 0 ? std::string(meridiem ? "" : " o'clock")
                                        : " " + (mm >= 10 ? cardinal(long(mm)) : std::string("oh ") + kOnes[mm]));
      if (!sec_zero) r += " and " + sec_words;
      return end;
    }
    return as_written(x, i, end, r);
  });

  // audit: a range before a time with a.m./p.m.: (?<![\w.,:-])(\d{1,2})( ?)[-–]\2(?=\d{1,2}(?::[0-5]\d)?\s*[aApP]\.?[mM]\b)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,:-", true)) return 0;
    const size_t a = run_of(x, i, 1, 2);
    if (!a) return 0;
    const bool sp = x.at(a) == ' ';
    size_t j = a + (sp ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else return 0;
    if (sp) {
      if (x.at(j) != ' ') return 0;
      ++j;
    }
    size_t k = run_of(x, j, 1, 2);
    if (!k) return 0;
    if (x.at(k) == ':' && x.at(k + 1) >= '0' && x.at(k + 1) <= '5' && x.is_digit(k + 2)) k += 3;
    if (!meridiem_at(x, x.spaces_end(k))) return 0;
    if (x.in_link(i)) return as_written(x, i, j, r);
    r = std::string(x.s.substr(i, a - i)) + " to ";
    return j;
  });
  // times: \b([01]?\d|2[0-3]):([0-5]\d)\b(?=(\s*[aApP]\.?[mM]\b\.?)?)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    std::vector<size_t> hours;
    if ((x.at(i) == '0' || x.at(i) == '1') && x.is_digit(i + 1)) hours.push_back(2);
    if (x.is_digit(i)) hours.push_back(1);
    if (x.at(i) == '2' && x.at(i + 1) >= '0' && x.at(i + 1) <= '3') hours.push_back(2);
    for (size_t h : hours) {
      const size_t c = i + h;
      if (x.at(c) != ':' || !(x.at(c + 1) >= '0' && x.at(c + 1) <= '5') || !x.is_digit(c + 2)) continue;
      const size_t e = c + 3;
      auto meridiem_at = [&](size_t j) {  // [aApP]\.?[mM]\b at j
        const char a = x.at(j);
        if (a != 'a' && a != 'A' && a != 'p' && a != 'P') return false;
        size_t k = j + 1;
        if (x.at(k) == '.' && (x.at(k + 1) == 'm' || x.at(k + 1) == 'M')) k += 2;
        else if (x.at(k) == 'm' || x.at(k) == 'M') k += 1;
        else return false;
        return x.boundary(k);
      };
      // audit: a lap time's fraction, (\.\d+)? (given back when what follows it fails): "1:23.456"
      if (x.at(e) == '.' && x.is_digit(e + 1)) {
        const size_t f = x.digits_end(e + 1);
        if (x.boundary(f) || meridiem_at(f)) {
          const int hh = small(x.s.substr(i, h)), mm = small(x.s.substr(c + 1, 2));
          r = cardinal(long(hh)) + " " + (mm >= 10 ? cardinal(long(mm)) : spell_digits(x.s.substr(c + 1, 2), "oh")) +
              " point " + spell_digits(x.s.substr(e + 1, f - e - 1)) + glue(x, f, false);
          return f;
        }
      }
      if (!x.boundary(e) && !meridiem_at(e)) continue;  // audit: (?:\b|(?=[aApP]\.?[mM]\b)), "3:30pm"
      bool meridiem = false;  // the look-ahead's optional group
      size_t j = x.spaces_end(e);
      const char a = x.at(j);
      if (a == 'a' || a == 'A' || a == 'p' || a == 'P') {
        ++j;
        size_t k = j;
        if (x.at(k) == '.' && (x.at(k + 1) == 'm' || x.at(k + 1) == 'M')) k += 2;
        else if (x.at(k) == 'm' || x.at(k) == 'M') k += 1;
        else k = 0;
        meridiem = k && x.boundary(k);
      }
      const int hh = small(x.s.substr(i, h)), mm = small(x.s.substr(c + 1, 2));
      if (mm == 0) r = cardinal(long(hh)) + (meridiem ? "" : " o'clock");
      else r = cardinal(long(hh)) + " " + (mm >= 10 ? cardinal(long(mm)) : std::string("oh ") + kOnes[mm]);
      r += glue(x, e);
      return e;
    }
    return 0;
  });
  // audit: a time with a dot before a.m./p.m.: \b(1[0-2]|0?[1-9])\.([0-5]\d)(?=\s*[aApP]\.?[mM]\b)  (7.30pm)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    std::vector<size_t> hours;
    if (x.at(i) == '1' && x.at(i + 1) >= '0' && x.at(i + 1) <= '2') hours.push_back(2);
    if (x.at(i) == '0' && x.at(i + 1) >= '1' && x.at(i + 1) <= '9') hours.push_back(2);
    if (x.at(i) >= '1' && x.at(i) <= '9') hours.push_back(1);
    for (size_t h : hours) {
      const size_t c = i + h;
      if (x.at(c) != '.' || !(x.at(c + 1) >= '0' && x.at(c + 1) <= '5') || !x.is_digit(c + 2)) continue;
      const size_t e = c + 3;
      if (!meridiem_at(x, x.spaces_end(e))) continue;
      if (x.in_link(i)) return as_written(x, i, e, r);
      const int mm = small(x.s.substr(c + 1, 2));
      r = cardinal(x.s.substr(i, h)) +
          (mm == 0 ? std::string() : " " + (mm >= 10 ? cardinal(long(mm)) : std::string("oh ") + kOnes[mm])) + glue(x, e);
      return e;
    }
    return 0;
  });
  // audit: ratios, with a ratio word near ("ratio", "aspect", "odds", "scale", "mix"; otherwise as written):
  // (?<![\w.,:])(\d+):((?:[1-9]\d*|0)(?:\.\d+)?)\b(?![:,.]\d|:)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,:", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i || x.at(a) != ':' || !x.is_digit(a + 1)) return 0;
    size_t b = x.at(a + 1) == '0' ? a + 2 : x.digits_end(a + 1);
    if (x.at(b) == '.' && x.is_digit(b + 1)) b = x.digits_end(b + 1);
    if (!x.boundary(b) || x.at(b) == ':' || ((x.at(b) == ',' || x.at(b) == '.') && x.is_digit(b + 1))) return 0;
    if (x.in_link(i) || !any_of_words(words_near(x, i, b, true), {"ratio", "ratios", "aspect", "odds", "scale", "mix",
                                                                  "mixed", "mixture", "proportion", "proportions"}))
      return as_written(x, i, b, r);
    r = cardinal(x.s.substr(i, a - i)) + " to " + number(x.s.substr(a + 1, b - a - 1));
    return b;
  });
  // month abbreviations before a number: \b(Jan|...|Dec)\.(?=\s+\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> abbr[] = {
        {"Jan", "January"}, {"Feb", "February"}, {"Mar", "March"}, {"Apr", "April"}, {"Jun", "June"}, {"Jul", "July"},
        {"Aug", "August"}, {"Sep", "September"}, {"Sept", "September"}, {"Oct", "October"}, {"Nov", "November"},
        {"Dec", "December"}};
    if (!x.boundary(i)) return 0;
    for (const auto& [a, full] : abbr) {
      const size_t e = i + std::string_view(a).size();
      if (!x.starts(i, a) || x.at(e) != '.') continue;
      const size_t k = x.spaces_end(e + 1);
      if (k > e + 1 && x.is_digit(k)) { r = full; return e + 1; }
    }
    return 0;
  });
  // month-day dates: \b(January|...|December)\s+(3[01]|[12]\d|0?[1-9])(st|nd|rd|th)?\b(?![\d:])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    for (const char* month : kMonths) {
      const size_t m = std::string_view(month).size();
      if (!x.starts(i, month)) continue;
      const size_t d0 = x.spaces_end(i + m);
      if (d0 == i + m) continue;
      std::vector<size_t> days;
      if (x.at(d0) == '3' && (x.at(d0 + 1) == '0' || x.at(d0 + 1) == '1')) days.push_back(2);
      if ((x.at(d0) == '1' || x.at(d0) == '2') && x.is_digit(d0 + 1)) days.push_back(2);
      if (x.at(d0) == '0' && x.at(d0 + 1) >= '1' && x.at(d0 + 1) <= '9') days.push_back(2);
      if (x.at(d0) >= '1' && x.at(d0) <= '9') days.push_back(1);
      for (size_t d : days) {
        const size_t s = d0 + d;
        std::vector<size_t> ends;
        for (const char* suffix : {"st", "nd", "rd", "th"})
          if (x.starts(s, suffix)) { ends.push_back(s + 2); break; }
        ends.push_back(s);
        for (size_t e : ends)
          if (x.boundary(e) && !x.is_digit(e) && x.at(e) != ':') {
            r = std::string(month) + " " + ordinal(x.s.substr(d0, d));
            return e;
          }
      }
      return 0;  // the months are distinct words: no other alternative can match here
    }
    return 0;
  });
  // temperatures: (?<![\w.])([-−])?(\d+(?:\.\d+)?)\s*°(?:\s*([CF])\b)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t sign = minus_sign(x, i), n0 = i + sign;
    const size_t ne = decimal_end(x, n0);
    if (ne == n0) return 0;
    const size_t deg = x.spaces_end(ne);
    if (!x.starts(deg, "\xC2\xB0")) return 0;
    size_t e = deg + 2;
    const char* unit = "";
    const size_t u = x.spaces_end(e);
    if ((x.at(u) == 'C' || x.at(u) == 'F') && x.boundary(u + 1)) {
      unit = x.at(u) == 'C' ? " Celsius" : " Fahrenheit";
      e = u + 1;
    }
    const std::string_view value = x.s.substr(n0, ne - n0);
    r = std::string(sign ? "minus " : "") + number(value) + " degree" + (value == "1" ? "" : "s") + unit;
    return e;
  });
  // audit: a percent range, ascending, no space or a space on both sides: (?<![\w.,-])(\d+(?:\.\d+)?)( ?)[-–]\2(?=\d+(?:\.\d+)?\s*%)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,-", true)) return 0;
    const size_t a = decimal_end(x, i);
    if (a == i) return 0;
    const bool sp = x.at(a) == ' ';
    size_t j = a + (sp ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else return 0;
    if (sp) {
      if (x.at(j) != ' ') return 0;
      ++j;
    }
    const size_t b = decimal_end(x, j);
    if (b == j || x.at(x.spaces_end(b)) != '%') return 0;
    if (x.in_link(i) || !decimal_less(x.s.substr(i, a - i), x.s.substr(j, b - j))) return as_written(x, i, j, r);
    r = std::string(x.s.substr(i, a - i)) + " to ";
    return j;
  });
  // percentages: (?<![\w.])([-−])?(\d[\d,]*(?:\.\d+)?)\s*%
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t sign = minus_sign(x, i), n0 = i + sign;
    const size_t ne = amount_end(x, n0);
    if (ne == n0) return 0;
    const size_t p = x.spaces_end(ne);
    if (x.at(p) != '%') return 0;
    r = std::string(sign ? "minus " : "") + number(x.s.substr(n0, ne - n0)) + " percent";
    return p + 1;
  });
  // addresses: \b(\d{1,5}) ((?:[A-Z][a-z]+ )+)(St|Street|...)\b(\.)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> street[] = {
        {"St", "Street"}, {"Street", "Street"}, {"Ave", "Avenue"}, {"Avenue", "Avenue"}, {"Rd", "Road"},
        {"Road", "Road"}, {"Blvd", "Boulevard"}, {"Boulevard", "Boulevard"}, {"Dr", "Drive"}, {"Drive", "Drive"},
        {"Ln", "Lane"}, {"Lane", "Lane"}, {"Way", "Way"}, {"Ct", "Court"}, {"Court", "Court"}, {"Pl", "Place"},
        {"Place", "Place"}};
    if (!x.boundary(i)) return 0;
    const size_t de = x.digits_end(i);
    if (de == i || de - i > 5 || x.at(de) != ' ') return 0;
    std::vector<size_t> names;  // the end of each "Name " (after its space), greedy
    for (size_t j = de + 1; upper(x.at(j)) && lower(x.at(j + 1));) {
      size_t k = j + 1;
      while (lower(x.at(k))) ++k;
      if (x.at(k) != ' ') break;
      names.push_back(k + 1);
      j = k + 1;
    }
    for (size_t k = names.size(); k-- > 0;) {
      const size_t at = names[k];
      for (const auto& [key, full] : street) {
        const size_t e = at + std::string_view(key).size();
        if (!x.starts(at, key) || !x.boundary(e)) continue;
        const bool dot = x.at(e) == '.';
        const size_t end = e + (dot ? 1 : 0);
        r = pairs(small(x.s.substr(i, de - i))) + " " + std::string(x.s.substr(de + 1, at - de - 1)) + full +
            (dot && sentence_end(x, end, false) ? "." : "");
        return end;
      }
    }
    return 0;
  });
  // decades: \b(1\d|20)(\d0)s\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    const bool first = (x.at(i) == '1' && x.is_digit(i + 1)) || (x.at(i) == '2' && x.at(i + 1) == '0');
    if (!first || !x.is_digit(i + 2) || x.at(i + 3) != '0' || x.at(i + 4) != 's' || !x.boundary(i + 5)) return 0;
    r = plural_pairs(pairs(small(x.s.substr(i, 4))));
    return i + 5;
  });
  // audit: (?<![\w.])'?([1-9]0)s\b  ('90s, 80s)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t j = i + (x.at(i) == '\'' ? 1 : 0);
    if (!(x.at(j) >= '1' && x.at(j) <= '9') || x.at(j + 1) != '0' || x.at(j + 2) != 's' || !x.boundary(j + 3)) return 0;
    if (x.in_link(i)) return as_written(x, i, j + 3, r);
    r = plural_pairs(cardinal(long(x.at(j) - '0') * 10));
    return j + 3;
  });
  // numbers with units: (?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(units, longest first)\b(\.)?
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const char* const units[][3] = {
        {"mph", "mile per hour", "miles per hour"}, {"kph", "kilometer per hour", "kilometers per hour"},
        {"lbs", "pound", "pounds"}, {"kWh", "kilowatt hour", "kilowatt hours"}, {"hrs", "hour", "hours"},
        {"min", "minute", "minutes"}, {"sec", "second", "seconds"}, {"km", "kilometer", "kilometers"},
        {"kg", "kilogram", "kilograms"}, {"cm", "centimeter", "centimeters"}, {"mm", "millimeter", "millimeters"},
        {"mg", "milligram", "milligrams"}, {"ml", "milliliter", "milliliters"}, {"ft", "foot", "feet"},
        {"lb", "pound", "pounds"}, {"oz", "ounce", "ounces"}, {"mi", "mile", "miles"}, {"GB", "gigabyte", "gigabytes"},
        {"MB", "megabyte", "megabytes"}, {"hr", "hour", "hours"}};
    if (!not_after(x, i, ".", true)) return 0;
    const size_t ne = amount_end(x, i);
    if (ne == i) return 0;
    for (size_t at : {ne + (x.is_space(ne) ? 1 : 0), ne}) {
      for (const auto& u : units) {
        const size_t e = at + std::string_view(u[0]).size();
        if (!x.starts(at, u[0]) || !x.boundary(e)) continue;
        const std::string_view value = x.s.substr(i, ne - i);
        const bool dot = x.at(e) == '.';
        const size_t end = e + (dot ? 1 : 0);
        r = number(value) + " " + (value == "1" ? u[1] : u[2]) + (dot && sentence_end(x, end, true) ? "." : "");
        return end;
      }
      if (at == ne) break;
    }
    return 0;
  });
  // ordinals: \b(\d+)(st|nd|rd|th)\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i)) return 0;
    const size_t de = x.digits_end(i);
    if (de == i) return 0;
    for (const char* suffix : {"st", "nd", "rd", "th"})
      if (x.starts(de, suffix) && x.boundary(de + 2)) { r = ordinal(x.s.substr(i, de - i)); return de + 2; }
    return 0;
  });
  // audit: a range to a mixed number first: (?<![\w.,])(\d{1,2})( ?)[-–]\2(?=\d{1,2} \d+/(?:2|3|4|8|16)\b(?!/)) -> "\1 to "
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,", true)) return 0;
    const size_t a = run_of(x, i, 1, 2);
    if (!a) return 0;
    const bool sp = x.at(a) == ' ';
    size_t j = a + (sp ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else return 0;
    if (sp) {
      if (x.at(j) != ' ') return 0;
      ++j;
    }
    const size_t w = run_of(x, j, 1, 2);
    if (!w || x.at(w) != ' ') return 0;
    const size_t n = x.digits_end(w + 1);
    if (n == w + 1 || x.at(n) != '/') return 0;
    const size_t d = x.digits_end(n + 1);
    const std::string_view den = x.s.substr(n + 1, d - n - 1);
    if (!(den == "2" || den == "3" || den == "4" || den == "8" || den == "16") || !x.boundary(d) || x.at(d) == '/') return 0;
    if (x.in_link(i)) return as_written(x, i, j, r);
    r = std::string(x.s.substr(i, a - i)) + " to ";
    return j;
  });
  // audit: mixed numbers: (?<![\w.,\-–−])(\d{1,2}) (\d+)/(\d+)\b(?!/), a proper fraction in halves, thirds, quarters,
  // eighths or sixteenths ("2 1/2"; not "the 300 2/3 voted", "Week 3 4/5", "-1 1/2")
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,-", true)) return 0;
    if (i >= 3 && (x.s.substr(i - 3, 3) == "\xE2\x80\x93" || x.s.substr(i - 3, 3) == "\xE2\x88\x92")) return 0;
    const size_t w = run_of(x, i, 1, 2);
    if (!w || x.at(w) != ' ') return 0;
    const size_t a = x.digits_end(w + 1);
    if (a == w + 1 || x.at(a) != '/') return 0;
    const size_t b = x.digits_end(a + 1);
    if (b == a + 1 || !x.boundary(b) || x.at(b) == '/') return 0;
    const std::string_view n = x.s.substr(w + 1, a - w - 1), d = x.s.substr(a + 1, b - a - 1);
    const bool denominator = equals(d, 2) || equals(d, 3) || equals(d, 4) || equals(d, 8) || equals(d, 16);
    if (x.in_link(i) || is_zero(n) || compare_numbers(n, d) >= 0 || !denominator) return as_written(x, i, b, r);
    std::string part;
    if (equals(n, 1)) {
      part = equals(d, 2) ? "half" : equals(d, 4) ? "quarter" : ordinal(d);
      part = (std::string_view("aeiou").find(part[0]) != std::string_view::npos ? "an " : "a ") + part;
    } else {
      part = fraction_words(n, d);
    }
    r = cardinal(x.s.substr(i, w - i)) + " and " + part;
    return b;
  });
  // fractions: (?<![\w/.])(\d+)/(\d+)\b(?!/) ; audit: 24/7, 50/50, 20/20 vision, 4/4 time as said; n/n "out of" after a
  // score word; n/1 as odds after "odds"; in a link or in arithmetic as before
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, "/.", true)) return 0;
    const size_t a = x.digits_end(i);
    if (a == i || x.at(a) != '/') return 0;
    const size_t b = x.digits_end(a + 1);
    if (b == a + 1 || !x.boundary(b) || x.at(b) == '/') return 0;
    const std::string_view n = x.s.substr(i, a - i), d = x.s.substr(a + 1, b - a - 1);
    // red-team round 3: in a link or in arithmetic (= * × + ( ) next to it) as before the audit
    // = * × + next to it, or a parenthesis with one of them beyond it ("(3/3)*4")
    auto op_at = [&](size_t p) { return std::string_view("=*+").find(x.at(p)) != std::string_view::npos || x.starts(p, "\xC3\x97"); };
    auto back = [&](size_t p) { while (p > 0 && space(x.s[p - 1])) --p; return p; };
    size_t q = back(i);
    if (q > 0 && x.s[q - 1] == '(') q = back(q - 1);
    size_t f = x.spaces_end(b);
    if (x.at(f) == ')') f = x.spaces_end(f + 1);
    const bool math = (q > 0 && (std::string_view("=*+").find(x.s[q - 1]) != std::string_view::npos ||
                                 (q >= 2 && x.s.substr(q - 2, 2) == "\xC3\x97"))) || op_at(f);
    if (x.in_link(i) || math) { r = fraction_words(n, d); return b; }
    const auto near = words_near(x, i, b, true);
    if ((equals(n, 24) && equals(d, 7)) || (equals(n, 50) && equals(d, 50)) ||
        (equals(n, 20) && equals(d, 20) && any_of_words(near, {"vision"})) ||
        (equals(n, 4) && equals(d, 4) &&
         any_of_words(near, {"time", "signature", "rhythm", "meter", "metre", "beat", "beats", "measure", "groove", "tempo"})))
      r = cardinal(n) + " " + cardinal(d);  // 24/7, 50/50, 20/20 vision, 4/4 time
    else if (compare_numbers(n, d) == 0 && compare_numbers(d, "1") > 0 &&
             any_of_words(near, {"rated", "rating", "ratings", "score", "scored", "scores", "star", "stars", "grade",
                                 "graded", "review", "reviews", "reviewed"}))
      r = cardinal(n) + " out of " + cardinal(d);  // "rated 10/10", "5/5 stars"
    else if (equals(d, 1) && compare_numbers(n, "1") > 0 && any_of_words(near, {"odds"}))
      r = cardinal(n) + " to one";  // "odds of 9/1" (not "Starting 9/1": a date)
    else r = fraction_words(n, d);
    return b;
  });
  // audit: dimensions: (?<![\w.,])(?!0x)N[x×]N(?:[x×]N)?(?:(s)\b|\b|(?=[A-Za-z]))(?!,\d), N = (\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)
  // (4x4, 2,560×1,440, 4x4x4, 4x4s; not hex: 0x10)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,*=+", true) || (x.at(i) == '0' && x.at(i + 1) == 'x')) return 0;
    auto ends = [&](size_t p) {  // N's ends at p, in the regex's backtracking order
      std::vector<size_t> out;
      const size_t de = x.digits_end(p);
      if (de == p) return out;
      if (de - p <= 3) {
        std::vector<size_t> chain;
        for (size_t e = de; x.at(e) == ',' && x.is_digit(e + 1) && x.is_digit(e + 2) && x.is_digit(e + 3);) chain.push_back(e += 4);
        for (size_t k = chain.size(); k-- > 0;) out.push_back(chain[k]);
      }
      const size_t dec = decimal_end(x, p);
      out.push_back(dec);
      if (dec != de) out.push_back(de);
      return out;
    };
    auto times = [&](size_t p) -> size_t {  // [x×] at p: its length
      if (x.at(p) == 'x') return 1;
      return x.starts(p, "\xC3\x97") ? 2 : 0;
    };
    auto free_after = [&](size_t e) {  // (?!,\d)(?![*=+(]): not a grouped number, not arithmetic ("24×7*3", "3x10(8)")
      return !(x.at(e) == ',' && x.is_digit(e + 1)) && std::string_view("*=+(").find(x.at(e)) == std::string_view::npos;
    };
    auto tail = [&](size_t e, bool& plural) -> size_t {  // (?:(s)\b|\b|(?=[A-Za-z]))(?!,\d)(?![*=+(]) at e: the end, or 0
      if (x.at(e) == 's' && x.boundary(e + 1) && free_after(e + 1)) { plural = true; return e + 1; }
      plural = false;
      if (x.boundary(e) && free_after(e)) return e;
      return alpha(x.at(e)) ? e : 0;
    };
    for (size_t e1 : ends(i)) {
      const size_t t1 = times(e1);
      if (!t1) continue;
      const size_t j1 = e1 + t1;
      for (size_t e2 : ends(j1)) {
        std::vector<std::pair<size_t, size_t>> parts{{i, e1}, {j1, e2}};
        bool plural = false;
        size_t end = 0;
        if (const size_t t2 = times(e2)) {
          for (size_t e3 : ends(e2 + t2))
            if ((end = tail(e3, plural))) { parts.push_back({e2 + t2, e3}); break; }
        }
        if (!end) end = tail(e2, plural);
        if (!end) continue;
        if (x.in_link(i)) return as_written(x, i, end, r);
        r.clear();
        for (size_t k = 0; k < parts.size(); ++k) {
          std::string w = number(x.s.substr(parts[k].first, parts[k].second - parts[k].first));
          if (k + 1 == parts.size() && plural) w = plural_words(w);
          r += (k ? " by " : "") + w;
        }
        r += glue(x, end, false);  // "6.3cm" -> "six point three cm"
        return end;
      }
    }
    return 0;
  });
  // ranges: (?<![\w.,-])(\d+)\s?[-–]\s?(\d+)\b(?![-\d])
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,-", true)) return 0;
    // audit: decimal ends, (\d+(?:\.\d+)?) each (the first given back when a dash does not follow it)
    for (const size_t a : {decimal_end(x, i), x.digits_end(i)}) {
    if (a == i) return 0;
    size_t j = a + (x.is_space(a) ? 1 : 0);
    if (x.at(j) == '-') j += 1;
    else if (x.starts(j, "\xE2\x80\x93")) j += 3;
    else continue;
    j += x.is_space(j) ? 1 : 0;
    const size_t bd = decimal_end(x, j), bi = x.digits_end(j);
    for (const size_t b : {bd, bi}) {
    if (b == j || !x.boundary(b) || x.at(b) == '-' || x.is_digit(b)) continue;
    if (x.at(b) == '.' && x.is_digit(b + 1)) continue;  // audit: (?!\.\d), not inside a decimal
    const std::string_view p = x.s.substr(i, a - i), q = x.s.substr(j, b - j);
    if (equation_after(x, b)) {  // audit: (?:(\s*=\s*)(?=[-−]?\d))?, a subtraction: "20 - 1 = 19", "3-1=2"
      r = number(p) + " minus " + number(q) + " equals ";
      return x.spaces_end(x.spaces_end(b) + 1);
    }
    if (p.find('.') != std::string_view::npos || q.find('.') != std::string_view::npos) {
      // audit, a decimal end ("1-1.25", "CI 1.2-3.4"): ascending, the same spacing on both sides, no year end; else as
      // before the audit ("1.5 -5", "610.12-1990", "5.0-0")
      auto year = [](std::string_view v) { return v.find('.') == std::string_view::npos && v.size() == 4 && in_range(v, 1000, 2099); };
      const bool before_dash = x.is_space(a);  // (\s?) on each side of the dash: both or neither
      const size_t dash = a + (before_dash ? 1 : 0), dash_len = x.at(dash) == '-' ? 1 : 3;
      const bool after_dash = j > dash + dash_len;
      if (before_dash != after_dash || year(p) || year(q) || !decimal_less(p, q)) return as_written(x, i, b, r);
      r = number(p) + " to " + number(q);
      return b;
    }
    const bool years = in_range(p, 1000, 2099) && in_range(q, 1000, 2099);
    r = (years ? pairs(small(strip_zeros(p))) : cardinal(p)) + " to " + (years ? pairs(small(strip_zeros(q))) : cardinal(q));
    return b;
    }
    }
    return 0;
  });
  // "#12": #\s?(\d+)\b ; "No. 14": \bNo\.\s?(\d+)\b
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (x.at(i) != '#') return 0;
    const size_t j = i + 1 + (x.is_space(i + 1) ? 1 : 0), e = x.digits_end(j);
    if (e == j || !x.boundary(e)) return 0;
    r = "number " + cardinal(x.s.substr(j, e - j));
    return e;
  });
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!x.boundary(i) || !x.starts(i, "No.")) return 0;
    const size_t j = i + 3 + (x.is_space(i + 3) ? 1 : 0), e = x.digits_end(j);
    if (e == j || !x.boundary(e)) return 0;
    r = "number " + cardinal(x.s.substr(j, e - j));
    return e;
  });
  // audit, in an equation (no space or a space on both sides): (?<=\d)( ?)[-−–]\1(?=\d[\d,.]*\s*=\s*[-−]?\d) -> " minus " ;
  // (?<=\d)( ?)/\1(?=...) -> " divided by " ; then "=" between digits: (?<=\d)=(?=[-−]?\d) -> " equals "
  for (const bool divide : {false, true}) {
    t = pass(t, [divide](const Text& x, size_t i, std::string& r) -> size_t {
      if (i == 0 || !digit(x.s[i - 1])) return 0;
      const bool sp = x.at(i) == ' ';
      const size_t o = i + (sp ? 1 : 0);
      size_t m = divide ? (x.at(o) == '/' ? 1 : 0) : minus_sign(x, o);
      if (!divide && !m && x.starts(o, "\xE2\x80\x93")) m = 3;  // an en dash too: "5 – 4 = 1"
      if (!m) return 0;
      size_t j = o + m;
      if (sp) {
        if (x.at(j) != ' ') return 0;
        ++j;
      }
      if (!x.is_digit(j)) return 0;
      const size_t start = j;
      while (x.is_digit(j) || x.at(j) == ',' || x.at(j) == '.') ++j;
      if (!equation_after(x, j)) return 0;
      if (x.in_link(i)) return as_written(x, i, start, r);
      r = divide ? " divided by " : " minus ";
      return start;
    });
  }
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (i == 0 || !digit(x.s[i - 1]) || x.at(i) != '=' || !x.is_digit(i + 1 + minus_sign(x, i + 1))) return 0;
    if (x.in_link(i)) return as_written(x, i, i + 1, r);
    r = " equals ";
    return i + 1;
  });
  // symbols between whitespace, one pass each: (?<=\s)=(?=\s), + , ×, &
  for (const auto& [sym, name] : {std::pair<const char*, const char*>{"=", "equals"}, {"+", "plus"},
                                  {"\xC3\x97", "times"}, {"&", "and"}}) {
    const std::string_view symbol(sym), word_(name);
    t = pass(t, [&](const Text& x, size_t i, std::string& r) -> size_t {
      if (i == 0 || !space(x.s[i - 1]) || !x.starts(i, symbol) || !x.is_space(i + symbol.size())) return 0;
      r = word_;
      return i + symbol.size();
    });
  }
  // negative numbers: (?<![\w.])[-−](?=\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t s = minus_sign(x, i);
    if (!s || !x.is_digit(i + s)) return 0;
    r = "minus ";
    return i + s;
  });
  // decimals: (?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\.(\d+)(?![\d_])   (audit: a grouped whole part; letters may follow)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true)) return 0;
    const size_t de = x.digits_end(i);
    if (de == i) return 0;
    std::vector<size_t> wholes;  // the alternation's ends, in its order
    if (de - i <= 3) {
      const size_t e = x.chain_end(de);
      if (e > de) wholes.push_back(e);
    }
    wholes.push_back(de);
    for (size_t a : wholes) {
      if (x.at(a) != '.') continue;
      const size_t b = x.digits_end(a + 1);
      if (b == a + 1 || x.at(b) == '_') continue;
      r = number(x.s.substr(i, b - i)) + glue(x, b, false);
      return b;
    }
    return 0;
  });

  // audit: plurals of numbers: (?<![\w.,'])(\d+)s\b  (747s, 6s, under-21s, 100s; not zero-led or long digit strings)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".,'", true)) return 0;
    const size_t e = x.digits_end(i);
    if (e == i || x.at(e) != 's' || !x.boundary(e + 1)) return 0;
    const std::string_view d = x.s.substr(i, e - i);
    if ((d.size() > 1 && d[0] == '0') || d.size() >= 7 || d == "911" || x.in_link(i)) return as_written(x, i, e + 1, r);
    std::string w = cardinal(d);
    if ((d == "100" || d == "1000" || d == "1000000") && word_before(x, i) != "the" && !(i > 0 && x.s[i - 1] == '-'))
      w = w.substr(4);  // "100s of millions": "hundreds"; "the 100s" (a class, a century): "the one hundreds"
    r = plural_words(w);
    return e + 1;
  });

  // integers: (?<![\w.])\d{1,3}(?:,\d{3})+(?![\d,]*\d) | (?<![\w.,])\d+(?![\d,]*\d)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    // (?![\d,]*\d) at e: the run of digits and commas from e holds no digit
    auto clear = [&](size_t e) { while (x.at(e) == ',') ++e; return !x.is_digit(e); };
    size_t end = 0;
    const size_t de = x.digits_end(i);
    if (de == i) return 0;
    if (not_after(x, i, ".", true) && de - i <= 3) {
      size_t e = de, groups = 0;
      while (x.at(e) == ',' && x.digits_end(e + 1) >= e + 4) e += 4, ++groups;
      if (groups && clear(e)) end = e;
    }
    if (!end && not_after(x, i, ".,", true) && clear(de)) end = de;
    if (!end) return 0;
    const std::string_view s = x.s.substr(i, end - i);
    if (s.find(',') != std::string_view::npos) { r = cardinal(without_commas(s)) + glue(x, end); return end; }
    if (s.size() > 1 && s[0] == '0') { r = spell_digits(s) + glue(x, end); return end; }
    if (s.size() >= 7) {  // digit by digit, but a number in arithmetic ("2000000 minus 400000") whole
      auto op = [](const std::string& w) { return w == "minus" || w == "plus" || w == "equals" || w == "times"; };
      std::string after;
      if (x.at(end) == ' ') {
        size_t j = end;
        while (x.at(j) == ' ') ++j;
        size_t k = j;
        while (lower(x.at(k))) ++k;
        after = std::string(x.s.substr(j, k - j));  // ( +([a-z]+)): the lowercase run
      }
      r = (word_before(x, i) == "equals" || op(after) ? cardinal(s) : spell_digits(s)) + glue(x, end);
      return end;
    }
    const auto window = x.s.substr(back_code_points(x.s, i, 30), i - back_code_points(x.s, i, 30));
    const auto words = letter_runs(window);
    bool zip = false;
    for (size_t k = words.size() > 3 ? words.size() - 3 : 0; k < words.size(); ++k) {
      const auto w = ascii_lower(words[k]);
      zip = zip || w == "zip" || w == "postal";
    }
    if (s == "911" || (s.size() == 5 && zip)) { r = spell_digits(s) + glue(x, end); return end; }
    const long n = small(s);
    if (s.size() == 4 && n >= 1000 && n <= 2099) {
      static const char* const cues[] = {"in", "since", "by", "from", "until", "till", "of", "during", "year", "circa",
                                         "before", "after", "around", "early", "late", "mid", "spring", "summer",
                                         "fall", "autumn", "winter", "through", "to"};
      const size_t before_end = python_rstrip(x.s, i);
      const std::string_view before = x.s.substr(0, before_end);
      bool operand = !before.empty() && (before.back() == '+' || before.back() == '*' ||
                                         (before.size() >= 2 && before.substr(before.size() - 2) == "\xC3\x97"));
      if (!operand && !before.empty() && before.back() == '=') {  // "=" after a digit or a one-letter variable
        const std::string_view p = before.substr(0, python_rstrip(before, before.size() - 1));
        if (!p.empty() && digit(p.back())) operand = true;
        else if (!p.empty() && alpha(p.back()))
          operand = p.size() == 1 || !(alpha(p[p.size() - 2]) || digit(p[p.size() - 2]));
      }
      if (operand) {
        r = cardinal(s) + glue(x, end);  // "3.5x=1050.", ".20*1800": an operand, not a year
        return end;
      }
      const size_t b20 = back_code_points(before, before.size(), 20);
      const auto prev = letter_runs(before.substr(b20));
      bool year = false;
      if (!prev.empty()) {
        const auto w = ascii_lower(prev.back());
        for (const char* c : cues) year = year || w == c;
      }
      if (!year) {  // (January|...|December)(?: [\w-]+)?,?$ at the end of `before` (audit: "thirty-first,", "May")
        auto ends_with_month = [](std::string_view b) {
          for (const char* month : kMonths) {
            const std::string_view m(month);
            if (b.size() >= m.size() && b.substr(b.size() - m.size()) == m) return true;
          }
          return false;
        };
        std::string_view b = before;
        for (int comma = 0; comma < 2 && !year; ++comma) {
          if (comma) { if (b.empty() || b.back() != ',') break; b.remove_suffix(1); }
          if (ends_with_month(b)) {  // a bare month: not when a lowercase word follows ("In March 1234 people")
            size_t k = end;
            while (x.at(k) == ' ' && k - end < 64) ++k;
            if (!(k > end && k - end < 64 && lower(x.at(k)))) year = true;
            break;
          }
          const size_t w = x.hyphen_run_start(b.size());  // b is a prefix of the text
          if (w == b.size() || w == 0 || b[w - 1] != ' ') continue;
          year = ends_with_month(b.substr(0, w - 1));
        }
      }
      if (!year) {  // re.match(r"\s*[.,;:!?]|\s*$", t[end:end+2]) and no digit in before[-3:]
        size_t stop = end;  // two code points after the number
        for (int k = 0; k < 2 && stop < x.n(); ++k) {
          ++stop;
          while (stop < x.n() && cont(x.s[stop])) ++stop;
        }
        const std::string_view after = x.s.substr(end, stop - end);
        bool punct = false;
        {
          size_t q = 0;
          while (q < after.size() && space(after[q])) ++q;
          punct = q < after.size() && std::string_view(".,;:!?").find(after[q]) != std::string_view::npos;
          if (!punct) punct = q == after.size();
        }
        bool digit_before = false;
        for (char c : before.substr(back_code_points(before, before.size(), 3))) digit_before = digit_before || digit(c);
        year = punct && !digit_before;
      }
      if (year) { r = pairs(n) + glue(x, end); return end; }
    }
    r = cardinal(s) + glue(x, end);
    return end;
  });

  // titles and abbreviations: (?<![\w.])(keys, longest first)\.(?=[\s,;:!?)]|$)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    static const std::pair<const char*, const char*> titles[] = {
        {"vs", "versus"}, {"Mr", "Mister"}, {"Mrs", "Missus"}, {"Ms", "Miz"}, {"Dr", "Doctor"}, {"Prof", "Professor"},
        {"Mt", "Mount"}, {"Jr", "Junior"}, {"Sr", "Senior"}, {"Gov", "Governor"}, {"Sen", "Senator"},
        {"Rep", "Representative"}, {"Gen", "General"}, {"Capt", "Captain"}, {"Lt", "Lieutenant"}, {"Sgt", "Sergeant"}};
    static const std::pair<const char*, const char*> others[] = {
        {"etc", "et cetera"}, {"e.g", "for example"}, {"i.e", "that is"}, {"approx", "approximately"},
        {"Apt", "Apartment"}, {"Dept", "Department"}, {"Inc", "Incorporated"}, {"Corp", "Corporation"},
        {"Co", "Company"}, {"Ave", "Avenue"}, {"Blvd", "Boulevard"}, {"Rd", "Road"}, {"est", "established"},
        {"min", "minutes"}, {"max", "maximum"}, {"govt", "government"}, {"St", ""}};
    if (!not_after(x, i, ".", true)) return 0;
    // the longest key that matches with its period and look-ahead (keys of equal length cannot both match)
    const char* best = nullptr;
    const char* expansion = nullptr;
    bool title = false;
    auto consider = [&](const char* key, const char* full, bool is_title) {
      const size_t e = i + std::string_view(key).size();
      if (!x.starts(i, key) || x.at(e) != '.') return;
      const size_t f = e + 1;
      const bool ok = f == x.n() || x.is_space(f) || std::string_view(",;:!?)").find(x.at(f)) != std::string_view::npos;
      if (ok && (!best || std::string_view(key).size() > std::string_view(best).size()))
        best = key, expansion = full, title = is_title;
    };
    for (const auto& [k, v] : titles) consider(k, v, true);
    for (const auto& [k, v] : others) consider(k, v, false);
    if (!best) return 0;
    const std::string_view key(best);
    const size_t end = i + key.size() + 1;
    auto name_follows = [&] { const size_t j = x.spaces_end(end); return j > end && upper(x.at(j)); };
    if (key == "St") r = name_follows() ? "Saint" : std::string("Street") + (sentence_end(x, end, true) ? "." : "");
    else if (key == "Dr" && !name_follows()) r = std::string("Drive") + (sentence_end(x, end, true) ? "." : "");
    else if (title) r = expansion;
    else r = std::string(expansion) + (sentence_end(x, end, true) ? "." : "");
    return end;
  });
  // (?<![\w.])vs(?=\s)
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    if (!not_after(x, i, ".", true) || !x.starts(i, "vs") || !x.is_space(i + 2)) return 0;
    r = "versus";
    return i + 2;
  });
  // [ \t]{2,} -> " "
  t = pass(t, [](const Text& x, size_t i, std::string& r) -> size_t {
    size_t j = i;
    while (x.at(j) == ' ' || x.at(j) == '\t') ++j;
    if (j - i < 2) return 0;
    r = " ";
    return j;
  });
  return t;
}

}  // namespace lfm::speech_normalize
