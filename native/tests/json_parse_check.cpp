// lfm::Json::parse (bulk string runs, moved containers; 2026-11-17) against the frozen reference parser
// (json_reference_v1.hpp): 1.2 M random and edge-case documents, a 13 MB image body and long strings with escapes at
// random places; every value (strings byte for byte, numbers exactly with their integral flag) and every error
// message identical. Prints both parsers' time on the 13 MB body.
#include "lfm/model_config.hpp"
#include "json_reference_v1.hpp"
#include <chrono>
#include <cstdio>
#include <random>
template <class J> std::string show(const J& j) {  // canonical dump, exact for strings and doubles
  std::string o;
  std::visit([&](const auto& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, std::nullptr_t>) o = "null";
    else if constexpr (std::is_same_v<T, bool>) o = v ? "true" : "false";
    else if constexpr (std::is_same_v<T, double>) { char b[64]; std::snprintf(b, 64, "%a/%d", v, (int)j.integral); o = b; }
    else if constexpr (std::is_same_v<T, std::string>) { o = "s" + std::to_string(v.size()) + ":" + v; }
    else if constexpr (std::is_same_v<T, typename J::Array>) { o = "["; for (auto& e : v) o += show(e) + ","; o += "]"; }
    else { o = "{"; for (auto& [k, e] : v) o += "s" + std::to_string(k.size()) + ":" + k + "=" + show(e) + ","; o += "}"; }
  }, j.value);
  return o;
}
template <class J> std::string run(const std::string& t) {
  try { return "OK " + show(J::parse(t)); } catch (const std::exception& e) { return std::string("ERR ") + e.what(); }
}
int main() {
  std::mt19937 g(20261117);
  const std::vector<std::string> atoms = {"\"", "\\", "\\\"", "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t", "\\u0041",
    "\\u00e9", "\\u20ac", "\\ud83d\\ude00", "\\ud83d", "\\ude00", "\\ud83dx", "\\u12", "\\x", "a", "abc", "\x01", "\x1f",
    "\x7f", "\xc3\xa9", "\xff", " ", "{", "}", "[", "]", ":", ",", "0", "-1", "1.5e3", "true", "null", "false", "\n", "\t", ""};
  long n = 0, bad = 0;
  auto check = [&](const std::string& t) {
    ++n;
    const auto a = run<lfm_old::Json>(t), b = run<lfm::Json>(t);
    if (a != b) { if (++bad <= 5) std::printf("DIFFER on %zu bytes: old %.80s | new %.80s\n", t.size(), a.c_str(), b.c_str()); }
  };
  // strings built from atoms, alone and inside structures
  for (int i = 0; i < 300000; ++i) {
    std::string inner; int k = g() % 12;
    for (int j = 0; j < k; ++j) inner += atoms[g() % atoms.size()];
    check("\"" + inner + "\"");
    check("{\"k\":\"" + inner + "\",\"a\":[1,\"" + inner + "\",{\"x\":null}]}");
    check(inner);  // raw fragments: mostly errors, whose messages must match too
    std::string t = "{\"m\":[{\"c\":\"" + inner + "\"}]}";
    if (!t.empty()) { t.resize(g() % (t.size() + 1)); check(t); }  // truncations
  }
  // big strings: a 13 MB base64 part, and long runs with escapes/specials at random places
  std::string b64(13'000'000, 'A');
  for (auto& c : b64) c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[g() % 64];
  check("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"image\":\"data:image/png;base64," + b64 + "\"}]}]}");
  for (int i = 0; i < 200; ++i) {
    std::string s(1 + g() % 200000, 'x');
    for (int j = 0; j < 20; ++j) { auto a = atoms[g() % atoms.size()]; s.insert(g() % s.size(), a); }
    check("[\"" + s + "\"]");
  }
  std::printf("%ld cases, %ld differ\n", n, bad);
  // speed on the 13 MB body
  std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"image\":\"data:image/png;base64," + b64 + "\"}]}]}";
  auto t = [&](auto f) { auto a = std::chrono::steady_clock::now(); f(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };
  for (int r = 0; r < 3; ++r) std::printf("13 MB: old %.1f ms, new %.1f ms\n", t([&] { lfm_old::Json::parse(body); }), t([&] { lfm::Json::parse(body); }));
  std::printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
