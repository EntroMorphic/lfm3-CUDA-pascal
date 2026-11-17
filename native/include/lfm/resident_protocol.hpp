#pragma once
// Experimental, CPU-only framing/admission. No CUDA or model math here.
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lfm::resident {
constexpr size_t max_line = 65536;
constexpr size_t max_tokens = 32768;
struct Startup {
  bool stream_on = true;
  std::vector<char*> native_argv;
};
// Intercept worker-only settings before the native parser/model initialization.
inline Startup startup(int argc, char** argv) {
  Startup result;
  bool seen = false;
  if (argc > 0) result.native_argv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) != "--worker-stream") {
      result.native_argv.push_back(argv[i]);
      continue;
    }
    if (seen || i + 1 >= argc) throw std::runtime_error("worker_stream_mode");
    seen = true;
    const std::string_view mode(argv[++i]);
    if (mode != "on" && mode != "off") throw std::runtime_error("worker_stream_mode");
    result.stream_on = mode == "on";
  }
  return result;
}
struct Request {
  uint64_t id = 0, seed = 0;
  float temperature = 0;
  int output = 0;
  std::vector<int> tokens;
};
inline uint64_t integer(std::string_view text) {
  uint64_t value = 0;
  if (text.empty() || (text.size() > 1 && text.front() == '0'))
    throw std::runtime_error("integer");
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    throw std::runtime_error("integer");
  return value;
}
inline std::vector<std::string_view> fields(const std::string& line) {
  if (line.empty() || line.size() > max_line) throw std::runtime_error("line");
  std::vector<std::string_view> result;
  size_t start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i < line.size() && (line[i] < 32 || line[i] > 126))
      throw std::runtime_error("ascii");
    if (i == line.size() || line[i] == ' ') {
      if (start == i) throw std::runtime_error("spacing");
      result.emplace_back(line.data() + start, i - start);
      start = i + 1;
    }
  }
  return result;
}
inline Request parse(const std::string& line, int context, int vocabulary,
                     int output_cap = 0) {
  const auto f = fields(line);
  if (f.size() < 8 || f[0] != "V1" || f[1] != "RUN")
    throw std::runtime_error("grammar");
  Request r;
  r.id = integer(f[2]); r.seed = integer(f[3]);
  // Decimal only: reject exponent, locale punctuation, NaN, Inf, signs.
  const std::string temperature(f[4]);
  bool dot = false, digit = false;
  for (char c : temperature) {
    if (c == '.' && !dot) dot = true;
    else if (c >= '0' && c <= '9') digit = true;
    else throw std::runtime_error("temperature");
  }
  if (!digit) throw std::runtime_error("temperature");
  r.temperature = std::strtof(temperature.c_str(), nullptr);
  if (!std::isfinite(r.temperature)) throw std::runtime_error("temperature");
  const auto output = integer(f[5]), count = integer(f[6]);
  if (!r.id || context <= 1 || vocabulary <= 0 || count == 0 ||
      count > max_tokens || count >= static_cast<uint64_t>(context) ||
      f.size() != 7 + count || output > static_cast<uint64_t>(context) - count ||
      (output_cap > 0 && (output == 0 || output > static_cast<uint64_t>(output_cap))))
    throw std::runtime_error("bounds");
  r.output = static_cast<int>(output);
  for (size_t i = 7; i < f.size(); ++i) {
    const auto token = integer(f[i]);
    if (token >= static_cast<uint64_t>(vocabulary)) throw std::runtime_error("token");
    r.tokens.push_back(static_cast<int>(token));
  }
  return r;
}
// Reads at most 64 KiB into memory, drains an oversized record. Unterminated
// records fail closed, including EOF following otherwise valid RUN bytes.
inline bool read_line(std::istream& input, std::string& line) {
  line.clear(); bool oversized = false; char c;
  while (input.get(c)) {
    if (c == '\n') {
      if (oversized) line.clear();
      return true;
    }
    if (line.size() < max_line) line += c;
    else oversized = true;
  }
  if (!line.empty() || oversized) throw std::runtime_error("truncated_record");
  return false;
}
// One reader, one active request. No user-space request queue. Clients MUST
// await END before sending another record; concurrent admission is unsupported.
// Runner(request, token_callback, completion_callback) returns a fixed end reason.
// Completion borrows the native generated vector: no protocol token buffer.
template<class Runner>
void serve(std::istream& input, std::ostream& output, int context, int vocabulary,
           int output_cap, Runner&& run, bool stream_on = true) {
  output << "V1 EXPERIMENTAL B1 SEQUENTIAL CANCEL_UNSUPPORTED LOADS 1 STREAM "
         << (stream_on ? "on" : "off") << '\n' << std::flush;
  uint64_t last_id = 0;
  std::string line;
  while (read_line(input, line)) {
    if (line == "V1 QUIT") { output << "V1 BYE\n" << std::flush; return; }
    if (line == "V1 CANCEL") {
      output << "V1 ERROR 0 CANCEL_UNSUPPORTED\n" << std::flush; continue;
    }
    Request r;
    try {
      r = parse(line, context, vocabulary, output_cap);
      if (r.id <= last_id) throw std::runtime_error("replayed_id");
    } catch (const std::exception&) {
      output << "V1 ERROR 0 ADMISSION\n" << std::flush; continue;
    }
    last_id = r.id;
    output << "V1 BEGIN " << r.id << '\n' << std::flush;
    size_t count = 0;
    bool completed = false;
    const size_t budget = r.output ? static_cast<size_t>(r.output) :
        static_cast<size_t>(context) - r.tokens.size();
    // Runtime errors are fatal: do not reuse potentially poisoned CUDA state.
    const auto reason = run(r, [&](int token) {
      if (completed || count >= budget || token < 0 || token >= vocabulary)
        throw std::runtime_error("token_callback_bounds");
      if (stream_on)
        output << "V1 TOKEN " << r.id << ' ' << count << ' ' << token << '\n' << std::flush;
      ++count;
      if (!output) throw std::runtime_error("output_closed");
    }, [&](const std::vector<int>& tokens) {
      if (completed || tokens.size() != count || tokens.size() > budget)
        throw std::runtime_error("completion_bounds");
      for (int token : tokens)
        if (token < 0 || token >= vocabulary) throw std::runtime_error("completion_token");
      completed = true;
      if (!stream_on) {
        output << "V1 TOKENS " << r.id << ' ' << tokens.size();
        for (int token : tokens) output << ' ' << token;
        output << '\n' << std::flush;
        if (!output) throw std::runtime_error("output_closed");
      }
    });
    if (!completed) throw std::runtime_error("missing_completion");
    output << "V1 END " << r.id << ' ' << count << ' ' << reason << '\n' << std::flush;
    if (!output) throw std::runtime_error("output_closed");
  }
}
}  // namespace lfm::resident
