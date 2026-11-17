#pragma once
// Loopback HTTP text serving for liquid-lfm3-server: a native port of tools/lfm3_server.py (Service,
// LocalHTTPServer, Handler) and the request rules it takes from tools/lfm3_chat.py and
// tools/lfm3_worker_client.py. Same endpoints, statuses, error codes, admission and lease rules, and
// response bodies byte for byte (Python json.dumps(ensure_ascii=False, separators=(",", ":"))).
//
// Host-only: the engine is behind Backend (one resident-protocol line in, one reply line out, called
// only from the engine-owner thread), so this layer is tested on any host against a fake engine.
//
// Deliberate differences from the Python server, each at a boundary no conforming client reaches
// (tests/test_lfm3_http_framing.py and tools/lfm3_http_diff.py measure the rest as identical):
//  - Number literals that underflow a double (|x| < 2.2e-308, e.g. 1e-400) are rejected; Python reads
//    them as 0.0 or a subnormal. lfm::Json's existing rule.
//  - Where http.server answers with an HTML error page, or with a body and no status line (HTTP/0.9
//    style, e.g. two-word request lines and HTTP/2.0), the native server sends a status line and a JSON
//    error. Statuses are the same except that HTTP/2.0 gets 505 with a status line. Unsupported methods
//    get 501 as JSON.
//  - A frontend failure that is not a validation error (none is known) answers 400 instead of
//    dropping the connection.
//  - Text is decoded by the engine, outside the frontend lock, so a request arriving during another
//    request's decode is not refused with 429 frontend_busy.
//  - Shutdown requests cancellation of running generations at their next token boundary instead of
//    waiting for them to finish; a second SIGTERM/SIGINT exits at once.
//  - After a fatal engine error, including an engine call longer than engine_timeout, the process
//    keeps answering 503 as the Python server does, but keeps its GPU allocations until it exits (the
//    engine is in-process; there is no worker to kill).
//  - Text generation is sliced (validation/text-wait-v1): ops text_begin and text_step, interleaved one
//    token per speech step while speech rows run, so text no longer waits for the speech batch to
//    drain, and session operations run between speech steps. Every reply is the same as alone. Three
//    behaviors differ from before:
//    * a text generation that passes its deadline while running is cancelled at its next slice (the
//      client's 504 is unchanged; its session's cache is reset instead of kept);
//    * an active text generation has left the queue, as an admitted speech row has, so one more
//      request is in flight at the same --queue-size;
//    * generate_batch, and op speech without speech rows, still wait for the rows (and any text
//      generation) to finish, and hold back later speech meanwhile.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "lfm/base64.hpp"
#include "lfm/image_decode.hpp"
#include "lfm/image_preprocess.hpp"
#include "lfm/image_handoff.hpp"
#include "lfm/lru_bytes.hpp"
#include "lfm/lfm3_text.hpp"
#include "lfm/speech_run.hpp"
#include "lfm/speech_text.hpp"
#include "lfm/model_config.hpp"

namespace lfm::http {

using Clock = std::chrono::steady_clock;
inline constexpr size_t kMaxBody = 4 * 1024 * 1024;
inline constexpr size_t kMaxResponse = 4 * 1024 * 1024;
// POST /speech (#2): the WAV travels base64 in the JSON, so its replies (engine frame and HTTP body) get a larger
// limit: 64 MiB of base64 holds about 17.5 minutes of 24 kHz 16-bit audio, far more than the step budget allows.
// kSpeechCapacity is the engine's per-chunk capacity (WorkerCore::kSpeechCapacity). The engine bounds a request's
// work itself (WorkerCore::kSpeechStepBudget, 6144 decode steps across all chunks, then truncation); this layer bounds
// the rest of it: at most kSpeechMaxChunks chunks and kSpeechMaxPromptTokens prompt tokens (prefill), and an engine
// request frame within kMaxBody. Worst case, at the slower rates assumed (34 frames/s, 4,700 prefill tokens/s,
// 0.035 s per chunk): 6144/34 + 512 x 0.035 + 131072/4700 = about 227 s, under 80% of the default engine timeout
// (validation/speech-server-v1).
inline constexpr size_t kMaxSpeechResponse = 64 * 1024 * 1024;
inline constexpr int kSpeechCapacity = 4096;
inline constexpr size_t kSpeechMaxChunks = 512;
inline constexpr size_t kSpeechMaxPromptTokens = 131072;
inline constexpr double kSpeechMinEngineTimeout = 300;  // seconds; /speech is refused below it
inline constexpr int kAudioOutToken = 125032;
// The text of a /speech request's messages (audio aside), checked before normalization and the split: the step budget
// (6,144 frames, about 8 minutes of audio) reads about 8 KB, and a chunk holds 4,096 tokens, so 64 KiB is generous; it
// bounds the frontend's work and memory for any request (audio-assurance-v1 red-team: a 2 MB text took 60 GB)
inline constexpr size_t kSpeechMaxTextBytes = 64 * 1024;
inline constexpr size_t kSpeechMaxClips = 64;  // audio clips per /speech request (WorkerCore::kSpeechMaxClips)
// Image input (#4, validation/images-v1): images travel base64 in the JSON (JPEG or PNG bytes), so a /speech body may
// be up to kMaxSpeechBody (Tripp, 2026-11-15: a full-resolution phone photo or two, sent as they are); every other
// endpoint keeps kMaxBody. The engine request frame carries the same bytes, so it gets the same room plus the prompts.
// A decoded image may hold at most kSpeechMaxImagePixels (a 48-megapixel photo fits; Pillow allows 179M, which would
// let one request take over half a gigabyte to decode): larger is refused as image_too_large. A request's images
// share its 4,096-token prompt (a full-size image is 2,048 tokens), so kSpeechMaxImages only bounds tiny ones.
inline constexpr size_t kMaxSpeechBody = 16 * 1024 * 1024;
inline constexpr size_t kMaxSpeechFrame = kMaxSpeechBody + 1024 * 1024;
inline constexpr size_t kSpeechMaxImages = 16;  // images per /speech request (WorkerCore::kSpeechMaxImages)
inline constexpr std::uint64_t kSpeechMaxImagePixels = 64ull * 1024 * 1024;

// Whether a /speech body's messages hold an image part ({"type": "image", ...} in a content list): only such a body
// may exceed kMaxBody. A shape check only; validate_speech checks the messages themselves.
inline bool has_image_part(const Json& request) {
  const auto* o = std::get_if<Json::Object>(&request.value);
  if (!o || !o->count("messages")) return false;
  const auto* list = std::get_if<Json::Array>(&o->at("messages").value);
  if (!list) return false;
  for (const auto& m : *list) {
    const auto* mo = std::get_if<Json::Object>(&m.value);
    if (!mo || !mo->count("content")) continue;
    const auto* parts = std::get_if<Json::Array>(&mo->at("content").value);
    if (!parts) continue;
    for (const auto& part : *parts) {
      const auto* po = std::get_if<Json::Object>(&part.value);
      if (!po || !po->count("type")) continue;
      const auto* type = std::get_if<std::string>(&po->at("type").value);
      if (type && *type == "image") return true;
    }
  }
  return false;
}

// An HTTP-visible refusal: status and error code, exactly as lfm3_server.APIError.
struct ApiError : std::exception {
  int status;
  std::string code;
  ApiError(int s, std::string c) : status(s), code(std::move(c)) {}
  const char* what() const noexcept override { return code.c_str(); }
};
// Engine/transport/protocol failure (lfm3_worker_client.WorkerFailure): no cache state is trusted.
struct WorkerFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------------ JSON out (Python json.dumps)

inline void append_py_string(std::string& o, std::string_view s) {
  static const char* hex = "0123456789abcdef";
  o += '"';
  for (const unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; }
        else o += static_cast<char>(c);
    }
  }
  o += '"';
}

// Builds one JSON object in insertion order with compact separators.
class Obj {
 public:
  Obj& str(std::string_view k, std::string_view v) { key(k); append_py_string(s_, v); return *this; }
  Obj& num(std::string_view k, long long v) { key(k); s_ += std::to_string(v); return *this; }
  Obj& boolean(std::string_view k, bool v) { key(k); s_ += v ? "true" : "false"; return *this; }
  Obj& ids(std::string_view k, const std::vector<int>& v) {
    key(k);
    s_ += '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) s_ += ','; s_ += std::to_string(v[i]); }
    s_ += ']';
    return *this;
  }
  Obj& raw(std::string_view k, std::string_view json) { key(k); s_ += json; return *this; }
  std::string done() const { return s_ + "}"; }

 private:
  void key(std::string_view k) { s_ += first_ ? "" : ","; first_ = false; append_py_string(s_, k); s_ += ':'; }
  std::string s_ = "{";
  bool first_ = true;
};

inline std::string error_body(std::string_view code) { return Obj().str("error", code).done(); }

// A double as the engine's JSON reader gets it back exactly (17 significant digits round-trip).
inline std::string exact_number(double v) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

// A finite double as Python's json.dumps writes it (float.__repr__): the shortest digits that round-trip, in fixed
// notation for decimal exponents -4..15 ("0.0001", "1.0", "123.5") and otherwise "d[.ddd]e[+-]XX" ("1e-05",
// "1.5e+16"). Responses carry floats only from /speech (seconds).
inline std::string py_float(double v) {
  if (!std::isfinite(v)) throw std::invalid_argument("non-finite number in a response");
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round-trip
  std::string sci(buf, r.ptr);
  std::string sign;
  if (sci[0] == '-') { sign = "-"; sci.erase(0, 1); }
  const size_t epos = sci.find('e');
  const int exp = std::stoi(sci.substr(epos + 1));
  std::string digits = sci.substr(0, epos);
  if (digits.size() > 1) digits.erase(1, 1);  // drop the '.'
  std::string out;
  if (exp >= -4 && exp < 16) {
    if (exp < 0) {
      out = "0." + std::string(size_t(-exp - 1), '0') + digits;
    } else if (size_t(exp) + 1 >= digits.size()) {
      out = digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
    }
  } else {
    out = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
    char e[16];
    std::snprintf(e, sizeof e, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    out += e;
  }
  return sign + out;
}

// ------------------------------------------------------------------ JSON in (strict_json, integer, ...)

// lfm3_worker_client.strict_json: strict UTF-8, nesting at most 64, no duplicate keys, no NaN/Infinity,
// finite numbers, an object at the top.
inline Json strict_json(std::string_view raw) {
  if (!lfm3_text::valid_utf8(raw)) throw std::invalid_argument("invalid UTF-8");
  bool quoted = false, escaped = false;
  int depth = 0;
  for (const char c : raw) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == '[' || c == '{') { if (++depth > 64) throw std::invalid_argument("JSON nesting exceeds limit"); }
    else if (c == ']' || c == '}') --depth;
  }
  Json j = Json::parse(std::string(raw));
  if (!std::holds_alternative<Json::Object>(j.value)) throw std::invalid_argument("expected JSON object");
  return j;
}

inline const Json::Object& object_of(const Json& j) {
  if (const auto* o = std::get_if<Json::Object>(&j.value)) return *o;
  throw std::invalid_argument("expected object");
}
inline bool is_int(const Json& j) { return std::holds_alternative<double>(j.value) && j.integral; }
// lfm3_chat.integer: a JSON integer (not a float, not a boolean) in low..high.
inline long long integer(const Json& j, long long low, long long high) {
  if (!is_int(j)) throw std::invalid_argument("expected integer");
  const double v = std::get<double>(j.value);
  if (!(v >= double(low) && v <= double(high))) throw std::invalid_argument("integer out of range");
  return static_cast<long long>(v);
}
inline std::set<std::string> keys_of(const Json::Object& o) {
  std::set<std::string> k;
  for (const auto& [key, value] : o) k.insert(key);
  return k;
}

struct Sampling {
  bool do_sample = false;
  double temperature = 1.0, top_p = 1.0, min_p = 0.0, repetition_penalty = 1.0;
  long long top_k = 0, seed = 0;
};

// lfm3_chat.sampling_options.
inline Sampling sampling_options(const Json* options, int vocabulary) {
  Sampling s;
  if (!options) return s;
  static const std::set<std::string> known = {"do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed"};
  const auto& o = object_of(*options);
  for (const auto& [key, value] : o)
    if (!known.count(key)) throw std::invalid_argument("unknown sampling option");
  if (o.count("do_sample")) {
    const auto* b = std::get_if<bool>(&o.at("do_sample").value);
    if (!b) throw std::invalid_argument("do_sample must be boolean");
    s.do_sample = *b;
  }
  if (o.count("top_k")) s.top_k = integer(o.at("top_k"), 0, vocabulary);
  if (o.count("seed")) s.seed = integer(o.at("seed"), 0, 2147483647LL);
  const auto real = [&](const char* key, double& out) {
    if (!o.count(key)) return;
    const auto* d = std::get_if<double>(&o.at(key).value);
    if (!d) throw std::invalid_argument(std::string("invalid ") + key);
    out = *d;
  };
  real("temperature", s.temperature);
  real("top_p", s.top_p);
  real("min_p", s.min_p);
  real("repetition_penalty", s.repetition_penalty);
  for (const auto& [key, value] : {std::pair<const char*, double>{"temperature", s.temperature}, {"top_p", s.top_p},
                                   {"min_p", s.min_p}, {"repetition_penalty", s.repetition_penalty}}) {
    const std::string k = key;
    if (!std::isfinite(value) || value < 0 || (k != "min_p" && value == 0) || ((k == "top_p" || k == "min_p") && value > 1))
      throw std::invalid_argument("invalid " + k);
  }
  return s;
}

inline bool valid_session_id(std::string_view sid) {
  if (sid.size() != 32) return false;
  for (const char c : sid)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
  return true;
}
inline std::string session_id_of(const Json& j) {
  const auto* s = std::get_if<std::string>(&j.value);
  if (!s || !valid_session_id(*s)) throw std::invalid_argument("invalid session ID");
  return *s;
}

// secrets.token_urlsafe(24).
inline std::string token_urlsafe24() {
  unsigned char b[24];
  size_t got = 0;
  while (got < sizeof b) {
    const auto n = ::getrandom(b + got, sizeof b - got, 0);
    if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("getrandom failed"); }
    got += size_t(n);
  }
  static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  std::string out;
  for (size_t i = 0; i < sizeof b; i += 3) {
    const unsigned v = (unsigned(b[i]) << 16) | (unsigned(b[i + 1]) << 8) | b[i + 2];
    out += a[v >> 18]; out += a[(v >> 12) & 63]; out += a[(v >> 6) & 63]; out += a[v & 63];
  }
  return out;
}

inline bool valid_token(std::string_view t) {
  if (t.size() < 32 || t.size() > 256) return false;
  for (const char c : t)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
  return true;
}

// lfm3_server.token_file: an owned, regular, mode-0600 file of at most 257 bytes.
inline std::string token_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open token file");
  struct stat info {};
  std::string raw(258, '\0');
  ssize_t n = -1;
  if (::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && (info.st_mode & 07777) == 0600 && info.st_uid == ::getuid() &&
      info.st_size <= 257) {
    size_t got = 0;
    for (;;) {
      const auto r = ::read(fd, raw.data() + got, raw.size() - got);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) { n = r < 0 ? -1 : ssize_t(got); break; }
      got += size_t(r);
      if (got == raw.size()) { n = ssize_t(got); break; }
    }
  } else {
    ::close(fd);
    throw std::runtime_error("token file must be owned regular mode-0600 file");
  }
  ::close(fd);
  if (n < 0) throw std::runtime_error("cannot read token file");
  raw.resize(size_t(n));
  for (const unsigned char c : raw)
    if (c >= 0x80) throw std::runtime_error("token file must be ASCII");
  if (!raw.empty() && raw.back() == '\n') raw.pop_back();
  if (!valid_token(raw)) throw std::runtime_error("bearer token must be 32..256 URL-safe characters");
  return raw;
}

// The default token file: $XDG_CONFIG_HOME/liquid-lfm3/token, or ~/.config/liquid-lfm3/token. tools/lfm3_token.py
// finds the same path, so clients need no setup.
inline std::string default_token_path() {
  const char* xdg = std::getenv("XDG_CONFIG_HOME");
  if (xdg && *xdg == '/') return std::string(xdg) + "/liquid-lfm3/token";
  const char* home = std::getenv("HOME");
  if (!home || *home != '/') throw std::runtime_error("no token given and no HOME for the default token file");
  return std::string(home) + "/.config/liquid-lfm3/token";
}

// The token at `path`, created on first use: a directory this user owns that no one else can write (made mode 0700
// if missing; its parent too), and in it a new mode-0600 file holding a fresh random token (O_EXCL, no symlinks).
// An existing file is read under token_file's checks. `created` says which happened.
inline std::string load_or_create_token(const std::string& path, bool& created) {
  created = false;
  const auto slash = path.rfind('/');
  if (slash == std::string::npos || slash == 0) throw std::runtime_error("token path must be absolute");
  const std::string dir = path.substr(0, slash);
  auto ensure_dir = [](const std::string& d) {
    if (::mkdir(d.c_str(), 0700) != 0 && errno != EEXIST) throw std::runtime_error("cannot create " + d);
    struct stat info {};
    if (::lstat(d.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != ::getuid() || (info.st_mode & 022))
      throw std::runtime_error(d + " must be a directory owned by this user that no one else can write");
  };
  const auto parent_slash = dir.rfind('/');
  if (parent_slash != std::string::npos && parent_slash > 0) {  // ~/.config: made if missing; may be any owned dir
    const std::string parent = dir.substr(0, parent_slash);
    if (::mkdir(parent.c_str(), 0700) != 0 && errno != EEXIST) throw std::runtime_error("cannot create " + parent);
  }
  ensure_dir(dir);
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) {
    if (errno != EEXIST) throw std::runtime_error("cannot create token file");
    return token_file(path);  // made earlier (or by a server starting at the same moment): its checks apply
  }
  const std::string token = token_urlsafe24();
  const std::string line = token + "\n";
  bool ok = ::fchmod(fd, 0600) == 0;
  for (size_t done = 0; ok && done < line.size();) {
    const auto n = ::write(fd, line.data() + done, line.size() - done);
    if (n < 0 && errno == EINTR) continue;
    ok = n > 0;
    if (ok) done += size_t(n);
  }
  ok = ::fsync(fd) == 0 && ok;
  ::close(fd);
  if (!ok) { ::unlink(path.c_str()); throw std::runtime_error("cannot write token file"); }
  created = true;
  return token;
}

// The server's token: --token-file FILE, else LFM3_SERVER_TOKEN, else the default file (created on first use).
// `note` is a line for stderr when the default file was used (its path; never the token).
inline std::string startup_token(const std::string& token_path, std::string& note) {
  note.clear();
  if (!token_path.empty()) return token_file(token_path);
  if (const char* env = std::getenv("LFM3_SERVER_TOKEN")) {
    if (!valid_token(env)) throw std::runtime_error("bearer token must be 32..256 URL-safe characters");
    return env;
  }
  const auto path = default_token_path();
  bool created = false;
  auto token = load_or_create_token(path, created);
  note = std::string(created ? "created the bearer token file " : "bearer token from ") + path;
  return token;
}

// ------------------------------------------------------------------ engine boundary

// The resident protocol (liquid-lfm3-worker, protocol 2) behind an in-process call.
class Backend {
 public:
  virtual ~Backend() = default;
  // One request line (no newline) in, one reply line out. Engine-owner thread only.
  virtual std::string call(const std::string& line) = 0;
  // As call, for a request that streams (op "speech" with "stream": true): the engine's progress lines (no newline)
  // go to `progress`, in order, on the engine-owner thread, before the reply returns. A throw from `progress` fails
  // the call. Backends that cannot stream throw.
  using Progress = std::function<void(const std::string&)>;
  virtual std::string call_streaming(const std::string& line, const Progress& progress) {
    (void)line, (void)progress;
    throw std::runtime_error("this engine cannot stream");
  }
  // Write end of the engine's nonblocking cancellation pipe (8-byte little-endian frames).
  virtual int cancellation_fd() const = 0;
  // After a fatal failure: refuse every later call. Engine-owner thread only, or after it stopped.
  virtual void close() = 0;
};

struct Settings {
  int max_sessions = 2, total_capacity = 32768, queue_size = 4, handlers = 8;
  double request_timeout = 330, socket_timeout = 10, session_idle_timeout = 300;
  // One engine call (lfm3_worker_client's per-RPC worker timeout): past it the service fails closed.
  double engine_timeout = 300;
  // Tokens per text slice while no speech row runs (validation/text-wait-v1): new speech and session operations wait
  // at most one slice (16 tokens: about 0.26 s on Thor); while rows run, a slice is one token.
  int text_slice = 16;
  // validation/image-cache-v1: the image caches' budget in MiB (each side's: the front end's token counts here, the
  // engine's encoder rows on the GPU); 0: no caching
  int image_cache_mb = 512;
  void validate() const {
    const auto in = [](long long v, long long lo, long long hi) { return v >= lo && v <= hi; };
    if (!in(max_sessions, 1, 2) || !in(total_capacity, 2, 32768) || !in(queue_size, 1, 64) || !in(handlers, 1, 128) ||
        !in(text_slice, 1, 4096) || !in(image_cache_mb, 0, 65536))
      throw std::invalid_argument("invalid server settings");
    for (const double t : {request_timeout, socket_timeout, session_idle_timeout, engine_timeout})
      if (!std::isfinite(t) || !(t > 0 && t <= 86400)) throw std::invalid_argument("invalid timeout");
  }
};

// lfm3_chat.policy: output vocabulary, modality placeholders and stop ids (sorted, unique).
struct Policy {
  int vocabulary = 0;
  std::vector<int> suppress, stops;
  bool speech = false;  // the engine can run op "speech" (WorkerCore::speech_supported); /speech is refused otherwise
  bool audio_input = false;  // op "speech" takes audio (WorkerCore::audio_input_supported); audio parts get 501 otherwise
  bool image_input = false;  // op "speech" takes images (WorkerCore::image_input_supported); image parts get 501 otherwise
  // Concurrent speech (validation/speech-batch-v1): the engine's speech batch rows (ops speech_admit/speech_step,
  // WorkerCore::speech_rows); 0: one /speech at a time through op "speech".
  int speech_rows = 0;
  // validation/image-queue-v1 part 2: the engine runs in this process and finds a decoded image's pixel values in
  // lfm::image::shared_pixel_handoff (the handler thread decodes and preprocesses; the engine does not again)
  bool image_handoff = false;
};

inline std::chrono::nanoseconds seconds(double s) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(s));
}

// ------------------------------------------------------------------ Service

class Service {
 public:
  using Event = std::shared_ptr<std::atomic<bool>>;

  Service(Backend& backend, const lfm3_text::Tokenizer& tokenizer, Policy policy, Settings settings)
      : backend_(backend), tokenizer_(tokenizer), policy_(std::move(policy)), settings_(settings) {
    settings_.validate();
    image_token_cache_.set_budget(size_t(settings_.image_cache_mb) << 20);
    owner_ = std::thread([this] {
      run();
      std::lock_guard g(done_lock_);
      owner_done_ = true;
      done_cv_.notify_all();
    });
    watchdog_ = std::thread([this] { watch(); });
  }
  ~Service() {
    stopping_ = true;
    {
      std::lock_guard g(done_lock_);
      watchdog_stop_ = true;
      done_cv_.notify_all();
    }
    if (watchdog_.joinable()) watchdog_.join();
    if (owner_.joinable()) owner_.join();
  }
  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;

  const Settings& settings() const { return settings_; }
  bool ready() {
    std::lock_guard g(lock_);
    return !failed_ && !stopping_;
  }
  bool failed() {
    std::lock_guard g(lock_);
    return failed_;
  }
  bool cleanup_failed() {
    std::lock_guard g(lock_);
    return cleanup_failed_;
  }
  size_t session_count() {
    std::lock_guard g(lock_);
    return sessions_.size();
  }

  std::string create(const Json& body) {
    const auto& o = object_of(body);
    if (keys_of(o) != std::set<std::string>{"capacity"}) throw std::invalid_argument("create requires capacity only");
    const int capacity = int(integer(o.at("capacity"), 2, settings_.total_capacity));
    const std::string sid = token_urlsafe24();
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      int reserved = 0;
      for (const auto& [id, s] : sessions_) reserved += s.capacity;
      if (int(sessions_.size()) >= settings_.max_sessions || reserved + capacity > settings_.total_capacity)
        throw ApiError(429, "session_quota");
      sessions_[sid] = Session{capacity, true, "create", nullptr, Clock::now()};
    }
    auto created = std::make_shared<bool>(false);
    auto run = [this, sid, capacity, created] {
      const auto result = rpc(Obj().str("op", "create").str("session_id", sid).num("capacity", capacity));
      const auto* id = result.has("session_id") ? std::get_if<std::string>(&result.at("session_id").value) : nullptr;
      if (!id || *id != sid) throw WorkerFailure("session mismatch");
      *created = true;
      return Obj().str("session_id", sid).num("capacity", capacity).boolean("gpu_batching", false).done();
    };
    auto cleanup = [this, sid, created] {
      std::lock_guard g(lock_);
      auto it = sessions_.find(sid);
      if (it == sessions_.end()) return;
      if (*created) { it->second.busy = false; it->second.operation.clear(); it->second.last_used = Clock::now(); }
      else sessions_.erase(it);
    };
    return submit(run, cleanup, [this, sid] { close_allocation(sid); });
  }

  std::string cancel(const std::string& sid, const Json& body) {
    if (!object_of(body).empty()) throw std::invalid_argument("cancel requires an empty object");
    if (!valid_session_id(sid)) throw std::invalid_argument("invalid session ID");
    std::lock_guard g(lock_);
    if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) throw ApiError(404, "unknown_session");
    const auto& s = it->second;
    if (!s.busy || s.operation != "generate" || !s.cancel_event) throw ApiError(409, "generation_not_active");
    s.cancel_event->store(true);
    // Nonblocking targeted pipe notification only. A failed notification is not acknowledged; the event
    // stays set, and only the owner completes the generation or reset.
    try { send_cancel(sid); }
    catch (const WorkerFailure&) { throw ApiError(503, "cancellation_transport_failed"); }
    return Obj().str("session_id", sid).boolean("cancel_requested", true).boolean("gpu_batching", false).done();
  }

  std::string lifecycle(const std::string& sid, const std::string& op, const Json& body) {
    if ((op != "reset" && op != "close") || !object_of(body).empty()) throw std::invalid_argument("invalid lifecycle request");
    lease(sid, op, nullptr);
    auto run = [this, sid, op] {
      const auto result = rpc(Obj().str("op", op).str("session_id", sid));
      if (!flag(result, op == "reset" ? "reset" : "closed") || !session_matches(result, sid))
        throw WorkerFailure("invalid lifecycle reply");
      if (op == "close") {
        std::lock_guard g(lock_);
        sessions_.erase(sid);
      }
      return Obj().boolean("ok", true).done();
    };
    return submit(run, [this, sid] { release(sid, nullptr); }, nullptr);
  }

  // `body` is shared, not copied, and dropped as soon as the prompt is encoded.
  // `gone` (optional; polled about every 50 ms while the request waits): true when the client's connection is reset. The
  // generation is then cancelled at its next slice, as /cancel would (its cache reset), and its reply goes nowhere
  // (text-wait-v1 rule 6: before, it ran to its end for no one, its cache kept). A client that only half-closes keeps
  // its request.
  std::string generate(std::shared_ptr<const Json> body, const std::function<bool()>& gone = {}) {
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    auto request = std::make_shared<Generation>(validate_generation(*body, body));
    body.reset();
    const auto sid = request->sid;
    Event event = std::make_shared<std::atomic<bool>>(false);
    const int capacity = lease(sid, "generate", event);
    try {
      frontend_call([&] { request->input = encode(*request, capacity); }, deadline);
      request->drop_body();
    } catch (...) {
      release(sid, event);
      throw;
    }
    // run by the engine owner as a kText job (text_begin, then text_cycle), not through `function`; waited on as submit
    // waits, with the client's connection watched
    auto [job, future] = enqueue(nullptr, [this, sid, event] { release(sid, event); }, nullptr, deadline, Kind::kText,
                                 request, event);
    for (;;) {
      const auto tick = std::min(deadline, Clock::now() + std::chrono::milliseconds(50));
      if (future.wait_until(tick) == std::future_status::ready) return future.get();
      if (Clock::now() >= deadline) break;
      if (gone && !event->load() && gone()) event->store(true);  // as /cancel: cancelled at its next slice
    }
    if (abandon(*job, future)) return future.get();
    // Do not release a running generation lease or pretend cancellation.
    throw ApiError(504, "request_deadline_outcome_unknown");
  }

  std::string generate_batch(std::shared_ptr<const Json> body) {
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    const auto& o = object_of(*body);
    const auto* list = o.size() == 1 && o.count("requests") ? std::get_if<Json::Array>(&o.at("requests").value) : nullptr;
    if (!list || list->size() != 2) throw std::invalid_argument("batch requires exactly two requests");
    auto requests = std::make_shared<std::array<Generation, 2>>();
    for (size_t i = 0; i < 2; ++i) (*requests)[i] = validate_generation((*list)[i], body);
    body.reset();
    const std::array<std::string, 2> sids{(*requests)[0].sid, (*requests)[1].sid};
    if (sids[0] == sids[1]) throw std::invalid_argument("duplicate batch session");
    const std::array<Event, 2> events{std::make_shared<std::atomic<bool>>(false), std::make_shared<std::atomic<bool>>(false)};
    std::array<int, 2> capacities{};
    // Check both before mutating either lease: never release somebody else's busy lease when the second
    // lane cannot be admitted.
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      for (const auto& sid : sids)
        if (!sessions_.count(sid)) throw ApiError(404, "unknown_session");
      for (const auto& sid : sids)
        if (sessions_.at(sid).busy) throw ApiError(409, "session_busy");
      for (size_t i = 0; i < 2; ++i) {
        auto& s = sessions_.at(sids[i]);
        capacities[i] = s.capacity;
        s.busy = true; s.operation = "generate"; s.cancel_event = events[i];
      }
    }
    auto cleanup = [this, sids, events] { for (size_t i = 0; i < 2; ++i) release(sids[i], events[i]); };
    try {
      frontend_call([&] {
        for (size_t i = 0; i < 2; ++i) {
          if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
          (*requests)[i].input = encode((*requests)[i], capacities[i]);
        }
      }, deadline);
      for (auto& r : *requests) r.drop_body();
    } catch (...) {
      cleanup();
      throw;
    }
    auto run = [this, requests, events, deadline] {
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      std::string lanes = "[";
      for (size_t i = 0; i < 2; ++i) {
        if (i) lanes += ',';
        lanes += native_request((*requests)[i]).str("session_id", (*requests)[i].sid).done();
      }
      lanes += ']';
      const auto result = rpc(Obj().str("op", "generate_batch").raw("lanes", lanes),
                              {{(*requests)[0].sid, events[0]}, {(*requests)[1].sid, events[1]}}, true);
      const auto* gpu = result.has("gpu_batching") ? std::get_if<bool>(&result.at("gpu_batching").value) : nullptr;
      const long long paired = result.has("paired_decode_calls") && is_int(result.at("paired_decode_calls"))
                                   ? static_cast<long long>(std::get<double>(result.at("paired_decode_calls").value)) : -1;
      if (paired < 0 || !gpu || *gpu != (paired > 0)) throw WorkerFailure("invalid batch accounting");
      const auto* results = result.has("results") ? std::get_if<Json::Array>(&result.at("results").value) : nullptr;
      if (!results || results->size() != 2) throw WorkerFailure("invalid batch results");
      std::vector<Response> responses;
      for (size_t i = 0; i < 2; ++i) {
        const auto& lane = (*results)[i];
        const auto* lo = std::get_if<Json::Object>(&lane.value);
        const auto boolean_field = [&](const char* k) { return lo && lo->count(k) && std::holds_alternative<bool>(lo->at(k).value); };
        if (!lo || !boolean_field("gpu_batching") || (lo->count("ok") && !flag(lane, "ok"))
            || std::get<bool>(lo->at("gpu_batching").value) != *gpu || !lo->count("paired_decode_calls")
            || !is_int(lo->at("paired_decode_calls")) || std::get<double>(lo->at("paired_decode_calls").value) != double(paired)
            || !boolean_field("cancelled") || !boolean_field("cache_valid"))
          throw WorkerFailure("invalid lane accounting");
        auto response = generation_response((*requests)[i], lane, *events[i], *gpu);
        response.body.num("paired_decode_calls", paired);
        responses.push_back(std::move(response));
      }
      // Prefill samples the first token without a decode forward, so each paired forward needs another
      // output slot in both lanes. Cancellation may discard the token sampled after that forward.
      long long budget = std::numeric_limits<long long>::max();
      for (const auto& r : *requests) budget = std::min<long long>(budget, r.count - 1);
      if (paired > budget) throw WorkerFailure("paired calls exceed requested decode budget");
      bool any_cancelled = false;
      long long completed = std::numeric_limits<long long>::max();
      for (const auto& r : responses) {
        const long long maximum = (long long)r.generated.size() - (r.cancelled ? 0 : 1);
        if (paired > maximum) throw WorkerFailure("paired calls exceed lane decode work");
        any_cancelled = any_cancelled || r.cancelled;
        completed = std::min<long long>(completed, (long long)r.generated.size() - 1);
      }
      if (!any_cancelled && paired != completed) throw WorkerFailure("paired calls do not match completed group");
      std::string items = "[";
      for (size_t i = 0; i < 2; ++i) {
        const auto lane = responses[i].body.done();
        if (lane.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
        if (i) items += ',';
        items += lane;
      }
      items += ']';
      const auto out = Obj().raw("results", items).boolean("gpu_batching", *gpu).num("paired_decode_calls", paired).done();
      if (out.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
      return out;
    };
    return submit(run, cleanup, nullptr, deadline, Kind::kExclusive);
  }

  // Stops admission, fails every later request, cancels running generations at their next token
  // boundary and waits for the engine owner to finish its queue.
  void close() {
    cancel_all_ = true;  // running speech rows are cancelled at their next step
    {
      std::lock_guard g(lock_);
      stopping_ = true;
      failed_ = true;
      // A requested cancellation, exactly as /cancel requests it, so the reply is a solicited one.
      for (auto& [sid, s] : sessions_)
        if (s.busy && s.operation == "generate" && s.cancel_event) s.cancel_event->store(true);
      sessions_.clear();
    }
    {
      std::lock_guard g(active_lock_);
      for (auto& [sid, active] : active_) {
        if (active.sent) continue;
        try { write_frame(active.token); active.sent = true; }
        catch (const WorkerFailure&) {}  // the generation then finishes normally; shutdown still waits for it
      }
    }
    queue_cv_.notify_all();
    // As lfm3_server.Service.close: a bounded wait for the engine owner. An engine call that never
    // returns cannot be reclaimed in-process; the caller exits instead (cleanup_failed()).
    bool done;
    {
      std::unique_lock g(done_lock_);
      done = done_cv_.wait_for(g, seconds(settings_.engine_timeout + 5), [&] { return owner_done_; });
      watchdog_stop_ = true;
      done_cv_.notify_all();
    }
    if (watchdog_.joinable()) watchdog_.join();
    if (!done) {
      owner_.detach();
      std::lock_guard g(lock_);
      cleanup_failed_ = true;
      return;
    }
    owner_.join();
    fatal();
  }

 private:
  struct Session {
    int capacity = 0;
    bool busy = false;
    std::string operation;
    Event cancel_event;
    Clock::time_point last_used;
  };
  struct Generation {
    std::string sid;
    const Json* messages = nullptr;
    std::shared_ptr<const Json> body;  // the parsed request; keeps `messages` alive until encoding
    int count = 0;
    Sampling sampling;
    std::vector<int> input;
    void drop_body() { messages = nullptr; body.reset(); }
  };
  struct Speech {
    std::shared_ptr<const Json> body;  // keeps `messages` alive until encoding
    const Json* messages = nullptr;
    std::string mode;
    int count = 0;
    long long seed = 0;
    bool split = false;
    std::vector<std::vector<int>> prompts;
    std::vector<std::string> texts;  // each chunk's text, when split
    std::string chunks_json;         // the prompts as the engine request carries them
    std::string audio_json;          // audio input (#3): the clips (base64 WAV) as the engine request carries them
    bool audio_input = false;        // the messages hold audio parts
    std::string images_json;         // image input (#4): the images (base64 JPEG/PNG) as the engine request carries them
    bool image_input = false;        // the messages hold image parts
    std::vector<int> image_tokens;   // each image part's token count, in prompt order (decode_images)
    // each decoded image's pixel values left for the in-process engine, held until this request ends (image-queue-v1)
    std::vector<lfm::LeasedMap<lfm::image::PixelValues>::Lease> image_leases;
    bool normalize = false;          // numbers and abbreviations in words before the split (audio mode)
    bool stream = false;             // the reply as events while the engine generates (speech_stream)
  };
  // Image input (#4): an image part's data (base64 of JPEG or PNG bytes) decoded as the engine will
  // (lfm/image_decode.hpp: bit-identical to HF's front end, validation/images-v1 rule 1) and its token count from
  // smart_resize, preprocess's own count without the resize (lfm::image::tokens); or, for the in-process engine,
  // preprocessed here and handed over (decode_images). Every refusal is an ApiError with its own code, before the engine sees the request.
  // `then(rgb)` (the token count, or the preprocessing for the engine) runs under the same refusals: smart_resize's
  // aspect-ratio refusal is one of them.
  template <class F>
  static auto decode_image(const std::string& data, F&& then) {
    std::string bytes;
    try {
      bytes = lfm::base64::decode(data);
    } catch (const std::exception&) {
      throw ApiError(400, "invalid_image_base64");
    }
    try {
      lfm::image::Limits limits;
      limits.max_pixels = kSpeechMaxImagePixels;
      return then(lfm::image::load_rgb(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(), limits));
    } catch (const lfm::image::DecodeError& e) {
      // the decoder's code as the refusal: image_corrupt, image_truncated, image_too_large (413), image_aspect_ratio,
      // image_unsupported_format, image_unsupported_jpeg, image_unsupported_png
      std::string code = e.code;
      if (code.rfind("image_", 0) == 0) code = code.substr(6);
      if (code.size() > 6 && code.compare(code.size() - 6, 6, "_image") == 0) code.resize(code.size() - 6);
      throw ApiError(code == "too_large" ? 413 : 400, "image_" + code);
    }
  }
  // validation/image-queue-v1: each image part decoded and preprocessed on the request's own handler thread, before
  // the front end, at most kImageDecodeSlots at a time (bounding memory: ~200 MB each at the 64-megapixel cap). A
  // request waits for a slot within its deadline (504 frontend_deadline past it) instead of being refused, stops
  // waiting when the server stops (503) or its client is gone (499 client_gone, a reply no one reads), and refuses a bad
  // image with its own code as before. The images' order is encode_speech's (image input is never split).
  void decode_images(Speech& s, Clock::time_point deadline, const std::function<bool()>& gone) {
    if (!s.image_input) return;
    for (const auto& m : lfm3_text::parse_messages(*s.messages, /*allow_audio=*/true, /*allow_image=*/true))
      for (const auto& part : m.parts) {
        if (!part.image) continue;
        // image-cache-v1: these exact bytes were decoded before (and accepted: only a decoded image is kept): their
        // count, with no decode; the engine finds the image's rows in its own cache, or decodes it again itself
        // C3: in process, only when the engine also holds its rows (else the engine thread would decode it): decoded
        // here instead and handed over, as a miss
        if (const auto known = image_token_cache_.find(part.data);
            known && (!policy_.image_handoff || lfm::image::shared_held_images().holds(part.data))) {
          s.image_tokens.push_back(*known);
          continue;
        }
        {
          std::unique_lock g(image_slots_lock_);
          for (;;) {
            if (stopping_) throw ApiError(503, "worker_unavailable");
            if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
            if (gone && gone()) throw ApiError(499, "client_gone");
            if (image_slots_ > 0) break;
            image_slots_cv_.wait_until(g, std::min(deadline, Clock::now() + std::chrono::milliseconds(50)));
          }
          --image_slots_;
        }
        struct Slot {
          Service& service;  // released by the destructor, however the decode ends
          ~Slot() {
            {
              std::lock_guard g(service.image_slots_lock_);
              ++service.image_slots_;
            }
            service.image_slots_cv_.notify_one();
          }
        } slot{*this};
        // refusals: each its own code, before the engine
        int tokens;
        if (policy_.image_handoff) {  // part 2: preprocessed here, found by the engine (lfm/image_handoff.hpp)
          auto pixels = decode_image(part.data, [](const lfm::image::Rgb& rgb) {
            return std::make_shared<const lfm::image::PixelValues>(lfm::image::preprocess(rgb));
          });
          tokens = pixels->tokens;
          s.image_leases.push_back(lfm::image::shared_pixel_handoff().put(part.data, std::move(pixels)));
        } else {
          tokens = decode_image(part.data, [](const lfm::image::Rgb& rgb) { return lfm::image::tokens(rgb); });
        }
        s.image_tokens.push_back(tokens);
        image_token_cache_.put(part.data, std::make_shared<const int>(tokens), sizeof(int));
      }
  }
  Speech validate_speech(const Json& item) const {
    const auto& o = object_of(item);
    static const std::set<std::string> allowed = {"messages", "mode", "max_new_tokens", "seed", "split", "normalize",
                                                  "stream"};
    for (const auto& [key, value] : o)
      if (!allowed.count(key)) throw std::invalid_argument("invalid speech fields");
    for (const char* key : {"messages", "mode", "max_new_tokens"})
      if (!o.count(key)) throw std::invalid_argument("invalid speech fields");
    Speech s;
    const auto* mode = std::get_if<std::string>(&o.at("mode").value);
    if (!mode || (*mode != "audio" && *mode != "interleaved" && *mode != "text"))
      throw std::invalid_argument("mode must be audio, interleaved or text");
    s.mode = *mode;
    s.count = int(integer(o.at("max_new_tokens"), 1, kSpeechCapacity - 1));
    s.seed = o.count("seed") ? integer(o.at("seed"), 0, 2147483647) : 0;
    if (!std::holds_alternative<Json::Array>(o.at("messages").value)) throw std::invalid_argument("messages must be a list");
    s.messages = &o.at("messages");
    bool audio_input = false;
    size_t clips = 0, images = 0;
    // The normalizer is American English only (speech_normalize.hpp): it applies by default when there is no system
    // prompt or a system prompt names American or British English (the model card's TTS prompt), and is refused for
    // any other system prompt (audio-assurance-v1 rule 10: "Tengo 3 gatos" must not become "Tengo three gatos").
    bool has_system = false, english = false;
    size_t text_bytes = 0;
    for (const auto& m : lfm3_text::parse_messages(*s.messages, /*allow_audio=*/true, /*allow_image=*/true)) {
      for (const auto& part : m.parts) clips += part.audio, images += part.image;
      text_bytes += m.content.size() + m.thinking.size();
      for (const auto& part : m.parts) text_bytes += part.text.size();
      if (text_bytes > kSpeechMaxTextBytes) throw std::invalid_argument("speech text exceeds its byte limit");
      if (m.role != "system") continue;
      has_system = true;
      std::string text = m.content;
      for (const auto& part : m.parts) text += part.text;
      english |= text.find("American English") != std::string::npos || text.find("British English") != std::string::npos;
    }
    const bool normalizable = !has_system || english;
    audio_input = s.audio_input = clips > 0;
    if (clips > kSpeechMaxClips) throw std::invalid_argument("too many audio clips");
    s.image_input = images > 0;
    if (images > kSpeechMaxImages) throw ApiError(400, "too_many_images");
    if (o.count("split")) {
      const auto* b = std::get_if<bool>(&o.at("split").value);
      if (!b) throw std::invalid_argument("split must be a boolean");
      s.split = *b;
    } else {
      s.split = s.mode == "audio" && !audio_input && !s.image_input;
    }
    if (s.split && s.mode != "audio") throw std::invalid_argument("split is for audio mode");
    if (s.split && audio_input) throw std::invalid_argument("split is for text input; audio input is not split");
    if (s.split && s.image_input) throw std::invalid_argument("split is for text input; image input is not split");
    if (o.count("normalize")) {
      const auto* b = std::get_if<bool>(&o.at("normalize").value);
      if (!b) throw std::invalid_argument("normalize must be a boolean");
      s.normalize = *b;
    } else {
      s.normalize = s.mode == "audio" && normalizable;
    }
    if (s.normalize && s.mode != "audio") throw std::invalid_argument("normalize is for audio mode");
    if (s.normalize && !normalizable) throw std::invalid_argument("normalize is for American or British English");
    if (o.count("stream")) {
      const auto* b = std::get_if<bool>(&o.at("stream").value);
      if (!b) throw std::invalid_argument("stream must be a boolean");
      s.stream = *b;
    }
    return s;
  }
  // Render and tokenize each chunk as a single request (lfm3_text::encode_messages), split by sentences when asked.
  // Audio input (#3): each user audio part (base64 16 kHz mono WAV) renders as the template's single <|audio|>, then
  // expands to its clip's encoder token count (lfm3_text::expand_audio); the clips go to the engine as given.
  void encode_speech(Speech& s) const {
    auto messages = lfm3_text::parse_messages(*s.messages, /*allow_audio=*/!s.split, /*allow_image=*/!s.split);
    if (s.normalize) speech_text::normalize_user_text(messages);
    std::vector<size_t> clip_tokens;
    std::vector<lfm3_text::MediaTokens> media;  // every part's medium in prompt order (with images: expand_media)
    size_t image_count = 0;
    s.audio_json = "[";
    s.images_json = "[";
    for (const auto& m : messages)
      for (const auto& part : m.parts) {
        if (part.audio) {
          clip_tokens.push_back(lfm::audio_features::tokens(lfm3_speech::clip_from_base64(part.data).size()));
          s.audio_json += (clip_tokens.size() > 1 ? ",\"" : "\"") + part.data + "\"";  // checked base64: no escapes
          media.push_back({lfm3_text::kAudioToken, clip_tokens.back()});
        } else if (part.image) {
          // decoded and preprocessed before the front end (decode_images), as the engine will: its token count
          if (image_count >= s.image_tokens.size()) throw std::logic_error("encode_speech: an image not decoded");
          media.push_back({lfm3_text::kImageToken, size_t(s.image_tokens[image_count])});
          s.images_json += (++image_count > 1 ? ",\"" : "\"") + part.data + "\"";  // checked base64: no escapes
        }
      }
    if (image_count != s.image_tokens.size()) throw std::logic_error("encode_speech: decoded images differ from parts");
    s.audio_json += "]";
    s.images_json += "]";
    if (clip_tokens.empty()) s.audio_json.clear();
    if (!image_count) s.images_json.clear();
    const auto chats = s.split ? speech_text::sentence_requests(messages)
                               : std::vector<std::vector<lfm3_text::Message>>{messages};
    if (chats.size() > kSpeechMaxChunks) throw std::invalid_argument("speech request has too many sentences");
    size_t tokens = 0;
    for (const auto& chat : chats) {
      auto ids = lfm3_text::encode_messages(tokenizer_, chat, kSpeechCapacity, s.count);
      for (const int id : ids) {
        if (id < 0 || id >= policy_.vocabulary) throw std::invalid_argument("input_ids out of range");
        // the audio and image placeholders come only from their parts (text cannot hold them); the expansion counts them
        if (id != lfm3_text::kAudioToken && id != lfm3_text::kImageToken &&
            std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), id))
          throw std::invalid_argument("multimodal placeholder input is unsupported");
      }
      if (image_count) ids = lfm3_text::expand_media(ids, media, kSpeechCapacity, s.count);
      else if (!clip_tokens.empty()) ids = lfm3_text::expand_audio(ids, clip_tokens, kSpeechCapacity, s.count);
      tokens += ids.size();
      if (tokens > kSpeechMaxPromptTokens) throw std::invalid_argument("speech request has too many prompt tokens");
      s.prompts.push_back(std::move(ids));
      if (s.split) s.texts.push_back(chat.back().content);
    }
    // The engine request, built here so that one too large for an engine frame is a 400, not a fatal engine error.
    s.chunks_json = "[";
    for (size_t k = 0; k < s.prompts.size(); ++k) {
      s.chunks_json += k ? ",[" : "[";
      for (size_t i = 0; i < s.prompts[k].size(); ++i) s.chunks_json += (i ? "," : "") + std::to_string(s.prompts[k][i]);
      s.chunks_json += "]";
    }
    s.chunks_json += "]";
    if (s.chunks_json.size() + s.audio_json.size() + s.images_json.size() + 256 > kMaxSpeechFrame)
      throw std::invalid_argument("speech request exceeds the engine frame");
    s.messages = nullptr;
    s.body.reset();
  }
  // The reply; streamed, the done event: {"event": "done"} and the same fields but the audio.
  std::string speech_response(const Speech& s, const Json& r) const {
    if (!r.has("cancelled") || !std::holds_alternative<bool>(r.at("cancelled").value)) throw WorkerFailure("invalid speech reply");
    Obj head;
    if (s.stream) head.str("event", "done");
    if (std::get<bool>(r.at("cancelled").value))
      return head.str("mode", s.mode).boolean("cancelled", true).num("seed", s.seed).boolean("split", s.split)
          .boolean("normalize", s.normalize).done();
    std::vector<int> slots;
    for (const auto& v : r.at("slots").array()) {
      const int id = v.integer();
      if (id != kAudioOutToken) slots.push_back(id);
    }
    const auto& chunks = r.at("chunks").array();
    const bool truncated = r.at("truncated").boolean();
    if (chunks.empty() || chunks.size() > s.prompts.size() || (chunks.size() < s.prompts.size() && !truncated))
      throw WorkerFailure("invalid speech reply");  // fewer chunks only when truncated
    std::string list = "[";
    for (size_t k = 0; k < chunks.size(); ++k) {
      Obj c;
      if (s.split) c.str("text", s.texts[k]);
      c.num("frames", chunks[k].at("frames").integer()).boolean("ended_by_eos", chunks[k].at("ended_by_eos").boolean());
      list += (k ? "," : "") + c.done();
    }
    list += "]";
    head.str("mode", s.mode).str("text", s.mode == "audio" ? std::string() : tokenizer_.decode(slots, true));
    if (!s.stream) head.str("audio_wav_base64", r.at("wav_base64").string());
    const auto out = head.num("sample_rate", 24000)
        .raw("audio_seconds", py_float(std::get<double>(r.at("audio_seconds").value)))
        .num("frames", r.at("frames").integer()).raw("chunks", list).num("seed", s.seed).boolean("split", s.split)
        .boolean("normalize", s.normalize)
        .boolean("cancelled", false).boolean("truncated", truncated).raw("prefill_seconds", py_float(std::get<double>(r.at("prefill_seconds").value)))
        .raw("generate_seconds", py_float(std::get<double>(r.at("generate_seconds").value))).done();
    if (out.size() > kMaxSpeechResponse) throw WorkerFailure("response exceeds limit");
    return out;
  }
  // A streamed reply's event lines, from the engine-owner thread to the connection's handler.
  struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> lines;
    void push(std::string line) {
      { std::lock_guard g(m); lines.push_back(std::move(line)); }
      cv.notify_one();
    }
  };
  // One streamed request's engine progress, checked and turned into its events (engine-owner thread). A malformed
  // progress line, or a reply that disagrees with what was streamed, is an engine fault (WorkerFailure).
  class StreamState {
   public:
    StreamState(const Service& service, const Speech& request, std::shared_ptr<Lines> lines)
        : s_(service), request_(request), lines_(std::move(lines)), decoder_(service.tokenizer_, true),
          text_mode_(request.mode != "audio") {}
    // `o`: a progress object of `keys` keys (1, or 2 with the batch's "ticket")
    void event(const Json::Object& o, size_t keys) {
      if (o.size() != keys || chunk_ >= request_.prompts.size()) throw WorkerFailure("invalid speech progress");
      if (o.count("audio")) {
        const auto& b64 = o.at("audio").string();
        const size_t bytes = lfm::base64::decode(b64).size();  // strict: the client gets these characters as they are
        if (bytes == 0 || bytes % 2) throw WorkerFailure("invalid speech progress");
        samples_ += bytes / 2;
        lines_->push(Obj().str("event", "audio").num("chunk", (long long)chunk_).str("pcm_base64", b64).done());
      } else if (o.count("text")) {
        const auto& ids = o.at("text").array();
        if (ids.size() != 1) throw WorkerFailure("invalid speech progress");
        const int id = ids[0].integer();
        if (id < 0 || id >= s_.policy_.vocabulary || id == kAudioOutToken) throw WorkerFailure("invalid speech progress");
        text_slots_.push_back(id);
        if (!text_mode_) return;
        const auto piece = decoder_.push(id);
        if (!piece.empty()) lines_->push(Obj().str("event", "text").num("chunk", (long long)chunk_).str("text", piece).done());
      } else if (o.count("chunk")) {
        // the chunk's text is complete: bytes held back for an unfinished character end as decode() ends them
        // (text comes only unsplit, in one chunk, so this is decode() of all the text)
        if (text_mode_) {
          const auto rest = decoder_.finish();
          if (!rest.empty()) lines_->push(Obj().str("event", "text").num("chunk", (long long)chunk_).str("text", rest).done());
        }
        const auto& c = o.at("chunk");
        const long long frames = c.at("frames").integer();
        const bool eos = c.at("ended_by_eos").boolean();
        Obj e;
        e.str("event", "chunk").num("chunk", (long long)chunk_);
        if (request_.split) e.str("text", request_.texts[chunk_]);
        lines_->push(e.num("frames", frames).boolean("ended_by_eos", eos).done());
        ended_.emplace_back(frames, eos);
        ++chunk_;
      } else {
        throw WorkerFailure("invalid speech progress");
      }
    }
    // The reply must agree with what was streamed: the samples, the slots, the chunks.
    void check(const Json& result) const {
      if (!result.has("cancelled") || !std::holds_alternative<bool>(result.at("cancelled").value))
        throw WorkerFailure("invalid speech reply");
      if (std::get<bool>(result.at("cancelled").value)) return;
      if (result.has("wav_base64") || result.at("samples").integer() != (long long)samples_)
        throw WorkerFailure("invalid speech reply");
      std::vector<int> want;
      for (const auto& v : result.at("slots").array())
        if (v.integer() != kAudioOutToken) want.push_back(v.integer());
      const auto& chunks = result.at("chunks").array();
      if (want != text_slots_ || chunks.size() != ended_.size()) throw WorkerFailure("invalid speech reply");
      for (size_t k = 0; k < chunks.size(); ++k)
        if (chunks[k].at("frames").integer() != ended_[k].first || chunks[k].at("ended_by_eos").boolean() != ended_[k].second)
          throw WorkerFailure("invalid speech reply");
    }

   private:
    const Service& s_;
    const Speech& request_;
    std::shared_ptr<Lines> lines_;
    lfm3_text::Tokenizer::StreamDecoder decoder_;
    bool text_mode_;
    size_t chunk_ = 0, samples_ = 0;
    std::vector<int> text_slots_;
    std::vector<std::pair<long long, bool>> ended_;  // per chunk: frames, ended_by_eos
  };
  // A /speech request in the concurrent speech batch: queued, then running in a row (engine-owner thread), then
  // settled once (its reply, or an error) through `promise`.
  struct Ticket {
    long long id = -1;
    std::shared_ptr<Speech> request;
    Clock::time_point deadline, arrived;
    std::shared_ptr<Lines> lines;  // streamed: its events
    std::promise<std::string> promise;
    std::mutex completion;
    // under completion: admitted into a row (from then on it runs to its end, bounded by the step budget, whatever
    // its deadline: in a batch every row runs slower than alone), abandoned (past its deadline while queued), settled
    bool admitted = false, abandoned = false, settled = false;
    std::atomic<bool> cancel{false};          // the client is gone or abandoned it: cancel at the next step
    // engine-owner thread only
    std::unique_ptr<StreamState> state;
    bool cancel_sent = false;
  };

  struct Response {
    Obj body;
    std::vector<int> generated;
    bool cancelled = false;
  };
  // How the engine owner schedules a queued job (validation/text-wait-v1):
  //   kShort      a session operation (create, reset, close): one short engine call, run between speech steps;
  //   kText       a text generation: begun (prefill) between speech steps, then one token per speech step while speech
  //               rows run (a long slice when none do); one text generation at a time, first in first out;
  //   kExclusive  one whole engine call that waits for the running speech rows and text to finish, and holds back
  //               later speech meanwhile (generate_batch; op "speech" without speech rows): the batch drains for it.
  enum class Kind { kShort, kText, kExclusive };
  struct Job {
    std::function<std::string()> function;  // kShort, kExclusive
    std::promise<std::string> promise;
    Clock::time_point deadline, arrived;
    std::function<void()> cleanup, discard;
    std::mutex completion;
    bool abandoned = false;
    Kind kind = Kind::kShort;
    std::shared_ptr<Generation> text;  // kText: the request
    Event event;                       // kText: its cancellation
  };
  struct Active {
    std::uint64_t token = 0;
    bool sent = false;
  };

  static bool flag(const Json& j, const char* key) {
    if (!j.has(key)) return false;
    const auto* b = std::get_if<bool>(&j.at(key).value);
    return b && *b;
  }
  static bool session_matches(const Json& j, const std::string& sid) {
    if (!j.has("session_id")) return true;
    const auto* s = std::get_if<std::string>(&j.at("session_id").value);
    return s && *s == sid;
  }

 public:
  // POST /speech (#2, #3): text and speech in; speech, interleaved text and speech, or text out. No session: one
  // request at a time on the engine's speech runner, queued and timed like every engine job. Body: {"messages": [...]
  // (the chat, as for /generate; a user message's content list may also hold {"type": "audio", "audio": base64 of a
  // 16 kHz mono 16-bit PCM or 32-bit float WAV file}, at most 64 clips; the 4 MiB body holds about 90 s of 16-bit
  // audio; a 501 audio_input_unsupported when the engine has no audio encoder for the checkpoint), "mode":
  // "audio" | "interleaved" | "text", "max_new_tokens": 1..4095 (per chunk), "seed": 0..2^31-1 (default 0), "split":
  // bool (audio mode with text input only; default true there: long-form TTS by sentences, lfm/speech_text.hpp),
  // "normalize": bool (audio mode only; default true there: numbers, symbols and abbreviations in the user's text read
  // out as words first, lfm/speech_normalize.hpp, validation/speech-normalize-v1), "stream": bool (default false: with
  // true, the reply comes as events while the engine generates, speech_stream)}.
  // Spoken replies to audio input: use "interleaved", as the model card's voice chat does; in "audio" mode the model
  // answers speech with unusable speech, in HF as here (validation/audio-in-v1).
  // Reply: {"mode", "text" (the emitted text; "" in audio mode), "audio_wav_base64" (24 kHz 16-bit mono WAV; in text
  // mode a WAV with no samples, its 44-byte header),
  // "sample_rate", "audio_seconds", "frames", "chunks": [{"text" (split only), "frames", "ended_by_eos"}], "seed",
  // "split", "normalize", "cancelled", "truncated" (the step budget stopped it short of its text: chunks skipped, or the last one
  // cut), "prefill_seconds", "generate_seconds"}. The WAV is byte for byte lfm3-speech's for the same request.
  // `gone` (optional; polled about every 50 ms while a batched request waits): true when the client's connection is
  // broken; the request is then cancelled at the engine's next step and frees its row (audio-assurance-v1 red-team:
  // before, an unstreamed request ran to its end for no one). A client that only half-closes keeps its request.
  std::string speech(std::shared_ptr<const Json> body, const std::function<bool()>& gone = {}) {
    // the step budget's worst case (~227 s) is sized for the default engine timeout; a shorter one could let an
    // admitted request fail the service closed
    if (!policy_.speech || settings_.engine_timeout < kSpeechMinEngineTimeout) throw ApiError(501, "speech_unsupported");
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    auto request = std::make_shared<Speech>(validate_speech(*body));
    if (request->audio_input && !policy_.audio_input) throw ApiError(501, "audio_input_unsupported");
    if (request->image_input && !policy_.image_input) throw ApiError(501, "image_input_unsupported");
    request->body = std::move(body);  // `messages` points into it until encode_speech drops it
    decode_images(*request, deadline, gone);
    frontend_call([&] { encode_speech(*request); }, deadline, request->image_input);
    if (policy_.speech_rows > 0) {
      auto t = std::make_shared<Ticket>();
      t->request = request, t->deadline = deadline;
      auto future = t->promise.get_future();
      enqueue_ticket(t);
      for (;;) {
        const auto tick = std::min(deadline, Clock::now() + std::chrono::milliseconds(50));
        if (future.wait_until(tick) == std::future_status::ready) return future.get();
        if (Clock::now() >= deadline) break;
        if (gone && !t->cancel && gone()) t->cancel = true;  // its reply goes nowhere; the wait ends when it stops
      }
      if (abandon_ticket(*t, future)) return future.get();  // admitted: it runs to its end
      throw ApiError(504, "queue_deadline");  // still queued at its deadline: never started
    }
    Event event = std::make_shared<std::atomic<bool>>(false);
    auto run = [this, request, event, deadline] {
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      Obj frame;
      frame.str("op", "speech").raw("chunks", request->chunks_json).str("mode", request->mode)
          .num("max_new_tokens", request->count).num("seed", request->seed).boolean("chunked", request->split);
      if (!request->audio_json.empty()) frame.raw("audio", request->audio_json);
      if (!request->images_json.empty()) frame.raw("images", request->images_json);
      const auto result = rpc(std::move(frame),
                              {{"speech", event}}, false, kMaxSpeechResponse, nullptr, kMaxSpeechFrame);
      return speech_response(*request, result);
    };
    return submit(run, [] {}, nullptr, deadline, Kind::kExclusive);
  }

  // POST /speech with "stream": true (validation/speech-stream-v1): the same request, its reply as events while the
  // engine generates, each one JSON line given to `send` (false: the client is gone). The first call to `send` is the
  // client's first byte of the reply, so anything thrown before it is an ordinary error reply; once a line is sent this
  // returns normally, ending with a done event, or an error event if the request fails after the start. Events:
  //   {"event": "audio", "chunk": k, "pcm_base64": 24 kHz mono 16-bit little-endian PCM}: all of them in order are the
  //     samples of the WAV the request would return unstreamed;
  //   {"event": "text", "chunk": k, "text": the next piece of the decoded text} (interleaved and text modes);
  //   {"event": "chunk", "chunk": k, "text" (split only), "frames", "ended_by_eos"}: chunk k has ended;
  //   {"event": "done", ...}: the unstreamed reply's fields but "audio_wav_base64"; or {"event": "error", "error": code}.
  // A client that goes away is cancelled at the engine's next step. Events queue between the engine-owner thread and
  // this one: the engine never waits on a client.
  // The sink: `send` writes one line (false: the client is gone); `commit` sends the reply's head alone (false: gone);
  // `peer` (polled about every 50 ms, from before the request is queued) says whether the client's side has closed
  // (Fin: it sent its end, which a client that half-closes after its request also does) or the connection is broken
  // (Reset). Fin before anything was sent commits the reply: a half-closed client keeps reading it, and a client that
  // is gone answers that write with a reset, which ends the request.
  // A client must treat a stream without a final done or error line as failed: the server closes the connection when a
  // send makes no progress for the socket timeout (a reader that stalls), or at shutdown after its drain.
  // At most max(1, handlers / 2) streams at once (429 stream_limit): a client reading at playback speed holds its
  // handler for the length of the audio, and the other endpoints keep the rest.
  enum class Peer { Open, Fin, Reset };
  struct StreamSink {
    std::function<bool(const std::string&)> send;
    std::function<bool()> commit;
    std::function<Peer()> peer;
  };
  void speech_stream(std::shared_ptr<const Json> body, const StreamSink& sink) {
    if (!policy_.speech || settings_.engine_timeout < kSpeechMinEngineTimeout) throw ApiError(501, "speech_unsupported");
    const auto deadline = Clock::now() + seconds(settings_.request_timeout);
    auto request = std::make_shared<Speech>(validate_speech(*body));
    if (!request->stream) throw std::logic_error("speech_stream without stream");
    if (request->audio_input && !policy_.audio_input) throw ApiError(501, "audio_input_unsupported");
    if (request->image_input && !policy_.image_input) throw ApiError(501, "image_input_unsupported");
    // the cap after validation: a malformed request is a 400 whatever the load, never a retryable 429
    if (streams_.fetch_add(1) >= std::max(1, settings_.handlers / 2)) {
      --streams_;
      throw ApiError(429, "stream_limit");
    }
    struct Release {
      std::atomic<int>& n;
      ~Release() { --n; }
    } release{streams_};
    request->body = std::move(body);
    decode_images(*request, deadline, [&] { return sink.peer && sink.peer() == Peer::Reset; });
    frontend_call([&] { encode_speech(*request); }, deadline, request->image_input);
    auto lines = std::make_shared<Lines>();
    std::future<std::string> future;
    // past the deadline: true if the request goes on (it completed meanwhile, or, in the speech batch, was admitted: it
    // runs to its end), else it is abandoned and `deadline_code` is the error
    std::function<bool()> abandon_now;
    std::string deadline_code = "request_deadline_outcome_unknown";
    std::function<void()> leave;        // stop the generation at its next step
    if (policy_.speech_rows > 0) {
      auto t = std::make_shared<Ticket>();
      t->request = request, t->deadline = deadline, t->lines = lines;
      future = t->promise.get_future();
      enqueue_ticket(t);
      abandon_now = [t, &future] { return abandon_ticket(*t, future); };
      deadline_code = "queue_deadline";  // abandoned only while queued: it never started
      leave = [t] { t->cancel = true; };
    } else {
      Event event = std::make_shared<std::atomic<bool>>(false);
      const std::string key = "speech:" + std::to_string(++stream_serial_);  // this request's cancellation target
      auto run = [this, request, event, lines, key, deadline] {
        if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
        StreamState state(*this, *request, lines);
        const Backend::Progress progress = [&](const std::string& line) {
          if (line.size() > kMaxBody) throw WorkerFailure("progress frame exceeds limit");
          const Json p = strict_json(line);
          state.event(std::get<Json::Object>(p.value), 1);
        };
        Obj frame;
        frame.str("op", "speech").raw("chunks", request->chunks_json).str("mode", request->mode)
            .num("max_new_tokens", request->count).num("seed", request->seed).boolean("chunked", request->split)
            .boolean("stream", true);
        if (!request->audio_json.empty()) frame.raw("audio", request->audio_json);
        if (!request->images_json.empty()) frame.raw("images", request->images_json);
        const auto result = rpc(std::move(frame), {{key, event}}, false, kMaxBody, &progress, kMaxSpeechFrame);
        state.check(result);
        return speech_response(*request, result);
      };
      auto queued = enqueue(run, [] {}, nullptr, deadline, Kind::kExclusive);
      future = std::move(queued.second);
      auto job = queued.first;
      abandon_now = [job, &future] { return abandon(*job, future); };
      leave = [this, event, key] {
        event->store(true);
        try { send_cancel(key); } catch (const WorkerFailure&) {}  // then it finishes normally, unread
      };
    }
    const auto& send = sink.send;
    bool started = false, unbounded = false;
    const auto flush = [&] {
      std::deque<std::string> out;
      { std::lock_guard g(lines->m); out.swap(lines->lines); }
      for (auto& line : out) {
        if (!send(line)) return false;
        started = true;
      }
      return true;
    };
    for (;;) {
      const bool ready = future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      if (!flush()) { leave(); return; }  // every line queued before completion is flushed before the outcome
      if (ready) break;
      const Peer peer = sink.peer();
      if (peer == Peer::Reset) { leave(); return; }
      if (peer == Peer::Fin && !started) {
        if (!sink.commit()) { leave(); return; }
        started = true;
      }
      if (!unbounded && Clock::now() >= deadline) {
        if (abandon_now()) { unbounded = true; continue; }  // completed meanwhile, or runs on to its end
        leave();
        if (!started) throw ApiError(504, deadline_code);
        send(Obj().str("event", "error").str("error", deadline_code).done());
        return;
      }
      const auto tick = Clock::now() + std::chrono::milliseconds(50);
      std::unique_lock g(lines->m);
      lines->cv.wait_until(g, unbounded ? tick : std::min(deadline, tick), [&] { return !lines->lines.empty(); });
    }
    std::string done;
    try {
      done = future.get();
    } catch (const ApiError& e) {
      if (!started) throw;
      send(Obj().str("event", "error").str("error", e.code).done());
      return;
    }
    send(done);
  }


 private:
  // lfm3_server.Service._validate_generation (before any lease): fields, types, ranges, the messages
  // themselves and the sampling options.
  Generation validate_generation(const Json& item, std::shared_ptr<const Json> owner) const {
    const auto& o = object_of(item);
    static const std::set<std::string> allowed = {"session_id", "messages", "max_new_tokens", "sampling"};
    for (const auto& [key, value] : o)
      if (!allowed.count(key)) throw std::invalid_argument("invalid generation fields");
    for (const char* key : {"session_id", "messages", "max_new_tokens"})
      if (!o.count(key)) throw std::invalid_argument("invalid generation fields");
    if (o.count("sampling") && !std::holds_alternative<Json::Object>(o.at("sampling").value))
      throw std::invalid_argument("sampling must be an object");
    if (!std::holds_alternative<Json::Array>(o.at("messages").value)) throw std::invalid_argument("messages must be a list");
    Generation g;
    g.sid = session_id_of(o.at("session_id"));
    g.count = int(integer(o.at("max_new_tokens"), 1, 32768));
    g.body = std::move(owner);
    g.messages = &o.at("messages");
    (void)lfm3_text::parse_messages(*g.messages);
    g.sampling = sampling_options(o.count("sampling") ? &o.at("sampling") : nullptr, policy_.vocabulary);
    return g;
  }

  // lfm3_chat.make_request with the lease's capacity: render, tokenize, fit, placeholder rejection.
  std::vector<int> encode(const Generation& g, int capacity) const {
    const auto messages = lfm3_text::parse_messages(*g.messages);
    auto ids = lfm3_text::encode_messages(tokenizer_, messages, capacity, g.count);
    if (ids.empty() || ids.size() > size_t(capacity) || ids.size() + size_t(g.count) > size_t(capacity))
      throw std::invalid_argument("prompt plus generation exceeds native cache capacity; no truncation performed");
    for (const int id : ids) {
      if (id < 0 || id >= policy_.vocabulary) throw std::invalid_argument("input_ids out of range");
      if (std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), id))
        throw std::invalid_argument("multimodal placeholder input is unsupported in text-only mode");
    }
    return ids;
  }

  // The engine request make_request builds, minus capacity/suppress/stop/long-context; decoded text on.
  static Obj native_request(const Generation& g) {
    const auto& s = g.sampling;
    return Obj().ids("input_ids", g.input).num("max_new_tokens", g.count).boolean("do_sample", s.do_sample)
        .raw("temperature", exact_number(s.temperature)).num("top_k", s.top_k).raw("top_p", exact_number(s.top_p))
        .raw("min_p", exact_number(s.min_p)).raw("repetition_penalty", exact_number(s.repetition_penalty)).num("seed", s.seed)
        .boolean("return_text", true);
  }

  // lfm3_server.Service._generation_response plus the decoded text.
  Response generation_response(const Generation& g, const Json& result, const std::atomic<bool>& event, bool gpu_batching) const {
    const auto* o = std::get_if<Json::Object>(&result.value);
    if (!o || o->count("sampling_policy") || o->count("sampled_policy") || o->count("sampling"))
      throw WorkerFailure("ambiguous generation metadata");
    bool cancelled = false;
    if (o->count("cancelled")) {
      const auto* b = std::get_if<bool>(&o->at("cancelled").value);
      if (!b) throw WorkerFailure("invalid cancellation metadata");
      cancelled = *b;
    }
    const auto* cache_valid = o->count("cache_valid") ? std::get_if<bool>(&o->at("cache_valid").value) : nullptr;
    if (cancelled) {
      if (!event.load() || !cache_valid || *cache_valid) throw WorkerFailure("unsolicited or invalid cancellation");
    } else {
      const auto* finish = o->count("finish_reason") ? std::get_if<std::string>(&o->at("finish_reason").value) : nullptr;
      if ((o->count("cache_valid") && !(cache_valid && *cache_valid)) || (finish && *finish == "cancelled"))
        throw WorkerFailure("unsolicited cache invalidation");
    }
    std::vector<int> generated;
    const auto* ids = o->count("generated_ids") ? std::get_if<Json::Array>(&o->at("generated_ids").value) : nullptr;
    if (!ids || ids->size() > size_t(g.count)) throw WorkerFailure("invalid generation reply");
    for (const auto& id : *ids) {
      if (!is_int(id)) throw WorkerFailure("invalid generation reply");
      const double v = std::get<double>(id.value);
      if (!(v >= 0 && v <= double(policy_.vocabulary - 1))) throw WorkerFailure("invalid generation reply");
      generated.push_back(int(v));
    }
    const auto is_stop = [&](int t) { return std::binary_search(policy_.stops.begin(), policy_.stops.end(), t); };
    bool invalid = !session_matches_required(result, g.sid) || (generated.empty() && !cancelled);
    for (const int t : generated)
      if (std::binary_search(policy_.suppress.begin(), policy_.suppress.end(), t)) invalid = true;
    for (size_t i = 0; i + 1 < generated.size(); ++i)
      if (is_stop(generated[i])) invalid = true;
    if (!cancelled && !generated.empty() && generated.size() < size_t(g.count) && !is_stop(generated.back())) invalid = true;
    if (invalid) throw WorkerFailure("invalid generation reply");
    const auto* do_sample = o->count("do_sample") ? std::get_if<bool>(&o->at("do_sample").value) : nullptr;
    if (!do_sample || *do_sample != g.sampling.do_sample || !o->count("seed") || !is_int(o->at("seed"))
        || std::get<double>(o->at("seed").value) != double(g.sampling.seed))
      throw WorkerFailure("sampling metadata mismatch");
    const long long consumed = cancelled ? 0 : (long long)g.input.size() + (long long)generated.size() - 1;
    const auto exact_int = [&](const char* key, long long lo, long long hi) -> long long {
      if (!o->count(key) || !is_int(o->at(key))) throw WorkerFailure(std::string("invalid ") + key);
      const double v = std::get<double>(o->at(key).value);
      if (!(v >= double(lo) && v <= double(hi))) throw WorkerFailure(std::string("invalid ") + key);
      return (long long)v;
    };
    exact_int("consumed_tokens", consumed, consumed);
    const long long reused = exact_int("reused_tokens", 0, cancelled ? 0 : (long long)g.input.size());
    const auto* text = o->count("text") ? std::get_if<std::string>(&o->at("text").value) : nullptr;
    if (!text) throw WorkerFailure("missing decoded text");
    Response r;
    r.generated = generated;
    r.cancelled = cancelled;
    r.body.str("session_id", g.sid).str("text", *text).ids("generated_ids", generated).num("reused_tokens", reused)
        .num("consumed_tokens", consumed).boolean("do_sample", g.sampling.do_sample).num("seed", g.sampling.seed)
        .boolean("gpu_batching", gpu_batching).boolean("cancelled", cancelled).boolean("cache_valid", !cancelled)
        .str("finish_reason", cancelled ? "cancelled" : is_stop(generated.back()) ? "stop" : "length");
    return r;
  }
  static bool session_matches_required(const Json& j, const std::string& sid) {
    if (!j.has("session_id")) return false;
    const auto* s = std::get_if<std::string>(&j.at("session_id").value);
    return s && *s == sid;
  }

  // One serialized CPU frontend job at a time (lfm3_server.Service._frontend_call, wait=False):
  // a second concurrent encode is refused rather than queued. `wait` (image requests, validation/image-queue-v1: their
  // images already decoded, only rendering and tokenizing is left) waits for it within the deadline instead.
  template <class F>
  void frontend_call(F&& function, Clock::time_point deadline, bool wait = false) {
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
    std::unique_lock guard(frontend_lock_, std::defer_lock);
    if (wait ? !guard.try_lock_until(deadline) : !guard.try_lock())
      throw wait ? ApiError(504, "frontend_deadline") : ApiError(429, "frontend_busy");
    if (stopping_) throw ApiError(503, "worker_unavailable");
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
    function();
    if (Clock::now() >= deadline) throw ApiError(504, "frontend_deadline");
  }

  int lease(const std::string& sid, const std::string& operation, Event event) {
    if (!valid_session_id(sid)) throw std::invalid_argument("invalid session ID");
    std::lock_guard g(lock_);
    if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) throw ApiError(404, "unknown_session");
    if (it->second.busy) throw ApiError(409, "session_busy");
    it->second.busy = true;
    it->second.operation = operation;
    it->second.cancel_event = std::move(event);
    return it->second.capacity;
  }

  void release(const std::string& sid, const Event& event) {
    std::lock_guard g(lock_);
    auto it = sessions_.find(sid);
    if (it != sessions_.end() && (!event || it->second.cancel_event == event)) {
      it->second.busy = false;
      it->second.operation.clear();
      it->second.cancel_event = nullptr;
      it->second.last_used = Clock::now();
    }
  }

  void write_frame(std::uint64_t token) {
    unsigned char frame[8];
    for (int i = 0; i < 8; ++i) frame[i] = static_cast<unsigned char>(token >> (8 * i));
    ssize_t n;
    do n = ::write(backend_.cancellation_fd(), frame, sizeof frame);
    while (n < 0 && errno == EINTR);
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) throw WorkerFailure("cancellation control queue full");
    if (n < 0) throw WorkerFailure("cancellation control pipe failed");
    if (n != 8) throw WorkerFailure("partial cancellation control write");
  }

  // lfm3_worker_client.WorkerClient.cancel: request only; the RPC reply acknowledges termination.
  bool send_cancel(const std::string& sid) {
    std::lock_guard g(active_lock_);
    auto it = active_.find(sid);
    if (backend_closed_ || it == active_.end()) return false;
    if (it->second.sent) return true;
    write_frame(it->second.token);
    it->second.sent = true;
    return true;
  }

  // lfm3_worker_client.WorkerClient.call + Service._rpc: owner thread only. Any failure is fatal.
  Json rpc(Obj request, std::vector<std::pair<std::string, Event>> targets = {}, bool batch = false,
           size_t max_reply = kMaxBody, const Backend::Progress* progress = nullptr, size_t max_request = kMaxBody) {
    if (backend_closed_) throw WorkerFailure("worker closed");
    if (sequence_ >= 2147483647) throw WorkerFailure("RPC sequence exhausted; restart required");
    const std::uint64_t id = ++sequence_;
    const auto line = request.num("rpc_id", (long long)id).done();
    if (line.size() + 1 > max_request) throw WorkerFailure("request frame exceeds limit");
    {
      std::lock_guard g(active_lock_);
      active_.clear();
      for (size_t i = 0; i < targets.size(); ++i)
        active_[targets[i].first] = Active{batch ? ((std::uint64_t(i + 1) << 32) | id) : id, false};
    }
    struct Clear {
      Service* s;
      ~Clear() { std::lock_guard g(s->active_lock_); s->active_.clear(); }
    } clear{this};
    for (const auto& [sid, event] : targets)
      if (event && event->load() && !send_cancel(sid)) throw WorkerFailure("pre-dispatch cancellation was not admitted");
    std::string reply;
    call_started_ = Clock::now().time_since_epoch().count();
    in_call_ = true;
    try {
      reply = progress ? backend_.call_streaming(line, *progress) : backend_.call(line);
    } catch (const std::exception& e) {
      in_call_ = false;
      throw WorkerFailure(std::string("worker RPC failed: ") + e.what());
    }
    in_call_ = false;
    if (timed_out_) throw WorkerFailure("engine call exceeded the engine timeout");
    while (!reply.empty() && reply.back() == '\n') reply.pop_back();
    if (reply.size() > max_reply) throw WorkerFailure("worker frame exceeds limit");
    Json result;
    try { result = strict_json(reply); }
    catch (const std::exception&) { throw WorkerFailure("invalid worker reply"); }
    const auto* ok = result.has("ok") ? std::get_if<bool>(&result.at("ok").value) : nullptr;
    if (!ok || !result.has("rpc_id") || !is_int(result.at("rpc_id")) || std::get<double>(result.at("rpc_id").value) != double(id))
      throw WorkerFailure("invalid worker reply");
    // Even native execution errors are fail-closed, rather than guessing which CUDA errors might leave
    // the process or another cache usable.
    if (!*ok) throw WorkerFailure("native operation rejected");
    std::get<Json::Object>(result.value).erase("rpc_id");
    return result;
  }

  void close_allocation(const std::string& sid) {
    const auto result = rpc(Obj().str("op", "close").str("session_id", sid));
    if (!flag(result, "closed") || !session_matches(result, sid)) throw WorkerFailure("invalid close reply");
    std::lock_guard g(lock_);
    sessions_.erase(sid);
  }

  void fatal() {
    {
      std::lock_guard g(lock_);
      failed_ = true;
      sessions_.clear();
    }
    bool failed_cleanup = false;
    {
      std::lock_guard g(active_lock_);
      if (backend_closed_) return;
      backend_closed_ = true;
      try { backend_.close(); }
      catch (...) { failed_cleanup = true; }
    }
    if (failed_cleanup) {
      std::lock_guard g(lock_);
      cleanup_failed_ = true;
    }
  }

  // Concurrent speech: a ticket joins the speech queue (429 queue_full when queue_size requests, text or speech, wait).
  void enqueue_ticket(const std::shared_ptr<Ticket>& t) {
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      if (Clock::now() >= t->deadline) throw ApiError(504, "queue_deadline");
      std::lock_guard q(queue_lock_);
      if (int(queue_.size() + speech_queue_.size()) >= settings_.queue_size) throw ApiError(429, "queue_full");
      t->id = ++ticket_serial_;
      t->arrived = Clock::now();
      speech_queue_.push_back(t);
    }
    queue_cv_.notify_all();
  }
  // Past its deadline: marks a still-queued ticket abandoned (false: the caller answers 504 queue_deadline); true when it
  // settled meanwhile or was admitted (it then runs to its end: wait for it).
  static bool abandon_ticket(Ticket& t, std::future<std::string>& future) {
    std::lock_guard g(t.completion);
    if (t.admitted || future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) return true;
    t.abandoned = true;
    t.cancel = true;
    return false;
  }
  // Settle a ticket once: its reply, or an error.
  static void settle(Ticket& t, std::string reply) {
    std::lock_guard g(t.completion);
    if (t.settled) return;
    t.settled = true;
    t.promise.set_value(std::move(reply));  // an admitted ticket's reply stands, whatever its deadline
  }
  static void settle(Ticket& t, const ApiError& e) {
    std::lock_guard g(t.completion);
    if (t.settled) return;
    t.settled = true;
    t.promise.set_exception(std::make_exception_ptr(e));
  }
  // Whether queued speech may be admitted (caller holds queue_lock_): unless an exclusive job (kExclusive) that
  // arrived first waits in the queue, for which the rows drain. Session operations and text generations
  // run between speech steps (validation/text-wait-v1), so they hold nothing back.
  bool speech_first() const {
    if (speech_queue_.empty()) return false;
    for (const auto& job : queue_)
      if (job->kind == Kind::kExclusive) return speech_queue_.front()->arrived < job->arrived;
    return true;
  }
  // One cycle of the speech batch (engine-owner thread): drop queued tickets nobody will read; admit queued tickets
  // into free rows while they arrived before any waiting job (a job that arrived first waits only for the running rows:
  // the batch drains for it); then one step of every row, with the cancellations of tickets whose clients are gone or
  // past their deadline (all of them at shutdown). Any engine failure fails the service closed and every ticket with it.
  void speech_cycle() {
    try {
      {  // queued tickets nobody will read (the client left, or it is past its deadline) give up their places
        std::deque<std::shared_ptr<Ticket>> gone;
        {
          std::lock_guard q(queue_lock_);
          const auto now = Clock::now();
          for (auto it = speech_queue_.begin(); it != speech_queue_.end();)
            if ((*it)->cancel || now >= (*it)->deadline) gone.push_back(*it), it = speech_queue_.erase(it);
            else ++it;
        }
        for (auto& t : gone) settle(*t, ApiError(504, "queue_deadline"));
      }
      for (;;) {
        std::shared_ptr<Ticket> t;
        {
          std::lock_guard q(queue_lock_);
          if (!speech_first() || int(running_.size()) >= policy_.speech_rows) break;
          t = speech_queue_.front();
          speech_queue_.pop_front();
        }
        if (failed()) { settle(*t, ApiError(503, "worker_unavailable")); continue; }
        {
          std::lock_guard g(t->completion);
          if (t->abandoned || t->cancel || Clock::now() >= t->deadline) {
            t->settled = true;
            t->promise.set_exception(std::make_exception_ptr(ApiError(504, "queue_deadline")));
            continue;
          }
          t->admitted = true;
        }
        const Speech& r = *t->request;
        Obj frame;
        frame.str("op", "speech_admit").num("ticket", t->id).raw("chunks", r.chunks_json).str("mode", r.mode)
            .num("max_new_tokens", r.count).num("seed", r.seed).boolean("chunked", r.split).boolean("stream", r.stream);
        if (!r.audio_json.empty()) frame.raw("audio", r.audio_json);
        if (!r.images_json.empty()) frame.raw("images", r.images_json);
        if (r.stream) t->state = std::make_unique<StreamState>(*this, r, t->lines);
        running_[t->id] = t;  // before the call: a failure fails it with the rest
        const Backend::Progress none = [](const std::string&) { throw WorkerFailure("unexpected speech progress"); };
        const auto reply = rpc(std::move(frame), {}, false, kMaxBody, &none, kMaxSpeechFrame);
        if (!reply.has("ticket") || reply.at("ticket").integer() != t->id) throw WorkerFailure("invalid speech admit reply");
      }
      if (running_.empty()) return;
      std::string cancel = "[";
      for (auto& [id, t] : running_)
        if (!t->cancel_sent && (t->cancel || cancel_all_)) {
          cancel += (cancel.size() > 1 ? "," : "") + std::to_string(id);
          t->cancel_sent = true;
        }
      cancel += "]";
      const Backend::Progress progress = [&](const std::string& line) {
        if (line.size() > kMaxSpeechResponse) throw WorkerFailure("progress frame exceeds limit");
        const Json p = strict_json(line);
        const auto& o = std::get<Json::Object>(p.value);
        const auto it = o.count("ticket") ? running_.find(o.at("ticket").integer()) : running_.end();
        if (it == running_.end()) throw WorkerFailure("invalid speech progress");
        const auto t = it->second;
        // an event line is as bounded as on the single-request path; only a finished reply (the WAV) may be larger
        if (!o.count("finished") && line.size() > kMaxBody) throw WorkerFailure("progress frame exceeds limit");
        if (o.count("finished")) {
          if (o.size() != 2) throw WorkerFailure("invalid speech progress");
          const Json& result = o.at("finished");
          if (t->state) t->state->check(result);
          std::string reply = speech_response(*t->request, result);
          running_.erase(it);
          settle(*t, std::move(reply));
          return;
        }
        if (!t->state) throw WorkerFailure("invalid speech progress");  // events only for a streamed ticket
        t->state->event(o, 2);
      };
      Obj frame;
      frame.str("op", "speech_step");
      if (cancel.size() > 2) frame.raw("cancel", cancel);
      const auto reply = rpc(std::move(frame), {}, false, kMaxBody, &progress);
      if (!reply.has("busy") || reply.at("busy").integer() != (long long)running_.size())
        throw WorkerFailure("invalid speech step reply");
    } catch (...) {
      fatal();
      std::deque<std::shared_ptr<Ticket>> queued;
      {
        std::lock_guard q(queue_lock_);
        queued.swap(speech_queue_);
      }
      for (auto& [id, t] : running_) settle(*t, ApiError(503, "worker_unavailable"));
      for (auto& t : queued) settle(*t, ApiError(503, "worker_unavailable"));
      running_.clear();
    }
  }

  std::string submit(std::function<std::string()> function, std::function<void()> cleanup, std::function<void()> discard,
                     std::optional<Clock::time_point> when = std::nullopt, Kind kind = Kind::kShort,
                     std::shared_ptr<Generation> text = nullptr, Event event = nullptr) {
    const auto deadline = when ? *when : Clock::now() + seconds(settings_.request_timeout);
    auto [job, future] = enqueue(std::move(function), std::move(cleanup), std::move(discard), deadline, kind,
                                 std::move(text), std::move(event));
    if (future.wait_until(deadline) == std::future_status::ready) return future.get();
    if (abandon(*job, future)) return future.get();
    // Do not release a running generation lease or pretend cancellation.
    throw ApiError(504, "request_deadline_outcome_unknown");
  }
  // Past its deadline: marks the job abandoned, unless it completed meanwhile (then false: its result stands).
  static bool abandon(Job& job, std::future<std::string>& future) {
    std::lock_guard g(job.completion);
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) return true;
    job.abandoned = true;
    return false;
  }
  std::pair<std::shared_ptr<Job>, std::future<std::string>> enqueue(std::function<std::string()> function,
                                                                    std::function<void()> cleanup,
                                                                    std::function<void()> discard,
                                                                    Clock::time_point deadline,
                                                                    Kind kind = Kind::kShort,
                                                                    std::shared_ptr<Generation> text = nullptr,
                                                                    Event event = nullptr) {
    auto job = std::make_shared<Job>();
    job->function = std::move(function);
    job->kind = kind;
    job->text = std::move(text);
    job->event = std::move(event);
    job->deadline = deadline;
    job->arrived = Clock::now();
    job->cleanup = cleanup;
    job->discard = std::move(discard);
    auto future = job->promise.get_future();
    try {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) throw ApiError(503, "worker_unavailable");
      if (Clock::now() >= deadline) throw ApiError(504, "queue_deadline");
      std::lock_guard q(queue_lock_);
      // one limit for every waiting request, text and speech alike
      if (int(queue_.size() + speech_queue_.size()) >= settings_.queue_size) throw ApiError(429, "queue_full");
      queue_.push_back(job);
    } catch (...) {
      cleanup();
      throw;
    }
    queue_cv_.notify_all();
    return {job, std::move(future)};
  }

  // lfm3_worker_client's per-RPC deadline: an engine call running past engine_timeout fails the
  // service closed (503 for everything, sessions dropped) and requests cancellation of whatever is
  // running. The late reply is then refused as a failure.
  void watch() {
    std::unique_lock g(done_lock_);
    while (!watchdog_stop_) {
      done_cv_.wait_for(g, std::chrono::milliseconds(100));
      if (watchdog_stop_ || !in_call_ || timed_out_) continue;
      const auto started = Clock::time_point(Clock::duration(call_started_.load()));
      if (Clock::now() - started < seconds(settings_.engine_timeout)) continue;
      timed_out_ = true;
      {
        std::lock_guard l(lock_);
        failed_ = true;
        sessions_.clear();
      }
      std::lock_guard a(active_lock_);
      for (auto& [sid, active] : active_) {
        if (active.sent) continue;
        try { write_frame(active.token); active.sent = true; } catch (const WorkerFailure&) {}
      }
    }
  }

  void reap_idle() {
    std::vector<std::string> expired;
    {
      std::lock_guard g(lock_);
      if (failed_ || stopping_) return;
      const auto now = Clock::now();
      for (auto& [sid, s] : sessions_)
        if (!s.busy && now - s.last_used >= seconds(settings_.session_idle_timeout)) {
          s.busy = true;
          expired.push_back(sid);
        }
    }
    for (const auto& sid : expired) close_allocation(sid);
  }

  // The engine owner (validation/text-wait-v1). Each pass, in this order:
  //   1. close idle sessions;
  //   2. at most one queued job, the first in arrival order that can run: a session operation (kShort) at once; a text
  //      generation (kText) when no other is active; an exclusive job (kExclusive) once no speech row and no text
  //      generation is running (nothing behind it runs before it);
  //   3. one speech cycle when rows run or queued speech may be admitted (speech_first);
  //   4. one text slice when a text generation is active: one token while speech rows run, a long slice otherwise.
  // Every engine call is still made by this thread, one at a time; a session's text and the speech batch share no
  // engine state but the read-only weights, so the order of their calls changes no byte (TEXT_WAIT_DESIGN.md §2.D).
  void run() {
    for (;;) {
      {
        std::unique_lock q(queue_lock_);
        if (stopping_ && queue_.empty() && speech_queue_.empty() && running_.empty() && !text_) return;
      }
      try { reap_idle(); }
      catch (...) { fatal(); }
      std::shared_ptr<Job> job;
      bool speech = false;
      {
        std::unique_lock q(queue_lock_);
        queue_cv_.wait_for(q, std::chrono::milliseconds(50), [&] {
          return !queue_.empty() || !speech_queue_.empty() || !running_.empty() || text_ || stopping_.load();
        });
        // the first job, in arrival order, that can run now: a text generation waiting for the active one does not hold
        // back a later session operation (a session's lease already orders its own operations), but an exclusive job
        // holds back everything behind it, so it cannot be starved
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
          const auto kind = (*it)->kind;
          if (kind == Kind::kText && text_) continue;
          if (kind == Kind::kExclusive && (!running_.empty() || text_ ||
                                           (!speech_queue_.empty() && speech_queue_.front()->arrived < (*it)->arrived)))
            break;  // the rows (and earlier speech) first: the batch drains for it
          job = *it;
          queue_.erase(it);
          break;
        }
        speech = !running_.empty() || speech_first();
      }
      if (job) {
        if (job->kind == Kind::kText) begin_text(job);
        else run_job(job);
      }
      if (speech) speech_cycle();
      if (text_) text_cycle();
    }
  }

  // A kShort or kExclusive job: its function, once.
  void run_job(const std::shared_ptr<Job>& job) {
    std::string result;
    try {
      if (failed()) throw ApiError(503, "worker_unavailable");
      if (Clock::now() >= job->deadline) throw ApiError(504, "queue_deadline");
      result = job->function();
    } catch (const ApiError&) {
      job->cleanup();
      job->promise.set_exception(std::current_exception());
      return;
    } catch (...) {
      fatal();
      job->cleanup();
      job->promise.set_exception(std::make_exception_ptr(ApiError(503, "worker_unavailable")));
      return;
    }
    complete(job, std::move(result));
  }

  // Publish completion and observe abandonment atomically. Only this owner thread may release an abandoned native
  // allocation.
  void complete(const std::shared_ptr<Job>& job, std::string result) {
    bool abandoned = false;
    {
      std::lock_guard g(job->completion);
      const bool expired = job->abandoned || Clock::now() >= job->deadline;
      abandoned = job->discard && expired;
      if (expired) {
        if (!abandoned) job->cleanup();
        job->promise.set_exception(std::make_exception_ptr(ApiError(504, "request_deadline_outcome_unknown")));
      } else {
        job->cleanup();
        job->promise.set_value(std::move(result));
      }
    }
    if (abandoned) {
      try { job->discard(); }
      catch (...) { fatal(); }
      job->cleanup();
    }
  }

  // A text generation's reply from the engine's final reply (op generate's, from text_begin or text_step).
  std::string text_reply(const Job& job, const Json& result) {
    auto response = generation_response(*job.text, result, *job.event, false);
    auto out = response.body.done();
    if (out.size() > kMaxResponse) throw WorkerFailure("response exceeds limit");
    return out;
  }

  // A kText job leaves the queue: checks, then op text_begin (the prefill, one call). Cancelled before it starts,
  // the session is reset instead, as before. A failure fails the service closed (any engine refusal does).
  void begin_text(const std::shared_ptr<Job>& job) {
    const auto& request = *job->text;
    std::string result;
    try {
      if (failed()) throw ApiError(503, "worker_unavailable");
      if (Clock::now() >= job->deadline) throw ApiError(504, "queue_deadline");
      if (job->event->load()) {
        const auto reset = rpc(Obj().str("op", "reset").str("session_id", request.sid));
        if (!flag(reset, "reset") || !session_matches(reset, request.sid)) throw WorkerFailure("invalid cancellation reset reply");
        result = Obj().str("session_id", request.sid).str("text", "").ids("generated_ids", {}).num("consumed_tokens", 0)
            .num("reused_tokens", 0).boolean("cancelled", true).str("finish_reason", "cancelled").boolean("cache_valid", false)
            .boolean("do_sample", request.sampling.do_sample).num("seed", request.sampling.seed).boolean("gpu_batching", false)
            .done();
      } else {
        const auto reply = rpc(native_request(request).str("op", "text_begin").str("session_id", request.sid),
                               {{request.sid, job->event}});
        if (flag(reply, "begun")) {
          if (!session_matches(reply, request.sid) || reply.has("generated_ids")) throw WorkerFailure("invalid text_begin reply");
          text_ = job;
          return;
        }
        result = text_reply(*job, reply);  // cancelled during its prefill: the generation's final reply
      }
    } catch (const ApiError&) {
      job->cleanup();
      job->promise.set_exception(std::current_exception());
      return;
    } catch (...) {
      fatal();
      job->cleanup();
      job->promise.set_exception(std::make_exception_ptr(ApiError(503, "worker_unavailable")));
      return;
    }
    complete(job, std::move(result));
  }

  // One slice of the active text generation: one token while speech rows run, settings_.text_slice otherwise. A client's
  // cancellation, the job's deadline, its abandonment or shutdown cancels it at this slice (op generate's cancelled
  // reply: the cache reset); /cancel can also reach a slice in progress through the pipe (its rpc's targets).
  void text_cycle() {
    const auto job = text_;
    const auto& request = *job->text;
    std::string result;
    try {
      // shutdown marks the service failed as it sets cancel_all_: the generation is still cancelled at this slice
      // (its cancelled reply, as /cancel would give), while the engine is open; any other failure answers 503
      if (failed() && !cancel_all_) throw ApiError(503, "worker_unavailable");
      bool abandoned;
      {
        std::lock_guard g(job->completion);
        abandoned = job->abandoned;
      }
      // the owner's own cancellations are requested as /cancel requests one (the event set first), so the cancelled
      // reply is a solicited one (generation_response refuses any other)
      if (cancel_all_ || abandoned || Clock::now() >= job->deadline) job->event->store(true);
      const bool cancel = job->event->load();
      const int tokens = running_.empty() ? settings_.text_slice : 1;
      const auto reply = rpc(Obj().str("op", "text_step").str("session_id", request.sid).num("tokens", tokens)
                                 .boolean("cancel", cancel),
                             {{request.sid, job->event}});
      if (flag(reply, "busy")) {
        if (!session_matches(reply, request.sid) || reply.has("generated_ids")) throw WorkerFailure("invalid text_step reply");
        return;
      }
      result = text_reply(*job, reply);
    } catch (const ApiError&) {
      text_.reset();
      job->cleanup();
      job->promise.set_exception(std::current_exception());
      return;
    } catch (...) {
      text_.reset();
      fatal();
      job->cleanup();
      job->promise.set_exception(std::make_exception_ptr(ApiError(503, "worker_unavailable")));
      return;
    }
    text_.reset();
    complete(job, std::move(result));
  }

  Backend& backend_;
  const lfm3_text::Tokenizer& tokenizer_;
  const Policy policy_;
  Settings settings_;
  std::mutex lock_;
  std::map<std::string, Session> sessions_;
  bool failed_ = false, cleanup_failed_ = false;
  std::atomic<bool> stopping_{false};
  std::timed_mutex frontend_lock_;
  static constexpr int kImageDecodeSlots = 2;  // images decoded at once (decode_images)
  std::mutex image_slots_lock_;
  std::condition_variable image_slots_cv_;
  int image_slots_ = kImageDecodeSlots;
  lfm::LockedLruBytes<int> image_token_cache_;  // image-cache-v1: an image part's base64 text -> its token count
  std::mutex queue_lock_;
  std::condition_variable queue_cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::mutex active_lock_;
  std::map<std::string, Active> active_;
  bool backend_closed_ = false;  // guarded by active_lock_
  std::uint64_t sequence_ = 0;   // owner thread only
  std::atomic<std::uint64_t> stream_serial_{0};
  std::atomic<int> streams_{0};  // streamed /speech requests in progress
  // concurrent speech: the queue (guarded by queue_lock_), the tickets running in rows (engine-owner thread; read
  // under queue_lock_ by that thread only), and shutdown's cancellation of them all
  std::deque<std::shared_ptr<Ticket>> speech_queue_;
  std::map<long long, std::shared_ptr<Ticket>> running_;
  // the active text generation (engine-owner thread only; read under queue_lock_ by that thread only)
  std::shared_ptr<Job> text_;
  std::atomic<long long> ticket_serial_{0};
  std::atomic<bool> cancel_all_{false};
  std::atomic<bool> in_call_{false}, timed_out_{false};
  std::atomic<Clock::rep> call_started_{0};
  std::mutex done_lock_;
  std::condition_variable done_cv_;
  bool owner_done_ = false, watchdog_stop_ = false;  // guarded by done_lock_
  std::thread owner_, watchdog_;
};

// ------------------------------------------------------------------ HTTP

inline const char* reason_phrase(int status) {
  switch (status) {
    case 200: return "OK"; case 202: return "Accepted"; case 400: return "Bad Request"; case 401: return "Unauthorized";
    case 404: return "Not Found"; case 408: return "Request Timeout"; case 409: return "Conflict";
    case 413: return "Request Entity Too Large"; case 414: return "Request-URI Too Long"; case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests"; case 431: return "Request Header Fields Too Large";
    case 499: return "Client Closed Request";
    case 500: return "Internal Server Error"; case 501: return "Not Implemented"; case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout"; case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

inline std::string lower(std::string s) {
  for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return s;
}

struct Request {
  std::string method, path;
  std::vector<std::pair<std::string, std::string>> headers;  // lower-case name, raw value
  std::vector<std::string> all(const std::string& name) const {
    std::vector<std::string> v;
    for (const auto& [k, value] : headers) if (k == name) v.push_back(value);
    return v;
  }
};

// One accepted connection: HTTP/1.0, one request, then close (Handler with protocol_version HTTP/1.0).
class Connection {
 public:
  Connection(int fd, Service& service, std::string token) : fd_(fd), service_(service), token_(std::move(token)) {
    const double t = service.settings().socket_timeout;
    timeval tv{};
    tv.tv_sec = time_t(t);
    tv.tv_usec = suseconds_t((t - double(tv.tv_sec)) * 1e6);
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  }
  ~Connection() {
    ::shutdown(fd_, SHUT_WR);
    ::close(fd_);
  }

  void handle() {
    Request request;
    // The whole request line and header block must arrive within one socket timeout; a peer trickling
    // one byte per timeout must not hold a handler forever.
    const auto header_deadline = Clock::now() + seconds(service_.settings().socket_timeout);
    int status = 0;
    try {
      status = read_head(request, header_deadline);
    } catch (const std::exception&) {
      return;  // timeout or reset while reading headers: close without a reply
    }
    if (status == -1) return;
    if (status) { reply(status, error_body(status == 431 ? "header_limit" : status == 414 ? "request_line_limit"
                                           : status == 505 ? "http_version" : "bad_request")); return; }
    if (request.method == "GET") { get(request); return; }
    if (request.method == "POST") { post(request); return; }
    reply(501, error_body("unsupported_method"));
  }

 private:
  // One raw line as BufferedReader.readline(limit) returns it: up to and including '\n', at most
  // `limit` bytes, shorter at end of stream (empty: the stream ended).
  std::string read_line(size_t limit, Clock::time_point deadline) {
    std::string line;
    char c;
    while (line.size() < limit && read_byte(c, deadline)) {
      line += c;
      if (c == '\n') break;
    }
    return line;
  }

  // BaseHTTPRequestHandler.handle_one_request + parse_request, and http.client.parse_headers followed
  // by email.feedparser (compat32) header parsing. Returns 0 on success, -1 to close silently, or an
  // HTTP status to refuse with.
  int read_head(Request& r, Clock::time_point deadline) {
    const auto request_line = read_line(65537, deadline);
    if (request_line.empty()) return -1;
    if (request_line.size() > 65536) return 414;
    std::string line = request_line;
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    // str.split() on the latin-1 decoded line: ASCII whitespace, \x1c-\x1f, \x85 and \xa0.
    std::vector<std::string> words;
    {
      std::string w;
      for (const unsigned char c : line) {
        const bool space = c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0;
        if (space) { if (!w.empty()) { words.push_back(w); w.clear(); } }
        else w += char(c);
      }
      if (!w.empty()) words.push_back(w);
    }
    if (words.empty()) return -1;
    if (words.size() != 3) return 400;  // two words is HTTP/0.9, which only http.server's GET path accepts
    const auto& version = words[2];
    if (version.rfind("HTTP/", 0) != 0) return 400;
    const auto number = version.substr(5);
    const auto dot = number.find('.');
    if (dot == std::string::npos || number.find('.', dot + 1) != std::string::npos) return 400;
    const auto major = number.substr(0, dot), minor = number.substr(dot + 1);
    for (const auto& part : {major, minor}) {
      if (part.empty() || part.size() > 10) return 400;
      for (const char c : part) if (c < '0' || c > '9') return 400;
    }
    if (std::stoll(major) >= 2) return 505;
    r.method = words[0];
    r.path = words[1];
    if (r.path.rfind("//", 0) == 0) {
      const auto first = r.path.find_first_not_of('/');
      r.path = "/" + (first == std::string::npos ? std::string() : r.path.substr(first));
    }
    // http.client.parse_headers: raw lines until a bare line ending or end of stream.
    std::string block;
    for (int count = 0;;) {
      const auto raw = read_line(65537, deadline);
      if (raw.size() > 65536) return 431;
      if (++count > 100) return 431;  // http.client counts the terminating line too: 99 headers at most
      if (raw.empty() || raw == "\r\n" || raw == "\n") break;
      block += raw;
    }
    // email.feedparser: logical lines split on \r\n, \r or \n (each keeps its ending).
    std::vector<std::string> lines;
    for (size_t i = 0; i < block.size();) {
      size_t j = i;
      while (j < block.size() && block[j] != '\r' && block[j] != '\n') ++j;
      if (j < block.size()) j += (block[j] == '\r' && j + 1 < block.size() && block[j + 1] == '\n') ? 2 : 1;
      lines.push_back(block.substr(i, j - i));
      i = j;
    }
    const auto header_start = [](const std::string& l) {
      // headerRE: '^(From |[\041-\071\073-\176]*:|[\t ])'
      if (l.rfind("From ", 0) == 0 || (!l.empty() && (l[0] == ' ' || l[0] == '\t'))) return true;
      for (const unsigned char c : l) {
        if (c == ':') return true;
        if (c < 0x21 || c > 0x7e) return false;
      }
      return false;
    };
    std::vector<std::vector<std::string>> fields;  // source lines of each header
    for (size_t i = 0; i < lines.size(); ++i) {
      const auto& l = lines[i];
      if (!header_start(l)) break;  // a blank or malformed line ends the headers; the rest is body
      if (l[0] == ' ' || l[0] == '\t') {
        if (!fields.empty()) fields.back().push_back(l);  // a leading continuation is dropped
        continue;
      }
      if (l.rfind("From ", 0) == 0) continue;  // unix-from line (or defect): not a header
      if (l[0] == ':') continue;               // missing header name: dropped, parsing continues
      fields.push_back({l});
    }
    for (const auto& source : fields) {
      const auto colon = source[0].find(':');
      std::string value = source[0].substr(colon + 1);
      value.erase(0, std::min(value.size(), value.find_first_not_of(" \t")));
      for (size_t i = 1; i < source.size(); ++i) value += source[i];
      while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) value.pop_back();
      r.headers.emplace_back(lower(source[0].substr(0, colon)), value);
    }
    return 0;
  }

  bool read_byte(char& c, Clock::time_point deadline) {
    if (pos_ == buffer_.size()) {
      buffer_.clear();
      pos_ = 0;
      if (!fill(deadline, 4096)) return false;
    }
    c = buffer_[pos_++];
    return true;
  }
  // Reads at least one byte before `deadline`; false on orderly EOF; throws on timeout or error.
  bool fill(Clock::time_point deadline, size_t want) {
    for (;;) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
      if (remaining <= 0) throw std::runtime_error("socket deadline");
      pollfd p{fd_, POLLIN, 0};
      const int ready = ::poll(&p, 1, int(std::min<long long>(remaining, 1 << 30)));
      if (ready < 0 && errno == EINTR) continue;
      if (ready <= 0) throw std::runtime_error("socket deadline");
      char block[65536];
      const auto n = ::recv(fd_, block, std::min(want, sizeof block), 0);
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (n < 0) throw std::runtime_error("socket error");
      if (n == 0) return false;
      buffer_.append(block, size_t(n));
      return true;
    }
  }

  static std::string head(int status) {
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    ::gmtime_r(&now, &tm);
    std::strftime(date, sizeof date, "%a, %d %b %Y %H:%M:%S GMT", &tm);
    std::string out = "HTTP/1.0 " + std::to_string(status) + " " + reason_phrase(status) + "\r\n";
    out += "Server: LFM3Local \r\nDate: ";  // http.server: server_version + " " + sys_version ("")
    out += date;
    return out;
  }
  // false: the peer is gone or not reading (a send failed or timed out).
  // `until` (optional) bounds the whole send: SO_SNDTIMEO restarts whenever a byte moves, so a reader taking a few
  // bytes a second could hold the handler for hours (audio-assurance-v1 red-team).
  bool send_all(const std::string& out, std::optional<Clock::time_point> until = std::nullopt) {
    size_t sent = 0;
    while (sent < out.size()) {
      if (until && Clock::now() >= *until) return false;
      const auto n = ::send(fd_, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return false;
      sent += size_t(n);
    }
    return true;
  }
  void reply(int status, std::string body, size_t limit = kMaxResponse) {
    if (body.size() > limit) { status = 500; body = "{\"error\":\"response_limit\"}"; }
    std::string out = head(status);
    out += "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) +
           "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + body;
    // a disconnected or non-reading peer cannot receive a second reply; 60 s plus a second per MiB is far beyond any
    // reader that is reading
    send_all(out, Clock::now() + std::chrono::seconds(60 + (out.size() >> 20)));
  }
  // The client reset its connection (an unstreamed request's `gone`); a half-close (FIN only) is not a reset.
  bool reset_by_peer() const {
    pollfd p{fd_, POLLRDHUP, 0};
    return ::poll(&p, 1, 0) > 0 && (p.revents & (POLLHUP | POLLERR)) != 0;
  }
  // A streamed /speech reply: 200 and newline-delimited JSON, no Content-Length (HTTP/1.0: the body ends when the
  // connection closes). The status line goes out with the first event.
  void stream_speech(std::shared_ptr<const Json> parsed) {
    bool headed = false;
    const auto header = [&] {
      std::string out;
      if (!headed) {
        out = head(200) + "\r\nContent-Type: application/x-ndjson; charset=utf-8\r\nCache-Control: no-store\r\n"
                          "Connection: close\r\n\r\n";
        headed = true;
      }
      return out;
    };
    Service::StreamSink sink;
    sink.send = [&](const std::string& line) {
      std::string out = header();
      out += line;
      out += '\n';
      return send_all(out);
    };
    sink.commit = [&] { return send_all(header()); };
    sink.peer = [&] {
      pollfd p{fd_, POLLRDHUP, 0};
      if (::poll(&p, 1, 0) <= 0) return Service::Peer::Open;
      if (p.revents & (POLLHUP | POLLERR)) return Service::Peer::Reset;
      return p.revents & POLLRDHUP ? Service::Peer::Fin : Service::Peer::Open;
    };
    service_.speech_stream(std::move(parsed), sink);
  }

  void get(const Request& r) {
    if (r.path != "/health") { reply(404, error_body("not_found")); return; }
    const bool ready = service_.ready();
    reply(ready ? 200 : 503, Obj().boolean("ready", ready).boolean("gpu_batching", false).done());
  }

  static bool constant_time_equal(const std::string& a, const std::string& b) {
    // hmac.compare_digest: time depends only on the length of `a` (the presented value).
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < a.size(); ++i) diff |= static_cast<unsigned char>(a[i] ^ (i < b.size() ? b[i] : 0));
    return diff == 0;
  }

  void post(const Request& r) {
    try {
      const auto auth = r.all("authorization");
      bool ascii = auth.size() == 1;
      if (ascii) for (const unsigned char c : auth[0]) if (c >= 0x80) ascii = false;
      if (auth.size() != 1 || auth[0].size() > 263 || !ascii || !constant_time_equal(auth[0], "Bearer " + token_))
        throw ApiError(401, "unauthorized");
      const auto lengths = r.all("content-length");
      if (!r.all("transfer-encoding").empty() || lengths.size() != 1) throw ApiError(400, "content_length_required");
      const auto& length_text = lengths[0];
      bool digits = !length_text.empty() && length_text.size() <= 8;
      for (const char c : length_text) if (c < '0' || c > '9') digits = false;
      if (!digits) throw ApiError(400, "invalid_content_length");
      const size_t length = std::stoul(length_text);
      // /speech may carry images (kMaxSpeechBody); a /speech body over kMaxBody without an image part is refused below,
      // once parsed, so audio and text requests keep the limit they had
      if (length < 1 || length > (r.path == "/speech" ? kMaxSpeechBody : kMaxBody)) throw ApiError(413, "body_limit");
      const auto types = r.all("content-type");
      if (types.size() != 1 || lower(types[0]) != "application/json") throw ApiError(415, "application_json_required");
      const auto deadline = Clock::now() + seconds(service_.settings().socket_timeout);
      std::string body = buffer_.substr(pos_);
      while (body.size() < length) {
        if (Clock::now() >= deadline) throw ApiError(408, "body_deadline");
        buffer_.clear();
        pos_ = 0;
        try {
          if (!fill(deadline, std::min<size_t>(65536, length - body.size()))) throw ApiError(400, "incomplete_body");
        } catch (const ApiError&) {
          throw;
        } catch (const std::exception&) {
          throw ApiError(408, "socket_deadline");
        }
        body += buffer_;
      }
      body.resize(length);
      // A parsed body is a tree several times the size of its text. Bodies over 1 MiB are parsed one
      // at a time; with the try-only frontend lock this bounds the trees alive at once.
      std::shared_ptr<const Json> parsed;
      {
        static std::mutex large;
        std::unique_lock guard(large, std::defer_lock);
        if (length > (size_t(1) << 20)) guard.lock();
        parsed = std::make_shared<const Json>(strict_json(body));
      }
      std::string().swap(body);
      std::string().swap(buffer_);
      const Json& request = *parsed;
      std::string result;
      if (r.path == "/sessions") result = service_.create(request);
      else if (r.path == "/generate") result = service_.generate(std::move(parsed), [this] { return reset_by_peer(); });
      else if (r.path == "/generate_batch") result = service_.generate_batch(std::move(parsed));
      else if (r.path == "/speech") {
        if (length > kMaxBody && !has_image_part(request)) throw ApiError(413, "body_limit");
        // "stream" is checked as a field by validate_speech: only a literal true streams
        bool stream = false;
        if (const auto* o = std::get_if<Json::Object>(&request.value)) {
          const auto it = o->find("stream");
          stream = it != o->end() && std::holds_alternative<bool>(it->second.value) && std::get<bool>(it->second.value);
        }
        if (stream) {
          stream_speech(std::move(parsed));
          return;
        }
        reply(200, service_.speech(std::move(parsed), [this] { return reset_by_peer(); }), kMaxSpeechResponse);
        return;
      }
      else {
        // /sessions/<32 URL-safe characters>/(reset|close|cancel)
        const std::string prefix = "/sessions/";
        if (r.path.size() < prefix.size() + 33 || r.path.compare(0, prefix.size(), prefix) != 0 ||
            !valid_session_id(r.path.substr(prefix.size(), 32)) || r.path[prefix.size() + 32] != '/')
          throw ApiError(404, "not_found");
        const auto sid = r.path.substr(prefix.size(), 32);
        const auto op = r.path.substr(prefix.size() + 33);
        if (op == "cancel") { reply(202, service_.cancel(sid, request)); return; }
        if (op != "reset" && op != "close") throw ApiError(404, "not_found");
        result = service_.lifecycle(sid, op, request);
      }
      reply(200, result);
    } catch (const ApiError& e) {
      reply(e.status, error_body(e.code));
    } catch (const std::exception&) {
      reply(400, error_body("invalid_request"));
    }
  }

  int fd_;
  Service& service_;
  const std::string token_;
  std::string buffer_;
  size_t pos_ = 0;
};

// LocalHTTPServer: literal IPv4 loopback only, a bounded number of concurrent handlers, and on
// saturation no thread, parsing, authentication or admission work (the connection is closed).
class Server {
 public:
  Server(Service& service, std::string token, int port) : service_(service), token_(std::move(token)) {
    if (!valid_token(token_)) throw std::invalid_argument("bearer token must be 32..256 URL-safe characters");
    if (port < 0 || port > 65535) throw std::invalid_argument("port must be an integer in 0..65535");
    listener_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener_ < 0) throw std::runtime_error("socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(std::uint16_t(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // the accept queue holds a crowd of clients arriving at once (http.server's 5 overflowed under concurrent speech,
    // and an overflowing queue can reset a connection: validation/speech-batch-v1); handlers bound the work, not this
    const int backlog = std::max(16, service.settings().handlers);
    if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 || ::listen(listener_, backlog) != 0) {
      ::close(listener_);
      throw std::runtime_error(std::string("cannot listen on 127.0.0.1: ") + std::strerror(errno));
    }
    socklen_t size = sizeof address;
    ::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size);
    port_ = ntohs(address.sin_port);
    if (::pipe2(wake_, O_CLOEXEC | O_NONBLOCK) != 0) { ::close(listener_); throw std::runtime_error("pipe failed"); }
  }
  ~Server() {
    if (listener_ >= 0) ::close(listener_);
    ::close(wake_[0]);
    ::close(wake_[1]);
  }
  int port() const { return port_; }

  // Accepts until stop(). Running handlers continue; see drain().
  void serve() {
    for (;;) {
      pollfd p[2] = {{listener_, POLLIN, 0}, {wake_[0], POLLIN, 0}};
      if (::poll(p, 2, -1) < 0) { if (errno == EINTR) continue; break; }
      if (p[1].revents) break;
      if (!(p[0].revents & POLLIN)) continue;
      const int fd = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) continue;
      {
        std::lock_guard g(lock_);
        if (active_ >= service_.settings().handlers) { ::close(fd); continue; }
        ++active_;
      }
      try {
        std::thread([this, fd] {
          try { Connection(fd, service_, token_).handle(); } catch (...) {}
          std::lock_guard g(lock_);
          --active_;
          idle_.notify_all();
        }).detach();
      } catch (...) {
        ::close(fd);
        std::lock_guard g(lock_);
        --active_;
      }
    }
    // Admission is over: refuse new connections at once rather than leaving them in the backlog.
    ::close(listener_);
    listener_ = -1;
  }
  // Waits up to `limit` for running handlers to send their replies.
  void drain(std::chrono::nanoseconds limit) {
    std::unique_lock g(lock_);
    idle_.wait_for(g, limit, [&] { return active_ == 0; });
  }
  // Async-signal-safe.
  void stop() {
    const char c = 1;
    [[maybe_unused]] const auto n = ::write(wake_[1], &c, 1);
  }
  int active_handlers() {
    std::lock_guard g(lock_);
    return active_;
  }

 private:
  Service& service_;
  std::string token_;
  int listener_ = -1, port_ = 0;
  int wake_[2] = {-1, -1};
  std::mutex lock_;
  std::condition_variable idle_;
  int active_ = 0;
};

}  // namespace lfm::http
