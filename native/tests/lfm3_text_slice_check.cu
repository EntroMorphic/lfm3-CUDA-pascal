// lfm3-text-slice-check (validation/text-wait-v1 rules 1 and 8): the worker's sliced text generation against its whole
// one, on the GPU, through WorkerCore::handle exactly as liquid-lfm3-server calls it.
//
//   lfm3-text-slice-check MODEL_DIRECTORY OUT_JSON [--small]
//
// 1. Text, alone: each case as op generate (the whole call), then as op text_begin and text_step with k = 1, 3, 16
//    and unbounded tokens per step, each on a fresh session. Every reply (ids, text, last_logits printed to the last
//    bit, consumed and reused counts) must equal op generate's. Cases: greedy, sampled (temperature 0.528, seed 42),
//    a second generation on the same session that reuses its cache (prefix continuation), and a request too long to
//    finish in one slice.
// 2. Cancellation: a sliced generation cancelled at its 6th step (text_step "cancel": true) answers op generate's
//    cancelled reply (the cache reset); the session's next generation equals a fresh session's.
// 3. Interleaved with speech: 4 TTS rows admitted staggered (2, then 2 more after 3 steps) are stepped to their end
//    alone; then the same schedule again with two text generations sliced one token per speech step (the first begun
//    after step 2, the second when the first ends). Every row's finished reply (its WAV) and every text reply must
//    equal its reply alone.
// --small: shorter texts (16 tokens) and short TTS, for the sanitizers (rule 8).
// Prints one line per check; exits 0 when every check passes.
#define LFM3_WORKER_CORE_ONLY
#include "../src/lfm3_worker.cu"

#include <fcntl.h>
#include <unistd.h>

namespace {

struct Harness {
  int pipe_[2] = {-1, -1};
  std::unique_ptr<WorkerCore> core;
  int rpc = 0;
  std::map<long long, std::string> finished;  // speech ticket -> its finished reply (progress line)
  explicit Harness(const std::string& model) {
    if (::pipe2(pipe_, O_NONBLOCK | O_CLOEXEC) != 0) throw std::runtime_error("pipe failed");
    core = std::make_unique<WorkerCore>(model, "", false, pipe_[0]);
    core->set_speech_rows(4);
    core->progress = [this](const std::string& line) {
      const auto j = lfm::Json::parse(line);
      if (j.has("finished")) finished[j.at("ticket").integer()] = line;
    };
  }
  // One protocol line; returns the reply line. Any refusal ("ok": false) is a failure of the check.
  std::string call(std::string body) {
    body.insert(1, "\"rpc_id\":" + std::to_string(++rpc) + ",");
    std::ostringstream out;
    if (!core->handle(body, false, out)) throw std::runtime_error("engine shut down");
    const auto reply = out.str();
    if (reply.find("\"ok\":true") == std::string::npos) throw std::runtime_error("refused: " + body.substr(0, 200) + " -> " + reply);
    return reply;
  }
  std::string create(int capacity = 4096) {
    static int serial = 0;
    const std::string sid = "slice-check-" + std::to_string(++serial);
    call("{\"op\":\"create\",\"session_id\":\"" + sid + "\",\"capacity\":" + std::to_string(capacity) + "}");
    return sid;
  }
  void close(const std::string& sid) { call("{\"op\":\"close\",\"session_id\":\"" + sid + "\"}"); }
};

// A reply without its rpc_id and session_id (they differ between runs of the same request).
std::string comparable(const std::string& reply) {
  auto j = lfm::Json::parse(reply);
  auto& o = std::get<lfm::Json::Object>(j.value);
  o.erase("rpc_id");
  o.erase("session_id");
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  std::function<void(const lfm::Json&)> dump = [&](const lfm::Json& v) {
    if (const auto* m = std::get_if<lfm::Json::Object>(&v.value)) {
      out << '{';
      bool first = true;
      for (const auto& [k, x] : *m) { out << (first ? "" : ",") << lfm::lfm3_text::json_string(k) << ':'; dump(x); first = false; }
      out << '}';
    } else if (const auto* a = std::get_if<lfm::Json::Array>(&v.value)) {
      out << '[';
      for (size_t i = 0; i < a->size(); ++i) { if (i) out << ','; dump((*a)[i]); }
      out << ']';
    } else if (const auto* s = std::get_if<std::string>(&v.value)) out << lfm::lfm3_text::json_string(*s);
    else if (const auto* b = std::get_if<bool>(&v.value)) out << (*b ? "true" : "false");
    else if (const auto* d = std::get_if<double>(&v.value)) out << *d;
    else out << "null";
  };
  dump(j);
  return out.str();
}

// A speech row's finished reply without its wall-clock fields (prefill_seconds, generate_seconds); the WAV, slots,
// chunks, frames and samples stay.
std::string speech_comparable(const std::string& line) {
  auto j = lfm::Json::parse(line);
  auto& finished = std::get<lfm::Json::Object>(std::get<lfm::Json::Object>(j.value).at("finished").value);
  finished.erase("prefill_seconds");
  finished.erase("generate_seconds");
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  std::function<void(const lfm::Json&)> dump = [&](const lfm::Json& v) {
    if (const auto* m = std::get_if<lfm::Json::Object>(&v.value)) {
      out << '{';
      bool first = true;
      for (const auto& [k, x] : *m) { out << (first ? "" : ",") << lfm::lfm3_text::json_string(k) << ':'; dump(x); first = false; }
      out << '}';
    } else if (const auto* a = std::get_if<lfm::Json::Array>(&v.value)) {
      out << '[';
      for (size_t i = 0; i < a->size(); ++i) { if (i) out << ','; dump((*a)[i]); }
      out << ']';
    } else if (const auto* s = std::get_if<std::string>(&v.value)) out << lfm::lfm3_text::json_string(*s);
    else if (const auto* b = std::get_if<bool>(&v.value)) out << (*b ? "true" : "false");
    else if (const auto* d = std::get_if<double>(&v.value)) out << *d;
    else out << "null";
  };
  dump(j);
  return out.str();
}

std::string ids_json(const std::vector<int>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s + "]";
}

// A generation request's fields (no op, no session).
std::string fields(const std::string& messages_or_ids, int tokens, bool sampled) {
  return messages_or_ids + ",\"max_new_tokens\":" + std::to_string(tokens) + ",\"return_logits\":true,\"return_text\":true" +
         (sampled ? ",\"do_sample\":true,\"temperature\":0.528,\"seed\":42" : ",\"do_sample\":false");
}

std::string whole(Harness& h, const std::string& sid, const std::string& f) {
  return h.call("{\"op\":\"generate\",\"session_id\":\"" + sid + "\"," + f + "}");
}

// text_begin, then text_step(k) until the generation finishes; `cancel_at` cancels at that step (0: never).
std::string sliced(Harness& h, const std::string& sid, const std::string& f, long long k, int cancel_at = 0,
                   int* steps = nullptr) {
  const auto begun = h.call("{\"op\":\"text_begin\",\"session_id\":\"" + sid + "\"," + f + "}");
  if (begun.find("\"begun\":true") == std::string::npos) return begun;
  for (int n = 1;; ++n) {
    const bool cancel = cancel_at && n == cancel_at;
    const auto r = h.call("{\"op\":\"text_step\",\"session_id\":\"" + sid + "\",\"tokens\":" + std::to_string(k) +
                          ",\"cancel\":" + (cancel ? "true" : "false") + "}");
    if (r.find("\"busy\":true") == std::string::npos) {
      if (steps) *steps = n;
      return r;
    }
    if (cancel) throw std::runtime_error("a cancelled step did not end the generation");
  }
}

int failures = 0;
std::string stage = "alone";  // named in an error, to place it
std::ostringstream report;
void check(const std::string& name, bool ok, const std::string& detail = "") {
  std::cout << (ok ? "PASS " : "FAIL ") << name << (detail.empty() ? "" : " (" + detail + ")") << "\n" << std::flush;
  report << (report.tellp() > 1 ? "," : "") << "{\"check\":" << lfm::lfm3_text::json_string(name)
         << ",\"pass\":" << (ok ? "true" : "false") << ",\"detail\":" << lfm::lfm3_text::json_string(detail) << "}";
  if (!ok) ++failures;
}

std::vector<int> generated_of(const std::string& reply) {
  std::vector<int> ids;
  const auto parsed = lfm::Json::parse(reply);  // held: a range-for over a temporary's member dangles in C++20
  for (const auto& v : parsed.at("generated_ids").array()) ids.push_back(int(v.integer()));
  return ids;
}

// Token ids of a TTS chunk, as the server's frontend encodes it (encode_speech, unsplit, no normalization).
std::string tts_chunk(Harness& h, const std::string& voice, const std::string& text, int count) {
  std::vector<lfm::lfm3_text::Message> messages(2);
  messages[0].role = "system";
  messages[0].content = "Perform TTS in American English using a " + voice + " voice.";
  messages[1].role = "user";
  messages[1].content = text;
  return "[" + ids_json(lfm::lfm3_text::encode_messages(h.core->tokenizer(), messages, 4096, count)) + "]";
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--small"))
    throw std::runtime_error("usage: lfm3-text-slice-check MODEL_DIRECTORY OUT_JSON [--small]");
  const bool small = argc == 4;
  Harness h(argv[1]);
  const int tokens = small ? 16 : 96;
  const std::string greedy_messages =
      "\"messages\":[{\"role\":\"user\",\"content\":\"Explain in one paragraph how a lighthouse warns ships at night.\"}]";
  const std::string sampled_messages =
      "\"messages\":[{\"role\":\"user\",\"content\":\"List three ways to keep bread fresh longer.\"}]";
  struct Case { std::string name, f; };
  std::vector<Case> cases{{"greedy", fields(greedy_messages, tokens, false)},
                          {"sampled", fields(sampled_messages, tokens, true)},
                          {"long", fields(greedy_messages, small ? 40 : 300, false)}};

  // 1. alone: whole against sliced, each on a fresh session
  for (const auto& c : cases) {
    std::string sid = h.create();
    const auto want = comparable(whole(h, sid, c.f));
    h.close(sid);
    for (long long k : {1LL, 3LL, 16LL, 2147483647LL}) {  // INT_MAX: unbounded, as op generate
      sid = h.create();
      int steps = 0;
      const auto got = comparable(sliced(h, sid, c.f, k, 0, &steps));
      h.close(sid);
      check("alone " + c.name + " k=" + (k > 1000 ? std::string("unbounded") : std::to_string(k)), got == want,
            "steps " + std::to_string(steps));
    }
  }
  // reuse: a second generation that continues the first's transcript, whole and sliced
  {
    std::vector<std::string> got;
    for (long long k : {0LL, 1LL, 3LL}) {
      const auto sid = h.create();
      const auto first = whole(h, sid, cases[0].f);
      // the continuation: the first request's prompt ids are not known here, so the second request is the first's
      // messages again plus its reply and a follow-up, through the frontend; the ledger reuses the common prefix
      const auto reply_text = lfm::Json::parse(first).at("text").string();
      const std::string follow = "\"messages\":[{\"role\":\"user\",\"content\":\"Explain in one paragraph how a lighthouse "
          "warns ships at night.\"},{\"role\":\"assistant\",\"content\":" + lfm::lfm3_text::json_string(reply_text) +
          "},{\"role\":\"user\",\"content\":\"Now say it in one sentence.\"}]";
      const auto f = fields(follow, tokens, false);
      const auto second = k == 0 ? whole(h, sid, f) : sliced(h, sid, f, k);
      got.push_back(comparable(second));
      if (k == 0) check("reuse happened", lfm::Json::parse(second).at("reused_tokens").integer() > 0,
                        "reused " + std::to_string(lfm::Json::parse(second).at("reused_tokens").integer()));
      h.close(sid);
    }
    check("reuse k=1", got[1] == got[0]);
    check("reuse k=3", got[2] == got[0]);
  }
  // 2. cancellation at the 6th step, then the session's next generation against a fresh session's
  stage = "cancellation";
  {
    auto sid = h.create();
    const auto cancelled = sliced(h, sid, cases[2].f, 1, 6);
    stage = "cancellation: parse";
    const auto j = lfm::Json::parse(cancelled);
    stage = "cancellation: fields";
    const bool flagged = j.has("cancelled") && j.at("cancelled").boolean() && !j.at("cache_valid").boolean();
    stage = "cancellation: ids";
    const size_t got = generated_of(cancelled).size();
    check("cancelled reply", flagged && got == 5, cancelled.substr(0, 160));
    stage = "cancellation: next generation";
    const auto next = comparable(whole(h, sid, cases[0].f));
    h.close(sid);
    stage = "cancellation: fresh generation";
    sid = h.create();
    const auto fresh = comparable(whole(h, sid, cases[0].f));
    h.close(sid);
    check("after cancellation equals fresh", next == fresh);
  }
  // 3. interleaved with speech rows
  stage = "speech";
  const int speech_tokens = small ? 128 : 512;
  const std::vector<std::pair<std::string, std::string>> speech{
      {"female", "The quick brown fox jumps over the lazy dog."},
      {"male", "The quick brown fox jumps over the lazy dog."},
      {"female", small ? "A lighthouse keeps ships safe." : "Lighthouses were once the most important safety systems on any coast."},
      {"male", small ? "Bread needs flour and water." : "Bread begins with only four ingredients: flour, water, salt and yeast."}};
  auto admit = [&](int i) {
    h.call("{\"op\":\"speech_admit\",\"ticket\":" + std::to_string(i + 1) + ",\"chunks\":" +
           tts_chunk(h, speech[i].first, speech[i].second, speech_tokens) + ",\"mode\":\"audio\",\"max_new_tokens\":" +
           std::to_string(speech_tokens) + ",\"seed\":" + std::to_string(i + 1) + ",\"chunked\":false,\"stream\":false}");
  };
  auto run_speech = [&](bool with_text, std::vector<std::string>& texts) {
    h.finished.clear();
    admit(0);
    admit(1);
    std::string sid;
    size_t text_index = 0;
    bool text_active = false;
    std::vector<std::string> text_sessions;
    for (int step = 1;; ++step) {
      if (step == 4) { admit(2); admit(3); }
      const auto r = h.call("{\"op\":\"speech_step\"}");
      if (with_text) {
        if (!text_active && text_index < 2 && step >= 3) {
          sid = h.create();
          text_sessions.push_back(sid);
          const auto begun = h.call("{\"op\":\"text_begin\",\"session_id\":\"" + sid + "\"," + cases[text_index].f + "}");
          if (begun.find("\"begun\":true") == std::string::npos) throw std::runtime_error("text_begin did not begin");
          text_active = true;
        } else if (text_active) {
          const auto s = h.call("{\"op\":\"text_step\",\"session_id\":\"" + sid + "\",\"tokens\":1,\"cancel\":false}");
          if (s.find("\"busy\":true") == std::string::npos) {
            texts.push_back(comparable(s));
            text_active = false;
            ++text_index;
          }
        }
      }
      if (lfm::Json::parse(r).at("busy").integer() == 0 && step > 4) break;
    }
    while (text_active) {  // speech done first: the text runs on alone
      const auto s = h.call("{\"op\":\"text_step\",\"session_id\":\"" + sid + "\",\"tokens\":1,\"cancel\":false}");
      if (s.find("\"busy\":true") == std::string::npos) { texts.push_back(comparable(s)); text_active = false; ++text_index; }
    }
    for (const auto& s : text_sessions) h.close(s);
    std::map<long long, std::string> out;
    for (const auto& [t, line] : h.finished) out[t] = speech_comparable(line);
    return out;
  };
  std::vector<std::string> none, texts;
  const auto alone = run_speech(false, none);
  const auto again = run_speech(false, none);
  const auto mixed = run_speech(true, texts);
  check("speech alone twice is identical (the comparison is exact, timings aside)", alone == again);
  for (const auto& [t, reply] : alone)
    check("speech row " + std::to_string(t) + " carries a WAV", reply.find("\"wav_base64\":\"UklGR") != std::string::npos);
  check("speech rows finished", alone.size() == 4 && mixed.size() == 4,
        std::to_string(alone.size()) + " and " + std::to_string(mixed.size()));
  for (const auto& [t, reply] : alone)
    check("speech row " + std::to_string(t) + " with text equals alone", mixed.count(t) && mixed.at(t) == reply);
  check("both texts ran beside speech", texts.size() == 2, std::to_string(texts.size()));
  for (size_t i = 0; i < texts.size(); ++i) {
    const auto sid = h.create();
    const auto want = comparable(whole(h, sid, cases[i].f));
    h.close(sid);
    check("text " + cases[i].name + " beside speech equals alone", texts[i] == want);
  }
  std::ofstream(argv[2]) << "{\"small\":" << (small ? "true" : "false") << ",\"failures\":" << failures
                         << ",\"checks\":[" << report.str() << "]}\n";
  std::cout << (failures ? "TEXT SLICE CHECK FAIL" : "TEXT SLICE CHECK PASS") << "\n";
  return failures ? 1 : 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-text-slice-check: " << e.what() << " (during " << stage << ")\n";
  return 2;
}
