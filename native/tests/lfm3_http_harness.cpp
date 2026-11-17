// Host harness for lfm/lfm3_http.hpp: the real HTTP layer, Service, native tokenizer and session ledger,
// with a fake engine in place of the GPU. The fake speaks resident protocol 2 exactly as
// liquid-lfm3-worker does (same reply fields, same cancellation pipe reader), so every Service path,
// including deadlines, abandonment, cancellation, batching and fatal failure, runs on any host.
//
//   lfm3-http-harness CHECKPOINT_DIR [--mode phrase|stop] [--token-delay-ms N] [--create-delay-ms N]
//                     [--speech on|off] [--audio-input on|off] [--speech-truncate K] [--speech-stream-fault KIND] [--speech-stream-piece-bytes N]
//                     [--speech-rows N (default 2; 0: one /speech at a time through op "speech")] [--fail-op OP] [--fail-after N] [--port N] [--request-timeout S] [--socket-timeout S]
//                     [--session-idle-timeout S] [--total-capacity N] [--max-sessions N] [--queue-size N]
//                     [--handlers N] [--text-slice N] [--trace-ops FILE (each engine call's op, one per line)]
// Token from LFM3_SERVER_TOKEN. Prints {"ready":true,"port":N}. SIGTERM exits like liquid-lfm3-server.
#include <lfm/lfm3_http.hpp>
#include <lfm/speech_run.hpp>
#include <lfm/lfm3_text_request.hpp>
#include <lfm/text_batch_cancellation.hpp>
#include <lfm/text_cancellation.hpp>
#include <lfm/text_sessions.hpp>

#include <csignal>
#include <iostream>
#include <pthread.h>
#include <sstream>

namespace {

using lfm::Json;
using lfm::http::Obj;

// Generation content for --mode phrase: exercises non-ASCII, JSON escapes and a final stop.
const char* kPhrase = "Caf\xc3\xa9 \"quoted\" back\\slash\ttab\nline \xe2\x80\xa8 \x7f end \xf0\x9f\x99\x82 done.";

class FakeEngine final : public lfm::http::Backend {
 public:
  // op "speech", as WorkerCore answers it: each chunk's "frames" is its prompt length (so a test sees exactly the
  // prompts the Service sent, audio placeholders expanded), the slots are the phrase with an <|audio_out|> after each
  // token (interleaved), audio_out only (audio) or the phrase only (text), and the WAV is a fixed 4-sample file.
  // --token-delay-ms sleeps per chunk and honours cancellation there.
  // Streamed ("stream": true, through call_streaming), as WorkerCore streams: per chunk, after --token-delay-ms and a
  // cancellation check, the phrase's text slots (chunk 0 only), one audio piece of 2 samples (bytes k, seed % 256, 255,
  // 127 for chunk k) and the chunk's end; the reply then has no WAV and "samples" 2 per chunk run.
  // --speech-stream-fault KIND makes the stream disagree with the reply (KIND: samples, slots, chunk, malformed).
  // Concurrent speech (--speech-rows N, ops speech_admit/speech_step as WorkerCore's): each step runs one chunk of
  // every busy row (after one --token-delay-ms), with the same events and replies, the replies through `progress`.
  struct FakeSpeech {
    std::string mode;
    std::vector<size_t> lengths;
    bool truncated = false, stream = false;
    unsigned char seed = 0;
  };
  // WorkerCore's refusals, so a request the Service accepts but the engine would refuse fails the harness closed here
  // too (a test then sees 503)
  FakeSpeech check_speech(const Json& request, const Progress* progress) {
    FakeSpeech f;
    f.stream = request.has("stream") && request.at("stream").boolean();
    if (f.stream && !progress) throw std::runtime_error("speech streaming needs the in-process server");
    f.mode = request.at("mode").string();
    if (f.mode != "audio" && f.mode != "interleaved" && f.mode != "text")
      throw std::runtime_error("speech mode must be audio, interleaved or text");
    const int count = request.at("max_new_tokens").integer();
    if (count < 1 || count >= 4096) throw std::runtime_error("speech max_new_tokens out of range");
    if (request.has("seed") && request.at("seed").integer() < 0) throw std::runtime_error("speech seed must be nonnegative");
    f.seed = (unsigned char)(request.has("seed") ? request.at("seed").integer() : 0);
    const bool chunked = request.has("chunked") && request.at("chunked").boolean();
    if (request.at("chunks").array().empty() || (!chunked && request.at("chunks").array().size() != 1))
      throw std::runtime_error("speech needs one prompt, or chunks");
    std::vector<std::vector<int>> prompts;
    for (const auto& c : request.at("chunks").array()) {
      if (c.array().empty() || c.array().size() + size_t(count) > 4096) throw std::runtime_error("speech prompt empty or over capacity");
      prompts.emplace_back();
      for (const auto& x : c.array()) {
        if (x.integer() < 0 || x.integer() >= int(tokenizer_.token_count())) throw std::runtime_error("speech prompt id out of range");
        prompts.back().push_back(x.integer());
      }
      f.lengths.push_back(c.array().size());
    }
    // audio input (#3), as WorkerCore checks it: each clip decoded, and the spans matching the clips
    if (request.has("audio")) {
      if (!audio_input_) throw std::runtime_error("audio input is not supported by this checkpoint");
      if (request.at("audio").array().size() > 64) throw std::runtime_error("too many audio clips");
      if (chunked) throw std::runtime_error("chunks cannot take audio input");
      std::vector<size_t> clip_tokens;
      for (const auto& a : request.at("audio").array())
        clip_tokens.push_back(lfm::audio_features::tokens(lfm3_speech::clip_from_base64(a.string()).size()));
      lfm3_speech::check_audio_spans(prompts[0], clip_tokens);
    } else {
      for (const auto& p : prompts) lfm3_speech::check_audio_spans(p, {});
    }
    // image input (#4), as WorkerCore checks it: each image decoded and preprocessed, the spans matching the images
    if (request.has("images")) {
      if (!image_input_) throw std::runtime_error("image input is not supported by this checkpoint");
      if (request.at("images").array().size() > 16) throw std::runtime_error("too many images");
      if (chunked) throw std::runtime_error("chunks cannot take image input");
      std::vector<size_t> image_tokens;
      lfm::image::Limits limits;
      limits.max_pixels = lfm::http::kSpeechMaxImagePixels;
      // image-queue-v1 red-team: the handoff's entries when the engine meets this request (leases released on every
      // way a request ends leave only those of the requests still in flight)
      if (handoff_log_.is_open()) handoff_log_ << lfm::image::shared_pixel_handoff().size() << "\n" << std::flush;
      for (const auto& a : request.at("images").array()) {
        // --engine-cache-mb: the engine's image cache as WorkerCore keeps it (keys published to the front end, C3)
        if (const auto held = engine_cache_.find(a.string())) {
          if (image_log_.is_open()) image_log_ << "cached\n" << std::flush;
          image_tokens.push_back(size_t(*held));
          continue;
        }
        const std::string bytes = lfm::base64::decode(a.string());
        const auto pv = lfm::image::preprocess(
            lfm::image::load_rgb(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(), limits));
        // image-queue-v1 part 2: the pixel values the handler thread left must be exactly the engine's own
        const auto handed = lfm::image::shared_pixel_handoff().find(a.string());
        if (handed && (handed->values != pv.values || handed->grid_h != pv.grid_h || handed->grid_w != pv.grid_w ||
                       handed->tokens != pv.tokens))
          throw std::runtime_error("handed-over pixel values differ from the engine's decode");
        if (image_log_.is_open()) image_log_ << (handed ? "handoff\n" : "decode\n") << std::flush;
        image_tokens.push_back(size_t(pv.tokens));
        engine_cache_.put(a.string(), std::make_shared<const int>(pv.tokens), size_t(pv.tokens) * 4096);
      }
      lfm3_speech::check_image_spans(prompts[0], image_tokens);
    } else {
      for (const auto& p : prompts) lfm3_speech::check_image_spans(p, {});
    }
    f.truncated = speech_truncate_ > 0 && size_t(speech_truncate_) < f.lengths.size();
    if (f.truncated) f.lengths.resize(size_t(speech_truncate_));
    return f;
  }
  // chunk k's streamed events, each line opening with `tag` ("" or "\"ticket\":T,")
  void emit_chunk(const FakeSpeech& f, size_t k, const Progress& progress, const std::string& tag) {
    if (k == 0 && f.mode != "audio")
      for (const int id : phrase_) progress("{" + tag + "\"text\":[" + std::to_string(id) + "]}");
    if (stream_fault_ == "malformed") progress("{" + tag + "\"audio\":\"AAA\"}");
    std::string pcm = {char(k), char(f.seed), char(255), char(127)};
    pcm.resize(std::max<size_t>(4, size_t(stream_piece_bytes_)), char(k));  // --speech-stream-piece-bytes
    progress("{" + tag + "\"audio\":\"" + lfm::base64::encode(pcm) + "\"}");
    progress("{" + tag + "\"chunk\":{\"frames\":" + std::to_string(f.lengths[k] + (stream_fault_ == "chunk")) +
             ",\"slots\":1,\"ended_by_eos\":" + (k % 2 ? "false" : "true") + "}}");
  }
  // the reply's fields from "cancelled" on, as WorkerCore writes them
  std::string speech_fields(const FakeSpeech& f) {
    std::ostringstream out;
    out << "\"cancelled\":false,\"truncated\":" << (f.truncated ? "true" : "false") << ",\"slots\":[";
    bool first = true;
    for (const int id : phrase_) {
      if (f.stream && stream_fault_ == "slots" && first) { out << id; first = false; }  // one text slot too many
      if (f.mode != "audio") { out << (first ? "" : ",") << id; first = false; }
      if (f.mode != "text") { out << (first ? "" : ",") << lfm::http::kAudioOutToken; first = false; }
    }
    out << "],\"chunks\":[";
    size_t frames = 0;
    for (size_t k = 0; k < f.lengths.size(); ++k) {
      out << (k ? "," : "") << "{\"frames\":" << f.lengths[k] << ",\"slots\":1,\"ended_by_eos\":" << (k % 2 ? "false" : "true") << "}";
      frames += f.lengths[k];
    }
    if (f.stream) {
      out << "],\"frames\":" << frames << ",\"samples\":"
          << std::max<size_t>(4, size_t(stream_piece_bytes_)) / 2 * f.lengths.size() + (stream_fault_ == "samples")
          << ",\"audio_seconds\":0.000166666,\"prefill_seconds\":0.25,\"generate_seconds\":0.5";
      return out.str();
    }
    out << "],\"frames\":" << frames << ",\"samples\":4,\"audio_seconds\":0.000166666,\"prefill_seconds\":0.25,"
        << "\"generate_seconds\":0.5,\"wav_base64\":\"UklGRiwAAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YQgAAAAAAAEA//8AgA==\"";
    return out.str();
  }
  void speech(const Json& request, int rpc, std::ostringstream& out, const Progress* progress = nullptr) {
    const FakeSpeech f = check_speech(request, progress);
    for (size_t k = 0; k < f.lengths.size(); ++k) {
      if (token_delay_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));
      if (reader_->requested(rpc)) { out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"cancelled\":true}\n"; return; }
      if (f.stream) emit_chunk(f, k, *progress, "");
    }
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true," << speech_fields(f) << "}\n";
  }
  void speech_admit(const Json& request, int rpc, std::ostringstream& out, const Progress* progress) {
    if (!progress) throw std::runtime_error("concurrent speech needs the in-process server");
    const long long ticket = request.at("ticket").integer();
    if (ticket < 0) throw std::runtime_error("speech ticket must be nonnegative");
    FakeSpeech f = check_speech(request, progress);
    int row = -1;
    for (int r = 0; r < int(rows_.size()); ++r) {
      if (rows_[size_t(r)].ticket == ticket) throw std::runtime_error("speech ticket already running");
      if (row < 0 && rows_[size_t(r)].ticket < 0) row = r;
    }
    if (row < 0) throw std::runtime_error("no free speech row");
    rows_[size_t(row)] = FakeRow{ticket, std::move(f), 0, false};
    ++admitted_;
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"ticket\":" << ticket << ",\"row\":" << row << "}\n";
  }
  void speech_step(const Json& request, int rpc, std::ostringstream& out, const Progress* progress) {
    if (!progress) throw std::runtime_error("concurrent speech needs the in-process server");
    if (request.has("cancel"))
      for (const auto& t : request.at("cancel").array())
        for (auto& r : rows_) if (r.ticket == t.integer()) r.cancel = true;
    // as lfm3_speech::Batch: a cancelled row ends at the step's start, before any work
    for (auto& r : rows_)
      if (r.ticket >= 0 && r.cancel) {
        (*progress)("{\"ticket\":" + std::to_string(r.ticket) + ",\"finished\":{\"cancelled\":true}}");
        r = FakeRow{};
      }
    int busy = 0;
    for (const auto& r : rows_) busy += r.ticket >= 0;
    if (busy && token_delay_ms_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));
    busy = 0;
    for (auto& r : rows_) {
      if (r.ticket < 0) continue;
      const std::string tag = "\"ticket\":" + std::to_string(r.ticket) + ",";
      if (r.f.stream) emit_chunk(r.f, r.next, *progress, tag);
      if (++r.next == r.f.lengths.size()) {
        (*progress)("{" + tag + "\"finished\":{" + speech_fields(r.f) + "}}");
        r = FakeRow{};
        continue;
      }
      ++busy;
    }
    max_busy_ = std::max(max_busy_, busy);
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"busy\":" << busy << "}\n";
  }

  FakeEngine(const lfm::lfm3_text::Tokenizer& t, std::vector<int> stops, std::string mode, int token_delay_ms,
             int create_delay_ms, std::string fail_op, int fail_after)
      : tokenizer_(t), stops_(std::move(stops)), mode_(std::move(mode)), token_delay_ms_(token_delay_ms),
        create_delay_ms_(create_delay_ms), fail_op_(std::move(fail_op)), fail_after_(fail_after) {
    if (::pipe2(pipe_, O_NONBLOCK | O_CLOEXEC) != 0) throw std::runtime_error("pipe");
    reader_ = std::make_unique<lfm::text::CancellationReader>(pipe_[0]);
    phrase_ = tokenizer_.encode(kPhrase);
  }
  int cancellation_fd() const override { return pipe_[1]; }
  void close() override { closed_ = true; }

  std::string call(const std::string& line) override { return dispatch(line, nullptr); }
  std::string call_streaming(const std::string& line, const Progress& progress) override { return dispatch(line, &progress); }
  std::string dispatch(const std::string& line, const Progress* progress) {
    if (closed_) throw std::runtime_error("engine closed");
    std::ostringstream out;
    int rpc = 0;
    try {
      const auto request = Json::parse(line);
      rpc = request.at("rpc_id").integer();
      if (rpc != last_rpc_ + 1) throw std::runtime_error("out-of-sequence RPC ID");
      last_rpc_ = rpc;
      const auto op = request.at("op").string();
      if (trace_) trace_ << op << "\n" << std::flush;  // --trace-ops: the order of engine calls, for the scheduler tests
      if (op == fail_op_ && ++fail_count_ >= fail_after_) throw std::runtime_error("injected failure");
      if (op == "generate_batch") { batch(request, rpc, out); return out.str(); }
      if (op == "speech") { speech(request, rpc, out, progress); return out.str(); }
      if (op == "speech_admit") { speech_admit(request, rpc, out, progress); return out.str(); }
      if (op == "speech_step") { speech_step(request, rpc, out, progress); return out.str(); }
      const auto id = request.at("session_id").string();
      // as WorkerCore: a sliced generation owns its session until its last text_step
      if (pending_.count(id) && op != "text_step") throw std::runtime_error("session has a text generation in progress");
      if (op == "text_begin") {
        Lane lane = start(request);
        if (reader_->requested(rpc)) { lane.lease->abort(); reply(out, rpc, lane, true, -1); return out.str(); }
        pending_.emplace(id, std::move(lane));
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << id << "\",\"begun\":true}";
        return out.str();
      }
      if (op == "text_step") {
        auto it = pending_.find(id);
        if (it == pending_.end()) throw std::runtime_error("no text generation in progress");
        const long long k = request.at("tokens").integer();
        if (k < 1) throw std::runtime_error("text_step tokens must be positive");
        const bool cancel = request.has("cancel") && request.at("cancel").boolean();
        Lane& lane = it->second;
        for (long long n = 0; n < k; ++n) {  // op generate's loop below, k iterations of it
          if (reader_->requested(rpc) || cancel) {
            lane.lease->abort();
            reply(out, rpc, lane, true, -1);
            pending_.erase(it);
            return out.str();
          }
          if (step(lane)) {
            finish(lane);
            reply(out, rpc, lane, false, -1);
            pending_.erase(it);
            return out.str();
          }
        }
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << id << "\",\"busy\":true}";
        return out.str();
      }
      if (op == "create") {
        std::this_thread::sleep_for(std::chrono::milliseconds(create_delay_ms_));
        ledger_.create(id, request.at("capacity").integer());
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << id << "\"}\n";
      } else if (op == "close") {
        ledger_.close(id);
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"closed\":true}\n";
      } else if (op == "reset") {
        ledger_.reset(id);
        out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"reset\":true}\n";
      } else if (op == "generate") {
        Lane lane = start(request);
        for (;;) {
          if (reader_->requested(rpc)) { lane.lease->abort(); reply(out, rpc, lane, true, -1); return out.str(); }
          if (step(lane)) break;
        }
        finish(lane);
        reply(out, rpc, lane, false, -1);
      } else throw std::runtime_error("unknown operation");
    } catch (const std::exception& e) {
      out.str("");
      out << "{\"rpc_id\":" << rpc << ",\"ok\":false,\"error\":";
      std::string message;
      lfm::http::append_py_string(message, e.what());
      out << message << "}\n";
    }
    return out.str();
  }

 private:
  struct Lane {
    std::string id;
    std::vector<int> input, generated;
    int count = 0;
    bool do_sample = false;
    long long seed = 0;
    size_t reused = 0;
    std::unique_ptr<lfm::text::SessionLedger::Lease> lease;
    bool done = false, cancelled = false;
  };
  Lane start(const Json& r) {
    Lane lane;
    lane.id = r.at("session_id").string();
    for (const auto& v : r.at("input_ids").array()) lane.input.push_back(v.integer());
    lane.count = r.at("max_new_tokens").integer();
    lane.do_sample = r.at("do_sample").boolean();
    lane.seed = r.at("seed").integer();
    if (!r.at("return_text").boolean()) throw std::runtime_error("server must ask for text");
    lane.lease = std::make_unique<lfm::text::SessionLedger::Lease>(
        ledger_.begin(lane.id, lfm::text::SessionLedger::Tokens(lane.input.begin(), lane.input.end()), lane.count));
    lane.reused = lane.lease->plan().full_prefill ? 0 : lane.lease->plan().reuse_tokens;
    return lane;
  }
  // Emits one token; true when the lane is complete.
  bool step(Lane& lane) {
    std::this_thread::sleep_for(std::chrono::milliseconds(token_delay_ms_));
    const size_t i = lane.generated.size();
    int token = mode_ == "stop" ? stops_.front() : i < phrase_.size() ? phrase_[i] : stops_.front();
    lane.generated.push_back(token);
    return std::find(stops_.begin(), stops_.end(), token) != stops_.end() || lane.generated.size() == size_t(lane.count);
  }
  void finish(Lane& lane) {
    std::vector<int> consumed = lane.input;
    consumed.insert(consumed.end(), lane.generated.begin(), lane.generated.end() - 1);
    lane.lease->commit(lfm::text::SessionLedger::Tokens(lane.generated.begin(), lane.generated.end()),
                       lfm::text::SessionLedger::Tokens(consumed.begin(), consumed.end()));
    lane.done = true;
  }
  void reply(std::ostream& out, int rpc, const Lane& lane, bool cancelled, long long paired) {
    std::string text;
    lfm::http::append_py_string(text, lfm::lfm3_text::generated_text(tokenizer_, lane.generated, stops_));
    std::string ids = "[";
    for (size_t i = 0; i < lane.generated.size(); ++i) ids += (i ? "," : "") + std::to_string(lane.generated[i]);
    ids += "]";
    const size_t consumed = cancelled ? 0 : lane.input.size() + lane.generated.size() - 1;
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"session_id\":\"" << lane.id << "\",\"generated_ids\":" << ids
        << ",\"reused_tokens\":" << (cancelled ? 0 : lane.reused) << ",\"consumed_tokens\":" << consumed
        << ",\"do_sample\":" << (lane.do_sample ? "true" : "false") << ",\"seed\":" << lane.seed;
    if (cancelled || paired >= 0) out << ",\"cancelled\":" << (cancelled ? "true" : "false") << ",\"cache_valid\":" << (cancelled ? "false" : "true");
    if (paired >= 0) out << ",\"gpu_batching\":" << (paired ? "true" : "false") << ",\"paired_decode_calls\":" << paired;
    out << ",\"live_token_count\":125033,\"text\":" << text << "}";
  }
  void batch(const Json& request, int rpc, std::ostream& out) {
    const auto& lanes_json = request.at("lanes").array();
    for (const auto& lane : lanes_json)
      if (pending_.count(lane.at("session_id").string())) throw std::runtime_error("session has a text generation in progress");
    std::array<Lane, 2> lanes{start(lanes_json[0]), start(lanes_json[1])};
    long long paired = 0;
    for (;;) {
      const auto c = reader_->requested_pair(rpc);
      for (size_t i = 0; i < 2; ++i)
        if (!lanes[i].done && c[i]) { lanes[i].lease->abort(); lanes[i].cancelled = lanes[i].done = true; }
      if (lanes[0].done && lanes[1].done) break;
      int stepping = 0;
      for (auto& lane : lanes)
        if (!lane.done) { ++stepping; if (step(lane)) finish(lane); }
      // Two lanes that both still need a forward after this token share one paired call.
      if (stepping == 2 && !lanes[0].done && !lanes[1].done) ++paired;
    }
    out << "{\"rpc_id\":" << rpc << ",\"ok\":true,\"gpu_batching\":" << (paired ? "true" : "false")
        << ",\"paired_decode_calls\":" << paired << ",\"results\":[";
    for (size_t i = 0; i < 2; ++i) {
      if (i) out << ',';
      reply(out, rpc, lanes[i], lanes[i].cancelled, paired);
    }
    out << "]}\n";
  }

  const lfm::lfm3_text::Tokenizer& tokenizer_;
  std::vector<int> stops_, phrase_;
  std::string mode_;
  int token_delay_ms_, create_delay_ms_;
  std::string fail_op_;
 public:
  int speech_truncate_ = 0;
  std::string stream_fault_;  // --speech-stream-fault
  int stream_piece_bytes_ = 4;  // --speech-stream-piece-bytes: each audio piece's size (even, at least 4)
  struct FakeRow {
    long long ticket = -1;
    FakeSpeech f;
    size_t next = 0;
    bool cancel = false;
  };
  std::vector<FakeRow> rows_;  // --speech-rows
  int admitted_ = 0, max_busy_ = 0;
  std::ofstream trace_;  // --trace-ops FILE
  bool image_input_ = true;  // --image-input: WorkerCore::image_input_supported
  lfm::LruBytes<int> engine_cache_;  // --engine-cache-mb N (default 0: none): a stand-in for WorkerCore::image_cache_
  std::ofstream image_log_;  // --image-log FILE: per image, "handoff" (found handed over) or "decode"
  std::ofstream handoff_log_;  // --handoff-log FILE: per image request, the handoff's entry count as the engine meets it
  bool audio_input_ = true;  // --audio-input: WorkerCore::audio_input_supported  // --speech-truncate K: answer only the first K chunks, as a request the step budget cut
 private:
  int fail_after_, fail_count_ = 0;
  int pipe_[2] = {-1, -1};
  std::unique_ptr<lfm::text::CancellationReader> reader_;
  lfm::text::SessionLedger ledger_{2, 32768};
  std::map<std::string, Lane> pending_;  // sliced generations (ops text_begin, text_step)
  int last_rpc_ = 0;
  bool closed_ = false;
};

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2 || argc % 2 != 0) throw std::runtime_error("usage: lfm3-http-harness CHECKPOINT_DIR [--option value]...");
  const std::string directory = argv[1];
  lfm::http::Settings settings;
  std::string mode = "phrase", fail_op;
  int port = 0, token_delay = 0, create_delay = 0, fail_after = 1;
  std::string token_path;
  bool speech = true;  // --speech off: the engine reports no speech support (a text-only checkpoint or build)
  bool audio_input = true;  // --audio-input off: speech without audio input (no encoder for the checkpoint)
  bool image_input = true;  // --image-input off: speech without image input (no vision encoder for the checkpoint)
  bool image_handoff = false;  // --image-handoff on: as the in-process server (image-queue-v1 part 2)
  std::string image_log, handoff_log;
  int engine_cache_mb = 0;
  int speech_truncate = 0;
  std::string stream_fault;
  int stream_piece_bytes = 4, speech_rows = 2;
  std::string trace_ops;
  for (int i = 2; i < argc; i += 2) {
    const std::string o = argv[i], v = argv[i + 1];
    if (o == "--mode") mode = v;
    else if (o == "--token-file") token_path = v;
    else if (o == "--token-delay-ms") token_delay = std::stoi(v);
    else if (o == "--create-delay-ms") create_delay = std::stoi(v);
    else if (o == "--fail-op") fail_op = v;
    else if (o == "--speech") speech = v == "on";
    else if (o == "--audio-input") audio_input = v == "on";
    else if (o == "--image-input") image_input = v == "on";
    else if (o == "--image-handoff") image_handoff = v == "on";
    else if (o == "--image-log") image_log = v;
    else if (o == "--handoff-log") handoff_log = v;
    else if (o == "--image-cache-mb") settings.image_cache_mb = std::stoi(v);
    else if (o == "--engine-cache-mb") engine_cache_mb = std::stoi(v);
    else if (o == "--speech-truncate") speech_truncate = std::stoi(v);
    else if (o == "--speech-stream-fault") stream_fault = v;
    else if (o == "--speech-stream-piece-bytes") stream_piece_bytes = std::stoi(v);
    else if (o == "--speech-rows") speech_rows = std::stoi(v);
    else if (o == "--trace-ops") trace_ops = v;
    else if (o == "--fail-after") fail_after = std::stoi(v);
    else if (o == "--port") port = std::stoi(v);
    else if (o == "--request-timeout") settings.request_timeout = std::stod(v);
    else if (o == "--socket-timeout") settings.socket_timeout = std::stod(v);
    else if (o == "--session-idle-timeout") settings.session_idle_timeout = std::stod(v);
    else if (o == "--total-capacity") settings.total_capacity = std::stoi(v);
    else if (o == "--max-sessions") settings.max_sessions = std::stoi(v);
    else if (o == "--queue-size") settings.queue_size = std::stoi(v);
    else if (o == "--handlers") settings.handlers = std::stoi(v);
    else if (o == "--engine-timeout") settings.engine_timeout = std::stod(v);
    else if (o == "--text-slice") settings.text_slice = std::stoi(v);
    else throw std::runtime_error("unsupported option " + o);
  }
  std::string note;  // as liquid-lfm3-server: --token-file, LFM3_SERVER_TOKEN, or the default file
  const std::string token = lfm::http::startup_token(token_path, note);
  if (!note.empty()) std::cerr << "lfm3-http-harness: " << note << "\n";
  sigset_t signals;
  sigemptyset(&signals); sigaddset(&signals, SIGTERM); sigaddset(&signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  std::signal(SIGPIPE, SIG_IGN);

  const auto config = lfm::ModelConfig::load(directory + "/config.json", 32768);
  std::ifstream raw_config(directory + "/config.json");
  const auto model = Json::parse(std::string(std::istreambuf_iterator<char>(raw_config), {}));
  lfm::http::Policy policy;
  policy.vocabulary = config.shape.vocabulary;
  policy.speech = speech;
  policy.audio_input = speech && audio_input;
  policy.image_input = speech && image_input;
  policy.speech_rows = speech ? speech_rows : 0;
  policy.image_handoff = image_handoff;
  policy.stops = config.eos;
  for (auto name : {"audio_token_id", "audio_out_token_id", "image_token_id", "video_token_id", "mixed_start_token_id", "audio_start_token_id"})
    policy.suppress.push_back(model.at(name).integer());
  policy.stops.push_back(model.at("text_end_token_id").integer());
  for (auto* ids : {&policy.suppress, &policy.stops}) {
    std::sort(ids->begin(), ids->end());
    ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
  }
  const auto tokenizer = lfm::lfm3_text::Tokenizer::load(directory + "/tokenizer.json");
  FakeEngine engine(tokenizer, policy.stops, mode, token_delay, create_delay, fail_op, fail_after);
  engine.speech_truncate_ = speech_truncate;
  engine.stream_fault_ = stream_fault;
  engine.stream_piece_bytes_ = stream_piece_bytes;
  engine.rows_.assign(size_t(std::max(0, speech_rows)), {});
  engine.audio_input_ = speech && audio_input;
  engine.image_input_ = speech && image_input;
  if (!image_log.empty()) engine.image_log_.open(image_log);
  engine.engine_cache_.set_observer([](const std::string& key, bool kept) { lfm::image::shared_held_images().kept(key, kept); });
  engine.engine_cache_.set_budget(size_t(engine_cache_mb) << 20);
  if (!handoff_log.empty()) engine.handoff_log_.open(handoff_log);
  if (!trace_ops.empty()) {
    engine.trace_.open(trace_ops);
    if (!engine.trace_) throw std::runtime_error("cannot open --trace-ops file");
  }
  auto service = std::make_unique<lfm::http::Service>(engine, tokenizer, policy, settings);
  auto server = std::make_unique<lfm::http::Server>(*service, token, port);
  std::thread waiter([&] {
    int received = 0;
    sigwait(&signals, &received);
    server->stop();
    sigwait(&signals, &received);
    std::_Exit(1);
  });
  std::cout << "{\"ready\":true,\"port\":" << server->port() << "}\n" << std::flush;
  server->serve();
  service->close();
  server->drain(lfm::http::seconds(settings.socket_timeout));
  std::cout << std::flush;
  std::_Exit(service->cleanup_failed() ? 1 : 0);
} catch (const std::exception& e) {
  std::cerr << "lfm3-http-harness: " << e.what() << "\n";
  return 1;
}
