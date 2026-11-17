#pragma once
// The host-side parts of the native speech path (#2): request options and results, the chunk loop over a speech
// session (long-form TTS by sentences, with the server's step budget) and the WAV writer. Host-only, so the chunk
// loop is tested on any host with a fake session (tests/test_speech_run.py); src/lfm3_speech.cuh holds the CUDA
// Session and includes this.
#include <lfm/audio_features.hpp>
#include <lfm/base64.hpp>
#include <lfm/wav.hpp>
#include <lfm/audio_modes.hpp>
#include <lfm/audio_sampling.hpp>
#include <lfm/image_preprocess.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm3_speech {

struct ImageRows;  // an image's encoder output on the GPU (src/lfm3_speech.cuh)

struct ChunkInfo { size_t frames, slots; bool ended_by_eos; };

struct Options {
  lfm::audio::Requested mode = lfm::audio::Requested::Audio;
  int max_new_tokens = 512;
  uint64_t seed = 0;
  bool return_audio_eos = false;
  // forced replay: the text tokens and the raw frames to use, in order (empty: pick and sample)
  std::vector<int> forced_tokens;
  std::vector<lfm::audio::Frame> forced_frames;
  bool record_hidden = false;
  // audio input (#3): the clips, 16 kHz mono samples, one per <|audio|> span of the prompt in order (each span holds
  // lfm::audio_features::tokens(samples) placeholders); record_audio keeps every clip's stages in Result::audio_dumps
  std::vector<std::vector<float>> audio;
  bool record_audio = false;
  // image input (#4): the images after the native front end (lfm/image_preprocess.hpp), one per <|image|> span of the
  // prompt in order (each span holds PixelValues::tokens placeholders); audio and image spans may interleave
  // Shared and immutable (validation/image-cache-v1 C4): the in-process handoff's pixels are used, not copied.
  std::vector<std::shared_ptr<const lfm::image::PixelValues>> images;
  // validation/image-cache-v1: per image (empty: none of this), its key for the engine's image cache (the image part's
  // base64 text; empty: not cached), and its encoder rows when the cache had them (null: encode). With rows, images[i]
  // holds only its grid and tokens, no pixel values.
  std::vector<std::string> image_keys;
  std::vector<std::shared_ptr<const ImageRows>> image_rows;
  // long-form TTS by sentences (tools/lfm3_speech_chunked_hf.py): each sentence is its own request, and the codec
  // stream runs on across them: codec_continue skips the reset, codec_finalize = false keeps the stream's tail open
  bool codec_continue = false, codec_finalize = true;
  // polled before every step (null: never); true ends the call with Cancelled, the codec stream left closed
  std::function<bool()> cancelled;
  // Streaming observers (null: none). They only read, in generation order: on_audio after each codec call with the
  // samples it made final (Result::wav grows by exactly these), on_slot with each emitted slot (Result::slots grows by
  // exactly these), and, from run_chunks, on_chunk as chunk k ends. A throw from one ends the call like any failure.
  std::function<void(const float* samples, size_t count)> on_audio;
  std::function<void(int slot)> on_slot;
  std::function<void(size_t chunk, const ChunkInfo& info)> on_chunk;
};

struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("speech generation cancelled at a step boundary") {}
};

struct Result {
  std::vector<int> prompt, slots, text_tokens;
  std::vector<lfm::audio::Frame> frames;    // emitted (codec input)
  std::vector<lfm::audio::Frame> raw_frames;  // as sampled, EOS frames included
  std::vector<float> wav;
  std::vector<std::vector<uint16_t>> hidden;  // per frame step, the BF16 bits of the state the depth decoder read
  std::vector<std::vector<uint16_t>> embeds;  // per decoded step (record_hidden), the BF16 input embedding fed
  struct StepRecord { bool frame, decode; int token; lfm::audio::Frame fed; };
  std::vector<StepRecord> steps;
  std::vector<lfm::audio_features::Dump> audio_dumps;
  int decode_steps = 0;
  double encode_seconds = 0;  // audio input: features and encoder (part of prefill_seconds)
  double prefill_seconds = 0, generate_seconds = 0;
  // where the generation time goes (host wall time, each phase synchronized at its end)
  double lm_step_seconds = 0, text_pick_seconds = 0, depth_seconds = 0, sample_host_seconds = 0, codec_seconds = 0;
};

// Media input (#3, #4): the prompt's `token` placeholders must be exactly the media's spans, in order: medium k is
// <|vision_start|>, counts[k] `token`, <|vision_end|> (the processor's expansion); no other `token`. `tag`, `medium`
// and `media` name the placeholder, one medium and the media in the refusals.
inline void check_spans(const std::vector<int>& prompt, int token, const std::vector<size_t>& counts,
                        const std::string& tag, const std::string& medium, const std::string& media) {
  const int start = 125009, end = 125010;  // vision_start/end
  size_t k = 0;
  for (size_t i = 0; i < prompt.size();) {
    if (prompt[i] != token) { ++i; continue; }
    size_t j = i;
    while (j < prompt.size() && prompt[j] == token) ++j;
    if (k >= counts.size()) throw std::runtime_error("speech: more " + tag + " spans than " + media);
    if (j - i != counts[k])
      throw std::runtime_error("speech: " + medium + " " + std::to_string(k) + " needs " + std::to_string(counts[k]) +
                               " placeholders, the prompt has " + std::to_string(j - i));
    if (i == 0 || prompt[i - 1] != start || j >= prompt.size() || prompt[j] != end)
      throw std::runtime_error("speech: an " + tag + " span must be wrapped in <|vision_start|> and <|vision_end|>");
    ++k, i = j;
  }
  if (k != counts.size()) throw std::runtime_error("speech: fewer " + tag + " spans than " + media);
}

// Audio input (#3): the prompt's <|audio|> placeholders must be exactly the clips' spans, in order: clip k is
// <|vision_start|>, clip_tokens[k] <|audio|>, <|vision_end|> (the processor's expansion); no other <|audio|>.
inline void check_audio_spans(const std::vector<int>& prompt, const std::vector<size_t>& clip_tokens) {
  check_spans(prompt, lfm::audio::ModeTokens{}.placeholders[0], clip_tokens, "<|audio|>", "audio clip", "audio clips");
}

// Image input (#4): the same for <|image|> and the images (PixelValues::tokens each).
inline constexpr int kImageToken = 125030;
inline void check_image_spans(const std::vector<int>& prompt, const std::vector<size_t>& image_tokens) {
  check_spans(prompt, kImageToken, image_tokens, "<|image|>", "image", "images");
}

// A clip as the server carries it (base64 WAV) -> its 16 kHz mono samples; refuses anything else.
inline std::vector<float> clip_from_base64(const std::string& b64) {
  auto a = lfm::wav::parse(lfm::base64::decode(b64));
  if (a.rate != lfm::audio_features::kRate)
    throw std::invalid_argument("audio must be 16000 Hz, got " + std::to_string(a.rate));
  if (a.samples.empty()) throw std::invalid_argument("audio clip has no samples");
  return std::move(a.samples);
}

// One request as one or more chunks (long-form TTS by sentences): chunk k is its own generation, seeded
// seed * 1000 + k when `chunked` (tools/lfm3_speech_chunked_hf.py), the codec stream running on across the chunks.
// The results concatenate (prompt: the first chunk's). A chunk "ended_by_eos" as the HF tool decides it: its last
// sampled frame is the EOS frame and its last slot is <|im_end|>.
//
// step_budget (0: none) bounds the steps of the whole request (sampled frames plus text picks), so its worst-case time
// is bounded (the server's engine timeout). A chunk whose max_new_tokens could use up what remains gets the remainder
// as its cap, finalizes the codec stream and is the last one run; *truncated then says whether the request stopped
// short of its text because of the budget: chunks skipped, or the chunk whose cap the budget lowered used all of it
// without ending by EOS. (A chunk cut by its own max_new_tokens shows only in its ended_by_eos.) A request that never
// comes near the budget runs exactly as without one.
// The chunk loop one chunk at a time: next() gives the next chunk's prompt and options (false: the request is
// complete), done() takes its result. A batch row runs it between its chunks; run_chunks runs it to the end.
class ChunkRun {
 public:
  ChunkRun(std::vector<std::vector<int>> prompts, Options o, bool chunked, size_t step_budget)
      : prompts_(std::move(prompts)), o_(std::move(o)), chunked_(chunked), budget_(step_budget), seed_(o_.seed),
        requested_(o_.max_new_tokens) {
    if (prompts_.empty()) throw std::runtime_error("speech: no prompt");
    if (chunked && (!o_.forced_tokens.empty() || !o_.forced_frames.empty())) throw std::runtime_error("chunks cannot be forced");
    if (chunked && !o_.audio.empty()) throw std::runtime_error("chunks cannot take audio input");
    if (chunked && !o_.images.empty()) throw std::runtime_error("chunks cannot take image input");
  }
  bool next(const std::vector<int>*& prompt, Options& o) {
    if (complete_) return false;
    if (chunked_) o_.seed = seed_ * 1000 + k_;
    final_ = k_ + 1 == prompts_.size(), capped_ = false;
    o_.max_new_tokens = requested_;
    if (budget_) {
      const size_t remaining = budget_ > used_ ? budget_ - used_ : 0;
      if (remaining == 0) throw std::runtime_error("speech: step budget exhausted before a chunk");  // unreachable
      if (remaining <= size_t(requested_)) {
        o_.max_new_tokens = int(remaining);
        capped_ = remaining < size_t(requested_);  // the budget lowered its cap (at equality its own cap stands)
        if (!final_) truncated_ = true;
        final_ = true;
      }
    }
    o_.codec_continue = k_ > 0, o_.codec_finalize = final_;
    prompt = &prompts_[k_];
    if (prompts_.size() == 1) {  // the one chunk takes the media, moved (validation/image-cache-v1 C4: not copied)
      auto images = std::move(o_.images);
      auto keys = std::move(o_.image_keys);
      auto rows = std::move(o_.image_rows);
      auto audio = std::move(o_.audio);
      o_.images.clear(), o_.image_keys.clear(), o_.image_rows.clear(), o_.audio.clear();
      o = o_;
      o.images = std::move(images), o.image_keys = std::move(keys), o.image_rows = std::move(rows);
      o.audio = std::move(audio);
    } else {
      o = o_;
    }
    return true;
  }
  void done(Result c) {
    const size_t chunk_steps = c.raw_frames.size() + c.text_tokens.size();  // one per step (audio_modes.hpp)
    used_ += chunk_steps;
    const auto& last = c.raw_frames.empty() ? lfm::audio::Frame{} : c.raw_frames.back();
    const bool ended = !c.raw_frames.empty() &&
                       std::find(last.begin(), last.end(), lfm::audio::kEosCode) != last.end() &&
                       !c.slots.empty() && c.slots.back() == lfm::audio::ModeTokens{}.eos[0];
    info_.push_back({c.frames.size(), c.slots.size(), ended});
    if (o_.on_chunk) o_.on_chunk(k_, info_.back());
    if (k_ == 0) {
      r_ = std::move(c);
    } else {
      auto cat = [](auto& to, auto& from) { to.insert(to.end(), from.begin(), from.end()); };
      cat(r_.slots, c.slots), cat(r_.text_tokens, c.text_tokens), cat(r_.frames, c.frames), cat(r_.raw_frames, c.raw_frames);
      cat(r_.wav, c.wav), cat(r_.hidden, c.hidden), cat(r_.embeds, c.embeds), cat(r_.steps, c.steps);
      r_.decode_steps += c.decode_steps;
      r_.prefill_seconds += c.prefill_seconds, r_.generate_seconds += c.generate_seconds;
      r_.lm_step_seconds += c.lm_step_seconds, r_.text_pick_seconds += c.text_pick_seconds;
      r_.depth_seconds += c.depth_seconds, r_.sample_host_seconds += c.sample_host_seconds, r_.codec_seconds += c.codec_seconds;
    }
    if (final_) {
      // cut by its cap: it used all of it without ending (a chunk can also end by EOS on exactly its last step)
      if (capped_ && chunk_steps >= size_t(o_.max_new_tokens) && !info_.back().ended_by_eos) truncated_ = true;
      complete_ = true;
    } else {
      ++k_;
    }
  }
  bool complete() const { return complete_; }
  bool truncated() const { return truncated_; }
  const std::vector<ChunkInfo>& info() const { return info_; }
  Result take() { return std::move(r_); }

 private:
  std::vector<std::vector<int>> prompts_;
  Options o_;
  bool chunked_;
  size_t budget_;
  uint64_t seed_;
  int requested_;
  size_t k_ = 0, used_ = 0;
  bool final_ = false, capped_ = false, truncated_ = false, complete_ = false;
  std::vector<ChunkInfo> info_;
  Result r_;
};

template <class SessionT>
inline Result run_chunks(SessionT& session, const std::vector<std::vector<int>>& prompts, Options o, bool chunked,
                         std::vector<ChunkInfo>& info, size_t step_budget = 0, bool* truncated = nullptr) {
  ChunkRun run(prompts, std::move(o), chunked, step_budget);
  info.clear();
  if (truncated) *truncated = false;
  const std::vector<int>* prompt = nullptr;
  Options chunk;
  while (run.next(prompt, chunk)) run.done(session.generate(*prompt, std::move(chunk)));
  info = run.info();
  if (truncated) *truncated = run.truncated();
  return run.take();
}

// Samples as 16-bit little-endian PCM appended to `f` (clipped to [-1, 1], rounded to nearest): a WAV file's data, and
// a streamed reply's audio. A NaN or infinite sample is a numeric fault upstream and throws (it would otherwise become
// a full-scale click).
inline void append_pcm16(const float* x, size_t n, std::string& f) {
  static_assert(sizeof(int16_t) == 2);
  const size_t at = f.size();
  f.resize(at + 2 * n);
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) throw std::runtime_error("speech: non-finite sample in the waveform");  // a fault, not a click
    const float c = std::max(-1.0f, std::min(1.0f, x[i]));
    const uint16_t s = uint16_t(int16_t(std::lrint(c * 32767.0f)));
    f[at + 2 * i] = char(s & 0xff), f[at + 2 * i + 1] = char(s >> 8);
  }
}

// The waveform as a 16-bit mono PCM WAV file (append_pcm16's samples).
inline std::string wav_bytes(const std::vector<float>& x, int rate) {
  std::string f;
  auto u32 = [&](uint32_t v) { f.append(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](uint16_t v) { f.append(reinterpret_cast<const char*>(&v), 2); };
  if (x.size() > (size_t(1) << 30)) throw std::runtime_error("speech: audio too long for a WAV file");
  const uint32_t bytes = uint32_t(x.size() * 2);
  f.append("RIFF", 4); u32(36 + bytes); f.append("WAVEfmt ", 8); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2);
  u16(2); u16(16); f.append("data", 4); u32(bytes);
  append_pcm16(x.data(), x.size(), f);
  return f;
}

}  // namespace lfm3_speech
