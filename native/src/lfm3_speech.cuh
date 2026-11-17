#pragma once
// Text in -> text and speech out, end to end in the native engine (#2 step 4, journal/lfm3-tts-quality-bar.md).
// Include after lfm3_main.cu and lfm3_batch.cuh (LFM3_RUNTIME_ONLY, LFM3_NATIVE_GEMM).
//
// One request, HF `_generate` semantics (lfm/audio_modes.hpp holds the control flow, A4):
// - prefill: the validated Runner, with the mode's marker appended to the prompt;
// - text steps: greedy on the device over the language model's logits, with the mode machine's suppression set (HF
//   generates text with do_sample=False);
// - speech steps: the depth decoder (lfm/depth.cuh, A1/B1) on the language model's final-normed hidden state, the
//   frame sampler (lfm/audio_sampling.hpp, B2: temperature 0.4, penalty 1.2 over the last 64 accepted frames, a
//   random reveal order), the EOS rewrite, and the frame fed back as the sum of its 12 codebook rows (A3);
// - the language model's decode steps: lfm3_batch::BatchDecoder;
// - emitted frames stream into the codec (lfm/codec.cuh, A2) 12 at a time; the waveform is 24 kHz FP32.
// Forced mode replays given tokens and frames instead of picking and sampling, and records the hidden state every
// frame step reads: the deterministic check of the whole loop.
//
// Concurrent requests (validation/speech-batch-v1): a Batch holds up to `rows` requests, one per row, each with its
// own mode machine, frame sampler, codec stream and result. A step advances every busy row once: one greedy pick over
// the language model's logits for the rows on a text step (each with its own suppression mask), one batched depth
// decoder pass sequence for the rows sampling a frame, then one decode step of the BatchDecoder for every row that
// feeds an input. Rows are independent by construction (the batched operators compute each row as at one row), so a
// request's output does not depend on what shares its batch or when it joined. A row joins between steps (admit:
// prefilled by the Runner, then loaded into the row) and leaves when it finishes (take). Session, the single-request
// API, is a batch of one: one code path.
#include <lfm/audio_encoder.cuh>
#include <lfm/vision_encoder.cuh>
#include <lfm/audio_modes.hpp>
#include <lfm/speech_run.hpp>
#include <lfm/audio_sampling.hpp>
#include <lfm/codec.cuh>
#include <lfm/depth.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <lfm/lru_bytes.hpp>

namespace lfm3_speech {

// A persistent pool for the batch's per-row host work (frame sampling, validation/speech-speed-v1): run(n, f) calls
// f(i) for every i in [0, n) on the pool's threads and the caller's, and returns when all are done (rethrowing the
// first exception). Each call touches only row i's own state, so the results do not depend on the threads.
class RowPool {
 public:
  explicit RowPool(int threads) {
    try {
      for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { work(); });
    } catch (...) {  // a thread that could not start: stop the ones that did, then refuse (no std::terminate)
      stop();
      throw;
    }
  }
  ~RowPool() { stop(); }
  RowPool(const RowPool&) = delete;
  RowPool& operator=(const RowPool&) = delete;

  void run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
      for (int i = 0; i < n; ++i) f(i);
      return;
    }
    {
      std::lock_guard<std::mutex> l(m_);
      f_ = &f, n_ = n, next_ = 0, busy_ = int(workers_.size()), error_ = nullptr, ++generation_;
    }
    wake_.notify_all();
    drain();
    std::unique_lock<std::mutex> l(m_);
    done_.wait(l, [this] { return busy_ == 0; });
    f_ = nullptr;
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void drain() {
    for (int i; (i = next_.fetch_add(1)) < n_;) {
      try {
        (*f_)(i);
      } catch (...) {
        std::lock_guard<std::mutex> l(m_);
        if (!error_) error_ = std::current_exception();
      }
    }
  }
  void stop() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
  }
  void work() {
    unsigned long long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(m_);
        wake_.wait(l, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      drain();
      {
        std::lock_guard<std::mutex> l(m_);
        if (--busy_ == 0) done_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* f_ = nullptr;
  int n_ = 0, busy_ = 0;
  std::atomic<int> next_{0};
  unsigned long long generation_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

// A non-blocking CUDA stream (a row's codec: it overlaps the language model's step).
struct CodecStream {
  cudaStream_t s = nullptr;
  CodecStream() { cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~CodecStream() { if (s) cudaStreamDestroy(s); }
  CodecStream(const CodecStream&) = delete;
  CodecStream& operator=(const CodecStream&) = delete;
};

// validation/image-cache-v1: an image's encoder output on the GPU, as the engine's image cache keeps it.
struct ImageRows {
  int grid_h, grid_w, tokens;
  Buffer<bf16> rows;  // tokens x hidden
  ImageRows(int h, int w, int t, size_t n) : grid_h(h), grid_w(w), tokens(t), rows(n) {}
};
using ImageCache = lfm::LruBytes<ImageRows>;

class Batch {
 public:
  // The engine's image cache (null: none, as lfm3-speech runs): read by the request's parse, filled here after an
  // encode; one thread (the engine's) uses both.
  void set_image_cache(ImageCache* cache) { image_cache_ = cache; }
  Batch(Runner& runner, const std::string& model_dir, int rows)
      : model_dir_(model_dir), runner_(runner), rows_(rows), dec_(runner, rows), depth_(runner.weights, rows),
        logits_dev_(size_t(rows) * lfm::depth::kCodebooks * lfm::depth::kCodes),
        suppress_dev_(size_t(rows) * runner.s.vocabulary),
        host_logits_(size_t(rows) * lfm::depth::kCodebooks * lfm::depth::kCodes), row_(size_t(rows)),
        pool_(std::max(1, std::min({rows, 8, int(std::thread::hardware_concurrency())}))) {
    // every row's mask defined from the start: the greedy pick reads all rows (a frame row's pick is unused)
    cuda_check(cudaMemset(suppress_dev_.p, 0, size_t(rows) * runner.s.vocabulary));
    dec_.defer_head = true;  // the head only on steps with a text pick (validation/speech-speed-v1)
    for (int r = 0; r < rows; ++r) {
      streams_.push_back(std::make_unique<CodecStream>());
      codecs_.push_back(std::make_unique<lfm::codec::Decoder>(model_dir + "/audio_tokenizer", 12));
      codecs_.back()->set_stream(streams_.back()->s);
    }
  }

  int rows() const { return rows_; }
  bool busy(int r) const { return row_.at(size_t(r)).busy; }
  bool finished(int r) const { return row_.at(size_t(r)).finished; }
  bool cancelled(int r) const { return row_.at(size_t(r)).cancelled; }
  int free_row() const {
    for (int r = 0; r < rows_; ++r) if (!row_[size_t(r)].busy) return r;
    return -1;
  }

  // Starts `prompt` on the free row r: refused before any state changes (as Session::generate refused: an empty prompt;
  // a video or speech-output placeholder; <|audio|> spans that are not exactly the clips', <|image|> spans that are not
  // exactly the images'; continuing a codec stream the row's previous request did not leave open), then the prefill
  // (the media encoded in prompt order and injected in place of their placeholders) and the load into the row.
  void admit(int r, std::vector<int> prompt, Options o) {  // by value: the row keeps it without its media (C4)
    using namespace lfm::audio;
    if (r < 0 || r >= rows_ || row_[size_t(r)].busy) throw std::runtime_error("speech: admit to a busy or invalid row");
    Row& w = row_[size_t(r)];
    if (prompt.empty()) throw std::runtime_error("speech: empty prompt");
    const int audio_token = ModeTokens{}.placeholders[0];
    const int image_token = kImageToken;
    for (int id : prompt)
      for (int p : ModeTokens{}.placeholders)
        if (id == p && p != audio_token && p != image_token)
          throw std::runtime_error("speech: video or speech-output placeholder in the prompt (not supported as input)");
    {
      std::vector<size_t> clip_tokens, image_tokens;
      for (const auto& clip : o.audio) {
        if (clip.empty()) throw std::runtime_error("speech: empty audio clip");
        clip_tokens.push_back(lfm::audio_features::tokens(clip.size()));
      }
      if ((!o.image_keys.empty() && o.image_keys.size() != o.images.size()) ||
          (!o.image_rows.empty() && o.image_rows.size() != o.images.size()))
        throw std::runtime_error("speech: image keys or cached rows differ from the images");
      for (size_t k = 0; k < o.images.size(); ++k) {
        if (!o.images[k]) throw std::runtime_error("speech: an image is missing");
        const auto& im = *o.images[k];
        const ImageRows* cached = k < o.image_rows.size() ? o.image_rows[k].get() : nullptr;
        if (im.grid_t != 1 || im.grid_h <= 0 || im.grid_w <= 0 || im.grid_h % 2 || im.grid_w % 2 ||
            im.row_size != lfm3_vision::Encoder::kPatchDim || im.tokens != im.grid_h * im.grid_w / 4 ||
            (cached ? !im.values.empty() || cached->tokens != im.tokens || cached->grid_h != im.grid_h ||
                          cached->grid_w != im.grid_w
                    : im.values.size() != size_t(im.grid_h) * im.grid_w * size_t(im.row_size)))
          throw std::runtime_error("speech: an image is not the front end's pixel values");
        image_tokens.push_back(size_t(im.tokens));
      }
      check_audio_spans(prompt, clip_tokens);
      check_image_spans(prompt, image_tokens);
    }
    if (o.codec_continue && !w.stream_open) throw std::runtime_error("speech: no open codec stream to continue");
    if (o.record_hidden && rows_ != 1) throw std::runtime_error("speech: record_hidden needs a batch of one");
    // the encoder (and its checkpoint contract) before any state changes
    if (!o.audio.empty() && !encoder_) encoder_ = std::make_unique<lfm3_audio_in::Encoder>(runner_.weights, model_dir_);
    bool encode_images = false;
    for (size_t k = 0; k < o.images.size(); ++k) encode_images |= k >= o.image_rows.size() || !o.image_rows[k];
    if (encode_images && !vision_) {
      std::ifstream cf(model_dir_ + "/config.json");
      std::stringstream cs;
      cs << cf.rdbuf();
      if (!cf) throw std::runtime_error("speech: cannot read " + model_dir_ + "/config.json");
      vision_ = std::make_unique<lfm3_vision::Encoder>(runner_.weights, lfm::Json::parse(cs.str()));
    }
    w.stream_open = false;
    w.res = Result();
    lfm::audio::ModeOptions mo;
    mo.mode = o.mode, mo.max_new_tokens = o.max_new_tokens, mo.return_audio_eos = o.return_audio_eos;
    w.m.emplace(ModeTokens{}, mo);
    const int marker = w.m->begin(prompt.back());
    if (marker >= 0) prompt.push_back(marker);
    w.res.prompt = prompt;
    const auto t0 = Clock::now();
    runner_.reset();
    if (o.audio.empty() && o.images.empty()) {
      runner_.forward(prompt);
    } else {
      const auto e0 = Clock::now();
      size_t total = 0;
      for (const auto& clip : o.audio) total += lfm::audio_features::tokens(clip.size());
      for (const auto& im : o.images) total += size_t(im->tokens);
      if (audio_capacity_ < total)
        audio_rows_ = std::make_unique<Buffer<bf16>>(total * size_t(runner_.s.hidden)), audio_capacity_ = total;
      // each span's medium, in prompt order (the spans were checked above: one span per clip and per image)
      size_t at = 0, clip = 0, image = 0;
      for (size_t i = 0; i < prompt.size();) {
        const int id = prompt[i];
        if (id != audio_token && id != image_token) { ++i; continue; }
        bf16* dst = audio_rows_->p + at * runner_.s.hidden;
        if (id == audio_token) {
          lfm::audio_features::Dump dump;
          const auto mel = lfm::audio_features::log_mel(o.audio[clip++]);
          at += size_t(encoder_->encode(mel, dst, o.record_audio ? &dump : nullptr));
          if (o.record_audio) w.res.audio_dumps.push_back(std::move(dump));
        } else {
          const size_t k = image++;
          const auto& im = *o.images[k];
          const size_t bytes = size_t(im.tokens) * runner_.s.hidden * sizeof(bf16);
          if (k < o.image_rows.size() && o.image_rows[k]) {  // the cache's rows (validation/image-cache-v1)
            cuda_check(cudaMemcpy(dst, o.image_rows[k]->rows.p, bytes, cudaMemcpyDeviceToDevice));
            at += size_t(im.tokens);
          } else {
            const int n = vision_->encode(im.values.data(), im.grid_h, im.grid_w, dst);
            if (n != im.tokens) throw std::runtime_error("speech: the encoder's rows differ from the image's tokens");
            // kept for the next time the same bytes come (only after a whole encode; the rows are the encoder's own)
            if (image_cache_ && k < o.image_keys.size() && !o.image_keys[k].empty() &&
                o.image_keys[k].size() + bytes <= image_cache_->budget()) {
              // an optional cache: failing to keep never fails the request (C5). cudaMalloc's out-of-memory is not
              // sticky; anything sticky surfaces at the synchronize below.
              try {
#ifdef LFM3_FAULT_IMAGE_KEEP  // validation/image-cache-v1 C5's fault-injection build: every keep fails
                throw std::runtime_error("injected: image rows not kept");
#endif
                auto kept = std::make_shared<ImageRows>(im.grid_h, im.grid_w, n, size_t(n) * runner_.s.hidden);
                cuda_check(cudaMemcpy(kept->rows.p, dst, bytes, cudaMemcpyDeviceToDevice));
                image_cache_->put(o.image_keys[k], std::move(kept), bytes);
              } catch (const std::exception&) {
                (void)cudaGetLastError();
              }
            }
            at += size_t(n);
          }
        }
        while (i < prompt.size() && prompt[i] == id) ++i;
      }
      if (at != total || clip != o.audio.size() || image != o.images.size())  // unreachable
        throw std::runtime_error("speech: encoded media rows differ from the placeholders");
      cuda_check(cudaDeviceSynchronize());
      w.res.encode_seconds = secs(e0);
      runner_.forward(prompt, audio_rows_->p, std::vector<int>{audio_token, image_token});
    }
    dec_.load(r);
    try {  // the row is active in the decoder from here: a throw must not leave it decoding (audio-assurance-v1)
      if (!o.codec_continue) codecs_[size_t(r)]->reset();
      cuda_check(cudaDeviceSynchronize());
    } catch (...) {
      dec_.finish(r);
      throw;
    }
    w.t1 = Clock::now();
    w.res.prefill_seconds = std::chrono::duration<double>(w.t1 - t0).count();
    // the row keeps its options without the media: after the prefill nothing reads them, and kept they would hold
    // pixels, keys and cached GPU rows until the row is next used, outside every budget (validation/image-cache-v1 C4)
    o.images.clear(), o.image_keys.clear(), o.image_rows.clear(), o.audio.clear();
    o.images.shrink_to_fit(), o.image_keys.shrink_to_fit(), o.image_rows.shrink_to_fit(), o.audio.shrink_to_fit();
    w.o = std::move(o);
    w.sampler.emplace(FrameSamplingOptions{0.4, 1.2, 64, o.seed});
    w.pending.clear();
    w.forced_token = w.forced_frame = 0;
    w.last_suppressed.clear();
    w.mask_loaded = false;
    w.busy = true, w.finished = w.cancelled = false;
  }

  // One step of every busy, unfinished row. Returns the rows that finished in it (their results wait in take()).
  // A throw leaves every row closed: the caller treats the engine as faulted.
  std::vector<int> step() {
    try {
      return step_rows();
    } catch (...) {
      for (int r = 0; r < rows_; ++r) {
        Row& w = row_[size_t(r)];
        if (w.busy && !w.finished) dec_.finish(r);
        w.busy = w.finished = false, w.stream_open = false;
        codecs_[size_t(r)]->abandon();  // a chunk in flight is waited for and dropped, so the codec can be reset
      }
      dec_.embed_dump = nullptr;
      throw;
    }
  }

  // A finished row's result; frees the row (its codec stream stays open for a continuing chunk when not finalized).
  Result take(int r) {
    Row& w = row_.at(size_t(r));
    if (!w.busy || !w.finished) throw std::runtime_error("speech: take from a row that has not finished");
    w.busy = w.finished = false;
    return std::move(w.res);
  }

 private:
  using Clock = std::chrono::steady_clock;
  static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

  struct Row {
    bool busy = false, finished = false, cancelled = false, stream_open = false, mask_loaded = false;
    Options o;
    std::optional<lfm::audio::ModeMachine> m;
    std::optional<lfm::audio::FrameSampler> sampler;
    Result res;
    std::vector<int32_t> pending;  // emitted frames not yet given to the codec
    size_t forced_token = 0, forced_frame = 0;
    std::vector<int> last_suppressed;
    Clock::time_point t1;
    lfm::audio::Step st = lfm::audio::Step::Done;
    lfm::audio::Frame raw{};
    lfm::audio::FrameSampler::Draw draw;
    lfm::audio::Outcome out;
  };

  bool forced(const Row& w) const { return !w.o.forced_tokens.empty() || !w.o.forced_frames.empty(); }

  // The row's last call: codec flush (finalizing or keeping the stream open), the forced replay's check, timings.
  void finish_row(int r) {
    Row& w = row_[size_t(r)];
    dec_.finish(r);
    if (forced(w) && (w.forced_token != w.o.forced_tokens.size() || w.forced_frame != w.o.forced_frames.size()))
      throw std::runtime_error("speech: forced replay ended with tokens or frames unused");
    const size_t before = w.res.wav.size();
    codecs_[size_t(r)]->decode(w.pending.empty() ? nullptr : w.pending.data(), int(w.pending.size() / 12),
                               w.o.codec_finalize, w.res.wav);
    cuda_check(cudaDeviceSynchronize());
    w.pending.clear();
    w.stream_open = !w.o.codec_finalize;
    w.res.generate_seconds = std::chrono::duration<double>(Clock::now() - w.t1).count();
    if (w.o.on_audio && w.res.wav.size() > before) w.o.on_audio(w.res.wav.data() + before, w.res.wav.size() - before);
    w.finished = true;
  }

  std::vector<int> step_rows() {
    using namespace lfm::audio;
    std::vector<int> done, text_rows, frame_rows;
    for (int r = 0; r < rows_; ++r) {
      Row& w = row_[size_t(r)];
      if (!w.busy || w.finished) continue;
      w.st = w.m->next();
      if (w.st == Step::Done) {
        finish_row(r);
        done.push_back(r);
        continue;
      }
      if (w.o.cancelled && w.o.cancelled()) {  // the codec stream left closed
        dec_.finish(r);
        w.cancelled = w.finished = true, w.stream_open = false;
        w.pending.clear();
        done.push_back(r);
        continue;
      }
      (w.st == Step::Text ? text_rows : frame_rows).push_back(r);
    }
    if (text_rows.empty() && frame_rows.empty()) return done;

    // Text: one greedy pick over every row's logits, each row with its own suppression mask.
    std::vector<int> picks;
    bool need_pick = false;
    for (int r : text_rows) {
      Row& w = row_[size_t(r)];
      if (forced(w)) continue;
      need_pick = true;
      const auto sup = w.m->suppressed();
      if (!w.mask_loaded || sup != w.last_suppressed) {
        std::vector<unsigned char> mask(runner_.s.vocabulary, 0);
        for (int id : sup) mask[id] = 1;
        cuda_check(cudaMemcpy(suppress_dev_.p + size_t(r) * runner_.s.vocabulary, mask.data(), mask.size(),
                              cudaMemcpyHostToDevice));
        w.last_suppressed = sup, w.mask_loaded = true;
      }
    }
    if (need_pick) {
      const auto a = Clock::now();
      picks = dec_.select_greedy_rows(suppress_dev_.p);
      const double t = secs(a);
      for (int r : text_rows) if (!forced(row_[size_t(r)])) row_[size_t(r)].res.text_pick_seconds += t;
    }
    for (int r : text_rows) {
      Row& w = row_[size_t(r)];
      int token;
      if (forced(w)) {
        if (w.forced_token >= w.o.forced_tokens.size()) throw std::runtime_error("speech: forced tokens exhausted");
        token = w.o.forced_tokens[w.forced_token++];
      } else {
        token = picks[size_t(r)];
      }
      w.out = w.m->on_text(token);
      w.res.text_tokens.push_back(token);
      w.res.steps.push_back({false, w.out.decode, w.out.token, {}});
      if (w.out.emitted) {
        w.res.slots.push_back(w.out.slot);
        if (w.o.on_slot) w.o.on_slot(w.out.slot);
      }
    }

    // Frames: the batched depth decoder over every row that samples one, pass by pass.
    std::vector<int> sampled;
    for (int r : frame_rows) {
      Row& w = row_[size_t(r)];
      if (w.o.record_hidden) {
        std::vector<uint16_t> bits(runner_.s.hidden);
        cuda_check(cudaMemcpy(bits.data(), dec_.hidden_state(r), bits.size() * 2, cudaMemcpyDeviceToHost));
        w.res.hidden.push_back(std::move(bits));
      }
      if (forced(w)) {
        if (w.forced_frame >= w.o.forced_frames.size()) throw std::runtime_error("speech: forced frames exhausted");
        w.raw = w.o.forced_frames[w.forced_frame++];
      } else {
        sampled.push_back(r);
      }
    }
    if (!sampled.empty()) {
      const auto a = Clock::now();
      double in_depth = 0;
      const int n = int(sampled.size());
      std::vector<const bf16*> hidden;
      for (int r : sampled) hidden.push_back(dec_.hidden_state(r)), row_[size_t(r)].draw = row_[size_t(r)].sampler->begin();
      auto b = Clock::now();
      depth_.begin_frames(hidden.data(), n);
      in_depth += secs(b);
      std::vector<int32_t> codes(size_t(n) * kCodebooks);
      std::unique_ptr<bool[]> masked(new bool[size_t(n) * kCodebooks]), score(new bool[size_t(n) * kCodebooks]);
      while (!row_[size_t(sampled[0])].draw.done()) {  // every draw has the same 3 passes
        for (int i = 0; i < n; ++i) {
          const auto& d = row_[size_t(sampled[size_t(i)])].draw;
          for (int c = 0; c < kCodebooks; ++c)
            codes[size_t(i) * kCodebooks + c] = d.codes[size_t(c)], masked[size_t(i) * kCodebooks + c] = d.masked[c],
            score[size_t(i) * kCodebooks + c] = d.picked[c];
        }
        b = Clock::now();
        depth_.pass_frames(n, codes.data(), masked.get(), score.get(), logits_dev_.p);
        cuda_check(cudaMemcpy(host_logits_.p, logits_dev_.p, size_t(n) * kCodebooks * kCodes * sizeof(float),
                              cudaMemcpyDeviceToHost));
        in_depth += secs(b);
        pool_.run(n, [&](int i) {  // each row's sampler, draw and logits are its own
          Row& w = row_[size_t(sampled[size_t(i)])];
          w.sampler->take(w.draw, host_logits_.p + size_t(i) * kCodebooks * kCodes);
        });
      }
      const double total = secs(a);
      for (int r : sampled) {
        Row& w = row_[size_t(r)];
        w.raw = w.draw.codes;
        w.res.depth_seconds += in_depth;
        w.res.sample_host_seconds += total - in_depth;
      }
    }
    for (int r : frame_rows) {
      Row& w = row_[size_t(r)];
      w.res.raw_frames.push_back(w.raw);
      w.out = w.m->on_frame(w.raw);
      w.res.steps.push_back({true, w.out.decode, -1, w.out.fed});
      if (!w.out.eos) w.sampler->accept(w.out.fed);
      if (w.out.emitted) {
        w.res.slots.push_back(w.out.slot);
        if (w.o.on_slot) w.o.on_slot(w.out.slot);
        if (!w.out.eos) {  // the codec takes speech frames only
          w.res.frames.push_back(w.out.fed);
          w.pending.insert(w.pending.end(), w.out.fed.begin(), w.out.fed.end());
          if (w.pending.size() == 12 * 12) {  // runs beside the decode step below; collected after it
            const auto a = Clock::now();
            codecs_[size_t(r)]->submit(w.pending.data(), 12, false), w.pending.clear();
            w.res.codec_seconds += secs(a);
          }
        }
      }
    }

    // One decode step for every row that feeds an input; a row whose step is not decoded (its last) leaves the batch.
    std::vector<int> tokens(size_t(rows_), 0), frames(size_t(rows_) * 12, -1);
    std::vector<int> stepping;
    bool any_frame = false;
    for (int r : text_rows) {
      Row& w = row_[size_t(r)];
      if (w.out.decode) tokens[size_t(r)] = w.out.token, stepping.push_back(r);
      else dec_.finish(r);
    }
    for (int r : frame_rows) {
      Row& w = row_[size_t(r)];
      if (w.out.decode) {
        tokens[size_t(r)] = ModeTokens{}.audio_out, stepping.push_back(r), any_frame = true;
        for (int c = 0; c < 12; ++c) frames[size_t(r) * 12 + c] = w.out.fed[size_t(c)];
      } else {
        dec_.finish(r);
      }
    }
    if (!stepping.empty()) {
      if (rows_ == 1) dec_.embed_dump = row_[0].o.record_hidden ? &row_[0].res.embeds : nullptr;
      const auto a = Clock::now();
      dec_.step(tokens, any_frame ? &frames : nullptr);
      cuda_check(cudaDeviceSynchronize());
      const double t = secs(a);
      dec_.embed_dump = nullptr;
      for (int r : stepping) row_[size_t(r)].res.lm_step_seconds += t, ++row_[size_t(r)].res.decode_steps;
    }
    // The codec chunks submitted this step: their audio, in each row's own event order (after the row's slot).
    for (int r : frame_rows) {
      Row& w = row_[size_t(r)];
      if (!codecs_[size_t(r)]->pending()) continue;
      const auto a = Clock::now();
      const size_t before = w.res.wav.size();
      codecs_[size_t(r)]->collect(w.res.wav);
      w.res.codec_seconds += secs(a);
      if (w.o.on_audio && w.res.wav.size() > before) w.o.on_audio(w.res.wav.data() + before, w.res.wav.size() - before);
    }
    return done;
  }

  std::string model_dir_;
  std::unique_ptr<lfm3_audio_in::Encoder> encoder_;
  std::unique_ptr<lfm3_vision::Encoder> vision_;
  ImageCache* image_cache_ = nullptr;
  std::unique_ptr<Buffer<bf16>> audio_rows_;
  size_t audio_capacity_ = 0;
  Runner& runner_;
  int rows_;
  lfm3_batch::BatchDecoder dec_;
  lfm::depth::Decoder depth_;
  std::vector<std::unique_ptr<CodecStream>> streams_;  // destroyed after the codecs that use them
  std::vector<std::unique_ptr<lfm::codec::Decoder>> codecs_;
  lfm::fp32::DeviceArray<float> logits_dev_;
  Buffer<unsigned char> suppress_dev_;
  lfm::codec::PinnedArray<float> host_logits_;
  std::vector<Row> row_;
  RowPool pool_;
};

// One request at a time: a batch of one.
class Session {
 public:
  Session(Runner& runner, const std::string& model_dir) : batch_(runner, model_dir, 1) {}
  void set_image_cache(ImageCache* cache) { batch_.set_image_cache(cache); }

  Result generate(std::vector<int> prompt, Options o) {
    batch_.admit(0, std::move(prompt), std::move(o));
    while (batch_.step().empty()) {}
    const bool cancelled = batch_.cancelled(0);
    Result r = batch_.take(0);
    if (cancelled) throw Cancelled();
    return r;
  }

 private:
  Batch batch_;
};

// (run_chunks and wav_bytes, used by lfm3-speech and liquid-lfm3-server alike, are in lfm/speech_run.hpp)
}  // namespace lfm3_speech
