// lfm3-speech: text in -> text and speech out, natively (#2 step 4; src/lfm3_speech.cuh).
//
//   lfm3-speech MODEL_DIR REQUEST.json OUT_STEM
//   lfm3-speech MODEL_DIR --batch LIST   (each line: REQUEST.json OUT_STEM; the model is loaded once per capacity)
//   lfm3-speech MODEL_DIR --concurrent ROWS LIST [STAGGER_SEED]
//      concurrent speech (validation/speech-batch-v1): every request of LIST runs in one batch of ROWS rows, a row
//      refilled as its request finishes (each request's chunks run in its row), the outputs as a lone run writes
//      them. STAGGER_SEED: the requests join in a random order, each 0-60 steps after the one before. Every request
//      must have the same capacity; record_hidden is refused (it needs a batch of one). Prints the aggregate audio
//      seconds per wall second of the batch loop.
// REQUEST: {"prompt_ids": [...], "mode": "audio" | "interleaved" | "text" | "none", "max_new_tokens": N, "seed": S,
//           "capacity": C (default 4096), "forced_tokens": [...], "forced_frames": [[12 codes]...],
//           "record_hidden": false, "chunks": [[ids...], ...]}
//      or, rendered natively: {"messages": [...], "split": false, ...}: the chat (lfm/lfm3_text.hpp, the text
//      binary's "messages" contract) instead of prompt_ids; "normalize": true (audio mode) reads numbers and
//      abbreviations in the user's text as words first (lfm/speech_normalize.hpp); with "split": true, [system, user] or [user] is read
//      sentence by sentence (lfm/speech_text.hpp), each sentence one chunk as below, and the chunk json adds "text".
// "chunks" (long-form TTS by sentences): each entry is one sentence's whole prompt, generated as its own request
// with seed SEED * 1000 + k; the codec stream runs on across them, so OUT_STEM.wav is one stream. The outputs
// concatenate the chunks (prompt: the first chunk's), and the json adds "chunks": [{frames, slots, ended_by_eos}].
// A chunk "ended_by_eos" as tools/lfm3_speech_chunked_hf.py decides it: its last sampled frame is the EOS frame and
// its last slot is <|im_end|>. "slots" follow HF's default return_audio_eos=False (an EOS frame emits no slot); the
// chunked HF tool passes True, so its summary counts one more slot per EOS frame.
// Refused before the model loads or anything is generated: an unknown mode, an empty prompt or chunk, and a prompt
// that cannot fit max_new_tokens in the capacity. Earlier outputs of OUT_STEM are removed first, so a failed run
// leaves none of them behind looking current; the exit status is nonzero on any failure, including a failed write.
// OUT_STEM.json: prompt (with the mode marker), slots, text tokens, frame counts, timings (real_time_factor: audio
// over generate_seconds, the decode loop; real_time_factor_end_to_end adds the prefills);
// OUT_STEM.frames.i32 (emitted frames, [frame][12]); OUT_STEM.raw.i32 (sampled frames, EOS frame included);
// OUT_STEM.wav (24 kHz 16-bit PCM, for listening) and OUT_STEM.f32 (the exact FP32 waveform);
// OUT_STEM.hidden.bf16 when record_hidden ([frame step][2048], the state the depth decoder read), and
// OUT_STEM.embeds.bf16 ([decoded step][2048], the input embedding fed). steps: [frame, decoded, token, fed codes...].
// Audio input (#3): "audio": [WAV paths] (16 kHz mono, 16-bit PCM or 32-bit float), one per <|audio|> span of
// prompt_ids in order (tools/lfm3_audio_in_hf.py writes such requests); or, with messages, user content parts
// {"type": "audio", "audio": WAV path}, rendered and expanded natively. Not with chunks or split. "record_audio": true
// also writes OUT_STEM.audioK.{mel,down,encoded,features}.f32, clip K's stages (lfm/audio_encoder.cuh).
// Image input (#4): "images": [image paths] (PNG or JPEG, read by the native front end: lfm/image_decode.hpp then
// lfm/image_preprocess.hpp), one per <|image|> span of prompt_ids in order; or, with messages, user content parts
// {"type": "image", "image": path}, expanded natively (audio and image parts may interleave). Not with chunks or split.
// "record_image": true also writes OUT_STEM.imageK.pixel_values.f32, image K's front-end output ([rows][1536]).
#define LFM3_RUNTIME_ONLY 1
#include "../src/lfm3_main.cu"
#include "../src/lfm3_batch.cuh"
#include "../src/lfm3_speech.cuh"
#include <lfm/image_decode.hpp>
#include <lfm/image_preprocess.hpp>
#include <lfm/speech_text.hpp>
#include <lfm/wav.hpp>

#include <chrono>
#include <cstdio>
#include <optional>
#include <random>

namespace {
void write_wav(const std::string& path, const std::vector<float>& x, int rate) {
  const auto bytes = lfm3_speech::wav_bytes(x, rate);
  std::ofstream f(path, std::ios::binary);
  f.write(bytes.data(), std::streamsize(bytes.size()));
  if (!f) throw std::runtime_error("cannot write " + path);
}
template <class T>
void write_raw(const std::string& path, const T* p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(p), std::streamsize(n * sizeof(T)));
  if (!f) throw std::runtime_error("cannot write " + path);
}
}  // namespace

// The model, loaded once per capacity (a batch reuses it across requests of the same capacity).
struct Loaded {
  int capacity = -1;
  std::unique_ptr<lfm3_speech::Session> session;
  std::unique_ptr<Runner> runner;
  lfm::ModelConfig cfg;
};

// A request read and rendered, ready to generate.
struct Job {
  std::string stem, mode;
  lfm3_speech::Options o;
  std::vector<std::vector<int>> prompts;
  std::vector<std::string> chunk_texts;
  std::shared_ptr<const lfm::lfm3_text::Tokenizer> text_tokenizer;
  bool chunked = false, record_image = false;
  double front_end_seconds = 0;  // image input: reading, decoding and preprocessing the images (CPU)
  int capacity = 4096;
};
void write_outputs(const Job& job, const lfm3_speech::Result& r, const std::vector<lfm3_speech::ChunkInfo>& chunk_info);

// Reads, checks and renders a request (loading the model for its capacity when it differs); removes OUT_STEM's
// earlier outputs.
Job prepare(const char* model_dir, const char* request_path, const std::string& stem, Loaded& m, bool one_capacity = false) {
  char* argv[] = {nullptr, const_cast<char*>(model_dir), const_cast<char*>(request_path)};
  const auto req = read_json(argv[2]);
  const std::string mode = req.has("mode") ? req.at("mode").string() : "audio";
  if (mode != "audio" && mode != "interleaved" && mode != "text" && mode != "none")
    throw std::runtime_error("mode must be audio, interleaved, text or none");
  const auto outputs = {".wav", ".f32", ".frames.i32", ".raw.i32", ".hidden.bf16", ".embeds.bf16", ".json"};
  // inputs the outputs must not be: the request (read already) and its audio clips (read later); removing one as an
  // "earlier output" would lose it
  std::vector<std::string> inputs{argv[2]};
  if (req.has("audio"))
    for (auto& path : req.at("audio").array()) inputs.push_back(path.string());
  if (req.has("messages"))
    if (const auto* list = std::get_if<lfm::Json::Array>(&req.at("messages").value))
      for (const auto& m : *list)
        if (const auto* o = std::get_if<lfm::Json::Object>(&m.value); o && o->count("content"))
          if (const auto* parts = std::get_if<lfm::Json::Array>(&o->at("content").value))
            for (const auto& part : *parts)
              if (const auto* po = std::get_if<lfm::Json::Object>(&part.value))
                for (const char* key : {"audio", "image"})
                  if (po->count(key))
                    if (const auto* a = std::get_if<std::string>(&po->at(key).value)) inputs.push_back(*a);
  if (req.has("images"))
    for (auto& path : req.at("images").array()) inputs.push_back(path.string());
  for (const char* ext : outputs)
    for (const auto& in : inputs)
      if (std::filesystem::weakly_canonical(stem + ext) == std::filesystem::weakly_canonical(in))
        throw std::runtime_error("an input (" + in + ") is one of OUT_STEM's outputs; use another stem");
  for (const char* ext : outputs) std::filesystem::remove(stem + ext);
  std::vector<std::string> media_outputs;
  for (int k = 0; k < 64; ++k) {
    for (const char* part : {"mel", "down", "encoded", "features"})
      media_outputs.push_back(stem + ".audio" + std::to_string(k) + "." + part + ".f32");
    media_outputs.push_back(stem + ".image" + std::to_string(k) + ".pixel_values.f32");
  }
  for (const auto& out : media_outputs)
    for (const auto& in : inputs)
      if (std::filesystem::weakly_canonical(out) == std::filesystem::weakly_canonical(in))
        throw std::runtime_error("an input (" + in + ") is one of OUT_STEM's outputs; use another stem");
  for (const auto& out : media_outputs) std::filesystem::remove(out);
  const int capacity = req.has("capacity") ? req.at("capacity").integer() : 4096;
  if (m.capacity != capacity) {
    // --concurrent: the runner is shared by every job prepared so far; a job of another capacity is refused before it
    // replaces the runner (audio-assurance-v1 red-team)
    if (one_capacity && m.runner) throw std::runtime_error("--concurrent needs one capacity");
    m.session.reset(), m.runner.reset();
    m.cfg = lfm::ModelConfig::load(std::string(argv[1]) + "/config.json", capacity);
    m.runner = std::make_unique<Runner>(argv[1], m.cfg);
    m.capacity = capacity;
  }
  const auto& cfg = m.cfg;
  Job job;
  job.stem = stem, job.mode = mode, job.capacity = capacity;
  lfm3_speech::Options& o = job.o;
  using lfm::audio::Requested;
  o.mode = mode == "audio" ? Requested::Audio : mode == "interleaved" ? Requested::Interleaved
         : mode == "text" ? Requested::Text : Requested::None;
  o.max_new_tokens = req.at("max_new_tokens").integer();
  o.seed = req.has("seed") ? uint64_t(req.at("seed").integer()) : 0;
  o.record_hidden = req.has("record_hidden") && req.at("record_hidden").boolean();
  o.record_audio = req.has("record_audio") && req.at("record_audio").boolean();
  job.record_image = req.has("record_image") && req.at("record_image").boolean();
  auto read_clips = [&](const std::vector<std::string>& paths) {
    if (paths.size() > 64) throw std::runtime_error("at most 64 audio clips");
    for (const auto& path : paths) {
      auto a = lfm::wav::read(path);
      if (a.rate != lfm::audio_features::kRate)
        throw std::runtime_error(path + ": expected 16000 Hz audio, got " + std::to_string(a.rate));
      if (a.samples.empty()) throw std::runtime_error(path + ": no samples");
      o.audio.push_back(std::move(a.samples));
    }
  };
  auto read_image = [&](const std::string& path) {
    if (o.images.size() >= 64) throw std::runtime_error("at most 64 images");
    const auto t0 = std::chrono::steady_clock::now();
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
      o.images.push_back(std::make_shared<const lfm::image::PixelValues>(lfm::image::preprocess(lfm::image::load_rgb(bytes.data(), bytes.size()))));
    } catch (const std::exception& e) {
      throw std::runtime_error(path + ": " + e.what());
    }
    job.front_end_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  if (req.has("audio")) {
    if (req.has("messages")) throw std::runtime_error("with messages, audio clips are the messages' audio parts");
    std::vector<std::string> paths;
    for (auto& path : req.at("audio").array()) paths.push_back(path.string());
    read_clips(paths);
  }
  if (req.has("images")) {
    if (req.has("messages")) throw std::runtime_error("with messages, images are the messages' image parts");
    for (auto& path : req.at("images").array()) read_image(path.string());
  }
  if (req.has("forced_tokens")) for (auto& x : req.at("forced_tokens").array()) o.forced_tokens.push_back(x.integer());
  if (req.has("forced_frames"))
    for (auto& f : req.at("forced_frames").array()) {
      lfm::audio::Frame fr{};
      const auto& a = f.array();
      if (a.size() != 12) throw std::runtime_error("a forced frame needs 12 codes");
      for (int c = 0; c < 12; ++c) fr[c] = a[c].integer();
      o.forced_frames.push_back(fr);
    }
  auto& prompts = job.prompts;
  auto& chunk_texts = job.chunk_texts;
  auto& text_tokenizer = job.text_tokenizer;  // with messages: "text" in the json, as the server decodes it (slots
                                              // without <|audio_out|>, special tokens skipped)
  const bool split = req.has("split") && req.at("split").boolean();
  // numbers and abbreviations in words before the split (lfm/speech_normalize.hpp); default off here, so the validated
  // requests run as they did (the server defaults it on in audio mode)
  const bool normalize = req.has("normalize") && req.at("normalize").boolean();
  if (normalize && (mode != "audio" || !req.has("messages")))
    throw std::runtime_error("normalize is for audio mode with messages");
  const bool chunked = job.chunked = req.has("chunks") || split;
  if (chunked && (!o.forced_tokens.empty() || !o.forced_frames.empty())) throw std::runtime_error("chunks cannot be forced");
  if (req.has("messages")) {
    if (req.has("prompt_ids") || req.has("chunks")) throw std::runtime_error("messages cannot be combined with prompt_ids or chunks");
    const std::string path = std::string(argv[1]) + "/tokenizer.json";
    const std::shared_ptr<const lfm::lfm3_text::Tokenizer> tok =
        lfm::lfm3_text::load_checked(path, lfm::tokenizer_token_count(path, cfg.shape.vocabulary));
    text_tokenizer = tok;
    auto messages = lfm::lfm3_text::parse_messages(req.at("messages"), /*allow_audio=*/!split, /*allow_image=*/!split);
    if (normalize) lfm::speech_text::normalize_user_text(messages);
    std::vector<std::string> paths;  // audio parts: WAV paths
    for (const auto& m : messages)
      for (const auto& part : m.parts)
        if (part.audio) paths.push_back(part.data);
    read_clips(paths);
    std::vector<size_t> clip_tokens;
    for (const auto& clip : o.audio) clip_tokens.push_back(lfm::audio_features::tokens(clip.size()));
    // with images, every part's medium in prompt order (expand_media); audio alone stays expand_audio's
    std::vector<lfm::lfm3_text::MediaTokens> media;
    size_t clip_at = 0;
    for (const auto& m : messages)
      for (const auto& part : m.parts)
        if (part.audio) {
          media.push_back({lfm::lfm3_text::kAudioToken, clip_tokens.at(clip_at++)});
        } else if (part.image) {
          read_image(part.data);
          media.push_back({lfm::lfm3_text::kImageToken, size_t(o.images.back()->tokens)});
        }
    const auto chats = split ? lfm::speech_text::sentence_requests(messages)
                             : std::vector<std::vector<lfm::lfm3_text::Message>>{messages};
    for (const auto& c : chats) {
      prompts.push_back(lfm::lfm3_text::encode_messages(*tok, c, capacity, o.max_new_tokens));
      if (!o.images.empty())
        prompts.back() = lfm::lfm3_text::expand_media(prompts.back(), media, capacity, o.max_new_tokens);
      else if (!clip_tokens.empty())
        prompts.back() = lfm::lfm3_text::expand_audio(prompts.back(), clip_tokens, capacity, o.max_new_tokens);
      if (split) chunk_texts.push_back(c.back().content);
    }
  } else if (split) {
    throw std::runtime_error("split needs messages");
  } else if (req.has("chunks")) {
    for (auto& c : req.at("chunks").array()) {
      prompts.emplace_back();
      for (auto& x : c.array()) prompts.back().push_back(x.integer());
    }
    if (prompts.empty()) throw std::runtime_error("chunks is empty");
  } else {
    prompts.emplace_back();
    for (auto& x : req.at("prompt_ids").array()) prompts.back().push_back(x.integer());
  }
  for (const auto& p : prompts) {  // (the messages path has checked this already, in encode_messages)
    if (p.empty()) throw std::runtime_error("empty prompt");
    if (o.max_new_tokens < 1 || p.size() + size_t(o.max_new_tokens) > size_t(capacity))
      throw std::runtime_error("prompt (" + std::to_string(p.size()) + ") + max_new_tokens exceeds capacity (" +
                               std::to_string(capacity) + ")");
  }
  return job;
}

void run(const char* model_dir, const char* request_path, const std::string& stem, Loaded& m) {
  const Job job = prepare(model_dir, request_path, stem, m);
  if (!m.session) m.session = std::make_unique<lfm3_speech::Session>(*m.runner, model_dir);
  std::vector<lfm3_speech::ChunkInfo> chunk_info;
  const auto r = lfm3_speech::run_chunks(*m.session, job.prompts, job.o, job.chunked, chunk_info);
  write_outputs(job, r, chunk_info);
}

void write_outputs(const Job& job, const lfm3_speech::Result& r, const std::vector<lfm3_speech::ChunkInfo>& chunk_info) {
  const auto& stem = job.stem;
  const auto& o = job.o;
  const auto& mode = job.mode;
  const bool chunked = job.chunked;
  const auto& chunk_texts = job.chunk_texts;
  const auto& text_tokenizer = job.text_tokenizer;
  write_wav(stem + ".wav", r.wav, 24000);
  write_raw(stem + ".f32", r.wav.data(), r.wav.size());
  std::vector<int32_t> fr, raw;
  for (auto& f : r.frames) fr.insert(fr.end(), f.begin(), f.end());
  for (auto& f : r.raw_frames) raw.insert(raw.end(), f.begin(), f.end());
  write_raw(stem + ".frames.i32", fr.data(), fr.size());
  write_raw(stem + ".raw.i32", raw.data(), raw.size());
  if (o.record_hidden) {
    std::vector<uint16_t> h;
    for (auto& x : r.hidden) h.insert(h.end(), x.begin(), x.end());
    write_raw(stem + ".hidden.bf16", h.data(), h.size());
    std::vector<uint16_t> e;
    for (auto& x : r.embeds) e.insert(e.end(), x.begin(), x.end());
    write_raw(stem + ".embeds.bf16", e.data(), e.size());
  }
  for (size_t k = 0; k < r.audio_dumps.size(); ++k) {
    const auto& d = r.audio_dumps[k];
    const std::string a = stem + ".audio" + std::to_string(k) + ".";
    write_raw(a + "mel.f32", d.mel.data(), d.mel.size());
    write_raw(a + "down.f32", d.down.data(), d.down.size());
    write_raw(a + "encoded.f32", d.encoded.data(), d.encoded.size());
    write_raw(a + "features.f32", d.features.data(), d.features.size());
  }
  if (job.record_image)
    for (size_t k = 0; k < o.images.size(); ++k)
      write_raw(stem + ".image" + std::to_string(k) + ".pixel_values.f32", o.images[k]->values.data(),
                o.images[k]->values.size());
  std::ofstream f(stem + ".json");
  f << "{\"mode\":" << lfm::lfm3_text::json_string(mode) << ",\"prompt\":";
  print_ids(f, r.prompt);
  f << ",\"slots\":"; print_ids(f, r.slots);
  f << ",\"text_tokens\":"; print_ids(f, r.text_tokens);
  f << ",\"steps\":[";
  for (size_t i = 0; i < r.steps.size(); ++i) {
    const auto& s = r.steps[i];
    f << (i ? "," : "") << "[" << (s.frame ? 1 : 0) << "," << (s.decode ? 1 : 0) << "," << s.token;
    if (s.frame) for (int c : s.fed) f << "," << c;
    f << "]";
  }
  f << "]";
  if (chunked) {
    f << ",\"chunks\":[";
    for (size_t k = 0; k < chunk_info.size(); ++k)
      f << (k ? "," : "") << "{\"frames\":" << chunk_info[k].frames << ",\"slots\":" << chunk_info[k].slots
        << ",\"ended_by_eos\":" << (chunk_info[k].ended_by_eos ? "true" : "false")
        << (chunk_texts.empty() ? "" : ",\"text\":" + lfm::lfm3_text::json_string(chunk_texts[k])) << "}";
    f << "]";
  }
  if (text_tokenizer) {
    std::vector<int> text_slots;
    for (int id : r.slots)
      if (id != lfm::audio::ModeTokens{}.audio_out) text_slots.push_back(id);
    f << ",\"text\":" << lfm::lfm3_text::json_string(text_tokenizer->decode(text_slots, true));
  }
  f << ",\"frames\":" << r.frames.size() << ",\"raw_frames\":" << r.raw_frames.size() << ",\"samples\":" << r.wav.size()
    << ",\"audio_seconds\":" << r.wav.size() / 24000.0 << ",\"decode_steps\":" << r.decode_steps
    << ",\"encode_seconds\":" << r.encode_seconds << ",\"prefill_seconds\":" << r.prefill_seconds << ",\"generate_seconds\":" << r.generate_seconds
    << (o.images.empty() ? std::string() : ",\"front_end_seconds\":" + std::to_string(job.front_end_seconds))
    << ",\"lm_step_seconds\":" << r.lm_step_seconds << ",\"text_pick_seconds\":" << r.text_pick_seconds
    << ",\"depth_seconds\":" << r.depth_seconds << ",\"sample_host_seconds\":" << r.sample_host_seconds
    << ",\"codec_seconds\":" << r.codec_seconds
    << ",\"real_time_factor\":" << (r.wav.empty() ? 0.0 : (r.wav.size() / 24000.0) / r.generate_seconds)
    << ",\"real_time_factor_end_to_end\":"
    << (r.wav.empty() ? 0.0 : (r.wav.size() / 24000.0) / (r.prefill_seconds + r.generate_seconds)) << "}\n";
  f.close();
  if (!f) throw std::runtime_error("cannot write " + stem + ".json");
  std::cout << "{\"frames\":" << r.frames.size() << ",\"audio_seconds\":" << r.wav.size() / 24000.0
            << ",\"generate_seconds\":" << r.generate_seconds << ",\"slots\":" << r.slots.size() << "}\n";
}

// --concurrent: every request of LIST through one batch of `rows` rows; see the header.
int concurrent(const char* model_dir, int rows, const char* list_path, const char* stagger_seed) {
  std::ifstream list(list_path);
  if (!list) throw std::runtime_error(std::string("cannot read ") + list_path);
  Loaded m;
  std::vector<Job> jobs;
  std::string request, stem;
  int failed = 0, done = 0;
  while (list >> request >> stem) {
    try {
      jobs.push_back(prepare(model_dir, request.c_str(), stem, m, /*one_capacity=*/true));
    } catch (const std::exception& e) {
      std::cerr << "lfm3-speech: " << request << ": " << e.what() << "\n";
      ++failed;
    }
  }
  if (jobs.empty()) throw std::runtime_error("no request to run");
  // arrival order and gaps (in steps): the list's order at once, or shuffled with random gaps
  std::vector<size_t> order(jobs.size());
  std::vector<long> gap(jobs.size(), 0);
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  if (stagger_seed) {
    std::mt19937_64 rng(std::stoull(stagger_seed));
    std::shuffle(order.begin(), order.end(), rng);
    for (auto& g : gap) g = long(rng() % 61);
  }
  lfm3_speech::Batch batch(*m.runner, model_dir, rows);
  struct Running { size_t job; std::unique_ptr<lfm3_speech::ChunkRun> run; };
  std::vector<std::optional<Running>> lanes(static_cast<size_t>(rows));
  const size_t budget = 0;  // as lone lfm3-speech: no step budget
  size_t next = 0;
  long steps = 0, since = 0;
  double audio = 0;
  const auto t0 = std::chrono::steady_clock::now();
  auto start_chunk = [&](int r) {  // the row's next chunk, or its request written and the row freed
    auto& run = *lanes[size_t(r)]->run;
    const std::vector<int>* prompt = nullptr;
    lfm3_speech::Options o;
    if (run.next(prompt, o)) { batch.admit(r, *prompt, o); return; }
    const Job& job = jobs[lanes[size_t(r)]->job];
    auto result = run.take();
    audio += result.wav.size() / 24000.0;
    write_outputs(job, result, run.info());
    std::cout << "{\"done\":" << lfm::lfm3_text::json_string(job.stem) << ",\"row\":" << r << ",\"step\":" << steps
              << ",\"audio_seconds\":" << result.wav.size() / 24000.0 << "}\n" << std::flush;
    lanes[size_t(r)].reset();
    ++done;
  };
  for (;;) {
    // admit arrivals into free rows (one per step boundary while their gaps allow)
    while (next < order.size() && since >= gap[next]) {
      const int r = [&] { for (int i = 0; i < rows; ++i) if (!lanes[size_t(i)] && !batch.busy(i)) return i; return -1; }();
      if (r < 0) break;
      const size_t j = order[next++];
      since = 0;
      try {
        lanes[size_t(r)] = Running{j, std::make_unique<lfm3_speech::ChunkRun>(jobs[j].prompts, jobs[j].o, jobs[j].chunked, budget)};
        start_chunk(r);
      } catch (const std::exception& e) {
        std::cerr << "lfm3-speech: " << jobs[j].stem << ": " << e.what() << "\n";
        lanes[size_t(r)].reset();
        ++failed;
      }
    }
    bool any = false;
    for (int r = 0; r < rows; ++r) any |= bool(lanes[size_t(r)]);
    if (!any && next >= order.size()) break;
    if (!any) { ++since; continue; }  // waiting for the next arrival's gap
    for (int r : batch.step()) {
      if (batch.cancelled(r)) throw std::runtime_error("unexpected cancellation");
      lanes[size_t(r)]->run->done(batch.take(r));
      start_chunk(r);
    }
    ++steps, ++since;
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "{\"rows\":" << rows << ",\"requests\":" << done << ",\"steps\":" << steps << ",\"audio_seconds\":" << audio
            << ",\"wall_seconds\":" << wall << ",\"audio_per_wall_second\":" << audio / wall << "}\n";
  std::cerr << "lfm3-speech: concurrent: " << done << " done, " << failed << " failed\n";
  return failed ? 1 : 0;
}

int main(int argc, char** argv) try {
  if ((argc == 5 || argc == 6) && std::string(argv[2]) == "--concurrent") {
    const int rows = std::stoi(argv[3]);
    // 16: the most rows the engine check validates (the language model's tile table changes above 16)
    if (rows < 1 || rows > 16) throw std::runtime_error("--concurrent ROWS must be 1..16");
    return concurrent(argv[1], rows, argv[4], argc == 6 ? argv[5] : nullptr);
  }
  Loaded m;
  if (argc == 4 && std::string(argv[2]) == "--batch") {
    // each line of LIST: REQUEST.json OUT_STEM; a failed request prints its error and the batch goes on; the exit
    // status is nonzero if any failed
    std::ifstream list(argv[3]);
    if (!list) throw std::runtime_error(std::string("cannot read ") + argv[3]);
    std::string request, stem;
    int failed = 0, done = 0;
    while (list >> request >> stem) {
      try {
        run(argv[1], request.c_str(), stem, m);
        ++done;
      } catch (const std::exception& e) {
        std::cerr << "lfm3-speech: " << request << ": " << e.what() << "\n";
        ++failed;
      }
    }
    std::cerr << "lfm3-speech: batch: " << done << " done, " << failed << " failed\n";
    return failed ? 1 : 0;
  }
  if (argc != 4) throw std::runtime_error("usage: lfm3-speech MODEL_DIR REQUEST.json OUT_STEM | MODEL_DIR --batch LIST");
  run(argv[1], argv[2], argv[3], m);
  return 0;
} catch (const std::exception& e) {
  std::cerr << "lfm3-speech: " << e.what() << "\n";
  return 1;
}
